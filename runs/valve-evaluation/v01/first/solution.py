"""Valve MPC. Run from this directory to select weights and replay validation."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import process


def _prediction(x, d, dt, hold):
    """Build y = free + gain @ moves using the sampled plant update order."""
    n, m = hold.shape
    retention = np.exp(-dt / .75)
    a = np.array([[1 - .8 * dt, 1.1 * dt], [0., retention]])
    b = np.array([0., 1 - retention])
    free = np.empty(n)
    gain = np.empty((n, m))
    state = x.copy()
    sensitivity = np.zeros((2, m))
    for k in range(n):
        next_state = a @ state + np.array([dt * (.3 * d[k] + .09), 0.])
        next_sensitivity = a @ sensitivity + np.outer(b, hold[k])
        free[k] = next_state[0] + .12 * state[1]
        gain[k] = next_sensitivity[0] + .12 * sensitivity[1] + .05 * hold[k]
        state, sensitivity = next_state, next_sensitivity
    return free, gain


def plan(x, r, d, last, cfg):
    """Plan N commands. Infeasible hard limits raise ValueError.

    A numerical solver failure returns a feasible hold sequence with
    success=False and a diagnostic status string.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    n = len(r)
    m = int(cfg["M"])
    if x.shape != (2,) or r.shape != (n,) or d.shape != (n,):
        raise ValueError("Expected two states and equal, one-dimensional previews")
    if int(cfg.get("N", n)) != n or not 1 <= m <= n:
        raise ValueError("Require N preview samples and 1 <= M <= N")
    dt, rate, lo, hi = (float(cfg[key]) for key in ("dt", "rate", "lo", "hi"))
    q, ru, s = (float(cfg[key]) for key in ("q", "r_u", "s"))
    last = float(last)
    if not np.all(np.isfinite(np.r_[x, r, d, last, dt, rate, lo, hi, q, ru, s])):
        raise ValueError("Nonfinite controller inputs")
    if dt <= 0 or min(rate, q, ru, s) < 0 or lo > hi:
        raise ValueError("Invalid sampling time, weights or limits")
    du = dt * rate
    first_lo, first_hi = max(lo, last - du), min(hi, last + du)
    if first_lo > first_hi:
        raise ValueError("Infeasible first command: amplitude and slew intervals are disjoint")

    # Last column stays active for every step after the last independent move.
    indices = np.minimum(np.arange(n), m - 1)
    hold = np.eye(m)[indices]
    delta = hold.copy()
    delta[1:] -= hold[:-1]
    previous = np.zeros(n)
    previous[0] = last
    free, gain = _prediction(x, d, dt, hold)
    # Tracking, commands and increments form a single weighted least squares QP.
    mat = np.vstack((np.sqrt(q) * gain, np.sqrt(ru) * hold, np.sqrt(s) * delta))
    target = np.r_[np.sqrt(q) * (r - free), np.zeros(n), np.sqrt(s) * previous]
    initial = np.full(m, np.clip(last, first_lo, first_hi))
    moves = initial
    success, status = True, "optimal"
    if du > 0 and lo < hi:
        lower, upper = np.full(m, lo), np.full(m, hi)
        lower[0], upper[0] = first_lo, first_hi
        constraints = []
        if m > 1:
            constraints.append(LinearConstraint(delta[1:m], -du, du))
        gram = mat.T @ mat
        linear = mat.T @ target
        scale = max(float(np.max(np.abs(gram))), float(np.max(np.abs(linear))),
                    np.finfo(float).tiny)
        gram, linear = gram / scale, linear / scale

        def objective(v):
            gv = gram @ v
            return v @ (gv - 2 * linear), 2 * (gv - linear)

        try:
            result = minimize(objective, initial, jac=True, method="SLSQP",
                              bounds=Bounds(lower, upper), constraints=constraints,
                              options={"ftol": 1e-13, "maxiter": 400})
            candidate = result.x
            tolerance = 1e-8 * max(1., abs(lo), abs(hi), du)
            if (result.success and np.all(np.isfinite(candidate))
                    and np.all(candidate >= lower - tolerance)
                    and np.all(candidate <= upper + tolerance)
                    and np.all(np.abs(np.diff(candidate)) <= du + tolerance)):
                # Remove any solver roundoff at the hard limits before execution.
                moves = candidate.copy()
                prev = last
                for j in range(m):
                    moves[j] = np.clip(moves[j], max(lo, prev - du), min(hi, prev + du))
                    prev = moves[j]
            else:
                success, status = False, f"feasible fallback: {result.message}"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            success, status = False, f"feasible fallback: {exc}"
    u = moves[indices]
    y = free + gain @ moves
    increments = np.diff(np.r_[last, u])
    cost = q * np.sum((y - r) ** 2) + ru * (u @ u) + s * (increments @ increments)
    return {"u": u, "y": y, "J": float(cost), "success": success, "status": status}


def _simulate(cfg, name, use_mpc):
    condition = cfg[name]
    x = np.array(condition["x0"], dtype=float)
    last = float(condition["last"])
    rows = []
    fallback_steps = []
    for k in range(condition["steps"]):
        t = (k + np.arange(cfg["N"])) * cfg["dt"]
        r, d = process.scenario(name, t)
        if use_mpc:
            result = plan(x, r, d, last, cfg)
            u = float(result["u"][0])
            if not result["success"]:
                fallback_steps.append(k)
        else:
            u = process.baseline(x, r[0], last, cfg)
        x, y = process.step(x, u, d[0], cfg)
        rows.append((t[0], r[0], d[0], last, u, y))
        last = u
    _, r, _, prev, u, y = np.asarray(rows).T
    rmse = float(np.sqrt(np.mean((y - r) ** 2)))
    rms_u = float(np.sqrt(np.mean(u ** 2)))
    rms_du = float(np.sqrt(np.mean((u - prev) ** 2)))
    metrics = {"position_rmse": rmse, "rms_u": rms_u, "rms_du": rms_du,
               "score": rmse + .015 * rms_u + .03 * rms_du,
               "fallback_steps": fallback_steps}
    return rows, metrics


def main():
    folder = Path(__file__).resolve().parent
    cfg = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    _, design_baseline = _simulate(cfg, "design", False)
    candidates = []
    for i, weights in enumerate(cfg["candidates"]):
        _, metrics = _simulate({**cfg, **weights}, "design", True)
        candidates.append({"index": i, "weights": weights, **metrics})
    picked = min(range(len(candidates)), key=lambda i: candidates[i]["score"])
    _, baseline = _simulate(cfg, "validation", False)
    rows, mpc = _simulate({**cfg, **cfg["candidates"][picked]}, "validation", True)
    summary = {"picked": picked, "design": {"baseline": design_baseline,
               "candidates": candidates}, "validation": {"baseline": baseline, "mpc": mpc}}
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (folder / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("t", "r", "d", "previous", "u", "y"))
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
