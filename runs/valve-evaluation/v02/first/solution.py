"""Constrained preview MPC for the valve in process.py."""
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

import process


@lru_cache(maxsize=64)
def _model(dt, n, m):
    alpha = np.exp(-dt / .75)
    a = np.array([[1 - .8 * dt, 1.1 * dt], [0., alpha]])
    b = np.array([0., 1. - alpha])
    c = a[0] + np.array([0., .12])
    # T expands the M absolute commands into N with a held tail.
    hold = np.eye(m)[np.minimum(np.arange(n), m - 1)]
    difference = np.eye(m) - np.eye(m, k=-1)
    sensitivity = np.zeros((2, m))
    gain = np.empty((n, m))
    for k, row in enumerate(hold):
        gain[k] = c @ sensitivity + .05 * row
        sensitivity = a @ sensitivity + b[:, None] * row
    return a, c, hold, difference, gain


def _free(x, d, dt, a, c):
    state = x.copy()
    free = np.empty(len(d))
    for k, disturbance in enumerate(d):
        offset = dt * (.3 * disturbance + .09)
        free[k] = c @ state + offset
        state = a @ state
        state[0] += offset
    return free


def plan(x, r, d, last, cfg):
    """Return the optimal move-blocked plan. Infeasibility and failures raise."""
    x, r, d = (np.asarray(value, dtype=float) for value in (x, r, d))
    n, m = len(r), int(cfg["M"])
    if (r.ndim != 1 or d.shape != (n,) or x.shape != (2,) or
            cfg.get("N", n) != n or cfg["M"] != m or not 1 <= m <= n):
        raise ValueError("Require two states and equal N-step previews, 1 <= M <= N")
    dt, rate, lo, hi, q, ru, s = (float(cfg[key]) for key in
                                   ("dt", "rate", "lo", "hi", "q", "r_u", "s"))
    last = float(last)
    if (not np.all(np.isfinite(np.r_[x, r, d, dt, rate, lo, hi, q, ru, s, last]))
            or dt <= 0 or lo > hi or min(rate, q, ru, s) < 0):
        raise ValueError("Invalid bounds, input values, sample period or weights")
    change = dt * rate
    if max(lo, last - change) > min(hi, last + change):
        raise ValueError("Infeasible first command: range and slew intervals do not intersect")

    a, c, hold, diff, gain = _model(dt, n, m)
    free = _free(x, d, dt, a, c)
    prior = np.zeros(m)
    prior[0] = last
    z = np.full(m, np.clip(last, lo, hi))
    if lo < hi and change > 0 and max(q, ru, s) > 0:
        h = q * (gain.T @ gain) + ru * (hold.T @ hold) + s * (diff.T @ diff)
        g = q * (gain.T @ (free - r)) - s * (diff.T @ prior)
        # Drop the decision-independent constant before numerical optimization.
        scale = max(np.abs(h).max(), np.abs(g).max(), np.finfo(float).tiny)
        h, g = h / scale, g / scale

        def objective(moves):
            hm = h @ moves
            return (.5 * hm + g) @ moves, hm + g

        result = minimize(objective, z, jac=True, method="SLSQP",
                          bounds=[(lo, hi)] * m,
                          constraints=[LinearConstraint(diff, prior-change, prior+change)],
                          options={"ftol": 1e-14, "maxiter": 500})
        if not result.success:
            raise RuntimeError(f"MPC solver failed ({result.status}): {result.message}")
        z = result.x

    moves = hold @ z
    deltas = np.diff(moves, prepend=last)
    if (not np.all(np.isfinite(moves)) or moves.min() < lo - 1e-9 or
            moves.max() > hi + 1e-9 or abs(deltas).max() > change + 1e-9):
        raise RuntimeError("Solver returned a command outside a hard constraint")
    y = free + gain @ z
    err = y - r
    cost = q * (err @ err) + ru * (moves @ moves) + s * (deltas @ deltas)
    return {"u": moves, "y": y, "J": float(cost), "status": "optimal"}


def _simulate(name, cfg, base=False):
    case = cfg[name]
    state = np.array(case["x0"], dtype=float)
    last = float(case["last"])
    dt, n = cfg["dt"], cfg["N"]
    records = []
    for k in range(case["steps"]):
        t = (k + np.arange(n)) * dt
        r, d = process.scenario(name, t)
        if base:
            command = process.baseline(state, r[0], last, cfg)
        else:
            command = plan(state, r, d, last, cfg)["u"][0]
        state, measured = process.step(state, command, d[0], cfg)
        records.append((t[0], r[0], d[0], last, command, measured))
        last = command
    return np.array(records)


def _metrics(rows, cfg):
    _, ref, _, previous, u, y = rows.T
    du = u - previous
    rmse, rms_u, rms_du = (float(np.sqrt(np.mean(v*v))) for v in (y-ref, u, du))
    violation = max(0., cfg["lo"]-u.min(), u.max()-cfg["hi"],
                    abs(du).max()-cfg["rate"]*cfg["dt"])
    if violation > 1e-9:
        raise RuntimeError("Closed-loop trace violates command bounds")
    return {"rmse": rmse, "rms_u": rms_u, "rms_du": rms_du,
            "score": rmse + .015*rms_u + .03*rms_du,
            "max_constraint_violation": float(violation)}


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    design = [_metrics(_simulate("design", cfg | weights), cfg)
              for weights in cfg["candidates"]]
    picked = min(range(len(design)), key=lambda i: design[i]["score"])
    chosen = cfg | cfg["candidates"][picked]
    selected_rows = _simulate("validation", chosen)
    summary = {"picked": picked, "weights": cfg["candidates"][picked],
               "design": design,
               "design_baseline": _metrics(_simulate("design", cfg, base=True), cfg),
               "validation": {"selected": _metrics(selected_rows, cfg),
                              "baseline": _metrics(_simulate("validation", cfg, base=True), cfg)}}
    with (root / "validation.csv").open("w", encoding="utf-8", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(("t", "r", "d", "previous", "u", "y"))
        writer.writerows(selected_rows)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
