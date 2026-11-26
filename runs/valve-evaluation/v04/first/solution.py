"""Finite-horizon output-tracking MPC for the supplied valve process."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import process


def _output_map(state, disturbance, holds, dt):
    """Construct offset and G so that sensor history equals offset + G @ moves."""
    retention = np.exp(-dt / 0.75)
    transition = np.array([[1.0 - 0.8 * dt, 1.1 * dt], [0.0, retention]])
    input_gain = np.array([0.0, 1.0 - retention])
    sensor = transition[0] + np.array([0.0, 0.12])
    sensitivity = np.zeros((2, holds.shape[1]))
    offset = np.empty(len(holds))
    gain = np.empty_like(holds)
    for k, selector in enumerate(holds):
        bias = dt * (0.3 * disturbance[k] + 0.09)
        offset[k] = sensor @ state + bias
        gain[k] = sensor @ sensitivity + 0.05 * selector
        sensitivity = transition @ sensitivity + np.outer(input_gain, selector)
        state = transition @ state + np.array([bias, 0.0])
    return offset, gain


def plan(x, r, d, last, cfg):
    """Return forecasts, objective and solver status; raise for an empty feasible set."""
    state = np.asarray(x, dtype=float)
    reference = np.asarray(r, dtype=float)
    disturbance = np.asarray(d, dtype=float)
    horizon = int(cfg.get("N", len(reference)))
    moves = int(cfg["M"])
    if not 1 <= moves <= horizon or reference.shape != (horizon,) or disturbance.shape != (horizon,):
        raise ValueError("Expected equal N-element previews and 1 <= M <= N")
    if state.shape != (2,):
        raise ValueError("Expected x = [flow, opening]")

    dt, lo, hi, rate = (float(cfg[key]) for key in ("dt", "lo", "hi", "rate"))
    q, ru, s = (float(cfg[key]) for key in ("q", "r_u", "s"))
    last = float(last)
    if not np.isfinite(np.r_[state, reference, disturbance, last, dt, lo, hi, rate, q, ru, s]).all():
        raise ValueError("MPC inputs must be finite")
    if dt <= 0 or rate < 0 or lo > hi or min(q, ru, s) < 0:
        raise ValueError("Invalid sampling period, actuator constraints or weights")
    max_change = rate * dt
    if max(lo, last - max_change) > min(hi, last + max_change):
        raise ValueError("Infeasible first command: magnitude and slew intervals are disjoint")

    holds = np.eye(moves)[np.minimum(np.arange(horizon), moves - 1)]
    offset, gain = _output_map(state, disturbance, holds, dt)
    difference = np.eye(moves) - np.eye(moves, k=-1)
    previous = np.zeros(moves)
    previous[0] = last
    quadratic = q * gain.T @ gain + ru * holds.T @ holds + s * difference.T @ difference
    linear = q * gain.T @ (offset - reference) - s * difference.T @ previous
    # Scale both coefficients together to give SLSQP a comparable stopping tolerance.
    scale = max(np.linalg.norm(quadratic, ord=np.inf), np.linalg.norm(linear, ord=np.inf))
    command = np.full(moves, np.clip(last, lo, hi), dtype=float)
    status = "optimal"
    message = "Fixed feasible command or constant objective"
    if hi > lo and max_change > 0 and scale > 0:
        quadratic /= scale
        linear /= scale
        result = minimize(
            lambda v: v @ quadratic @ v + 2.0 * linear @ v,
            command,
            jac=lambda v: 2.0 * (quadratic @ v + linear),
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[LinearConstraint(difference, previous - max_change, previous + max_change)],
            options={"ftol": 1e-13, "maxiter": max(200, 10 * moves)},
        )
        message = str(result.message)
        tolerance = 1e-8
        feasible = (
            np.isfinite(result.x).all()
            and np.all(result.x >= lo - tolerance)
            and np.all(result.x <= hi + tolerance)
            and np.all(np.abs(difference @ result.x - previous) <= max_change + tolerance)
        )
        if result.success and feasible:
            # Remove round-off at active constraints, including the first slew limit.
            anchor = last
            for i, value in enumerate(result.x):
                command[i] = np.clip(value, max(lo, anchor - max_change), min(hi, anchor + max_change))
                anchor = command[i]
        else:
            status = "fallback"

    u = holds @ command
    y = offset + gain @ command
    delta = np.diff(u, prepend=last)
    cost = q * np.sum((y - reference) ** 2) + ru * (u @ u) + s * (delta @ delta)
    return {"u": u, "y": y, "J": float(cost), "status": status, "message": message}


def _simulate(cfg, case, weights=None):
    state = np.array(cfg[case]["x0"])
    last = cfg[case]["last"]
    rows = []
    fallbacks = 0
    controller_cfg = {**cfg, **(weights or {})}
    for k in range(cfg[case]["steps"]):
        times = (k + np.arange(cfg["N"])) * cfg["dt"]
        reference, disturbance = process.scenario(case, times)
        if weights is None:
            command = process.baseline(state, reference[0], last, cfg)
        else:
            proposal = plan(state, reference, disturbance, last, controller_cfg)
            command = proposal["u"][0]
            fallbacks += int(proposal["status"] == "fallback")
        state, sensor = process.step(state, command, disturbance[0], cfg)
        rows.append((times[0], reference[0], disturbance[0], last, command, sensor))
        last = command

    _, reference, _, previous, u, y = np.array(rows).T
    rmse = float(np.sqrt(np.mean((y - reference) ** 2)))
    rms_u = float(np.sqrt(np.mean(u**2)))
    rms_du = float(np.sqrt(np.mean((u - previous) ** 2)))
    metrics = {"rmse": rmse, "rms_u": rms_u, "rms_du": rms_du,
               "score": rmse + 0.015 * rms_u + 0.03 * rms_du}
    if weights is not None:
        metrics["fallback_count"] = fallbacks
    return rows, metrics


def main():
    folder = Path(__file__).resolve().parent
    cfg = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    design = [_simulate(cfg, "design", weights)[1] for weights in cfg["candidates"]]
    picked = min(range(len(design)), key=lambda i: design[i]["score"])
    rows, mpc = _simulate(cfg, "validation", cfg["candidates"][picked])
    _, baseline = _simulate(cfg, "validation")
    summary = {
        "picked": picked,
        "design": {"candidates": design, "baseline": _simulate(cfg, "design")[1]},
        "validation": {"mpc": mpc, "baseline": baseline},
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (folder / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", "r", "d", "previous", "u", "y"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
