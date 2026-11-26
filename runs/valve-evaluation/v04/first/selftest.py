"""Replay the delivered CSV and independently check convex first-order optimality."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

import process
import solution


def replay(x, u, d, cfg):
    y = []
    for command, disturbance in zip(u, d):
        x, measurement = process.step(x, command, disturbance, cfg)
        y.append(measurement)
    return np.array(y)


def case(cfg, last, rng):
    n, m = cfg["N"], cfg["M"]
    x, r, d = rng.normal(size=2), rng.normal(size=n), rng.normal(size=n)
    p = solution.plan(x, r, d, last, cfg)
    du = cfg["dt"] * cfg["rate"]
    assert p["status"] == "optimal"
    assert np.all(p["u"] >= cfg["lo"] - 1e-12) and np.all(p["u"] <= cfg["hi"] + 1e-12)
    assert np.max(np.abs(np.diff(p["u"], prepend=last))) <= du + 1e-12
    assert np.all(p["u"][m:] == p["u"][m - 1])
    np.testing.assert_allclose(replay(x, p["u"], d, cfg), p["y"], atol=1e-12)

    def objective(v):
        u = np.r_[v, np.repeat(v[-1], n-m)]
        error = replay(x, u, d, cfg) - r
        delta = np.diff(u, prepend=last)
        return cfg["q"] * (error @ error) + cfg["r_u"] * (u @ u) + cfg["s"] * (delta @ delta)

    v = p["u"][:m]
    np.testing.assert_allclose(objective(v), p["J"], rtol=1e-12, atol=1e-12)
    # Central difference is exact for a quadratic. Minimize this linearization
    # on the actuator polytope to bound the possible objective improvement.
    gradient = np.array([(objective(v + e) - objective(v - e)) / 2 for e in np.eye(m)])
    diff = np.eye(m) - np.eye(m, k=-1)
    anchor = np.zeros(m)
    anchor[0] = last
    linear = linprog(gradient, A_ub=np.vstack((diff, -diff)),
                     b_ub=np.r_[anchor + du, du - anchor], bounds=(cfg["lo"], cfg["hi"]))
    assert linear.success
    assert gradient @ v - linear.fun < 2e-5 * (1 + p["J"])


def main():
    folder = Path(__file__).resolve().parent
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["picked"] == np.argmin([c["score"] for c in summary["design"]["candidates"]])
    cfg = {**settings, **settings["candidates"][summary["picked"]]}
    with (folder / "validation.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["t", "r", "d", "previous", "u", "y"]
        rows = list(reader)
    t, r, d, previous, u, y = np.array([list(map(float, row.values())) for row in rows]).T
    np.testing.assert_allclose(t, np.arange(cfg["validation"]["steps"]) * cfg["dt"])
    np.testing.assert_allclose(np.array([r, d]), process.scenario("validation", t))
    np.testing.assert_allclose(previous, np.r_[cfg["validation"]["last"], u[:-1]])
    np.testing.assert_allclose(replay(cfg["validation"]["x0"], u, d, cfg), y, atol=1e-12)
    rmse = np.sqrt(np.mean((y-r)**2))
    rms_u = np.sqrt(np.mean(u**2))
    rms_du = np.sqrt(np.mean((u-previous)**2))
    metrics = summary["validation"]["mpc"]
    np.testing.assert_allclose([metrics[k] for k in ("rmse", "rms_u", "rms_du", "score")],
                               [rmse, rms_u, rms_du, rmse + .015 * rms_u + .03 * rms_du])
    rng = np.random.default_rng(236)
    for n, m, dt in [(1, 1, .12), (5, 1, .27), (11, 5, .09), (7, 7, .35)]:
        for last in [-.71, -.7, .25, .6]:
            case(dict(cfg, N=n, M=m, dt=dt, lo=-.7, hi=.6, rate=.32), last, rng)
    for overrides in [dict(rate=0), dict(lo=.1, hi=.1), dict(q=0, r_u=0, s=0)]:
        case({**cfg, **overrides}, .1, rng)

    zeros = np.zeros(cfg["N"])
    try:
        solution.plan([0, 0], zeros, zeros, cfg["hi"] + cfg["rate"]*cfg["dt"] + 1, cfg)
    except ValueError as ex:
        assert "Infeasible" in str(ex)
    else:
        raise AssertionError("Empty first-command interval was accepted")
    failure = SimpleNamespace(success=False, x=np.full(cfg["M"], np.nan), message="injected failure")
    with patch("solution.minimize", return_value=failure):
        p = solution.plan([0, 0], zeros, zeros, .15, cfg)
        assert p["status"] == "fallback"
        np.testing.assert_allclose(p["u"], .15)
    print("Passed validation replay, 19 independent optimality checks and both failure paths.")


if __name__ == "__main__":
    main()
