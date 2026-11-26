"""Self-checks using the supplied nonlinear-step interface as an independent oracle."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

import process
from solution import plan


def check_plan(x, r, d, last, cfg):
    p = plan(x, r, d, last, cfg)
    assert p["success"], p["status"]
    u, predicted = p["u"], p["y"]
    m, n = cfg["M"], cfg["N"]
    rate = cfg["dt"] * cfg["rate"]
    assert u.shape == predicted.shape == (n,)
    assert np.all(u >= cfg["lo"] - 1e-12) and np.all(u <= cfg["hi"] + 1e-12)
    assert np.all(np.abs(np.diff(np.r_[last, u])) <= rate + 1e-12)
    np.testing.assert_array_equal(u[m - 1:], np.full(n - m + 1, u[m - 1]))

    def rollout(v):
        state, prev, cost = x.copy(), last, 0.
        outputs = []
        for k in range(n):
            cmd = v[min(k, m - 1)]
            state, y = process.step(state, cmd, d[k], cfg)
            cost += cfg["q"] * (y - r[k])**2 + cfg["r_u"] * cmd**2 + cfg["s"] * (cmd - prev)**2
            prev = cmd
            outputs.append(y)
        return cost, outputs

    v = u[:m]
    cost, y = rollout(v)
    np.testing.assert_allclose(predicted, y, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(p["J"], cost, rtol=1e-10, atol=1e-11)
    # A centered difference at unit displacement is exact for a quadratic.
    grad = np.array([(rollout(v + e)[0] - rollout(v - e)[0]) / 2 for e in np.eye(m)])
    delta = np.eye(m)
    delta[1:] -= np.eye(m)[:-1]
    prev = np.zeros(m)
    prev[0] = last
    oracle = linprog(grad, A_ub=np.vstack([delta, -delta]),
                     b_ub=np.r_[prev + rate, -prev + rate],
                     bounds=[(cfg["lo"], cfg["hi"])] * m, method="highs")
    assert oracle.success
    # Convexity bounds the global suboptimality by this first-order gap.
    assert grad @ v - oracle.fun <= 4e-5 * (1 + abs(cost))


def main():
    root = Path(__file__).resolve().parent
    base = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    cfg = {**base, **base["candidates"][1]}
    rng = np.random.default_rng(118)
    for n, m, dt, rate in [(1, 1, .31, 1.3), (7, 7, .12, .7), (13, 2, .27, 2.2),
                            (5, 3, .07, 0.), (9, 1, .055, 1.6), (8, 4, .2, 1.05)]:
        for last in (0., cfg["hi"], cfg["lo"] - .5 * dt * rate):
            check_plan(rng.normal(size=2), rng.normal(size=n), rng.normal(size=n), last,
                       {**cfg, "N": n, "M": m, "dt": dt, "rate": rate})
    r, d = process.scenario("validation", np.arange(cfg["N"]) * cfg["dt"])
    x = np.asarray(cfg["validation"]["x0"])
    try:
        plan(x, r, d, 10., cfg)
    except ValueError as exc:
        assert "Infeasible" in str(exc)
    else:
        raise AssertionError("Expected infeasible first command to be reported")
    with patch("solution.minimize", return_value=SimpleNamespace(
            x=np.full(cfg["M"], np.nan), success=False, message="forced failure")):
        p = plan(x, r, d, -.08, cfg)
        assert not p["success"] and "forced failure" in p["status"]
        np.testing.assert_array_equal(p["u"], np.full(cfg["N"], -.08))

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    candidates = summary["design"]["candidates"]
    assert summary["picked"] == min(range(len(candidates)), key=lambda i: candidates[i]["score"])
    with (root / "validation.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["t", "r", "d", "previous", "u", "y"]
        rows = list(reader)
    assert len(rows) == cfg["validation"]["steps"]
    last = cfg["validation"]["last"]
    mpc_cfg = {**base, **base["candidates"][summary["picked"]]}
    for k, row in enumerate(rows):
        t = (k + np.arange(cfg["N"])) * cfg["dt"]
        r, d = process.scenario("validation", t)
        cmd = plan(x, r, d, last, mpc_cfg)["u"][0]
        x, y = process.step(x, cmd, d[0], cfg)
        np.testing.assert_allclose([float(a) for a in row.values()], [t[0], r[0], d[0], last, cmd, y])
        last = cmd
    data = np.array([[float(row[field]) for field in ("r", "previous", "u", "y")] for row in rows])
    r, prev, u, y = data.T
    metrics = np.sqrt(np.mean(np.square([y - r, u, u - prev]), axis=1))
    reported = summary["validation"]["mpc"]
    np.testing.assert_allclose(metrics, [reported[k] for k in ("position_rmse", "rms_u", "rms_du")])
    np.testing.assert_allclose(metrics @ [1, .015, .03], reported["score"])
    print("PASS: 18 independent horizon/cost/optimality checks, failure handling and 40-step replay")


if __name__ == "__main__":
    main()
