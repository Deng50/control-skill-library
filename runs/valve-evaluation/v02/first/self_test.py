"""Independent numerical checks for the valve MPC and exported validation trace."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import process
import solution


def replay(x, u, d, cfg):
    y = []
    for command, disturbance in zip(u, d):
        x, reading = process.step(x, command, disturbance, cfg)
        y.append(reading)
    return np.array(y)


def verify_plan(x, r, d, last, cfg):
    answer = solution.plan(x, r, d, last, cfg)
    u, y = answer["u"], answer["y"]
    m, n = cfg["M"], len(r)
    q, ru, s = (cfg[k] for k in ("q", "r_u", "s"))
    cap = cfg["dt"] * cfg["rate"]
    delta = np.diff(u, prepend=last)
    actual = replay(x, u, d, cfg)
    assert np.max(abs(y - actual)) < 1e-10
    assert abs(answer["J"] - (q*sum((actual-r)**2)+ru*sum(u**2)+s*sum(delta**2))) < 1e-9
    assert np.all(u[m-1:] == u[m-1])
    assert max(cfg["lo"]-min(u), max(u)-cfg["hi"], max(abs(delta))-cap) < 1e-8

    # Form independent gradients by probing each absolute command in the plant.
    z = u[:m]
    hold = np.vstack([np.eye(m), np.repeat(np.eye(m)[-1:], n-m, axis=0)])
    diff = np.eye(m) - np.eye(m, k=-1)
    y0 = replay(x, np.zeros(n), d, cfg)
    gains = np.column_stack([replay(x, col, d, cfg)-y0 for col in hold.T])
    grad = 2*(q*gains.T@(actual-r) + ru*hold.T@u + s*diff.T@delta[:m])
    all_normals = np.vstack((np.eye(m), -np.eye(m), diff, -diff))
    slack = np.r_[cfg["hi"]-z, z-cfg["lo"], cap-delta[:m], cap+delta[:m]]
    active = all_normals[slack < 1e-7]
    residual = nnls(active.T, -grad)[1] if len(active) else np.linalg.norm(grad)
    assert residual < 2e-5 * (1+np.linalg.norm(grad)), residual


def main():
    path = Path(__file__).resolve().parent
    cfg = json.loads((path / "settings.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    rng = np.random.default_rng(90877)
    for n in (1, 3, 8, 13):
        for m in sorted({1, min(2, n), n}):
            for dt in (.06, .2, .43):
                test_cfg = cfg | cfg["candidates"][1] | dict(N=n, M=m, dt=dt)
                x, r, d = rng.normal(size=2), rng.normal(size=n), rng.normal(size=n)
                last = rng.uniform(cfg["lo"], cfg["hi"])
                verify_plan(x, r, d, last, test_cfg)
                # Each decision column and a nontrivial combination are excited.
                a, c, hold, _, gain = solution._model(dt, n, m)
                free = solution._free(x, d, dt, a, c)
                for probe in [*np.eye(m), rng.normal(size=m)]:
                    actual = replay(x, hold @ probe, d, test_cfg)
                    np.testing.assert_allclose(free+gain@probe, actual, atol=2e-13)

    verify_plan(x, r, d, 0.23, test_cfg | dict(rate=0.))
    verify_plan(x, r, d, test_cfg["hi"]+.02, test_cfg)
    verify_plan(x, r, d, .2, test_cfg | dict(lo=.2, hi=.2))
    try:
        solution.plan(x, r, d, 100., test_cfg)
    except ValueError as failure:
        assert "Infeasible" in str(failure)
    else:
        raise AssertionError("Missing infeasibility signal")

    with (path / "validation.csv").open(newline="") as src:
        reader = csv.DictReader(src)
        assert reader.fieldnames == ["t", "r", "d", "previous", "u", "y"]
        rows = list(reader)
    case = cfg["validation"]
    assert len(rows) == case["steps"]
    picked = summary["picked"]
    assert picked == np.argmin([row["score"] for row in summary["design"]])
    chosen = cfg | cfg["candidates"][picked]
    last, x = case["last"], case["x0"]
    for k, row in enumerate(rows):
        t = (k + np.arange(cfg["N"])) * cfg["dt"]
        r, d = process.scenario("validation", t)
        expected_u = solution.plan(x, r, d, last, chosen)["u"][0]
        x, y = process.step(x, expected_u, d[0], cfg)
        np.testing.assert_allclose(list(map(float, row.values())),
                                   [t[0], r[0], d[0], last, expected_u, y], atol=1e-12)
        last = expected_u
    mat = np.array([list(map(float, row.values())) for row in rows])
    _, r, _, prior, u, y = mat.T
    score = np.sqrt(np.mean((y-r)**2))+.015*np.sqrt(np.mean(u*u))+.03*np.sqrt(np.mean((u-prior)**2))
    np.testing.assert_allclose(score, summary["validation"]["selected"]["score"], atol=1e-12)
    print("Passed prediction columns, direct process replay, objective, KKT, bounds, failure reporting and 40-row trace.")


if __name__ == "__main__":
    main()
