"""Independent checks using the original sampled process and a convex first-order bound."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

import process
import solution


def simulate(x, commands, d, cfg):
    sensors = []
    for command, disturbance in zip(commands, d):
        x, y = process.step(x, command, disturbance, cfg)
        sensors.append(y)
    return np.array(sensors)


def verify_plan(cfg, x, r, d, last):
    n, m = cfg['N'], cfg['M']
    result = solution.plan(x, r, d, last, cfg)
    assert result['status'] == 'optimal', result
    u = result['u']
    du = np.diff(u, prepend=last)
    slew = cfg['rate'] * cfg['dt']
    assert len(u) == n and np.all(u[m-1:] == u[m-1])
    assert max(abs(du)) <= slew + 1e-12
    assert u.min() >= cfg['lo'] - 1e-12 and u.max() <= cfg['hi'] + 1e-12
    y = simulate(x, u, d, cfg)
    np.testing.assert_allclose(result['y'], y, atol=3e-13)
    J = np.sum(cfg['q']*(y-r)**2 + cfg['r_u']*u**2 + cfg['s']*du**2)
    np.testing.assert_allclose(result['J'], J, rtol=2e-13, atol=2e-13)

    # Independently excite each command column of the actual process. The minimum
    # of the local linearization over the input polytope lower-bounds the convex QP.
    hold = np.vstack((np.eye(m), np.tile(np.eye(m)[-1], (n-m, 1))))
    zero = simulate(x, np.zeros(n), d, cfg)
    gain = np.array([simulate(x, impulse, d, cfg) - zero for impulse in hold.T]).T
    delta = np.diff(hold, axis=0, prepend=np.zeros((1, m)))
    grad = 2 * (cfg['q'] * gain.T @ (y-r) + cfg['r_u'] * hold.T @ u + cfg['s'] * delta.T @ du)
    offset = np.zeros(m)
    offset[0] = last
    lp = linprog(grad, A_ub=np.vstack((delta[:m], -delta[:m])),
                 b_ub=np.r_[slew + offset, slew - offset], bounds=[(cfg['lo'], cfg['hi'])]*m)
    assert lp.success, lp.message
    gap = grad @ u[:m] - lp.fun
    assert gap < 1e-5 * max(1., J), (gap, J)
    return max(abs(result['y'] - y)), max(0., gap)


def main():
    root = Path(__file__).resolve().parent
    base = json.loads((root / 'settings.json').read_text())
    rng = np.random.default_rng(53)
    results = []
    for i in range(35):
        n = int(rng.integers(1, 19))
        m = int(rng.integers(1, n+1))
        cfg = dict(base, N=n, M=m, dt=rng.uniform(.04, .55), lo=-.8, hi=.7,
                   rate=rng.uniform(.12, 3), q=rng.uniform(.2, 6),
                   r_u=rng.uniform(.001, .3), s=rng.uniform(.001, .4))
        if i == 0:
            cfg.update(N=1, M=1)
        if i == 1:
            cfg.update(rate=0)
        if i == 2:
            cfg.update(M=n, q=0, r_u=0, s=0)
        if i == 3:
            cfg.update(lo=.04, hi=.04)
        n = cfg['N']
        r, d = rng.normal(size=(2, n))
        last = rng.uniform(cfg['lo'], cfg['hi'])
        if i == 4:
            last = cfg['hi'] + .25 * cfg['rate'] * cfg['dt']
        results.append(verify_plan(cfg, rng.normal(size=2), r, d, last))

    cfg = dict(base, **base['candidates'][1])
    x = [.24, -.12]
    r, d = rng.normal(size=(2, cfg['N']))
    free, gain, expand = solution._prediction(x, d, cfg['dt'], cfg['N'], cfg['M'])
    for z in [*np.eye(cfg['M']), np.array([.1, -.23, .4])]:
        np.testing.assert_allclose(free + gain @ z, simulate(x, expand @ z, d, cfg), atol=1e-13)
    try:
        solution.plan(x, r, d, cfg['hi'] + 2 * cfg['rate'] * cfg['dt'], cfg)
    except ValueError as exc:
        assert 'Infeasible first command' in str(exc)
    else:
        raise AssertionError('Infeasibility was not observable')
    failed = SimpleNamespace(x=np.zeros(cfg['M']), success=False, message='simulated nonconvergence')
    with patch.object(solution, 'minimize', return_value=failed):
        result = solution.plan(x, r, d, .2, cfg)
    assert result['status'] == 'fallback_hold' and result['message'] == failed.message
    np.testing.assert_allclose(result['u'], .2)
    np.testing.assert_allclose(result['y'], simulate(x, result['u'], d, cfg))

    with (root/'validation.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['t', 'r', 'd', 'previous', 'u', 'y']
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
    assert len(rows) == cfg['validation']['steps']
    summary = json.loads((root/'summary.json').read_text())
    assert summary['picked'] == int(np.argmin([i['score'] for i in summary['design']['candidates']]))
    x, previous = cfg['validation']['x0'], cfg['validation']['last']
    for k, row in enumerate(rows):
        t = k * cfg['dt']
        r, d = process.scenario('validation', t)
        x, y = process.step(x, row['u'], d, cfg)
        np.testing.assert_allclose([row[key] for key in ('t','r','d','previous','y')], [t,r,d,previous,y])
        previous = row['u']
    data = np.array([[row[key] for key in ('r','previous','u','y')] for row in rows]).T
    r, prev, u, y = data
    rmse, rmsu, rmsdu = [np.sqrt(np.mean(v**2)) for v in [y-r, u, u-prev]]
    np.testing.assert_allclose(summary['validation']['selected']['score'], rmse+.015*rmsu+.03*rmsdu)
    print(f'35 plans verified; max sensor error {max(i[0] for i in results):.3g}; '
          f'max convex optimality gap {max(i[1] for i in results):.3g}.')
    print(f'All prediction columns, solver failure and infeasibility checked; {len(rows)} validation rows replayed.')


if __name__ == '__main__':
    main()
