"""Preview MPC for the sampled valve. Run this file to replay both experiments."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import process


def _prediction(state, disturbance, dt, horizon, moves):
    """Return free sensor response, command-to-sensor gain, and command expansion."""
    expand = np.eye(moves)[np.minimum(np.arange(horizon), moves - 1)]
    retention = np.exp(-dt / 0.75)
    flow, opening = state
    flow_gain = np.zeros(moves)
    opening_gain = np.zeros(moves)
    free = np.empty(horizon)
    gain = np.empty((horizon, moves))
    for k, drive in enumerate(expand):
        flow = (1.0 - 0.8 * dt) * flow + dt * (1.1 * opening + 0.3 * disturbance[k] + 0.09)
        flow_gain = (1.0 - 0.8 * dt) * flow_gain + 1.1 * dt * opening_gain
        free[k] = flow + 0.12 * opening
        gain[k] = flow_gain + 0.12 * opening_gain + 0.05 * drive
        opening *= retention
        opening_gain = retention * opening_gain + (1.0 - retention) * drive
    return free, gain, expand


def plan(x, r, d, last, cfg):
    """Compute a feasible command schedule and predict each returned process sensor value."""
    state, reference, disturbance = (np.asarray(v, dtype=float).reshape(-1) for v in (x, r, d))
    n, m = cfg['N'], cfg['M']
    dt, rate, lo, hi = (float(cfg[k]) for k in ('dt', 'rate', 'lo', 'hi'))
    q, ru, s = weights = np.array([cfg[k] for k in ('q', 'r_u', 's')], dtype=float)
    last = float(last)
    if (len(state) != 2 or len(reference) != n or len(disturbance) != n
            or not isinstance(n, (int, np.integer)) or not isinstance(m, (int, np.integer))
            or not 1 <= m <= n):
        raise ValueError('Expected two states, N preview values and integer 1 <= M <= N')
    if (not np.isfinite(np.r_[state, reference, disturbance, dt, rate, lo, hi, weights, last]).all()
            or dt <= 0 or rate < 0 or lo > hi or min(weights) < 0):
        raise ValueError('Invalid model parameter, bound, weight or signal')

    slew = dt * rate
    lower, upper = np.full(m, lo), np.full(m, hi)
    lower[0], upper[0] = max(lo, last - slew), min(hi, last + slew)
    if lower[0] > upper[0]:
        raise ValueError('Infeasible first command: amplitude and slew intervals do not overlap')
    hold = np.full(m, np.clip(last, lower[0], upper[0]))
    free, gain, expand = _prediction(state, disturbance, dt, n, m)
    delta = np.diff(expand, axis=0, prepend=np.zeros((1, m)))
    previous = np.zeros(n)
    previous[0] = last
    diff_moves = delta[1:m]

    # Cost is z.T H z + 2*g.T z plus a constant independent of z.
    hessian = q * gain.T @ gain + ru * expand.T @ expand + s * delta.T @ delta
    linear = q * gain.T @ (free - reference) - s * delta.T @ previous
    scale = max(float(np.linalg.norm(hessian, ord=np.inf)), float(np.linalg.norm(linear, ord=np.inf)))
    chosen, status, message = hold, 'optimal', ''
    if slew > 0 and lo < hi and scale > 0:
        hessian /= scale
        linear /= scale

        def objective(z):
            hz = hessian @ z
            return float(z @ (hz + 2 * linear)), 2 * (hz + linear)

        constraints = [LinearConstraint(diff_moves, -slew, slew)] if m > 1 else []
        result = minimize(objective, hold, jac=True, bounds=Bounds(lower, upper),
                          constraints=constraints, method='SLSQP',
                          options={'ftol': 1e-13, 'maxiter': max(250, 8 * m)})
        proposed = np.asarray(result.x)
        if (result.success and np.all(np.isfinite(proposed))
                and np.all(proposed >= lower - 1e-9) and np.all(proposed <= upper + 1e-9)
                and np.all(np.abs(diff_moves @ proposed) <= slew + 1e-9)):
            # Remove last-bit constraint residuals before predicting or issuing the commands.
            chosen = proposed.copy()
            prev = last
            for k in range(m):
                chosen[k] = np.clip(chosen[k], max(lo, prev - slew), min(hi, prev + slew))
                prev = chosen[k]
        else:
            status, message = 'fallback_hold', str(result.message)

    u = expand @ chosen
    y = free + gain @ chosen
    changes = np.diff(u, prepend=last)
    cost = q * np.sum((y - reference)**2) + ru * (u @ u) + s * (changes @ changes)
    return {'u': u, 'y': y, 'J': float(cost), 'status': status, 'message': message}


def _rollout(cfg, name, weights=None):
    local = dict(cfg, **(weights or {}))
    condition = cfg[name]
    state = np.array(condition['x0'], dtype=float)
    previous = condition['last']
    rows = []
    for k in range(condition['steps']):
        times = (k + np.arange(cfg['N'])) * cfg['dt']
        r, d = process.scenario(name, times)
        if weights is None:
            u = process.baseline(state, r[0], previous, local)
        else:
            result = plan(state, r, d, previous, local)
            if result['status'] != 'optimal':
                raise RuntimeError(f"{name} sample {k}: {result['status']} {result['message']}")
            u = result['u'][0]
        state, y = process.step(state, u, d[0], local)
        rows.append((times[0], r[0], d[0], previous, u, y))
        previous = u
    return np.array(rows)


def _metrics(rows, cfg):
    _, r, _, previous, u, y = rows.T
    def rms(a):
        return float(np.sqrt(np.mean(a**2)))
    tracking, effort, changes = rms(y - r), rms(u), rms(u - previous)
    return {'rmse': tracking, 'rms_u': effort, 'rms_du': changes,
            'score': tracking + 0.015 * effort + 0.03 * changes,
            'steps': len(rows), 'min_u': float(u.min()), 'max_u': float(u.max()),
            'max_abs_du': float(np.abs(u - previous).max()),
            'max_constraint_violation': max(0.0, float(cfg['lo'] - u.min()),
                                            float(u.max() - cfg['hi']),
                                            float(np.abs(u - previous).max() - cfg['dt'] * cfg['rate']))}


def main():
    directory = Path(__file__).resolve().parent
    cfg = json.loads((directory / 'settings.json').read_text(encoding='utf-8'))
    design = []
    for i, weights in enumerate(cfg['candidates']):
        data = _metrics(_rollout(cfg, 'design', weights), cfg)
        design.append(dict(candidate=i, weights=weights, **data))
    eligible = [item for item in design if item['max_constraint_violation'] <= 1e-9]
    picked = min(eligible, key=lambda item: item['score'])['candidate']
    validation = _rollout(cfg, 'validation', cfg['candidates'][picked])
    baseline = _metrics(_rollout(cfg, 'validation'), cfg)
    selected = _metrics(validation, cfg)
    summary = {'picked': picked,
               'design': {'candidates': design, 'baseline': _metrics(_rollout(cfg, 'design'), cfg)},
               'validation': {'selected': selected, 'baseline': baseline,
                              'absolute_score_reduction': baseline['score'] - selected['score']}}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (directory / 'validation.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('t', 'r', 'd', 'previous', 'u', 'y'))
        writer.writerows(validation)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
