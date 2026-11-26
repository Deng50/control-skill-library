"""Independent checks of exported NumPy models with closed-form ZOH references."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from collect import DEFAULT_OUT, HERE, write_json


def load_export(folder):
    spec = importlib.util.spec_from_file_location("standalone_plant", folder / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Plant(folder)


def exact_state(state, u, dt, p):
    a = np.array([[0., 1.], [p["c"], p["a"]]])
    equilibrium = np.array([-(p["b"]*u + p["f0"])/p["c"], 0.])
    return equilibrium + expm(a*dt) @ (state-equilibrium)


def sequence_check(plant, u, dt, p):
    reset = np.array([.34, -.67])
    truth = np.empty((len(u)+1, 2))
    truth[0] = reset
    for i, action in enumerate(u):
        truth[i+1] = exact_state(truth[i], action, dt, p)
    pred = plant.rollout(reset, u, dt)
    if not np.array_equal(pred[0], reset):
        raise AssertionError("Reset not reproduced")
    error = pred[1:] - truth[1:]
    return dict(samples=len(u), dt_s=dt, u_min=float(min(u)), u_max=float(max(u)),
                y_rmse=float(np.sqrt(np.mean(error[:, 0]**2))),
                v_rmse=float(np.sqrt(np.mean(error[:, 1]**2))), y_max_abs=float(max(abs(error[:, 0])))), truth, pred


def residual_check(plant, p):
    rng = np.random.default_rng(67)
    samples = rng.uniform([-.6, -1.2, -.95], [.8, 1.2, .95], (96, 3))
    residuals, discrepancies, exact_defects = [], [], []
    delta = .00005
    for s, t in zip(samples, rng.uniform(.005, .095, len(samples))):
        x, u = s[:2], s[2]
        y, v, acc = plant.at(x, u, t)
        predicted = [plant.at(x, u, t+dt)[0] for dt in (-delta, 0, delta)]
        reference = [exact_state(x, u, t+dt, p)[0] for dt in (-delta, 0, delta)]
        rhs = lambda y, v: p["a"]*v + p["c"]*y + p["b"]*u + p["f0"]
        def finite_defect(triple):
            left, center, right = triple
            return (left-2*center+right)/delta**2 - rhs(center, (right-left)/(2*delta))
        ode = acc-rhs(y, v)
        residuals.append(float(ode))
        discrepancies.append(abs(finite_defect(predicted)-ode))
        exact_defects.append(abs(finite_defect(reference)))
    return dict(points=len(samples), finite_difference_increment_s=delta,
                ode_rmse_Y_per_s2=float(np.sqrt(np.mean(np.square(residuals)))),
                max_difference_analytic_vs_finite=float(max(discrepancies)),
                max_abs_exact_solution_finite_defect=float(max(exact_defects)))


def verify(out):
    p = json.loads((HERE/"physics.json").read_text(encoding="utf-8"))
    paired = json.loads((out/"paired/report.json").read_text(encoding="utf-8"))
    missing = json.loads((out/"data-only/report.json").read_text(encoding="utf-8"))
    if missing["physics_status"] != "data_only" or set(missing["results"]) != {"nn"} or not missing["missing"]:
        raise AssertionError("Missing physics path failed")
    if paired["split"] != missing["split"] or missing["data_sha256"] != paired["data_sha256"]:
        raise AssertionError("Training branches used different trajectories")
    if (out/"data-only/pinn").exists():
        raise AssertionError("Unexpected physics bundle in missing-equation run")
    if paired["results"]["nn"] != missing["results"]["nn"]:
        raise AssertionError("Removing physics changed the supervised baseline")
    rng = np.random.default_rng(157)
    sequences = {"per_10ms": (.01, rng.uniform(-1., 1., 800)),
                 "per_50ms": (.05, rng.uniform(-1., 1., 160)),
                 "per_100ms": (.1, rng.uniform(-1., 1., 80))}
    details, forecasts = {}, {}
    for name in ("nn", "pinn"):
        plant = load_export(out/"paired"/name)
        scores = {}
        for case, (dt, u) in sequences.items():
            score, truth, pred = sequence_check(plant, u, dt, p)
            scores[case] = score
            forecasts[f"{name}_{case}"] = np.column_stack((np.arange(len(u))*dt, u, truth[1:], pred[1:]))
        x = [.34, -.67]
        common = [.2, -.1]
        a = plant.rollout(x, common+[.85]*4)
        b = plant.rollout(x, common+[-.85]*4)
        if not np.array_equal(a[:3], b[:3]) or not a[3, 0] > b[3, 0]:
            raise AssertionError("Control branch causality or sign failed")
        ode_check = residual_check(plant, p)
        if max(ode_check["max_difference_analytic_vs_finite"], ode_check["max_abs_exact_solution_finite_defect"]) > 3e-6:
            raise AssertionError("Dimensional ODE derivative check failed")
        if any(s["y_rmse"] > .005 or s["v_rmse"] > .05 for s in scores.values()):
            raise AssertionError("Numerical deployment regression")
        details[name] = dict(arbitrary_hold_sequences=scores, residual=ode_check,
                             changed_u_first_affected_step=3, changed_u_first_delta_y=float(a[3, 0]-b[3, 0]))
    with np.load(out/"paired/nn/weights.npz") as a, np.load(out/"data-only/nn/weights.npz") as b:
        if set(a.files) != set(b.files) or not all(np.array_equal(a[k], b[k]) for k in a.files):
            raise AssertionError("Baseline weights differed")
    np.savez_compressed(out/"independent_rollouts.npz", **forecasts)
    write_json(out/"independent.json", dict(results=details, identical_supervised_weights_without_equation=True,
               same_split=True, thresholds=dict(finite_difference_abs=3e-6, arbitrary_y_rmse=.005, arbitrary_v_rmse=.05),
               independent_rollout_columns=["left_t_s","held_u","right_y_exact","right_v_exact","right_y_model","right_v_model"]))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify(parser.parse_args().out)
