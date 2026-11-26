"""Verify thermal bundles through CLI, physical-unit finite differences, and held-out leakage probes."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def independent_predict(bundle):
    """Rebuild the exported tanh MLP in NumPy float64, independently of training code."""
    settings = read(bundle / "interface.json")["scalers"]
    params = torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True)
    layers = [(params[f"{i}.weight"].double().numpy(), params[f"{i}.bias"].double().numpy())
              for i in (0, 2, 4)]

    def predict(features):
        z = (features - settings["center"]) / settings["scales"]
        for i, (weight, bias) in enumerate(layers):
            z = z @ weight.T + bias
            if i < len(layers) - 1:
                z = np.tanh(z)
        return z[:, 0] * settings["y_scale"] + settings["y_mean"]
    return predict


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=root / "runs/thermal/independent.json")
    p.add_argument("--scratch-parent", type=Path, default=None)
    args = p.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="control_thermal_forward_", dir=args.scratch_parent))
    examples = root / "runs/thermal"
    src = examples / "data.csv"
    physics = root / "cases/thermal/physics.json"
    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    evidence = {"evidence": "executed", "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__, "scratch": str(scratch),
                "command": "python cases/thermal/independent_check.py",
                "seed": read(examples / "paired/report.json")["runtime"]["seed"]}

    def cli(*params):
        return subprocess.run([sys.executable, "-B", "-m", "controlkit.thermal", *map(str, params)],
                              cwd=root, text=True, capture_output=True)

    checks = {}
    paired = read(examples / "paired/report.json")
    no_physics = read(examples / "data-only/report.json")
    table(scratch / "reverse.csv", list(reversed(columns)), rows)
    run = cli("train", "--csv", scratch / "reverse.csv", "--physics", physics,
              "--epochs", paired["epochs"], "--out", scratch / "trained")
    run.check_returncode()
    errors = {}
    for model in ("nn", "pinn"):
        actual = np.loadtxt(scratch / f"trained/{model}/test.csv", delimiter=",", skiprows=1)
        original = np.loadtxt(examples / f"paired/{model}/test.csv", delimiter=",", skiprows=1)
        errors[model] = float(np.max(np.abs(actual-original)))
    evidence["column_permutation_max_abs_change"] = errors
    checks["column_order"] = max(errors.values()) < 1e-8

    infos = {}
    for rel in ["paired/pinn", "paired/nn", "data-only/nn"]:
        bundle = examples / rel
        saved = np.loadtxt(bundle / "test.csv", delimiter=",", skiprows=1)
        query = saved[1, :3]
        run = cli("infer", bundle, *query)
        run.check_returncode()
        infos[rel] = {"cli_max_error_at_query_K": abs(float(run.stdout.strip())-float(saved[1, 4])),
                      "numpy_bundle_max_error_K": float(np.max(np.abs(
                          independent_predict(bundle)(saved[:, :3]) - saved[:, 4])))}
    evidence["reload"] = infos
    checks["reload"] = all(v["cli_max_error_at_query_K"] < 5e-5 and
                           v["numpy_bundle_max_error_K"] < 5e-5 for v in infos.values())
    checks["data_only"] = set(no_physics["results"]) == {"nn"} and (
        no_physics["physics_sha256"] is None and
        no_physics["results"]["nn"]["ode_rmse_W"] is None)

    # Differentiate directly in SI seconds. One-sided endpoints remain inside the support.
    bundle = examples / "paired/pinn"
    predict = independent_predict(bundle)
    trials = np.loadtxt(bundle / "test.csv", delimiter=",", skiprows=1)[:, :3]
    h = 1e-3
    direction = np.array([h, 0.0, 0.0])
    t_min, t_max = min(trials[:, 0]), max(trials[:, 0])
    derivative = np.empty(len(trials))
    for i, z in enumerate(trials):
        z = z[None, :]
        if z[0, 0] == t_min:
            value = (-3*predict(z) + 4*predict(z+direction) - predict(z+2*direction))/(2*h)
        elif z[0, 0] == t_max:
            value = (3*predict(z) - 4*predict(z-direction) + predict(z-2*direction))/(2*h)
        else:
            value = (predict(z+direction) - predict(z-direction))/(2*h)
        derivative[i] = value[0]
    coef = read(physics)
    residual = coef["C_J_K"] * derivative + coef["G_W_K"]*(predict(trials) - coef["Ta_K"]) - trials[:, 2]
    measured = float(np.sqrt(np.mean(residual**2)))
    logged = paired["results"]["pinn"]["ode_rmse_W"]
    evidence["heldout_ode"] = dict(samples=len(trials), derivative_step_seconds=h,
                                    difference_W=abs(measured-logged), finite_difference_W=measured,
                                    reported_autograd_W=logged, tolerance_W=1e-4)
    checks["ode_units_and_chain"] = abs(measured-logged) < 1e-4

    # One bad row with the same trajectory ID, then a point-for-point copy under a different ID.
    row_mix = [dict(row) for row in rows]
    row_mix[0]["partition"] = "test"
    other_id = [*rows, *[dict(row, trajectory="duplicate_id", partition="test") for row in rows
                          if row["trajectory"] == rows[0]["trajectory"]]]
    for name, dataset in (("mixed_id",row_mix), ("cloned_trajectory",other_id)):
        table(scratch/f"{name}.csv", columns, dataset)
        run = cli("train","--csv",scratch/f"{name}.csv","--epochs",1,"--out",scratch/name)
        checks[name] = run.returncode != 0 and not (scratch/name/"report.json").is_file()
        evidence[name] = {"exit_code":run.returncode,
                          "diagnostic":(run.stderr.strip().splitlines() or [""])[-1]}
    evidence.update(checks=checks, status="passed" if all(checks.values()) else "failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(evidence["status"], args.out)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
