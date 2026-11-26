"""Fit paired interval flow models using only the archive-style trajectory contract."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import qmc
import torch
from torch import nn

from collect import CONTROL_DT, DEFAULT_OUT, HERE, NATIVE_DT, SUBSTEPS, sha, write_json
from predict import Plant

SEED = 901
WIDTH = 24
STAGES = 12
ITERATIONS_PER_STAGE = 25
PHYSICS_WEIGHT = 1.


def load_data(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if sha(folder / "trajectories.npz") != manifest["data_sha256"]:
        raise ValueError("Trajectory package checksum failed")
    if manifest["columns"] != ["t_s", "u", "y"] or manifest["native_dt_s"] != NATIVE_DT or manifest["control_dt_s"] != CONTROL_DT:
        raise ValueError("Unexpected trajectory layout or time base")
    with np.load(folder / "trajectories.npz", allow_pickle=False) as f:
        arrays = dict(f)
    seen = set()
    ids = set()
    groups = set()
    for row in manifest["rows"]:
        array = arrays[row["id"]]
        t, u, y = array.T
        ident = hashlib.sha256(array.tobytes()).hexdigest()
        condition = (tuple(row["reset"]), hashlib.sha256(u.tobytes()).hexdigest())
        if ident in seen or ident != row["trace_sha256"] or row["id"] in ids or condition in groups:
            raise ValueError("Repeated or altered trajectory/condition")
        seen.add(ident)
        ids.add(row["id"])
        groups.add(condition)
        if not np.isfinite(array).all() or t[0] != 0 or not np.allclose(np.diff(t), NATIVE_DT, atol=1e-12, rtol=0):
            raise ValueError("Unsafe time base or missing values")
        if y[0] != row["reset"][0] or (len(y)-1) % SUBSTEPS:
            raise ValueError("Reset or trajectory length mismatch")
        if np.any(np.ptp(u[:-1].reshape(-1, SUBSTEPS), axis=1) > 0):
            raise ValueError("Command switched inside a training hold")
    return manifest, arrays


def windows(rows, arrays):
    starts, targets = [], []
    for row in rows:
        _, u, y = arrays[row["id"]].T
        k = np.arange(0, len(y)-1, SUBSTEPS)
        v = np.empty(len(k))
        v[0] = row["reset"][1]
        j = k[1:]
        # Left derivative at a command edge; native samples belong to the previous hold.
        v[1:] = (25*y[j]-48*y[j-1]+36*y[j-2]-16*y[j-3]+3*y[j-4])/(12*NATIVE_DT)
        starts.append(np.column_stack((y[k], v, u[k])))
        targets.append(y[k[:, None]+np.arange(1, SUBSTEPS+1)])
    return np.vstack(starts), np.vstack(targets)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, WIDTH)
        self.head = nn.Linear(WIDTH, 3, bias=False)
        self.linear = nn.Linear(3, 3)

    def forward(self, z):
        return self.linear(z) + self.head(torch.tanh(self.hidden(z)))


def coefficients(network, features, scale):
    z = (features-scale["mean"])/scale["std"]
    return network(z)*scale["acc"]


def flow(q, features, tau):
    a0, a1, a2 = q.split(1, dim=1)
    s = tau/CONTROL_DT
    y = features[:, :1] + features[:, 1:2]*tau + tau**2*(a0/2 + s*a1/6 + s*s*a2/12)
    v = features[:, 1:2] + tau*(a0 + s*a1/2 + s*s*a2/3)
    return y, v, a0 + s*a1 + s*s*a2


def val_score(network, scale, validation, arrays):
    with torch.no_grad():
        state = torch.tensor([row["reset"] for row in validation])
        commands = torch.tensor(np.stack([arrays[row["id"]][:-1:SUBSTEPS, 1] for row in validation]))
        truth = torch.tensor(np.stack([arrays[row["id"]][SUBSTEPS::SUBSTEPS, 2] for row in validation]))
        preds = []
        for k in range(commands.shape[1]):
            x = torch.cat((state, commands[:, k:k+1]), dim=1)
            y, v, _ = flow(coefficients(network, x, scale), x, CONTROL_DT)
            state = torch.cat((y, v), dim=1)
            preds.append(y[:, 0])
        return float(torch.sqrt(torch.mean((torch.stack(preds, dim=1)-truth)**2)))


def export(bundle, network, scale, feature_range, model, data_hash, physics_hash):
    bundle.mkdir(parents=True, exist_ok=True)
    np.savez(bundle / "weights.npz", **{k: t.detach().numpy() for k, t in network.state_dict().items()})
    write_json(bundle / "scaler.json", dict(input_mean=scale["mean"].tolist(),
               input_std=scale["std"].tolist(), acceleration_scale=float(scale["acc"]), fit_partition="train"))
    write_json(bundle / "interface.json", dict(schema="second-order-interval-flow-v1", model=model,
               input=["y [Y]", "v [Y/s]", "u [U]"], outputs=["y_next [Y]", "v_next [Y/s]"],
               min_step_s=NATIVE_DT, max_hold_s=CONTROL_DT, hold="ZOH on [k,k+1)",
               hidden_width=WIDTH, activation="tanh", feature_support_train=feature_range.tolist(),
               reset="Known reset y and v, or estimated initial v; propagate both outputs afterwards",
               input_policy="finite commands; training ranges documented; extrapolation is unchecked",
               data_sha256=data_hash, physics_sha256=physics_hash, weights_sha256=sha(bundle / "weights.npz"),
               scaler_sha256=sha(bundle / "scaler.json")))
    shutil.copyfile(HERE / "predict.py", bundle / "predict.py")


def evaluate(plant, rows, arrays, out):
    metrics, raw = {}, {}
    for row in rows:
        a = arrays[row["id"]]
        commands = a[:-1:SUBSTEPS, 1]
        truth = a[::SUBSTEPS, 2]
        pred = plant.rollout(row["reset"], commands)
        x, target = windows([row], arrays)
        one_step = plant.step(x[:, :2], x[:, 2])[:, 0]
        err = pred[1:, 0] - truth[1:]
        raw[row["id"]] = np.column_stack((a[::SUBSTEPS, 0], a[::SUBSTEPS, 1], truth,
                                           pred[:, 0], pred[:, 1]))
        metrics[row["id"]] = dict(split=row["split"], steps=len(commands),
              one_step_y_rmse=float(np.sqrt(np.mean((one_step-target[:, -1])**2))),
              free_y_rmse=float(np.sqrt(np.mean(err**2))), free_y_max_abs=float(np.max(np.abs(err))),
              terminal_y_error=float(err[-1]))
    np.savez_compressed(out / "rollouts.npz", **raw)
    return metrics


def fit(name, root, manifest, arrays, physics, physics_hash):
    torch.manual_seed(SEED)
    train_rows = [r for r in manifest["rows"] if r["split"] == "train"]
    val_rows = [r for r in manifest["rows"] if r["split"] == "validation"]
    x_np, y_np = windows(train_rows, arrays)
    features, targets = torch.tensor(x_np), torch.tensor(y_np)
    tau = torch.arange(1, SUBSTEPS+1)*NATIVE_DT
    # Causal native-rate slopes define window starts. The acceleration scale uses train labels only.
    numerical_acc = 2*(y_np[:, -1]-x_np[:, 0]-CONTROL_DT*x_np[:, 1])/CONTROL_DT**2
    scale = dict(mean=features.mean(dim=0), std=features.std(dim=0, unbiased=False),
                 acc=float(np.sqrt(np.mean(numerical_acc**2))))
    feature_range = np.stack((x_np.min(0), x_np.max(0)))
    collocation = qmc.Sobol(4, scramble=True, seed=SEED+1).random_base2(10)
    support = torch.tensor(qmc.scale(collocation[:, :3], *feature_range))
    t_collocation = torch.tensor(collocation[:, 3:4]*CONTROL_DT)
    network = Network()
    opt = torch.optim.LBFGS(network.parameters(), lr=.8, max_iter=ITERATIONS_PER_STAGE,
                            tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def losses():
        q = coefficients(network, features, scale)
        y, _, _ = flow(q, features, tau)
        # Equalize sensitivity to average acceleration across observation offsets.
        data_loss = torch.mean(((y-targets)/(.5*tau**2*scale["acc"]))**2)
        if physics is None:
            return data_loss, torch.zeros(())
        yp, vp, acc = flow(coefficients(network, support, scale), support, t_collocation)
        rhs = physics["a"]*vp + physics["c"]*yp + physics["b"]*support[:, 2:3] + physics["f0"]
        return data_loss, torch.mean(((acc-rhs)/scale["acc"])**2)

    def closure():
        opt.zero_grad()
        data, phys = losses()
        objective = data + PHYSICS_WEIGHT*phys
        objective.backward()
        return objective

    history, best, chosen = [], np.inf, None
    for stage in range(1, STAGES+1):
        opt.step(closure)
        with torch.no_grad():
            data, phys = losses()
            score = val_score(network, scale, val_rows, arrays)
        history.append(dict(stage=stage, train_data_mse=float(data), train_ode_mse=float(phys) if physics else None,
                            validation_free_y_rmse=score))
        if score < best:
            best, chosen = score, (stage, copy.deepcopy(network.state_dict()))
    if chosen is None:
        raise ValueError("No finite validation candidate")
    network.load_state_dict(chosen[1])
    bundle = root / name
    export(bundle, network, scale, feature_range, name, manifest["data_sha256"], physics_hash)
    metrics = evaluate(Plant(bundle), [r for r in manifest["rows"] if r["split"] != "train"], arrays, bundle)
    test_errors = [m["free_y_rmse"] for m in metrics.values() if m["split"] == "test"]
    return dict(model=name, seed=SEED, stages=STAGES, max_iterations_per_stage=ITERATIONS_PER_STAGE,
                training_windows=len(features), collocation_points=len(support) if physics else 0,
                physics_loss_weight=PHYSICS_WEIGHT if physics else 0., selected_stage=chosen[0],
                selected_validation_free_y_rmse=best, pooled_test_free_y_rmse=float(np.sqrt(np.mean(np.square(test_errors)))),
                metrics=metrics, history=history)


def train(data, out, physics_path):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    manifest, arrays = load_data(data)
    physics = None
    if physics_path is not None:
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        if physics.get("schema") != "second-order-zoh-v1" or not all(
                isinstance(physics.get(k), (int, float)) and np.isfinite(physics[k]) for k in ("a", "c", "b", "f0")):
            raise ValueError("Expected the documented ODE and finite named coefficients")
    elif (out / "pinn").exists():
        raise ValueError("A PINN bundle already exists here; choose a separate output for the data-only run")
    out.mkdir(parents=True, exist_ok=True)
    results = {"nn": fit("nn", out, manifest, arrays, None, None)}
    if physics is not None:
        results["pinn"] = fit("pinn", out, manifest, arrays, physics, sha(physics_path))
    split = {s: [r["id"] for r in manifest["rows"] if r["split"] == s] for s in ("train", "validation", "test")}
    write_json(out / "report.json", dict(split=split, data_sha256=manifest["data_sha256"], results=results,
               physics_status="paired" if physics else "data_only",
               missing=[] if physics else ["source-backed second-order equation and coefficients a,c,b,f0"],
               framework={"numpy":np.__version__, "torch":torch.__version__, "dtype":"float64", "device":"CPU"}))
    print({k: v["pooled_test_free_y_rmse"] for k, v in results.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physics", type=Path)
    args = parser.parse_args()
    train(args.data, args.out, args.physics)
