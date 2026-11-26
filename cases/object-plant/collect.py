"""Collect native-rate y and applied u from the source-verified ODE3 equivalent."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SOURCE = ROOT.parent / "素材/55-系统辨识MPC-ADRC-PID"
DEFAULT_OUT = ROOT / "runs/object-plant"
SLX = "Linear_2nd_order_model4DataProduction.slx"
SCRIPT = "Linear_2nd_order_Sys_Model4Datacollection.m"
ARCHIVE = "Data4Ident.mat"
HASHES = {
    SLX: "dcd2b78165767af0503dfe35e342836d4115a65848d98a45ddb009278ea22a01",
    SCRIPT: "87708a3b18f4d2a279d1c5b57edd45a78c7811e90b42d53f7ece451f9cc973f7",
    ARCHIVE: "a954fa53c95453faf55ba728f77f2ef9e56019de4e478959903f5cf85cb6cbf1",
}
NATIVE_DT = .01
CONTROL_DT = .1
SUBSTEPS = 10


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source(source):
    observed = {name: sha(source / name) for name in HASHES}
    if observed != HASHES:
        raise ValueError(f"Source differs from verified release: {observed}")
    with zipfile.ZipFile(source / SLX) as z:
        cfg = ET.fromstring(z.read("simulink/configSet0.xml"))
        sys = ET.fromstring(z.read("simulink/systems/system_251.xml"))
    solver = cfg.find(".//P[@Name='SolverName']").text
    step = cfg.find(".//P[@Name='FixedStep']").text
    connections = {line.find("P[@Name='Src']").text: [p.text for p in line.findall(".//P[@Name='Dst']")]
                   for line in sys.findall("Line")}
    if solver != "ode3" or step != "T_sample" or connections["258#out:1"] != ["269#in:1", "268#in:1"]:
        raise ValueError("Unexpected source integration or state order")
    return {"source_sha256": observed, "solver": solver, "fixed_step_variable": step,
            "subsystem": "2nd_order_dynamic", "source_connections": connections}


def transition(physics, exact=False):
    m = np.zeros((4, 4))
    m[0, 1] = 1
    m[1] = [physics["c"], physics["a"], physics["b"], physics["f0"]]
    m *= NATIVE_DT
    if exact:
        return expm(m)
    return np.eye(4) + m + m @ m / 2 + m @ m @ m / 6


def simulate(operator, initial, commands):
    """commands[k] is held forward; last row contains an observed output only."""
    state = np.array([*initial, 0., 1.])
    y = np.empty(len(commands))
    for i, u in enumerate(commands):
        y[i] = state[0]
        state[2] = u
        state = operator @ state
    return y


def make_commands(family, seed, amplitude, offset, n):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * CONTROL_DT
    if family == "levels":
        blocks = []
        while len(blocks) < n:
            blocks.extend([rng.uniform(-1, 1)] * int(rng.integers(2, 13)))
        shape = np.asarray(blocks[:n])
    elif family == "sines":
        phase = rng.uniform(0, 2*np.pi, 2)
        shape = .65*np.sin(2*np.pi*.31*t+phase[0]) + .35*np.sin(2*np.pi*.83*t+phase[1])
    elif family == "alternating":
        shape = rng.choice([-1., 1.], n)
    elif family == "chirp":
        shape = np.sin(2*np.pi*(.12*t+.12*t*t))
    else:
        raise ValueError(f"Unknown input family {family}")
    macros = offset + amplitude * shape
    return np.r_[np.repeat(macros, SUBSTEPS), macros[-1]]


# All cases and numeric choices are local engineering demonstrations.
# The archive is an actual training trajectory and is never regenerated as a second condition.
def conditions():
    rng = np.random.default_rng(168)
    rows = []
    for split, count in (("train", 12), ("validation", 3), ("test", 4)):
        for i in range(count):
            families = ("levels", "sines", "alternating")
            family = families[i % 3] if split != "test" else ("levels", "sines", "alternating", "chirp")[i]
            rows.append(dict(id=f"{split}_{i:02d}_{family}", split=split, family=family,
                             seed=int(rng.integers(10_000, 900_000)),
                             reset=[float(rng.uniform(-.75, .85)), float(rng.uniform(-1.5, 1.5))],
                             amplitude=float(rng.uniform(.55, 1.25)), offset=float(rng.uniform(-.18, .18))))
    return rows


def collect(source, out):
    out.mkdir(parents=True, exist_ok=True)
    src = check_source(source)
    physics = json.loads((HERE / "physics.json").read_text(encoding="utf-8"))
    rk = transition(physics)
    exact = transition(physics, exact=True)
    archive = np.asarray(loadmat(source / ARCHIVE)["Data_collect"], dtype=float)
    if archive.shape != (2001, 3) or not np.isfinite(archive).all():
        raise ValueError("Archived Data_collect must be finite, 2001 x 3")
    t, u, y = archive.T
    t_expected = np.arange(2001) * NATIVE_DT
    u_expected = (np.arange(2001) % 500 < 250).astype(float)
    replay = simulate(rk, [0, 0], u_expected)
    errors = [float(np.max(np.abs(v))) for v in (t-t_expected, u-u_expected, y-replay)]
    if max(errors) >= 1e-10:
        raise ValueError(f"Archived time, u, y mismatch: {errors}")
    rows = [dict(id="original_mat", split="train", family="original_5s_square", reset=[0., 0.],
                 origin=ARCHIVE, samples=len(y))]
    arrays = {"original_mat": archive}
    exact_errors = {}
    for row in conditions():
        u = make_commands(row["family"], row["seed"], row["amplitude"], row["offset"], 80)
        y = simulate(rk, row["reset"], u)
        arrays[row["id"]] = np.column_stack((np.arange(len(u))*NATIVE_DT, u, y))
        exact_errors[row["id"]] = float(np.max(np.abs(y - simulate(exact, row["reset"], u))))
        row.update(samples=len(y), origin="verified_ode3_equivalent")
        rows.append(row)
    if max(exact_errors.values()) > 1e-5:
        raise ValueError("Independent exact-ZOH cross-check failed")
    np.savez_compressed(out / "trajectories.npz", **arrays)
    for row in rows:
        row["trace_sha256"] = hashlib.sha256(arrays[row["id"]].tobytes()).hexdigest()
    if len({r["trace_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trajectory")
    write_json(out / "manifest.json", dict(columns=["t_s", "u", "y"], native_dt_s=NATIVE_DT,
               control_dt_s=CONTROL_DT, rows=rows, data_sha256=sha(out / "trajectories.npz")))
    write_json(out / "source_check.json", dict(**src, original_rows=len(t),
               maximum_archive_error_t_u_y=errors, per_case_max_abs_vs_exact_zoh=exact_errors,
               reference="Each independently reset ODE3 run compared with scipy.linalg.expm at 10 ms"))
    print("Collected", len(rows), "trajectories. Archive max differences", errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    collect(args.source, args.out)
