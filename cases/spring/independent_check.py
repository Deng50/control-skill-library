"""Forward tests of the PGAC handoff, time alignment and independent autonomous rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
import torch


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def numpy_actor(folder):
    meta = read(folder/"interface.json")
    pars = torch.load(folder/"weights.pt", weights_only=True, map_location="cpu")
    layers = [(pars[f"net.{i}.weight"].double().numpy(), pars[f"net.{i}.bias"].double().numpy())
              for i in (0,2,4)]

    def predict(features):
        z = (features - meta["center"]) / meta["scale"]
        for weights, bias in layers:
            z = np.tanh(z @ weights.T + bias)
        return meta["force_limit"] * z[...,0]
    return predict


def discretize(path, dt):
    data = read(path)
    system = np.zeros((4,4))
    system[:2,:] = np.column_stack((data["A"],data["B"],data["E"]))
    return expm(dt * system)[:2,:]


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--out", type=Path, default=root/"runs/pgac/independent.json")
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="pgac_forward_", dir=args.scratch_parent))
    src, pgac, model = root/"runs/expert", root/"runs/pgac", root/"runs/imitation"
    ctrl = root/"runs/mpc/controller.json"
    case = root/"cases/spring"
    cfg, meta, report = read(ctrl), read(src/"data.json"), read(pgac/"report.json")
    fields = ["q_m","v_mps","reference_m","disturbance_N"]
    assert read(model/"interface.json")["features"] == fields
    evidence = dict(evidence="executed", platform=sys.platform, python=sys.version,
                    numpy=np.__version__, torch=torch.__version__, seed=report["runtime"]["seed"],
                    scratch=str(scratch), command="python cases/spring/independent_check.py",
                    source_actor_sha256=sha(model/"weights.pt"))
    assert evidence["source_actor_sha256"] == report["source_actor_sha256"]
    def cli(*params):
        return subprocess.run([sys.executable,"-B","-m","controlkit.policy",*map(str,params)],
                              cwd=root,capture_output=True,text=True)

    # Change the trained Actor's physical mapping while preserving weights and the real expert.
    altered = scratch/"altered"
    shutil.copytree(model, altered)
    interface = read(altered/"interface.json")
    interface["force_limit"] = cfg["force_limit"] / 2
    write(altered/"interface.json", interface)
    result = cli("optimize","--actor",altered,"--epochs",1,"--out",scratch/"bad_range")
    assert result.returncode and "Actuator range conflicts" in result.stderr, result.stderr
    evidence["range_rejection"] = result.stderr.strip().splitlines()[-1]

    # Deliberately re-hash a one-sample-delayed action label; its previous-action trace remains real.
    with np.load(src/"expert.npz", allow_pickle=False) as archive:
        data = dict(archive)
    ids = data["trajectory_id"]
    delayed = data["u"].copy()
    for gid in np.unique(ids):
        rows = np.flatnonzero(ids==gid)
        delayed[rows] = np.vstack(([meta["initial_previous_N"]],data["u"][rows[:-1]]))
    bad = scratch/"delayed.npz"
    np.savez_compressed(bad, **dict(data,u=delayed))
    write(scratch/"delayed.json", dict(meta,dataset_sha256=sha(bad)))
    result = cli("optimize","--data",bad,"--manifest",scratch/"delayed.json",
                 "--epochs",1,"--out",scratch/"bad_delay")
    assert result.returncode and "Previous-action alignment" in result.stderr, result.stderr
    evidence["delay_rejection"] = dict(message=result.stderr.strip().splitlines()[-1],
                                       changed_data_sha256=sha(bad))

    # Observe the actual models immediately before their first Adam update in a fresh one-epoch run.
    sys.path.insert(0,str(root))
    from controlkit import policy
    X = np.column_stack((data["x"],data["reference"],data["d"]))
    held = np.isin(ids,meta["splits"]["validation"])
    baseline = numpy_actor(model)
    expected = baseline(X[held])
    original_load, original_step = policy.load_actor, torch.optim.Adam.step
    models, snapshots = {},{}
    def observe_load(*params,**kwargs):
        net, desc = original_load(*params,**kwargs)
        models[id(next(net.parameters()))] = net
        return net, desc
    def observe_step(opt,*params,**kwargs):
        key = id(opt.param_groups[0]["params"][0])
        if key not in snapshots:
            with torch.no_grad():
                y = models[key](torch.tensor(X[held],dtype=torch.float32)).numpy().ravel()
            snapshots[key] = float(np.max(np.abs(y-expected)))
        return original_step(opt,*params,**kwargs)
    with patch.object(policy,"load_actor",observe_load), patch.object(torch.optim.Adam,"step",observe_step):
        policy.optimize(src/"expert.npz",src/"data.json",model,ctrl,scratch/"observed",1)
    assert len(snapshots)==2 and max(snapshots.values())<2e-6
    evidence["pre_update_max_error_against_independent_actor_N"] = list(snapshots.values())

    # Reproduce both validation Critic values independently on the exact same frozen Actor.
    nominal = discretize(case/"nominal.json",cfg["sample_time_s"])
    residual = np.load(pgac/"perception.npy",allow_pickle=False)
    def objective(coupled):
        features=X[held].copy()
        previous=data["previous_u"][held,0].copy()
        total=np.zeros(len(features))
        for _ in range(cfg["horizon"]):
            u=baseline(features)
            row=np.column_stack((features[:,:2],u,features[:,3]))
            nxt=row@nominal.T
            if coupled:
                nxt+=row@residual
            violation=np.maximum(np.abs(nxt)-[1.,1.5],0)
            total+=cfg["q"]*(nxt[:,0]-features[:,2])**2+cfg["r"]*(u/cfg["force_limit"])**2
            total+=cfg["s"]*((u-previous)/cfg["force_limit"])**2+100*(violation**2).sum(axis=1)
            previous,features[:,:2]=u,nxt
        return float(total.mean()/cfg["horizon"])
    values = dict(nominal=objective(False), coupled=objective(True))
    assert abs(values["nominal"]-values["coupled"])>1e-9
    np.testing.assert_allclose(list(values.values()),[report["critic_at_same_actor"][k] for k in values],
                               rtol=2e-5,atol=1e-9)
    evidence["independent_initial_critic_J"] = values

    # Independently integrate the reference continuous model and close each policy on its own state.
    reference = discretize(case/"reference.json",cfg["sample_time_s"])
    scenarios = read(case/"expert.json")["scenarios"]
    tolerance_N = 2e-5
    closed = {}
    for name in ["imitation","nominal_only","residual_coupled","selected"]:
        bundle=model if name=="imitation" else pgac/name
        actor=numpy_actor(bundle)
        source=report["selected"] if name=="selected" else name
        result=cli("infer",bundle,0.1,0,0.4,0.12)
        result.check_returncode()
        assert abs(float(result.stdout.strip()) - actor(np.array([.1,0,.4,.12]))) < tolerance_N
        closed[name]={}
        for gid in meta["splits"]["test"]:
            sc=scenarios[gid]
            x=np.array(sc["initial_state"])
            r,d=sc["reference"][0],sc["disturbance"][0]
            obs=[]
            for _ in range(sc["steps"]):
                u=float(actor(np.r_[x,r,d]))
                x=reference@np.r_[x,u,d]
                obs.append([u,*x])
            obs=np.array(obs)
            stored=np.loadtxt(pgac/f"{source}-{sc['name']}.csv",delimiter=",",skiprows=1)
            rmse=float(np.sqrt(np.mean((obs[:,1]-r)**2)))
            own=dict(rmse_m=rmse,max_action_delta_N=float(np.max(np.abs(obs[:,0]-stored[:,6]))),
                     max_state_delta=float(np.max(np.abs(obs[:,1:]-stored[:,7:9]))))
            assert own["max_action_delta_N"] < tolerance_N and own["max_state_delta"] < 2e-6
            assert abs(rmse-report["test"][source][sc["name"]]["rmse_m"]) < 2e-6
            closed[name][sc["name"]]=own
    evidence.update(heldout=closed, status="passed", action_tolerance_N=tolerance_N)
    write(args.out,evidence)
    print(args.out)


if __name__ == "__main__":
    main()
