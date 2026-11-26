"""Score exported neural transitions on the actual three Simulink input experiments."""
import importlib.util
from pathlib import Path
import numpy as np

from .io import read, write, digest

ROOT=Path(__file__).resolve().parents[1]


def run():
    training=ROOT/"runs/object-plant"
    traces=ROOT/"runs/plant-transfer"
    source=read(traces/"native.json")
    dt=source["dt"]
    limits=read(training/"independent.json")["thresholds"]
    checks={}
    for name in ("nn","pinn"):
        bundle=training/"paired"/name
        spec=importlib.util.spec_from_file_location("deploy_"+name,bundle/"predict.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        network=module.Plant(bundle)
        rows={}
        for file in sorted(traces.glob("trace*.csv")):
            data=np.genfromtxt(file,delimiter=",",names=True)
            x=np.column_stack((data["q"],data["v"]))
            u=data["u"][:-1]
            if not np.allclose(np.diff(data["t"]),dt,rtol=0,atol=1e-12):
                raise ValueError("Neural port requires a declared regular controller clock")
            teacher=network.step(x[:-1],u,dt)
            free=network.rollout(x[0],u,dt)[1:]
            rmse=np.sqrt(np.mean((free-x[1:])**2,axis=0))
            rows[file.name]=dict(reference_sha256=digest(file),initial_state=x[0],steps=len(free),dt=dt,
                teacher_forced_rmse=np.sqrt(np.mean((teacher-x[1:])**2,axis=0)),free_rmse=rmse,
                passed=bool(rmse[0]<limits["arbitrary_y_rmse"] and rmse[1]<limits["arbitrary_v_rmse"]))
        checks[name]=dict(weights_sha256=digest(bundle/"weights.npz"),trajectories=rows)
    passed=all(len(b["trajectories"])==3 and all(v["passed"] for v in b["trajectories"].values()) for b in checks.values())
    write(training/"native_traces.json",dict(passed=passed,threshold_source="independent.json",source_model=source["model"],results=checks))
    if not passed:
        raise ValueError("Native Simulink trajectory exceeded the previously declared tolerance")
    print("Native Simulink reference versus two deployed neural models verified")


if __name__=="__main__":
    run()
