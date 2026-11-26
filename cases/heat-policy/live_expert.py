"""Optional real-time benchmark for an installation that can load CasADi/IPOPT."""
import argparse
import importlib.metadata
import json
import time
import numpy as np
from heat_data import C0, DT, OUT, create_plant, reference_stage_functions, write


def measure(steps):
    if not 1<=steps<=700:
        raise ValueError("steps must be between 1 and the archived episode length")
    try:
        from nmpc_casadi_align_5s_fix_torque import create_default_controller
    except (ImportError,OSError) as err:
        result=dict(status="dependency_unavailable",casadi_version=importlib.metadata.version("casadi"),
                    error=str(err),solve_latency_ms=None)
        write("live_expert.json",result)
        print(json.dumps(result,ensure_ascii=True))
        return
    meta=json.loads((OUT/"data.json").read_text("utf-8"))
    data=np.load(OUT/"expert.npz",allow_pickle=False)
    # Training case only; no contact with held-out scenarios.
    info=meta["episodes"][0]
    ix=np.flatnonzero(data["eid"]==info["id"])
    plant=create_plant(data["x"][ix[0]],info["speed"])
    c=create_default_controller(dt=DT,horizon=50)
    stage=reference_stage_functions()
    rows=[]
    for i in ix[:steps]:
        t,ref,amb=data["t"][i],data["ref"][i],data["amb"][i]
        start=time.perf_counter_ns()
        u,plan=c.solve(plant.x,ref,amb,C0,plant.v_seq,t)
        elapsed=(time.perf_counter_ns()-start)*1e-6
        states=c.opti.value(c.X).T
        speeds=np.interp(t+np.arange(51)*DT,plant.v_seq[:,0],plant.v_seq[:,1])
        terms,_=stage(states[:-1,:3],plan.T,states[1:,:3],np.full(50,ref),speeds[:-1],speeds[1:])
        np.testing.assert_allclose(terms.sum(),c.opti.value(c.opti.f),rtol=1e-7,atol=1e-6)
        plant.step(*u,amb,C0,t_now=t)
        rows.append(dict(t_s=t,status=c.opti.stats()["return_status"],solve_ms=elapsed,
                         objective=float(c.opti.value(c.opti.f)),recomputed_objective=float(terms.sum()),
                         action=u.tolist(),archive_action=data["u"][i].tolist()))
    write("live_expert.json",dict(status="success",casadi_version=importlib.metadata.version("casadi"),
          episode=info["name"],first_solve_ms=rows[0]["solve_ms"],measurements=rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=3)
    measure(p.parse_args().steps)
