"""Import real NMPC records and independently replay them on the original plant."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
LIBRARY = Path(__file__).resolve().parents[2]
SOURCE = LIBRARY.parent / "素材" / "pgac"
ARCHIVE = LIBRARY.parent / "备份2026.5.15"
KERNEL = SOURCE / "pgac_auto_train_agent_v1" / "kernels"
OUT = LIBRARY / "runs" / "heat-policy"
DT = 5.0
C0 = 4.493 * 32
Q, R, S = 1 / 16, 1 / 1600000000, 300000000.0
LO = np.array([200., 0., 0.])
HI = np.array([1000., 4000., 1.])
PREVIOUS = np.array([200., 0., 1.])
# Split before reading trajectory values. The extended filenames correspond to
# the actual archive collector's __main__ loop (horizon=50, Tm0=Tamb+10 K).
SPLIT = {
    "train": [("30",253,283), ("60",253,288), ("90",253,293), ("udds",253,298),
              ("nedc",253,283), ("120",253,293), ("30",263,298), ("60",263,293),
              ("90",263,288), ("udds",263,283), ("nedc",263,293), ("120",263,283)],
    "validation": [("udds",253,288), ("60",263,298)],
    "test": [("nedc",263,288), ("90",253,298)],
}
EXPERT = "nmpc_casadi_align_5s_fix_torque.py"
PLANT = "system_test_align_delay.py"
MODEL = "thermal-management-model.py"

sys.path.insert(0, str(SOURCE))
from system_test_align_delay import SystemTestPlantAlign, load_thermal_model_class


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def create_plant(x, speed):
    return SystemTestPlantAlign(load_thermal_model_class(SOURCE / MODEL)(), x0=x, ts=DT,
                                speed_csv=KERNEL / "Speed_seq" / f"speed_{speed}.csv")


def power(u):
    n = u[..., 0]
    return u[..., 1] + 2e-5 * n**2 + .0038 * n + .2589


def reference_stage_functions():
    """Execute only original scalar cost/torque expressions, with NumPy operations.

    CasADi is needed for solving; these expressions also work numerically with
    NumPy and the source's unbounded regular-grid interpolants.
    """
    parsed = ast.parse((SOURCE / EXPERT).read_text(encoding="utf-8-sig"))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "MPCCasadiAlign")
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build")
    loop = next(n for n in build.body if isinstance(n, ast.For))
    stage = next(n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "J")
    p_btm = next(n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "p_btm")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    bodies = [methods[k] for k in ("calc_n_m", "calc_t_m", "calc_a_k", "calc_f_trac")]
    bodies += [n for n in build.body if isinstance(n, ast.FunctionDef) and n.name in ("calc_i", "capacity_loss")]
    model = create_plant([253.15, 263.15, 1.], "60").model
    ns = {"np": np, "ca": SimpleNamespace(if_else=np.where, fmax=np.maximum, sqrt=np.sqrt,
          fabs=np.abs, exp=np.exp, vertcat=lambda soc,tb: np.stack([soc,tb],axis=-1)),
          "ocv": model.ocv_interp, "rb": model.rb_interp}
    exec(compile(ast.Module(body=bodies,type_ignores=[]), str(SOURCE / EXPERT), "exec"), ns)
    obj = SimpleNamespace(model=model, dt=DT)
    from types import MethodType
    for name in ("calc_n_m","calc_t_m","calc_a_k","calc_f_trac"):
        setattr(obj,name,MethodType(ns[name],obj))
    ns["self"] = obj
    stage_expr = compile(ast.Expression(body=stage.value), str(SOURCE / EXPERT),"eval")
    power_expr = compile(ast.Expression(body=p_btm.value), str(SOURCE / EXPERT),"eval")

    def evaluate(x, u, xn, ref, vk, vkp1):
        # calc_f_trac() uses a scalar branch. Evaluate archived torque without changing it.
        nm = np.array([obj.calc_n_m(v) for v in vk])
        tm = np.array([obj.calc_t_m(a,b) for a,b in zip(vk,vkp1)])
        current = np.asarray(ns["calc_i"](x.T, u.T, nm, tm))
        dc = ns["capacity_loss"](xn[:,0],current)
        p = eval(power_expr, ns, {"uk": u.T})
        np.testing.assert_allclose(p, power(u), rtol=1e-14)
        terms = np.column_stack([Q*(xn[:,0]-ref)**2, R*p**2, S*dc**2])
        ns["self"].P_ref = ref
        j = eval(stage_expr, ns, dict(J=0,Q1=Q,R1=R,S1=S,tb_next=xn[:,0],p_btm=p,delta_c=dc))
        np.testing.assert_allclose(j,terms.sum(axis=1),rtol=1e-12)
        return terms, current
    return evaluate


def check_episode(x,xn,u,previous,times,delays,delays_next):
    if not all(np.isfinite(a).all() for a in (x,xn,u,previous,times,delays,delays_next)):
        raise ValueError("Non-finite transition")
    if not np.array_equal(np.diff(times), np.full(len(times)-1, DT)) or times[0] != 0:
        raise ValueError("Wrong sample time")
    if not np.array_equal(previous[0], PREVIOUS) or not np.array_equal(previous[1:],u[:-1]):
        raise ValueError("Previous executed action misaligned")
    if np.any(u < LO-1e-4) or np.any(u > HI+1e-4):
        raise ValueError("Action mapping/bounds mismatch")
    if np.max(np.abs(x[1:]-xn[:-1])) > 2e-7 or not np.array_equal(delays_next[:-1],delays[1:]):
        raise ValueError("Original state/delay replay mismatch")


def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    verified = {}
    for name in (EXPERT,PLANT,MODEL):
        if sha(ARCHIVE / name) != sha(SOURCE / name):
            raise ValueError(f"Archive source changed: {name}")
        verified[name] = sha(SOURCE / name)
    records, collected = [], {}
    for part,specs in SPLIT.items():
        for speed,tamb,tref in specs:
            eid = len(records)
            name = f"speed_{speed}_Tamb{tamb}_Tb0{tamb}_Tm0{tamb+10}_Tbref_{tref}"
            archive = ARCHIVE / "data_collection" / f"nmpc_data_{name}.npz"
            with np.load(archive,allow_pickle=False) as d:
                x = np.column_stack([d[k] for k in ("Tb","Tm","SOC")])
                u = np.column_stack([d[k] for k in ("pump_n","Ph","alpha")])
                t, amb, ref, velocity = (d[k] for k in ("Time","T_amb","Tb_ref","vehicle_speed"))
            plant = create_plant(x[0],speed)
            if sha(plant.speed_csv_path) != sha(ARCHIVE / "Speed_seq" / f"speed_{speed}.csv"):
                raise ValueError("Archive speed profile differs")
            speeds = np.interp(np.arange(len(t)+1)*DT,plant.v_seq[:,0],plant.v_seq[:,1])
            np.testing.assert_allclose(speeds[:-1],velocity,atol=1e-10)
            np.testing.assert_allclose(x[0],[tamb+.15,tamb+10+.15,1])
            np.testing.assert_allclose(amb,tamb+.15)
            np.testing.assert_allclose(ref,tref+.15)
            xn, delay, delay_n = [], [], []
            for k,ui in enumerate(u):
                delay.append(plant._x[3:].copy())
                xn.append(plant.step(*ui,amb[k],C0,t_now=t[k]))
                delay_n.append(plant._x[3:].copy())
            xn,delay,delay_n = map(np.asarray,(xn,delay,delay_n))
            previous = np.vstack([PREVIOUS,u[:-1]])
            check_episode(x,xn,u,previous,t,delay,delay_n)
            duplicate = (u[1:] == u[:-1]).all(axis=1)
            norm = (u-LO)/(HI-LO)
            records.append(dict(id=eid,split=part,name=name,speed=speed,rows=len(t),archive_sha256=sha(archive),
                                speed_sha256=sha(plant.speed_csv_path),max_state_replay_abs=np.max(np.abs(x[1:]-xn[:-1]),axis=0).tolist(),
                                exact_repeats=int(duplicate.sum()),near_low_2pct=(norm<=.02).sum(axis=0).tolist(),
                                near_high_2pct=(norm>=.98).sum(axis=0).tolist()))
            for key,val in dict(eid=np.full(len(t),eid),t=t,x=x,xn=xn,u=u,previous=previous,
                                delay=delay,delay_next=delay_n,ref=ref,amb=amb,vk=speeds[:-1],vk1=speeds[1:]).items():
                collected.setdefault(key,[]).append(val)
            print(part,name,len(t),records[-1]["max_state_replay_abs"],flush=True)
    np.savez_compressed(OUT/"expert.npz",**{k:np.concatenate(v) for k,v in collected.items()})
    write("data.json",dict(dt_s=DT,expert_horizon_steps=50,weights=[Q,R,S],C0_Ah=C0,
                           teacher="archived executed actions",source_sha256=verified,kernel_sha256=sha(KERNEL/"pgac_core.py"),
                           collector_sha256=sha(ARCHIVE/"data_collector.py"),npz_sha256=sha(OUT/"expert.npz"),episodes=records,
                           solver_status="unrecorded; collector repeats last successful u on failure",
                           per_solve_latency_ms=None,source_root=str(SOURCE),archive_root=str(ARCHIVE)))


if __name__ == "__main__":
    collect()
