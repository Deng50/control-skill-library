"""Exact ZOH spring plant, finite-horizon constrained MPC and independent closed-loop scoring."""
import argparse
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy.linalg import expm
from scipy.optimize import LinearConstraint, minimize

from .io import digest, read, runtime, write
from .prediction import lifted

CASE = Path(__file__).resolve().parents[1] / "cases" / "spring"


class Spring:
    def __init__(self, name="reference", dt=None):
        spec = read(CASE / (name + ".json"))
        self.dt = float(dt if dt is not None else spec["interface"]["sample_time"])
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        A, B, E = [np.asarray(spec[k]) for k in "ABE"]
        aug = np.zeros((4,4))
        aug[:2] = np.column_stack((A,B,E))
        step = expm(aug*self.dt)
        self.A, self.B, self.E = step[:2,:2], step[:2,2:3], step[:2,3:4]
        self.C = np.asarray(spec["C"])
        self.limit = np.array([v["upper"] for v in spec["interface"]["states"]])
        self.force = spec["interface"]["controls"][0]["upper"]

    def step(self, x, u, d):
        return self.A @ x + self.B[:,0]*u + self.E[:,0]*d


class Controller:
    """Open-loop solve over absolute forces; hold the last move after M."""
    def __init__(self, model, config):
        self.model = model
        n = int(config["horizon"])
        m = int(config.get("moves",n))
        self.force = float(config.get("force_limit",model.force))
        q,r,s = [float(config[key]) for key in ("q","r","s")]
        if not (1 <= m <= n and q > 0 and r > 0 and s >= 0 and 0 < self.force <= model.force):
            raise ValueError("Invalid horizon, cost or actuator bounds")
        self.n = n
        self.dt = model.dt
        self.expansion = np.eye(m)[np.minimum(np.arange(n),m-1)]
        self.O,T = lifted(model.A,model.B,np.eye(2),np.zeros((2,1)),n)
        _,Td = lifted(model.A,model.E,np.eye(2),np.zeros((2,1)),n)
        self.dist = Td @ np.ones(n)
        self.T = T @ self.expansion
        self.y_map = self.T[::2]
        self.diff = (np.eye(n)-np.eye(n,k=-1)) @ self.expansion
        self.lim = np.tile(model.limit,n)
        self.Wy = q / model.limit[0]**2
        self.Wdu = s / self.force**2
        self.H = (self.Wy*self.y_map.T@self.y_map
                  + r/self.force**2*self.expansion.T@self.expansion
                  + self.Wdu*self.diff.T@self.diff)

    def __call__(self, x, reference, disturbance, previous):
        free = self.O @ x + self.dist*disturbance
        f = self.Wy*self.y_map.T@(free[::2]-reference) - self.Wdu*self.diff[0]*previous
        plan = np.linalg.solve(self.H,-f)
        if np.max(np.abs(plan)) <= self.force and np.all(np.abs(free+self.T@plan) <= self.lim):
            return float(plan[0])
        constraint = LinearConstraint(self.T,-self.lim-free,self.lim-free)
        result = minimize(lambda u: 0.5*u@self.H@u + f@u, np.clip(plan,-self.force,self.force),
                          jac=lambda u:self.H@u+f, bounds=[(-self.force,self.force)]*len(plan),
                          constraints=[constraint], method="SLSQP", options={"ftol":1e-10,"maxiter":100})
        if not result.success or np.max(np.abs(free+self.T@result.x)-self.lim) > 1e-7:
            raise RuntimeError(f"MPC infeasible or solver failure: {result.message}")
        return float(result.x[0])


def rollout(plant, policy, scenario, force=None):
    if hasattr(policy,"dt") and not np.isclose(plant.dt,policy.dt):
        raise ValueError("Controller and plant sample periods differ; explicit resampling required")
    x = np.array(scenario["initial_state"],dtype=float)
    ref = float(scenario["reference"][0])
    d = float(scenario["disturbance"][0])
    previous = 0.
    rows, times = [], []
    for k in range(scenario["steps"]):
        start = perf_counter_ns()
        u = float(policy(x.copy(),ref,d,previous))
        times.append((perf_counter_ns()-start)/1e6)
        xn = plant.step(x,u,d)
        if not np.isfinite(xn).all():
            raise ValueError("Non-finite closed-loop state")
        rows.append([k*plant.dt,*x,ref,d,previous,u,*xn])
        x,previous = xn,u
    data = np.array(rows)
    errors = data[:,7]-ref
    lim = force if force is not None else plant.force
    metrics = dict(rmse_m=float(np.sqrt(np.mean(errors**2))),
                   total_variation_N=float(np.sum(np.abs(data[:,6]-data[:,5]))),
                   rms_force_N=float(np.sqrt(np.mean(data[:,6]**2))),
                   max_force_violation_N=float(max(0,np.max(abs(data[:,6]))-lim)),
                   max_position_violation_m=float(max(0,np.max(abs(data[:,7]))-plant.limit[0])),
                   max_speed_violation_mps=float(max(0,np.max(abs(data[:,8]))-plant.limit[1])),
                   mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)))
    return data,metrics


COLUMNS = "time_s,q_m,v_mps,reference_m,disturbance_N,previous_u_N,u_N,next_q_m,next_v_mps"


def feasible(m):
    return max(m["max_force_violation_N"],m["max_position_violation_m"],m["max_speed_violation_mps"]) <= 1e-7


def score(m):
    return m["rmse_m"]/1.0 + 0.002*m["rms_force_N"]/4 + 0.0001*m["total_variation_N"]/4


def load_controller_factory(entry):
    if entry is None:
        return Controller
    module, name = entry.split(":", 1)
    return getattr(import_module(module), name)


def tune(out, dt, config_path=CASE/"controller.json", factory=Controller):
    source = read(config_path)
    settings = source["parameters"]
    base = dict(settings, moves=settings.get("moves",settings["horizon"]),
                force_limit=settings.get("force_limit",4.))
    conditions = []
    for stage in ("tuning","verification","test"):
        group = source[stage+"_scenarios"]
        if len(group) != 1:
            raise ValueError("This benchmark expects one distinct condition per tuning, verification and test stage")
        sc = group[0]
        conditions.append(tuple(tuple(sc[key]) for key in ("initial_state","reference","disturbance")))
    if len(set(conditions)) != len(conditions):
        raise ValueError("Hold out distinct operating conditions for verification and test")
    model,plant = Spring("nominal",dt),Spring("reference",dt)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    def evaluate(cfg, stage):
        sc = source[stage+"_scenarios"][0]
        return rollout(plant,factory(model,cfg),sc,cfg["force_limit"])
    trials = []
    candidates = [(base["r"],base["s"]),*source.get("candidate_weights",[(.05,.03),(.015,.015),(.005,.01),(.002,.005),(.001,.003)])]
    space = source.get("parameter_space",{})
    budget = int(source.get("budget",len(candidates)))
    if budget < 1:
        raise ValueError("Tuning budget must include the baseline")
    for r,s in dict.fromkeys(tuple(pair) for pair in candidates):
        weights = {"r":r,"s":s}
        if any(not space[key]["lower"] <= weights[key] <= space[key]["upper"] for key in weights if key in space):
            if (r,s)==(base["r"],base["s"]):
                raise ValueError("Baseline weights are outside configured search bounds")
            continue
        if len(trials)>=budget:
            break
        cfg = dict(base,r=r,s=s)
        try:
            _,m = evaluate(cfg,"tuning")
            trials.append(dict(config=cfg,metrics=m,score=score(m),feasible=feasible(m)))
        except RuntimeError as exc:
            trials.append(dict(config=cfg,feasible=False,reason=str(exc)))
    pool = [t for t in trials if t["feasible"]]
    if not pool:
        raise RuntimeError("No feasible tuning candidate")
    candidate = min(pool,key=lambda t:t["score"])["config"]
    _,baseline_val = evaluate(base,"verification")
    _,candidate_val = evaluate(candidate,"verification")
    if feasible(candidate_val) and (not feasible(baseline_val) or score(candidate_val)<score(baseline_val)):
        chosen = candidate
    elif feasible(baseline_val):
        chosen = base
    else:
        raise RuntimeError("Neither controller passed independent validation")
    config = dict(chosen,sample_time_s=dt,model_sha256=digest(CASE/"nominal.json"))
    tests = {}
    for key,cfg in [("baseline",base),("selected",chosen)]:
        data,tests[key] = evaluate(cfg,"test")
        np.savetxt(out/(key+".csv"),data,header=COLUMNS,delimiter=",",comments="")
    mismatched = False
    try:
        rollout(Spring("reference",dt*2),factory(model,chosen),source["test_scenarios"][0])
    except ValueError as exc:
        mismatched = "sample periods" in str(exc)
    _,adapted = rollout(Spring("reference",dt*2),factory(Spring("nominal",dt*2),chosen),source["test_scenarios"][0],chosen["force_limit"])
    report = dict(runtime=runtime(),scenario_file_sha256=digest(config_path),controller_factory=f"{factory.__module__}:{factory.__name__}",trials=trials,
                  validation=dict(baseline=baseline_val,candidate=candidate_val),selected=config,
                  test=tests,period_mismatch_rejected=mismatched,rediscretized_double_dt=adapted)
    write(out/"report.json",report)
    if not mismatched or not feasible(tests["selected"]) or not feasible(adapted):
        raise RuntimeError("Closed-loop acceptance check failed")
    write(out/"controller.json",config)
    print({k:round(v["rmse_m"],6) for k,v in tests.items()})
    return config


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/mpc")
    p.add_argument("--dt",type=float,default=0.2)
    p.add_argument("--config",default=str(CASE/"controller.json"))
    p.add_argument("--controller-factory",help="Importable module:factory(model,config) returning a policy with dt")
    args=p.parse_args()
    tune(args.out,args.dt,args.config,load_controller_factory(args.controller_factory))
