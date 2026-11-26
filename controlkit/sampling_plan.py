"""Review a spring expert sampling design and pilot only the training conditions."""
import argparse
import numpy as np

from .io import digest, read, runtime, write
from .spring import CASE, Controller, Spring, feasible, rollout


def check_scenarios(scenarios, splits):
    ids=[i for group in ("train","validation","test") for i in splits[group]]
    if sorted(ids)!=list(range(len(scenarios))) or not all(splits.values()):
        raise ValueError("Supply a complete disjoint trajectory split")
    groups={i:group for group,indices in splits.items() for i in indices}
    seen={}
    names=set()
    interface=read(CASE/"reference.json")["interface"]
    fields={"initial_state":"states","reference":"outputs","disturbance":"disturbances"}
    for i,scene in enumerate(scenarios):
        if not scene["name"] or scene["name"] in names or type(scene["steps"]) is not int or scene["steps"]<1:
            raise ValueError("Each scenario needs a unique name and a positive number of steps")
        names.add(scene["name"])
        values=[]
        for key,channels in fields.items():
            signal=np.array(scene[key],dtype=float)
            bounds=interface[channels]
            low=[c["lower"] for c in bounds]
            high=[c["upper"] for c in bounds]
            if signal.shape!=(len(bounds),) or not np.all(np.isfinite(signal)&(signal>=low)&(signal<=high)):
                raise ValueError(f"{scene['name']}: invalid {key} dimensions or operating bounds")
            values.extend(signal)
        if seen.setdefault(tuple(values),groups[i])!=groups[i]:
            raise ValueError("Equivalent physical condition crosses expert trajectory partitions")


def run(controller, source, split, out, pilot_steps=6):
    if pilot_steps<1:
        raise ValueError("Pilot length must be positive")
    cfg=read(controller)
    if cfg["model_sha256"]!=digest(CASE/"nominal.json"):
        raise ValueError("Expert controller does not match the declared nominal model")
    scenes,groups=read(source)["scenarios"],read(split)
    check_scenarios(scenes,groups)
    plant=Spring("reference",cfg["sample_time_s"])
    expert=Controller(Spring("nominal",plant.dt),cfg)
    pilots=[]
    for i in groups["train"]:
        sc=scenes[i]
        preview=dict(sc,steps=min(pilot_steps,sc["steps"]))
        rows,metrics=rollout(plant,expert,preview,cfg["force_limit"])
        if not feasible(metrics):
            raise ValueError("Training preview violates a hard constraint in "+sc["name"])
        pilots.append(dict(id=i,name=sc["name"],rows=len(rows),initial_error_m=sc["reference"][0]-sc["initial_state"][0],
                           disturbance_N=sc["disturbance"][0],force_min_N=float(min(rows[:,6])),
                           force_max_N=float(max(rows[:,6]))))
    edge=.98*cfg["force_limit"]
    near_positive=[p["name"] for p in pilots if p["force_max_N"]>=edge]
    near_negative=[p["name"] for p in pilots if p["force_min_N"]<=-edge]
    advice=[]
    for names,sign in ((near_positive,1),(near_negative,-1)):
        if names:
            advice.append(dict(direction=sign,retain_boundary_conditions=names))
        else:
            advice.append(dict(direction=sign,recommended_trial=dict(initial_state=[-.3*sign,0],
                               reference=[.4*sign],disturbance=[-.12*sign],steps=36),
                               reason="Increase initial tracking error against disturbance; pilot before adding"))
    write(out,dict(runtime=runtime(),controller_sha256=digest(controller),scenario_sha256=digest(source),
          pilot_steps=pilot_steps,partitions=groups,training_pilots=pilots,recommendations=advice,
          near_limit_N=edge,design_units={"position":"m","velocity":"m/s","force":"N"}))
    print("Training pilots",len(pilots),"positive/negative limit coverage",len(near_positive),len(near_negative))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller",default="runs/mpc/controller.json")
    p.add_argument("--scenarios",default=str(CASE/"expert.json"))
    p.add_argument("--split",default=str(CASE/"split.json"))
    p.add_argument("--pilot-steps",type=int,default=6)
    p.add_argument("--out",default="runs/expert/plan.json")
    a=p.parse_args()
    run(a.controller,a.scenarios,a.split,a.out,a.pilot_steps)
