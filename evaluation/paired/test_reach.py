"""Replay submitted Python self-tests against isolated engineering faults after first-pass grading."""
import ast
import importlib.util
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch
from prepare import HERE, ROOT


def source_from_receipt(path):
    record=json.loads(path.read_text(encoding="utf-8-sig"))
    commands=record["tests"]
    if len(commands)!=1:
        raise ValueError("Review new test format before running")
    cmd=commands[0].replace("\r\n","\n")
    heredoc=re.search(r"(?ms)^@'\n(.*?)\n'@ \|",cmd)
    if heredoc:
        body=heredoc.group(1)
    else:
        body=cmd.split(" -c '",1)[1].rstrip("'").replace("''","'")
    tree=ast.parse(body)
    # Re-execute the submitted numeric test body, never its shell wrapper.
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            deps=[n.name for n in node.names]
        elif isinstance(node,ast.ImportFrom):
            deps=[node.module]
        else:
            continue
        if any(d.split(".")[0] not in {"torch","numpy","solution","actuator","teacher"} for d in deps):
            raise ValueError("Unexpected test dependency")
    return compile(tree,str(path),"exec")


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def inject(module,kind):
    heat,lift,cost=module.heat,module.lift,module.policy_cost
    if kind=="time_unit":
        module.heat=lambda net,z,c:heat(net,z,dict(c,time_scale_min=c.get("time_scale_min",1.)/60))
    if kind=="missing_equation":
        module.heat=lambda net,z,c:torch.zeros_like(z[:,:1]) if "C" not in c or "G" not in c else heat(net,z,c)
    if kind=="second_increment":
        def wrong_lift(config,n):
            f,p=lift(config,n)
            p=np.array(p)
            if n>=3:
                p[2:,1]=0
            return f,p
        module.lift=wrong_lift
    if kind=="feedthrough":
        module.lift=lambda c,n:lift(dict(c,d=0.),n)
    if kind=="perception_bypass":
        module.policy_cost=lambda a,p,x,r,u,c,coupled:cost(a,p,x,r,u,c,False)
    if kind=="actor_detached":
        module.policy_cost=lambda *args,**kwargs:cost(*args,**kwargs).detach().requires_grad_(True)


def run():
    folder=ROOT/"runs/paired"
    first=json.loads((folder/"results.json").read_text(encoding="utf-8"))
    kinds=("time_unit","missing_equation","second_increment","feedthrough","perception_bypass","actor_detached")
    observations={}
    summary={}
    for name,case in first["trials"].items():
        trial=folder/name/"first"
        load(trial/"actuator.py","actuator")
        load(trial/"teacher.py","teacher")
        body=source_from_receipt(trial/"receipt.json")
        observations[name]={}
        for kind in ("clean",*kinds):
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(470); np.random.seed(470); random.seed(470)
            solution=load(trial/"solution.py","solution")
            inject(solution,kind)
            try:
                exec(body,{"__name__":"__main__"})
                error=None
            except Exception as exc:
                error=type(exc).__name__+": "+str(exc).split("\n")[0]
            if kind=="clean" and error:
                raise ValueError(f"Self-test {name} fails on original: {error}")
            observations[name][kind]=dict(self_test_failed=error is not None,diagnostic=error)
        aggregate=summary.setdefault(case["group"],dict(detected=0,opportunities=0))
        aggregate["detected"]+=sum(observations[name][k]["self_test_failed"] for k in kinds)
        aggregate["opportunities"]+=len(kinds)
    (folder/"test-reach.json").write_text(json.dumps(dict(
        design="Post-hoc mutation analysis of independently submitted tests; clean build must pass first. Each mutant is introduced separately.",
        groups=summary,observations=observations),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(summary)


if __name__=="__main__":
    run()
