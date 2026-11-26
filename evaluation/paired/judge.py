"""Blinded numerical acceptance. Run outside the participant workspace."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from actuator import Actuator
from teacher import stage, stage_cost


def model(solution):
    linear = torch.nn.Linear(2,1,dtype=torch.float64)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[.71,-.23]]))
        linear.bias.fill_(.1)
    zp=torch.tensor([[.2,.13],[.6,-.18],[.9,.26]],dtype=torch.float64)
    cfg=dict(C=81.,G=3.4,time_scale_min=2.3,time_origin_min=7.,y_shift_C=18.,y_scale_K=9.,Ta_C=21.)
    residual=solution.heat(linear,zp.clone(),cfg)
    temperature=18.+9.*linear(zp)
    expected=81.*9.*linear.weight[0,0]/(2.3*60.)+3.4*(temperature-21.)-1000*zp[:,1:2]
    assert residual.shape==(3,1)
    torch.testing.assert_close(residual,expected)
    grad=torch.autograd.grad(residual.sum(),linear.weight)[0]
    torch.testing.assert_close(grad,torch.autograd.grad(expected.sum(),linear.weight)[0])
    assert solution.heat(linear,zp,{k:v for k,v in cfg.items() if k!="G"}) is None


def mpc(solution):
    rng=np.random.default_rng(871)
    for n in (1,3,7):
        for direct in (0.,.37):
            cfg=dict(a=.81,b=.17,c=-1.2,d=direct,g=.33)
            f,p=solution.lift(cfg,n)
            for _ in range(5):
                initial=rng.normal(size=3)
                changes=rng.normal(size=n)
                plant=Actuator(cfg,*initial[:2])
                obs=[plant.step(cmd) for cmd in initial[2]+np.cumsum(changes)]
                np.testing.assert_allclose(np.asarray(f)@initial+np.asarray(p)@changes,obs,atol=1e-10,rtol=0)


def pgac(solution):
    cfg=dict(N=4,beta=.32,a=.86,b=.17,limit=2.3,center=.14,scale=.77,q=2.7,r=.035,s=.18,discount=.91)
    x=torch.tensor([[.4],[-.7],[.2]],dtype=torch.float64)
    ref=torch.tensor([[.8],[.4],[-.1]],dtype=torch.float64)
    last=torch.tensor([[.13],[-.35],[.52]],dtype=torch.float64)
    actor=torch.nn.Linear(2,1,dtype=torch.float64)
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[1.2,.29]]))
        actor.bias.fill_(.13)
    perception=lambda z:.11*z[:,:1]-.037*z[:,1:]
    w=actor.weight.detach().numpy().ravel()
    for coupled in (False,True):
        def oracle(weights):
            costs=[]
            for state,target,prev in zip(x.flatten().tolist(),ref.flatten().tolist(),last.flatten().tolist()):
                score=0.
                for k in range(cfg["N"]):
                    nxt,action=stage(lambda arr: np.dot(weights,arr)+.13,state,target,prev,cfg)
                    if coupled:
                        nxt+=.11*state-.037*action
                    score+=cfg["discount"]**k*stage_cost(nxt,target,action,prev,cfg)
                    state,prev=nxt,action
                costs.append(score)
            return np.mean(costs)
        actual=solution.policy_cost(actor,perception,x,ref,last,cfg,coupled)
        assert actual.ndim==0
        np.testing.assert_allclose(float(actual.detach()),oracle(w),atol=1e-8)
        grad=torch.autograd.grad(actual,actor.weight)[0].detach().numpy().ravel()
        h=1e-5
        expected=[(oracle(w+v*h)-oracle(w-v*h))/(2*h) for v in np.eye(2)]
        np.testing.assert_allclose(grad,expected,rtol=1e-5,atol=1e-6)


def grade(folder):
    path=Path(folder).resolve()
    sys.path.insert(0,str(path))
    if (path/"library").exists():
        sys.path.insert(0,str(path/"library"))
    try:
        spec=importlib.util.spec_from_file_location("participant",path/"solution.py")
        solution=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
    except Exception as exc:
        return {name:{"pass":False,"error":repr(exc)} for name in ("model","mpc","pgac")}
    result={}
    for test in (model,mpc,pgac):
        try:
            test(solution)
            result[test.__name__]={"pass":True}
        except Exception as exc:
            result[test.__name__]={"pass":False,"error":repr(exc)}
    return result


if __name__=="__main__":
    print(json.dumps(grade(sys.argv[1]),indent=2))
