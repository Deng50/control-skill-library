"""External oracle for the complete unseen controller delivery; no participant implementation imports."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from process import step, baseline, scenario

HERE=Path(__file__).resolve().parent


def assess(folder):
    with open(HERE/"settings.json",encoding="utf-8") as f:
        base=json.load(f)
    spec=importlib.util.spec_from_file_location("entry",folder/"solution.py")
    impl=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(folder))
    spec.loader.exec_module(impl)
    def cost(x,r,d,last,cfg,U):
        Y=[]
        for u,dist in zip(U,d):
            x,y=step(x,u,dist,cfg)
            Y.append(y)
        y=np.array(Y)
        return y,float(np.sum(cfg["q"]*(y-r)**2+cfg["r_u"]*U**2+cfg["s"]*np.diff(np.r_[last,U])**2))
    def closed(name,cfg,use_mpc):
        profile=base[name]
        x,last=profile["x0"],profile["last"]
        rows=[]
        for k in range(profile["steps"]):
            t=k*cfg["dt"]
            r,d=scenario(name,t+cfg["dt"]*np.arange(cfg["N"]))
            u=float(impl.plan(x,r,d,last,cfg)["u"][0]) if use_mpc else baseline(x,r[0],last,cfg)
            x,y=step(x,u,d[0],cfg)
            rows.append([t,r[0],d[0],last,u,y])
            last=u
        a=np.array(rows)
        rms=lambda z: np.sqrt(np.mean(z*z))
        return a,float(rms(a[:,5]-a[:,1])+.015*rms(a[:,4])+.03*rms(a[:,4]-a[:,3]))
    outcome={}
    for n,m,dt,last in ((8,3,.2,.24),(5,2,.31,-.39),(3,1,.12,.81)):
        cfg=dict(base,**base["candidates"][1],N=n,M=m,dt=dt)
        x=np.array([-.37,.15])
        r=np.linspace(.94,.12,n);d=np.sin(np.arange(n)*.7)
        plan=impl.plan(x,r,d,last,cfg)
        U=np.asarray(plan["u"],float)
        y,J=cost(x,r,d,last,cfg,U)
        assert U.shape==(n,) and np.isfinite(U).all()
        np.testing.assert_allclose(plan["y"],y,rtol=0,atol=2e-7)
        np.testing.assert_allclose(plan["J"],J,rtol=1e-6,atol=1e-7)
        assert np.min(U)>=cfg["lo"]-1e-7 and np.max(U)<=cfg["hi"]+1e-7
        assert np.max(abs(np.diff(np.r_[last,U])))<=cfg["rate"]*dt+1e-7
        np.testing.assert_allclose(U[m-1:],U[m-1],atol=1e-8)
        expand=lambda v:np.r_[v,np.repeat(v[-1],n-m)]
        objective=lambda v:cost(x,r,d,last,cfg,expand(v))[1]
        du=cfg["rate"]*dt
        diff=lambda v:np.diff(np.r_[last,v])
        cons=[dict(type="ineq",fun=lambda v:du-diff(v)),dict(type="ineq",fun=lambda v:du+diff(v))]
        oracle=minimize(objective,np.full(m,last),method="SLSQP",constraints=cons,bounds=[(cfg["lo"],cfg["hi"])]*m,
                        options=dict(ftol=1e-12,maxiter=200))
        assert oracle.success and abs(J-oracle.fun)<2e-5
    outcome["multiple_periods_objective_bounds"]=True
    report=json.loads((folder/"summary.json").read_text(encoding="utf-8-sig"))
    scores=[closed("design",dict(base,**p),True)[1] for p in base["candidates"]]
    picked=int(report["picked"])
    assert picked in range(len(scores)) and scores[picked]<=min(scores)+1e-6
    cfg=dict(base,**base["candidates"][picked])
    a,sc=closed("validation",cfg,True)
    with (folder/"validation.csv").open(encoding="utf-8-sig") as f:
        stored=np.array([[float(row[key]) for key in ("t","r","d","previous","u","y")] for row in csv.DictReader(f)])
    np.testing.assert_allclose(a,stored,atol=2e-6,rtol=1e-6)
    outcome.update(selected=picked,design=scores,closed_loop_score=sc,baseline_score=closed("validation",cfg,False)[1])
    return outcome


if __name__=="__main__":
    try:
        value=assess(Path(sys.argv[1]).resolve())
        print(json.dumps(dict(passed=True,metrics=value)))
    except Exception as exc:
        print(json.dumps(dict(passed=False,error=repr(exc))))
