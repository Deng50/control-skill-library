"""Replay a four-parameter grey-box model on the archived second-order dataset."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.io import loadmat
from scipy.signal import lfilter, ss2tf

from .io import digest, runtime, write


def simulate(params,u,dt):
    if np.shape(params)!=(4,):
        raise ValueError("This model expects named [a,c,b,f0], not six A/B entries")
    a,c,b,f0=params
    aug=np.array([[0,1,0,0],[c,a,b,f0],[0,0,0,0],[0,0,0,0]],float)
    disc=expm(aug*dt)
    A,B=disc[:2,:2],disc[:2,2:]
    y=np.zeros(len(u))
    for j,inputs in enumerate((u,np.ones_like(u))):
        num,den=ss2tf(A,B,np.array([[1.,0.]]),np.zeros((1,2)),input=j)
        y+=lfilter(num[0],den,inputs)
    return y


def run(source,out):
    source=Path(source)
    t,u,y=loadmat(source/"Data4Ident.mat")["Data_collect"].T
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt) or t[0]!=0 or abs(y[0])>1e-9:
        raise ValueError("This archived case assumes a regular time axis and zero initial state")
    n=int(len(t)*.65)
    fit=least_squares(lambda p:simulate(p,u[:n],dt)-y[:n],[-2.,-5.,5.,0.],
                      bounds=([-30,-50,.01,-10],[-.1,-.1,50,10]))
    if not fit.success:
        raise RuntimeError(fit.message)
    pred=simulate(fit.x,u,dt)
    archived=loadmat(source/"model_param.mat")["x_model"].ravel()
    if archived.shape!=(4,) or not np.isfinite(archived).all():
        raise ValueError("Expected archived [a,c,b,f0] from the scoped four-parameter case")
    rank=int(np.linalg.matrix_rank(fit.jac))
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    np.savetxt(out/"trajectory.csv",np.column_stack((t,u,y,pred)),delimiter=",",comments="",header="t,u,y,y_free_run")
    rmse=float(np.sqrt(np.mean((pred[n:]-y[n:])**2)))
    write(out/"report.json",dict(runtime=runtime(),source_sha256=digest(source/"Data4Ident.mat"),
          dt_s=dt,names=["a","c","b","f0"],fitted=fit.x,archived=archived,train_rows=n,
          test_rows=len(t)-n,test_kind="chronological suffix of one simulated trajectory",
          heldout_rmse=rmse,sensitivity_rank=rank,source_model="zero initial state, piecewise constant input, exact ZOH",
          heldout_tolerance=.03,passed=rank==4 and rmse<.03))
    if rank!=4 or rmse>=.03:
        raise ValueError("Rank or held-out tracking tolerance failed")
    print("Fitted [a,c,b,f0]",fit.x,"held-out RMSE",rmse)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",help="directory containing Data4Ident.mat and model_param.mat")
    p.add_argument("--out",default="runs/identification")
    a=p.parse_args()
    run(a.source,a.out)
