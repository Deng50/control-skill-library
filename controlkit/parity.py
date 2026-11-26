"""Compare a stateful Python plant with full-state traces captured on the reference plant."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .io import digest, read, write


def compare(time, inputs, states, reset, step, tolerance, channels):
    """reset(full_state) fully resets the adapter. step(u,dt) returns post-step state."""
    t=np.asarray(time,float)
    u=np.asarray(inputs,float)
    x=np.asarray(states,float)
    tol=np.asarray(tolerance,float)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or x.shape!=(len(t),len(channels)):
        raise ValueError("Require ordered timestamps and one full state record per timestamp")
    if len(u)!=len(t)-1 or tol.shape!=(x.shape[1],) or np.any(tol<=0):
        raise ValueError("Require one applied input per interval and per-state physical tolerances")
    if not all(np.isfinite(a).all() for a in (t,u,x,tol)):
        raise ValueError("Reference contains nonfinite values")
    modes={}
    for mode in ("one_step","free_rollout"):
        reset(x[0].copy())
        predicted=[]
        for k,dt in enumerate(np.diff(t)):
            if mode=="one_step":
                reset(x[k].copy())
            xn=np.asarray(step(u[k],float(dt)),float).copy()
            if xn.shape!=(x.shape[1],):
                raise ValueError("Port returned an incorrect state dimension")
            predicted.append(xn)
        predicted=np.asarray(predicted)
        error=predicted-x[1:]
        bad=np.argwhere(~np.isfinite(error)|(np.abs(error)>tol))
        first=None
        if len(bad):
            k,j=map(int,bad[0])
            value=predicted[k,j]
            first=dict(step=k+1,time=float(t[k+1]),channel=channels[j],reference=float(x[k+1,j]),
                       actual=float(value) if np.isfinite(value) else None)
        maxima=[float(np.max(abs(col))) if np.isfinite(col).all() else None for col in error.T]
        modes[mode]=dict(passed=not len(bad),maximum_absolute_error=maxima,first_failure=first)
    return modes


class SecondOrder:
    """Original ode3 for a held input, physical state order [displacement, velocity]."""
    def reset(self,x):
        self.x=x.copy()

    def step(self,u,h):
        def f(x):
            q,v=x
            return np.array([v,-9*q-4.2*v+9*u+1])
        k1=f(self.x)
        k2=f(self.x+h*k1/2)
        k3=f(self.x+.75*h*k2)
        self.x=self.x+h*(2*k1/9+k2/3+4*k3/9)
        return self.x


def run(folder):
    folder=Path(folder)
    source=read(folder/"native.json")
    rows={}
    plant=SecondOrder()
    for i in range(1,len(source["trajectories"])+1):
        file=folder/f"trace{i}.csv"
        with file.open(encoding="utf-8-sig") as stream:
            data=np.array([[float(row[k]) for k in ("t","u","q","v")] for row in csv.DictReader(stream)])
        def replay(u):
            return compare(data[:,0],u,data[:,2:],plant.reset,plant.step,[1e-10,1e-10],["q","v"])
        commands=data[:-1,1]
        rows[file.name]=dict(sha256=digest(file),aligned=replay(commands),delayed=replay(np.r_[commands[0],commands[:-1]]))
    ok=all(all(check["passed"] for check in row["aligned"].values())
           and not row["delayed"]["free_rollout"]["passed"] for row in rows.values())
    write(folder/"parity.json",dict(passed=ok,reference=source["model"],source_sha256=digest(source["model"]),
          note="q/v in source model coordinates; second coordinate is q per second",traces=rows))
    if not ok:
        raise ValueError("Reference and port diverged or injected delay went undetected; inspect parity.json")
    print(folder/"parity.json")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",nargs="?",default="runs/plant-transfer")
    run(p.parse_args().folder)
