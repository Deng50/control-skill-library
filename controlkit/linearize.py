"""Local discrete prediction, independent perturbation check and switching-point diagnostic."""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from .io import digest, runtime, write


def numerical_jacobian(fn,z,steps):
    z,h=np.asarray(z,float),np.asarray(steps,float)
    if z.ndim!=1 or h.shape!=z.shape or not np.isfinite(z).all() or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError("Positive per-channel perturbation and finite operating point required")
    mid=fn(z)
    left,right=[],[]
    for axis,delta in zip(np.eye(len(z)),h):
        left.append((mid-fn(z-axis*delta))/delta)
        right.append((fn(z+axis*delta)-mid)/delta)
    L,R=np.column_stack(left),np.column_stack(right)
    return (L+R)/2,L,R


def linearize(fn,z,h,analytic=None):
    z,h=np.array(z,float),np.array(h,float)
    y=fn(z)
    J,L,R=numerical_jacobian(fn,z,h)
    Jhalf,_,_=numerical_jacobian(fn,z,h/2)
    chosen=Jhalf if analytic is None else np.asarray(analytic(z))
    if chosen.shape!=J.shape:
        raise ValueError("Analytic Jacobian dimension differs from numerical Jacobian")
    gap=np.abs(L-R)/np.maximum(np.maximum(abs(L),abs(R)),1e-8)
    return dict(jacobian=chosen,offset=y-chosen@z,at_point=y,
                left_right_relative_gap=gap,half_step_max_change=float(np.max(np.abs(J-Jhalf))),
                analytic_vs_numeric=float(np.max(np.abs(chosen-Jhalf))) if analytic is not None else None)


def run(source,out):
    path=Path(source)
    spec=importlib.util.spec_from_file_location("library_external_thermal",path)
    mod=importlib.util.module_from_spec(spec)
    original=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode=original
    model=mod.ThermalNominalModel()
    def step(z):
        tb,tm,soc,n,power,alpha,ambient=z
        return model.forward([tb,tm,soc],[n,power,alpha],
               dict(Vk=10.,Vkp1=10.05,T_amb=ambient,C0=143.776))
    z=np.array([292.3,300.5,.57,550.,1200.,.43,275.])
    h=np.array([.001,.001,.00001,.1,.1,.00005,.001])
    result=linearize(step,z,h)
    J=result["jacobian"]
    probes=np.random.default_rng(37).uniform(-1,1,(30,len(z)))*np.array([.2,.2,.005,10,15,.015,.2])
    residual=np.array([step(z+v)-(result["at_point"]+J@v) for v in probes])
    error=np.max(np.abs(residual),axis=0)
    # At a rule boundary the central slope can conceal incompatible directional models.
    kink=linearize(lambda u:np.maximum(u,0),[0.],[.001])["left_right_relative_gap"].item()
    analytic=linearize(lambda u:np.array([u[0]**2+u[1]]),[2.,1.],[1e-4,1e-4],
                        lambda u:np.array([[2*u[0],1.]]))
    budget=np.array([.002,.002,1e-6])
    passed=bool(np.all(error<budget) and kink>.9 and analytic["analytic_vs_numeric"]<1e-8)
    write(out,dict(runtime=runtime(37),source_sha256=digest(path),time_semantics="next_state",dt_s=model.Ts,
          operating_point=z,perturbations=h,A=J[:,:3],B=J[:,3:6],E=J[:,6:],bias=result["offset"],
          C=[[1,0,0]],D=[[0,0,0]],output="battery temperature K",validation_probes=probes,
          local_max_errors=error,local_budgets_K_K_SOC=budget,derivative_checks=result,
          piecewise_boundary_gap=kink,analytic_control=analytic["analytic_vs_numeric"],accepted=passed))
    if not passed:
        raise ValueError("Local fidelity or branch check failed; inspect before using predictor")
    print("Local maximum next-state errors",error)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--out",default="runs/linearization/report.json")
    a=p.parse_args()
    run(a.source,a.out)
