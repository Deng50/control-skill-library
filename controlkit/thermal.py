"""Generate constant-power thermal trajectories, fit a conditional PINN and a matched data baseline."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import digest, read, runtime, write

FIELDS = ["trajectory","partition","time_s","initial_K","power_W","temperature_K"]
PHYSICS = Path(__file__).resolve().parents[1]/"cases/thermal/physics.json"
SEED = 3109


def physical(path):
    p = read(path)
    if (p.get("equation") != "C*dT/dt = -G*(T-Ta)+P" or p.get("drive") != "constant_per_trajectory"
            or p.get("units") != dict(time_s="s",power_W="W",initial_K="K",temperature_K="K")):
        raise ValueError("Expected the constant-power thermal ODE with declared SI columns")
    values = [float(p[k]) for k in ("C_J_K","G_W_K","Ta_K")]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Thermal parameters must be positive and finite")
    return values


def sample(path, physics=PHYSICS):
    C,G,Ta = physical(physics)
    # Cross the training-condition extremes; reserve whole, distinct interior trajectories.
    conditions = [(a,b,"train") for a in (289.,295.,301.) for b in (-12.,0.,24.)]
    conditions += [(290.,18.,"validation"),(299.,-8.,"validation"),
                   (291.,-6.,"test"),(298.,15.,"test"),(300.,8.,"test")]
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(FIELDS)
        for i,(T0,power,split) in enumerate(conditions):
            eq = Ta+power/G
            for t in np.linspace(0,90,19):
                w.writerow([i,split,t,T0,power,eq+(T0-eq)*math.exp(-t*G/C)])
    return path


def load_rows(path):
    groups, partitions, signatures = {},{},{}
    with open(path,encoding="utf-8-sig",newline="") as src:
        rows = csv.DictReader(src)
        if not set(FIELDS).issubset(rows.fieldnames or []) or len(rows.fieldnames)!=len(set(rows.fieldnames)):
            raise ValueError("Expected uniquely named thermal data fields "+str(FIELDS))
        for row in rows:
            gid,part = row["trajectory"],row["partition"]
            if part not in ("train","validation","test") or partitions.setdefault(gid,part)!=part:
                raise ValueError("Invalid partition or trajectory leaks across partitions")
            values = [float(row[k]) for k in FIELDS[2:]]
            if not np.isfinite(values).all() or values[0]<0 or min(values[1],values[3])<=0:
                raise ValueError("Invalid time or temperature")
            if signatures.setdefault(gid,values[1:3])!=values[1:3]:
                raise ValueError("Power and initial temperature must be constant within each trajectory")
            groups.setdefault(gid,[]).append(values)
    condition_partitions = {}
    for gid, condition in signatures.items():
        part = partitions[gid]
        if condition_partitions.setdefault(tuple(condition),part) != part:
            raise ValueError("Thermal condition leaks across partitions under different trajectory IDs")
    out = {}
    for part in ("train","validation","test"):
        selection = []
        for gid,rows in groups.items():
            if partitions[gid]!=part:
                continue
            traj = np.asarray(sorted(rows))
            if traj[0,0]!=0 or np.any(np.diff(traj[:,0])<=0):
                raise ValueError("Each trajectory starts at zero and has increasing, unique timestamps")
            selection.append(traj)
        if not selection:
            raise ValueError("Missing whole trajectories for "+part)
        out[part] = np.concatenate(selection)
    return out,partitions


def network():
    return nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))


def tensor(a):
    return torch.as_tensor(a,dtype=torch.float32)


def scalers(train):
    inputs = train[:,:3]
    center = [0.,float(inputs[:,1].mean()),float(inputs[:,2].mean())]
    scales = [float(inputs[:,0].max()),float(inputs[:,1].std()),float(inputs[:,2].std())]
    if min(scales)<=0 or train[:,3].std()<=0:
        raise ValueError("Training cases lack variation in time, condition or output")
    return dict(center=center,scales=scales,y_mean=float(train[:,3].mean()),y_scale=float(train[:,3].std()),
                lower=inputs.min(axis=0).tolist(),upper=inputs.max(axis=0).tolist())


def normalized(arr,s):
    return (tensor(arr[:,:3])-tensor(s["center"]))/tensor(s["scales"])


def predict(model,z,s):
    return s["y_mean"]+s["y_scale"]*model(z)[:,0]


def ode(model,z,s,phys):
    C,G,Ta=phys
    z=z.detach().requires_grad_(True)
    T = predict(model,z,s)
    grad = torch.autograd.grad(T.sum(),z,create_graph=True)[0][:,0] / s["scales"][0]
    power = z[:,2]*s["scales"][2]+s["center"][2]
    return C*grad+G*(T-Ta)-power


def fit(data,s,phys,epochs,directory):
    torch.manual_seed(SEED)
    model=network()
    opt=torch.optim.Adam(model.parameters(),lr=0.004)
    train,valid,test = (data[k] for k in ("train","validation","test"))
    z = normalized(train,s)
    y = tensor(train[:,3])
    zv = normalized(valid,s)
    ztest = normalized(test,s)
    initial = torch.unique(z[:,1:],dim=0)
    z0 = torch.column_stack((torch.zeros(len(initial)),initial))
    # Collocation uses the training conditions and their own observed time ranges.
    points = []
    for c in initial:
        subset = z[(z[:,1:]==c).all(dim=1),0]
        points.append(torch.column_stack((torch.linspace(subset.min(),subset.max(),23),c.repeat(23,1))))
    collocation=torch.cat(points)
    best_val=float("inf")
    log=[]
    best_epoch=0
    for it in range(1,epochs+1):
        data_loss=((predict(model,z,s)-y)/s["y_scale"]).square().mean()
        loss=data_loss
        if phys:
            res=ode(model,collocation,s,phys)
            power_scale=max(phys[1]*s["y_scale"],s["scales"][2])
            lphy=(res/power_scale).square().mean()
            t0=z0[:,1]*s["scales"][1]+s["center"][1]
            lic=((predict(model,z0,s)-t0)/s["y_scale"]).square().mean()
            loss=loss+lphy+lic
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            val=float((predict(model,zv,s)-tensor(valid[:,3])).square().mean())
        if val<best_val:
            best_val,best_epoch=val,it
            best={k:t.detach().clone() for k,t in model.state_dict().items()}
        if it==1 or it%25==0 or it==epochs:
            log.append([it,float(data_loss.detach()),float(lphy.detach()) if phys else None,
                        float(lic.detach()) if phys else None,val])
    model.load_state_dict(best)
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(best,directory/"weights.pt")
    write(directory/"interface.json",dict(scalers=s,features=FIELDS[2:5],output=FIELDS[5],
          hidden=[32,32],drive="constant_per_trajectory",physics=phys,seed=SEED))
    pred=predict(model,ztest,s).detach().numpy()
    err=pred-test[:,3]
    np.savetxt(directory/"test.csv",np.column_stack((test,pred,err)),delimiter=",",comments="",
               header="time_s,initial_K,power_W,temperature_K,prediction_K,error_K")
    each={}
    for cond in np.unique(test[:,1:3],axis=0):
        e=err[(test[:,1:3]==cond).all(axis=1)]
        each[str(tuple(cond.tolist()))]=float(np.sqrt(np.mean(e**2)))
    write(directory/"loss.json",dict(columns=["epoch","data_scaled","physics_scaled","initial_scaled","val_K2"],rows=log))
    return dict(best_epoch=best_epoch,test_rmse_K=float(np.sqrt(np.mean(err**2))),per_condition_rmse_K=each,
                initial_max_error_K=float(max(abs(err[test[:,0]==0]))),
                ode_rmse_W=float(ode(model,ztest,s,phys).square().mean().sqrt().detach()) if phys else None)


def train(csv_path,phys_path,out,epochs):
    if epochs<1:
        raise ValueError("Positive training budget required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data,split=load_rows(csv_path)
    s=scalers(data["train"])
    out=Path(out)
    result={"nn":fit(data,s,None,epochs,out/"nn")}
    if phys_path:
        result["pinn"]=fit(data,s,physical(phys_path),epochs,out/"pinn")
    write(out/"report.json",dict(runtime=runtime(SEED),data_sha256=digest(csv_path),split=split,
          physics_sha256=digest(phys_path) if phys_path else None,results=result,epochs=epochs,
          physics_status="active" if phys_path else "missing: supply equation, parameters and units",
          optimizer="Adam",lr=.004,loss_weights=[1,1,1] if phys_path else [1,0,0]))
    print({k:v["test_rmse_K"] for k,v in result.items()})


def infer(bundle,t,T0,power):
    s=read(Path(bundle)/"interface.json")["scalers"]
    row=np.array([[t,T0,power]])
    if not np.isfinite(row).all() or np.any(row<np.array(s["lower"])) or np.any(row>np.array(s["upper"])):
        raise ValueError("Outside declared training feature ranges")
    model=network()
    model.load_state_dict(torch.load(Path(bundle)/"weights.pt",map_location="cpu",weights_only=True))
    return float(predict(model,normalized(row,s),s).detach()[0])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    cmds=p.add_subparsers(dest="mode",required=True)
    c=cmds.add_parser("sample")
    c.add_argument("--csv",required=True)
    c.add_argument("--physics",default=str(PHYSICS))
    c=cmds.add_parser("train")
    c.add_argument("--csv",required=True)
    c.add_argument("--physics",help="Omit for data-only training")
    c.add_argument("--epochs",type=int,default=500)
    c.add_argument("--out",required=True)
    c=cmds.add_parser("infer")
    c.add_argument("bundle")
    c.add_argument("condition",nargs=3,type=float,metavar="VALUE")
    a=p.parse_args()
    if a.mode=="sample":
        sample(a.csv,a.physics)
    elif a.mode=="train":
        train(a.csv,a.physics,a.out,a.epochs)
    else:
        print(infer(a.bundle,*a.condition))
