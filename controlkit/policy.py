"""Expert collection, behavior cloning and physics-guided actor updates for the spring benchmark."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import digest, read, runtime, write
from .sampling_plan import check_scenarios
from .spring import CASE, COLUMNS, Controller, Spring, feasible, rollout, score

FEATURES = ["q_m","v_mps","reference_m","disturbance_N"]
SEED = 145


def tensor(a):
    return torch.as_tensor(a,dtype=torch.float32)


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)


def collect(controller,out,scenarios_path=CASE/"expert.json",split_path=CASE/"split.json"):
    cfg = read(controller)
    if cfg["model_sha256"] != digest(CASE/"nominal.json"):
        raise ValueError("Nominal model version mismatch")
    dt=cfg["sample_time_s"]
    plant,expert=Spring("reference",dt),Controller(Spring("nominal",dt),cfg)
    scenarios=read(scenarios_path)["scenarios"]
    splits=read(split_path)
    check_scenarios(scenarios,splits)
    trajectories=[rollout(plant,expert,s,cfg["force_limit"])[0] for s in scenarios]
    rows=np.vstack(trajectories)
    ids=np.repeat(np.arange(len(scenarios)),[len(t) for t in trajectories])
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"expert.npz",x=rows[:,1:3],reference=rows[:,3:4],d=rows[:,4:5],
                        previous_u=rows[:,5:6],u=rows[:,6:7],next_x=rows[:,7:9],t=rows[:,0],trajectory_id=ids)
    write(out/"data.json",dict(features=FEATURES,target="force_N",force_limit=cfg["force_limit"],sample_time_s=dt,
          dataset_sha256=digest(out/"expert.npz"),plant_source_sha256=digest(CASE/"reference.json"),
          controller_sha256=digest(controller),initial_previous_N=0,
          design=dict(initial_states=[s["initial_state"] for s in scenarios],
                      references_m=sorted({s["reference"][0] for s in scenarios}),
                      disturbances_N=sorted({s["disturbance"][0] for s in scenarios})),
          rows=len(rows),samples_near_limit=int(np.sum(abs(rows[:,6])>=.98*cfg["force_limit"])),
          splits=splits,scenarios=scenarios,scenario_sha256=digest(scenarios_path),runtime=runtime()))
    print("expert samples",len(rows))


def dataset(file,manifest):
    meta=read(manifest)
    if meta["features"]!=FEATURES or meta["target"]!="force_N":
        raise ValueError("Spring feature order or action units differ; provide an explicit mapping")
    if meta["dataset_sha256"]!=digest(file):
        raise ValueError("Dataset does not match the manifest hash")
    with np.load(file,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    ids=data["trajectory_id"]
    n=len(ids)
    if ids.shape!=(n,) or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError("trajectory_id requires an integer vector")
    for k,w in [("x",2),("reference",1),("d",1),("previous_u",1),("u",1)]:
        if data[k].shape!=(n,w) or not np.isfinite(data[k]).all():
            raise ValueError(f"Invalid {k} shape or content")
    if not 0<meta["force_limit"]<=4 or meta["sample_time_s"]<=0 or np.max(np.abs(data["u"]))>meta["force_limit"]+1e-7:
        raise ValueError("Expert command outside declared force range, or invalid sample period")
    parts=meta["splits"]
    listed=[i for s in ("train","validation","test") for i in parts[s]]
    if len(listed)!=len(set(listed)) or set(listed)!=set(ids) or any(len(parts[s])==0 for s in parts):
        raise ValueError("Incomplete splits or trajectory leakage")
    for i in set(ids):
        ix=ids==i
        us=data["u"][ix,0]
        if not np.allclose(data["previous_u"][ix,0],np.r_[meta["initial_previous_N"],us[:-1]],rtol=0,atol=1e-7):
            raise ValueError(f"Previous-action alignment failed in trajectory {i}")
        if "t" in data and not np.allclose(np.diff(data["t"][ix]),meta["sample_time_s"]):
            raise ValueError("Sampling period disagrees with timestamps")
    masks={k:np.isin(ids,v) for k,v in parts.items()}
    features=np.column_stack((data["x"],data["reference"],data["d"]))
    return data,meta,masks,features


class Actor(nn.Module):
    def __init__(self,meta):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh(),nn.Linear(32,1))
        self.register_buffer("center",tensor(meta["center"]))
        self.register_buffer("scale",tensor(meta["scale"]))
        self.force=float(meta["force_limit"])

    def forward(self,features):
        return self.force*torch.tanh(self.net((features-self.center)/self.scale))


def save_actor(folder,model,meta):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),folder/"weights.pt")
    write(folder/"interface.json",meta)


def load_actor(folder,expected_force=None):
    meta=read(Path(folder)/"interface.json")
    if meta["features"]!=FEATURES or meta["target"]!="force_N" or meta["hidden"]!=[48,32]:
        raise ValueError("Unsupported policy interface")
    if expected_force is not None and expected_force!=meta["force_limit"]:
        raise ValueError("Actuator range conflicts with trained policy")
    net=Actor(meta)
    net.load_state_dict(torch.load(Path(folder)/"weights.pt",map_location="cpu",weights_only=True))
    return net.eval(),meta


def imitation(file,manifest,out,epochs):
    setup()
    data,info,masks,features=dataset(file,manifest)
    tr,va,te=[masks[k] for k in ("train","validation","test")]
    center,scale=features[tr].mean(axis=0),features[tr].std(axis=0)
    if np.min(scale)<1e-8:
        raise ValueError("Training feature lacks excitation")
    meta=dict(features=FEATURES,target="force_N",center=center,scale=scale,hidden=[48,32],
              force_limit=info["force_limit"],sample_time_s=info["sample_time_s"],dataset_sha256=digest(file))
    net=Actor(meta)
    X,Y=tensor(features),tensor(data["u"])
    opt=torch.optim.Adam(net.parameters(),lr=.003)
    best=float("inf")
    logs=[]
    for i in range(1,epochs+1):
        loss=(net(X[tr])-Y[tr]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v=float((net(X[va])-Y[va]).square().mean())
        if v<best:
            best,best_epoch=v,i
            weights={k:w.detach().clone() for k,w in net.state_dict().items()}
        if i%25==0 or i==1:
            logs.append([i,float(loss.detach()),v])
    net.load_state_dict(weights)
    save_actor(out,net,meta)
    out=Path(out)
    rmse=float((net(X[te])-Y[te]).square().mean().sqrt().detach())
    write(out/"report.json",dict(runtime=runtime(SEED),best_epoch=best_epoch,epochs=epochs,
          log_epoch_train_val_N2=logs,heldout_action_rmse_N=rmse))
    print("imitation heldout action RMSE (N)",rmse)


def act(net):
    @torch.inference_mode()
    def policy(x,r,d,previous):
        return net(tensor([[*x,r,d]])).item()
    return policy


class RolloutCritic:
    """Differentiable finite-horizon cost; residual is an optional frozen linear perception layer."""
    def __init__(self,nominal,residual,horizon,cfg):
        self.A,self.B,self.E=[tensor(v) for v in (nominal.A,nominal.B,nominal.E)]
        self.W=tensor(residual)
        self.horizon=horizon
        self.cfg=cfg

    def objective(self,net,X,previous,coupled):
        x,r,d=X[:,:2],X[:,2:3],X[:,3:4]
        limit=self.cfg["force_limit"]
        total=0.
        for _ in range(self.horizon):
            u=net(torch.cat((x,r,d),dim=1))
            nxt=x@self.A.T+u@self.B.T+d@self.E.T
            if coupled:
                nxt=nxt+torch.cat((x,u,d),dim=1)@self.W
            violation=torch.relu(nxt.abs()-tensor([1.,1.5]))
            term=(self.cfg["q"]*(nxt[:,:1]-r)**2+self.cfg["r"]*(u/limit)**2
                  +self.cfg["s"]*((u-previous)/limit)**2+100*violation.square().sum(dim=1,keepdim=True))
            total=total+term
            previous,x=u,nxt
        return total.mean()/self.horizon


def optimize(file,manifest,actor_dir,controller,out,epochs):
    setup()
    data,info,masks,features=dataset(file,manifest)
    if "next_x" not in data or data["next_x"].shape!=data["x"].shape or not np.isfinite(data["next_x"]).all():
        raise ValueError("Paired next states required for perception; imitation remains usable")
    cfg=read(controller)
    if digest(controller)!=info.get("controller_sha256") or info.get("plant_source_sha256")!=digest(CASE/"reference.json"):
        raise ValueError("Bind the original expert and evaluation plant before PGAC")
    if info["sample_time_s"]!=cfg["sample_time_s"]:
        raise ValueError("Controller and expert dataset sample periods differ")
    original,meta=load_actor(actor_dir,cfg["force_limit"])
    if meta["dataset_sha256"]!=digest(file):
        raise ValueError("Different expert dataset from imitation")
    nominal,plant=Spring("nominal",cfg["sample_time_s"]),Spring("reference",cfg["sample_time_s"])
    tr,va=masks["train"],masks["validation"]
    z=np.column_stack((data["x"],data["u"],data["d"]))
    nominal_next=data["x"]@nominal.A.T+data["u"]@nominal.B.T+data["d"]@nominal.E.T
    target=data["next_x"]-nominal_next
    for i in np.unique(data["trajectory_id"]):
        ix=data["trajectory_id"]==i
        if not np.allclose(data["next_x"][ix][:-1],data["x"][ix][1:],atol=1e-8,rtol=0):
            raise ValueError("Next state does not align to following observation")
    W,_,rank,_=np.linalg.lstsq(z[tr],target[tr],rcond=None)
    if rank!=z.shape[1]:
        raise ValueError("Insufficient independent excitation for perception regression")
    X,Y=tensor(features),tensor(data["u"])
    critic=RolloutCritic(nominal,W,cfg["horizon"],cfg)
    scenarios=info["scenarios"]
    def evaluation(policy,part):
        results={}
        for idx in info["splits"][part]:
            sc=scenarios[idx]
            ix=data["trajectory_id"]==idx
            if (not np.allclose(data["x"][ix][0],sc["initial_state"]) or
                    not np.allclose(data["reference"][ix],sc["reference"]) or
                    not np.allclose(data["d"][ix],sc["disturbance"]) or ix.sum()!=sc["steps"]):
                raise ValueError("Scenario does not match expert trajectory")
            policy(np.array(sc["initial_state"]),sc["reference"][0],sc["disturbance"][0],0.)
            traj,metrics=rollout(plant,policy,sc,cfg["force_limit"])
            results[sc["name"]]=(traj,metrics)
        return results
    def validation_score(net):
        rows=evaluation(act(net),"validation")
        if not all(feasible(m) for _,m in rows.values()):
            return float("inf")
        return np.mean([score(m) for _,m in rows.values()])
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/"perception.npy",W)
    previous=tensor(data["previous_u"])
    base_value=validation_score(original)
    if not np.isfinite(base_value):
        raise RuntimeError("Imitation violates the frozen validation constraints")
    with torch.no_grad():
        j0=float(critic.objective(original,X[va],previous[va],False))
        jp=float(critic.objective(original,X[va],previous[va],True))
    records={}
    contenders={"imitation":(original,base_value)}
    anchor=original(X[tr]).detach()
    for name,coupled in [("nominal_only",False),("residual_coupled",True)]:
        net,_=load_actor(actor_dir,cfg["force_limit"])
        start_error=float((net(X[va])-original(X[va])).abs().max().detach())
        if start_error!=0:
            raise RuntimeError("Warm-start actions changed before first policy update")
        opt=torch.optim.Adam(net.parameters(),lr=.0004)
        best,log=float("inf"),[]
        for i in range(1,epochs+1):
            J=critic.objective(net,X[tr],previous[tr],coupled)
            loss=J+.02*((net(X[tr])-anchor)/cfg["force_limit"]).square().mean()
            opt.zero_grad()
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5.))
            if not math_is_positive(norm):
                raise ValueError("Actor gradient is zero or nonfinite")
            opt.step()
            value=validation_score(net)
            log.append(dict(epoch=i,critic_J=float(J.detach()),gradient_norm=norm,
                            validation_score=float(value) if np.isfinite(value) else None))
            if value<best:
                best,best_epoch=value,i
                weights={k:w.detach().clone() for k,w in net.state_dict().items()}
        if not np.isfinite(best):
            records[name]=dict(feasible=False,updates=log)
            continue
        net.load_state_dict(weights)
        save_actor(out/name,net,meta)
        records[name]=dict(feasible=True,best_epoch=best_epoch,updates=log,warmstart_max_action_error_N=start_error)
        contenders[name]=(net,best)
    selected=min(contenders,key=lambda k:contenders[k][1])
    save_actor(out/"selected",contenders[selected][0],dict(meta,selected_from=selected))
    tests={}
    policies={"expert":Controller(nominal,cfg),**{k:act(t[0]) for k,t in contenders.items()}}
    for name,policy in policies.items():
        rows=evaluation(policy,"test")
        tests[name]={sc:m for sc,(traj,m) in rows.items()}
        for sc,(traj,m) in rows.items():
            np.savetxt(out/f"{name}-{sc}.csv",traj,header=COLUMNS,delimiter=",",comments="")
    violation=not all(feasible(m) for m in tests[selected].values())
    report=dict(runtime=runtime(SEED),controller_sha256=digest(controller),source_actor_sha256=digest(Path(actor_dir)/"weights.pt"),
                critic="finite-horizon physical cost",perception="linear supervised residual; frozen during actor training",
                val_nominal_next_rmse=np.sqrt(np.mean(target[va]**2,axis=0)),
                val_corrected_next_rmse=np.sqrt(np.mean((target[va]-z[va]@W)**2,axis=0)),
                critic_at_same_actor=dict(nominal=j0,coupled=jp),epochs=epochs,branches=records,selected=selected,
                accepted_on_test=not violation,validation_scores={k:v[1] for k,v in contenders.items()},test=tests)
    write(out/"report.json",report)
    if violation or abs(j0-jp)<1e-9:
        raise RuntimeError("Held-out constraint or perception coupling check failed")
    print("validation-selected policy",selected)


def math_is_positive(value):
    return np.isfinite(value) and value>0


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    c=p.add_subparsers(dest="mode",required=True)
    sub=c.add_parser("collect")
    sub.add_argument("--controller",default="runs/mpc/controller.json")
    sub.add_argument("--out",default="runs/expert")
    sub.add_argument("--scenarios",default=str(CASE/"expert.json"))
    sub.add_argument("--split",default=str(CASE/"split.json"))
    for name in ("imitate","optimize"):
        sub=c.add_parser(name)
        sub.add_argument("--data",default="runs/expert/expert.npz")
        sub.add_argument("--manifest",default="runs/expert/data.json")
        sub.add_argument("--out",default="runs/"+("imitation" if name=="imitate" else "pgac"))
        sub.add_argument("--epochs",type=int,default=600 if name=="imitate" else 20)
    sub.add_argument("--actor",default="runs/imitation")
    sub.add_argument("--controller",default="runs/mpc/controller.json")
    sub=c.add_parser("infer")
    sub.add_argument("bundle")
    sub.add_argument("features",nargs=4,type=float)
    args=p.parse_args()
    if getattr(args,"epochs",1)<1:
        p.error("epochs must be positive")
    if args.mode=="collect":
        collect(args.controller,args.out,args.scenarios,args.split)
    elif args.mode=="imitate":
        imitation(args.data,args.manifest,args.out,args.epochs)
    elif args.mode=="optimize":
        optimize(args.data,args.manifest,args.actor,args.controller,args.out,args.epochs)
    else:
        net,_=load_actor(args.bundle)
        print(net(tensor([args.features])).detach().item())
