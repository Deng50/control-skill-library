"""Prepare causal feature windows from grouped CSV and fit a data-only MLP baseline."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import digest, read, runtime, write


def checked_index(value):
    if not isinstance(value,int) or isinstance(value,bool) or value<0:
        raise ValueError("lag/ahead must be a non-negative integer in samples")
    return value


def windows(path,schema):
    features,targets=schema["features"],schema["targets"]
    if not features or not targets or not np.isfinite(schema["dt_s"]) or schema["dt_s"]<=0:
        raise ValueError("Features, targets, positive dt_s are required")
    for spec in features+targets:
        if not spec.get("unit"):
            raise ValueError("Missing unit for "+spec["column"])
    lags=[checked_index(f["lag"]) for f in features]
    offsets=[checked_index(t["ahead"]) for t in targets]
    if any(f["column"]==t["column"] and f["lag"]==0 and t["ahead"]==0 for f in features for t in targets):
        raise ValueError("Current target is an input feature")
    required={s["column"] for s in features+targets}|{schema[k] for k in ("trajectory","partition","time")}
    groups,partitions={},{}
    with open(path,encoding="utf-8-sig",newline="") as source:
        reader=csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []) or len(set(reader.fieldnames))!=len(reader.fieldnames):
            raise ValueError(f"Missing or ambiguous named columns. Required {sorted(required)}")
        for row in reader:
            gid,part=row[schema["trajectory"]],row[schema["partition"]]
            if part not in ("train","validation","test") or partitions.setdefault(gid,part)!=part:
                raise ValueError("Trajectory crosses partitions")
            groups.setdefault(gid,[]).append(row)
    chunks={p:[] for p in ("train","validation","test")}
    for gid,rows in groups.items():
        t=np.array([float(row[schema["time"]]) for row in rows])
        if len(t)<=max(lags)+max(offsets) or not np.isfinite(t).all() or not np.allclose(np.diff(t),schema["dt_s"],rtol=1e-6,atol=1e-9):
            raise ValueError(f"Time grid or window invalid in {gid}")
        for k in range(max(lags),len(rows)-max(offsets)):
            x=[float(rows[k-f["lag"]][f["column"]]) for f in features]
            y=[float(rows[k+s["ahead"]][s["column"]]) for s in targets]
            chunks[partitions[gid]].append(x+y)
    arrays={k:np.array(v) for k,v in chunks.items()}
    if any(not len(v) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Missing partition or nonfinite samples")
    return arrays,partitions


def net(nx,ny):
    return nn.Sequential(nn.Linear(nx,32),nn.Tanh(),nn.Linear(32,ny))


def train(source,spec_path,out,epochs):
    if epochs<1:
        raise ValueError("Positive training budget required")
    cfg=read(spec_path)
    arrays,part=windows(source,cfg)
    dim=len(cfg["features"])
    trainset=arrays["train"]
    mean,scale=trainset.mean(axis=0),trainset.std(axis=0)
    scale=np.where(scale>1e-9,scale,1.)
    tensors={k:torch.tensor((a-mean)/scale,dtype=torch.float32) for k,a in arrays.items()}
    torch.manual_seed(488)
    torch.set_num_threads(1)
    model=net(dim,len(cfg["targets"]))
    opt=torch.optim.Adam(model.parameters(),lr=.005)
    best=float("inf")
    for i in range(1,epochs+1):
        a=tensors["train"]
        loss=(model(a[:,:dim])-a[:,dim:]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v=tensors["validation"]
            val=float((model(v[:,:dim])-v[:,dim:]).square().mean())
        if val<best:
            best,epoch=val,i
            weights={k:t.detach().clone() for k,t in model.state_dict().items()}
    model.load_state_dict(weights)
    pred=model(tensors["test"][:,:dim]).detach().numpy()*scale[dim:]+mean[dim:]
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(weights,out/"weights.pt")
    write(out/"interface.json",dict(schema=cfg,mean=mean,scale=scale))
    write(out/"report.json",dict(runtime=runtime(488),method="supervised_only",source_sha256=digest(source),
          effective_split=part,rows={k:len(v) for k,v in arrays.items()},best_epoch=epoch,
          heldout_rmse_per_target=np.sqrt(np.mean((pred-arrays["test"][:,dim:])**2,axis=0))))
    np.savetxt(out/"test.csv",np.column_stack((arrays["test"][:,dim:],pred)),delimiter=",")
    print("test rows",len(pred))


def infer(bundle,row):
    p=Path(bundle)
    meta=read(p/"interface.json")
    dim=len(meta["schema"]["features"])
    if len(row)!=dim or not np.isfinite(row).all():
        raise ValueError("Provide finite values in feature/window order")
    mean,scale=np.array(meta["mean"]),np.array(meta["scale"])
    model=net(dim,len(mean)-dim)
    model.load_state_dict(torch.load(p/"weights.pt",map_location="cpu",weights_only=True))
    with torch.no_grad():
        z=torch.tensor((row-mean[:dim])/scale[:dim],dtype=torch.float32)
        return model(z).numpy()*scale[dim:]+mean[dim:]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="stage",required=True)
    for name in ("audit","train"):
        parser=sub.add_parser(name)
        parser.add_argument("csv")
        parser.add_argument("schema")
        parser.add_argument("--out",required=True)
    parser.add_argument("--epochs",type=int,default=300)
    parser=sub.add_parser("infer")
    parser.add_argument("bundle")
    parser.add_argument("row",nargs="+",type=float)
    args=p.parse_args()
    if args.stage=="train":
        train(args.csv,args.schema,args.out,args.epochs)
    elif args.stage=="audit":
        arrays,part=windows(args.csv,read(args.schema))
        write(args.out,dict(source_sha256=digest(args.csv),partitions=part,shapes={k:v.shape for k,v in arrays.items()}))
    else:
        print(infer(args.bundle,args.row))
