"""Fit the archive kernel's ActorNet, then optimize two paired heat PGAC candidates."""
import argparse
import copy
import csv
import json
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

from heat_data import (C0, DT, HI, KERNEL, LO, OUT, Q, R, S, SOURCE,
                       check_episode, create_plant, power, reference_stage_functions, sha, write)

sys.path.append(str(KERNEL))
import pgac_core as core

core.device = torch.device("cpu")
torch.set_num_threads(1)
SEED = 714
PG_HORIZON = 12
PG_UPDATES = 14
OBS = ["Tb_K","Tm_K","SOC","Tb_ref_K"]
ACTIONS = ["pump_rpm","heater_W","bypass_fraction"]
ARCH = {"x_dim":3,"r_dim":1,"hidden1":48,"hidden2":32,"action_dim":3}
SPEC = {"observations":OBS,"actions":ACTIONS,"dt_s":DT,"lower":LO.tolist(),
        "upper":HI.tolist(),"actor_arch":ARCH,"plant":"SystemTestPlantAlign",
        "kernel_sha256":sha(KERNEL/"pgac_core.py")}


def tensor(a):
    return torch.as_tensor(a,dtype=torch.float32)


class Policy(nn.Module):
    def __init__(self,center,scale):
        super().__init__()
        self.actor = core.ActorNet(**ARCH)
        self.register_buffer("center",center)
        self.register_buffer("scale",scale)
        self.register_buffer("low",tensor(LO))
        self.register_buffer("span",tensor(HI-LO))

    def forward(self,x,r):
        z = (torch.cat((x,r),dim=-1)-self.center)/self.scale
        return self.low + self.span*self.actor(z[:,:3],z[:,3:])


def save_pack(name,policy):
    folder = OUT/name
    folder.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), folder/"actor.pt")
    sample = (tensor([[253.15,263.15,1]]),tensor([[293.15]]))
    torch.jit.trace(policy.eval(), sample).save(str(folder/"inference.ts"))
    contents = dict(**SPEC,raw_feature_mean=policy.center.tolist(),raw_feature_scale=policy.scale.tolist(),
                    actor_weights_sha256=sha(folder/"actor.pt"),inference_sha256=sha(folder/"inference.ts"),
                    inference="torch.jit.load(inference.ts)(x[B,3], ref[B,1]) returns physical [B,3]",
                    training_archive_sha256=sha(OUT/"expert.npz"),
                    plant_sha256=json.loads((OUT/"data.json").read_text("utf-8"))["source_sha256"],
                    actor_fraction_range=[.005,.995],history="current state and scalar reference only")
    (folder/"interface.json").write_text(json.dumps(contents,indent=2),encoding="utf-8")


def load_policy(name):
    folder = OUT/name
    meta = json.loads((folder/"interface.json").read_text(encoding="utf-8"))
    if any(meta[k] != val for k,val in SPEC.items()) or sha(folder/"actor.pt") != meta["actor_weights_sha256"]:
        raise ValueError("Actor identity, interface or checkpoint mismatch")
    policy = Policy(tensor(meta["raw_feature_mean"]),tensor(meta["raw_feature_scale"]))
    policy.load_state_dict(torch.load(folder/"actor.pt",map_location="cpu",weights_only=True),strict=True)
    return policy.eval()


def fit(net, objective, train_ix, val_ix, updates, lr):
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    rng = np.random.default_rng(SEED)
    best, selected = float("inf"), 0
    history = []
    for k in range(1,updates+1):
        batch = rng.choice(train_ix,size=256,replace=True)
        optimizer.zero_grad()
        loss = objective(batch)
        loss.backward()
        optimizer.step()
        if k%50 == 0 or k==updates:
            with torch.no_grad():
                v = float(objective(val_ix))
            history.append(dict(update=k,validation_loss=v))
            if v<best:
                best,selected,weights = v,k,copy.deepcopy(net.state_dict())
    net.load_state_dict(weights)
    return dict(selected_update=selected,best_validation=best,history=history)


class Perception(nn.Module):
    """Original PerceptionNet with three state and six control/context channels."""
    def __init__(self,features,residual):
        super().__init__()
        self.net = core.PerceptionNet(x_dim=3,u_dim=6,hidden=40)
        for name,val in dict(center=features.mean(0),scale=features.std(0).clamp_min(1e-5),
                             delta_center=residual.mean(0),delta_scale=residual.std(0).clamp_min(1e-8)).items():
            self.register_buffer(name,val)

    def forward(self,features):
        z = (features-self.center)/self.scale
        return self.net(z[:,:3],z[:,3:])*self.delta_scale+self.delta_center


def features(x,u,env):
    return torch.cat([x,u,env["T_amb"][:,None],env["Vk"][:,None],env["Vkp1"][:,None]],1)


def nominal_step(nom,x,u,env,delay):
    # The original plant initializes all four zero delay components using THIS
    # action. Mid-episode samples start with their own recorded heat memories.
    empty = (delay==0).all(1)
    if empty.any():
        qb,qm = nom._core_terms(x,u,env)[4:6]
        start = torch.stack(nom.init_delay_state(qb,qm),1)
        delay = torch.where(empty[:,None],start,delay)
    xn,*rest = nom.step_with_delay(x,u,env,*delay.unbind(1))
    return xn,torch.stack(rest,1)


def expert_cost(nom,x,u,xn,ref,env):
    """Source NMPC electrical model. The original Torch nominal uses a different input power."""
    vk, vkp1 = env["Vk"],env["Vkp1"]
    m = nom.model
    n_m = vk*m.i/m.r/.105
    acc = (vkp1-vk)/DT
    resist = m.Cr*m.mass_v*m.g + .5*m.rho_air*m.Af*m.Cd*vk**2
    force = m.mass_v*acc + torch.where(acc>=0,resist,torch.zeros_like(resist))
    torque = force*m.r/(m.i*m.eta_T)
    motor_w = torch.where(torque>=0,torque*n_m/(9.55*m.eta_T),torque*n_m*m.eta_T/9.55)
    # All archived SOC/Tb are inside the calibrated electrical tables. Outside
    # the axes the expert extrapolates, unlike the stock Torch interpolator.
    def interp(mat):
        soc,tb = x[:,2],x[:,0]
        a,b = nom.SOC_vec,nom.T_vec
        i = (torch.searchsorted(a,soc.contiguous())-1).clamp(0,len(a)-2)
        j = (torch.searchsorted(b,tb.contiguous())-1).clamp(0,len(b)-2)
        wi,wj = (soc-a[i])/(a[i+1]-a[i]),(tb-b[j])/(b[j+1]-b[j])
        return (1-wi)*((1-wj)*mat[i,j]+wj*mat[i,j+1])+wi*((1-wj)*mat[i+1,j]+wj*mat[i+1,j+1])
    uocv, rb = interp(nom.V0_mat),interp(nom.Rb_mat).clamp_min(1e-6)
    p = power(u)
    total = motor_w+p
    root = (uocv**2-4*rb*total).clamp_min(1e-12).sqrt()
    # Rationalized source equation avoids float32 cancellation around zero.
    current = torch.where(uocv**2-4*rb*total>=1e-12,2*total/(uocv+root),(uocv-root)/(2*rb))
    dc = core.capacity_loss_torch(xn[:,0],current,DT)
    return torch.stack([Q*(xn[:,0]-ref)**2,R*p**2,S*dc**2],1)


def env_for(d,ix):
    return dict(Vk=d["vk"][ix],Vkp1=d["vk1"][ix],T_amb=d["amb"][ix],C0=C0)


def rollout(actor,perception,nom,d,starts,reset_history=False,hold_time=False):
    x,delay = d["x"][starts],d["delay"][starts]
    if reset_history:
        delay = torch.zeros_like(delay)
    total = torch.zeros(len(starts))
    for step in range(PG_HORIZON):
        ix = starts if hold_time else starts+step
        env,ref = env_for(d,ix),d["ref"][ix]
        u = actor(x,ref[:,None])
        xn,delay = nominal_step(nom,x,u,env,delay)
        if perception is not None:
            xn = xn+perception(features(x,u,env))
        total = total+expert_cost(nom,x,u,xn,ref,env).sum(1)
        x = xn
    return total.mean()


def grad_norm(net):
    return float(torch.sqrt(sum((p.grad**2).sum() for p in net.parameters() if p.grad is not None)))


def tune(name,perception,nom,d,train_starts,anchor):
    actor = load_policy("imitation")
    start_difference = float((actor(d["x"][anchor],d["ref"][anchor,None])-d["initial_anchor"]).abs().max().detach())
    if start_difference!=0:
        raise ValueError("Warm start does not reproduce imitation output")
    original = copy.deepcopy(actor.state_dict())
    opt = torch.optim.Adam(actor.parameters(),lr=.00012)
    rng = np.random.default_rng(SEED+4)
    history=[]
    for k in range(PG_UPDATES):
        ix = rng.choice(train_starts,64)
        opt.zero_grad()
        j = rollout(actor,perception,nom,d,ix)
        j.backward()
        norm = grad_norm(actor)
        if not np.isfinite(norm) or norm<=0:
            raise ValueError("Actor gradient invalid")
        torch.nn.utils.clip_grad_norm_(actor.parameters(),5.)
        opt.step()
        history.append(dict(update=k+1,J=float(j.detach()),gradient_norm=norm))
    delta = max(float((w-original[n]).abs().max()) for n,w in actor.state_dict().items())
    if delta==0:
        raise ValueError("Actor parameters did not change")
    save_pack(name,actor)
    return dict(preupdate_max_abs_physical=start_difference,updates=history,max_parameter_change=delta)


@torch.inference_mode()
def evaluate(policy,episodes,meta,data,stage,part,policy_name):
    rows=[]
    detail=[]
    for eid in episodes:
        ix = np.flatnonzero(data["eid"]==eid)
        x0 = data["x"][ix[0]]
        scenario = meta["episodes"][eid]
        plant = create_plant(x0,scenario["speed"])
        xs,us,xns,latencies = [],[],[],[]
        for i in ix:
            x,ref,amb,t = plant.x.copy(),data["ref"][i],data["amb"][i],data["t"][i]
            if policy is None:
                u = data["u"][i]
            else:
                start = time.perf_counter_ns()
                u = policy(tensor(x[None,:]),tensor([[ref]])).numpy()[0].astype(float)
                latencies.append((time.perf_counter_ns()-start)*1e-6)
            xn = plant.step(*u,amb,C0,t_now=t)
            xs.append(x); us.append(u); xns.append(xn)
        x,u,xn = map(np.asarray,(xs,us,xns))
        terms,current = stage(x,u,xn,data["ref"][ix],data["vk"][ix],data["vk1"][ix])
        errors = xn[:,0]-data["ref"][ix]
        if policy is None:
            np.testing.assert_allclose(x,data["x"][ix],rtol=0,atol=2e-7)
        violations = np.any((u<LO-1e-4)|(u>HI+1e-4),axis=1)
        bound_excess = np.maximum(np.maximum(LO-u,u-HI),0).max(0)
        rows.append(dict(eid=eid,scenario=scenario["name"],split=part,policy=policy_name,
                         steps=len(ix),J=float(terms.sum()),tracking_J=float(terms[:,0].sum()),
                         power_J=float(terms[:,1].sum()),aging_J=float(terms[:,2].sum()),
                         battery_MAE_K=float(np.abs(errors).mean()),battery_RMSE_K=float(np.sqrt(np.mean(errors**2))),
                         btm_Wh=float(power(u).sum()*DT/3600),motor_max_K=float(xn[:,1].max()),
                         motor_violation_steps=int((xn[:,1]>333.15).sum()),bound_violation_steps=int(violations.sum()),
                         max_physical_bound_excess=bound_excess.tolist(),
                         inference_p50_ms=float(np.median(latencies)) if latencies else None,
                         inference_p95_ms=float(np.percentile(latencies,95)) if latencies else None,
                         expert_action_RMSE=np.sqrt(np.mean((u-data["u"][ix])**2,axis=0)).tolist()))
        if part=="test":
            detail.extend(np.column_stack([np.full(len(ix),eid),data["t"][ix],x,u,xn,data["ref"][ix],
                                          data["vk"][ix],data["vk1"][ix],current,terms]).tolist())
    if detail:
        with (OUT/f"test-{policy_name}.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["eid","action_time_s","Tb_K","Tm_K","SOC",*ACTIONS,"next_Tb_K","next_Tm_K","next_SOC",
                             "ref_K","vk_mps","vk1_mps","expert_formula_A","tracking_J","power_J","aging_J"])
            writer.writerows(detail)
    return rows


def run():
    core.set_seed(SEED)
    raw = dict(np.load(OUT/"expert.npz",allow_pickle=False))
    meta = json.loads((OUT/"data.json").read_text("utf-8"))
    if sha(OUT/"expert.npz")!=meta["npz_sha256"]:
        raise ValueError("Archive digest changed")
    if any(sha(SOURCE/name)!=value for name,value in meta["source_sha256"].items()):
        raise ValueError("Expert, plant or calibration source changed")
    if sha(KERNEL/"pgac_core.py")!=meta["kernel_sha256"]:
        raise ValueError("PGAC kernel changed")
    d = {k:tensor(v) for k,v in raw.items() if k!="eid"}
    ids = {p:[e["id"] for e in meta["episodes"] if e["split"]==p] for p in ("train","validation","test")}
    ix = {p:np.flatnonzero(np.isin(raw["eid"],v)) for p,v in ids.items()}
    obs = torch.cat([d["x"],d["ref"][:,None]],1)
    train,val = ix["train"],ix["validation"]
    actor = Policy(obs[train].mean(0),obs[train].std(0).clamp_min(1e-5))
    def bc_loss(rows):
        return (((actor(d["x"][rows],d["ref"][rows,None])-d["u"][rows])/actor.span)**2).mean()
    bc = fit(actor,bc_loss,train,val,1800,.0018)
    save_pack("imitation",actor)
    print("imitation",bc["best_validation"],flush=True)
    nom = core.ThermalNominalModelTorchAlignDelay(create_plant([253.15,263.15,1],"60").model,dt=DT)
    env = env_for(d,np.arange(len(obs)))
    with torch.no_grad():
        nominal,delay = nominal_step(nom,d["x"],d["u"],env,d["delay"])
        residual = d["xn"]-nominal
        z = features(d["x"],d["u"],env)
    perc = Perception(z[train],residual[train])
    def p_loss(rows):
        return (((perc(z[rows])-residual[rows])/perc.delta_scale)**2).mean()
    per_fit = fit(perc,p_loss,train,val,1800,.002)
    perc.eval().requires_grad_(False)
    torch.save(perc.state_dict(),OUT/"perception.pt")
    with torch.no_grad():
        base_rmse = residual[val].square().mean(0).sqrt()
        fit_rmse = (residual[val]-perc(z[val])).square().mean(0).sqrt()
    per_fit.update(validation_nominal_RMSE=base_rmse.tolist(),validation_corrected_RMSE=fit_rmse.tolist(),
                   state_order=["Tb_K","Tm_K","SOC"],context=["Tb","Tm","SOC",*ACTIONS,"Tamb","Vk","Vkp1"])
    eligible=train[train+PG_HORIZON<len(obs)]
    eligible=eligible[raw["eid"][eligible]==raw["eid"][eligible+PG_HORIZON-1]]
    anchor=np.random.default_rng(SEED+11).choice(eligible,128,replace=False)
    d["initial_anchor"]=actor(d["x"][anchor],d["ref"][anchor,None]).detach()
    nom_j=rollout(actor,None,nom,d,anchor)
    actor.zero_grad()
    nom_j.backward()
    grad_nom=torch.cat([p.grad.flatten().clone() for p in actor.parameters()])
    actor.zero_grad()
    count=[]
    hook=perc.register_forward_hook(lambda *_:count.append(1))
    res_j=rollout(actor,perc,nom,d,anchor)
    hook.remove()
    res_j.backward()
    grad_res=torch.cat([p.grad.flatten() for p in actor.parameters()])
    probe=dict(nominal_J=float(nom_j.detach()),coupled_J=float(res_j.detach()),
               actor_gradient_delta_L2=float(torch.linalg.vector_norm(grad_nom-grad_res)),
               coupled_perception_calls=len(count),horizon=PG_HORIZON)
    if probe["actor_gradient_delta_L2"]<=0 or probe["nominal_J"]==probe["coupled_J"] or len(count)!=PG_HORIZON:
        raise ValueError("Perception did not affect the actor gradient")
    print("perception",base_rmse.tolist(),fit_rmse.tolist(),probe,flush=True)
    pg = {name:tune(name,p,nom,d,eligible,anchor) for name,p in (("nominal",None),("coupled",perc))}
    stage=reference_stage_functions()
    checks=fault_checks(raw,d,meta,nom,actor,perc,anchor,stage)
    policies = {"expert":None}
    for name in ("imitation","nominal","coupled"):
        policies[name]=torch.jit.load(str(OUT/name/"inference.ts")).eval()
    validation=[]
    for name,net in policies.items():
        validation.extend(evaluate(net,ids["validation"],meta,raw,stage,"validation",name))
    # Hard actuator and motor limits precede average validation trajectory cost.
    def score(name):
        metrics=[r for r in validation if r["policy"]==name]
        return (sum(r["motor_violation_steps"]+r["bound_violation_steps"] for r in metrics),
                float(np.mean([r["J"] for r in metrics])))
    selected=min(("imitation","nominal","coupled"),key=score)
    selected_record=dict(name=selected,validation_score=list(score(selected)),sha256=sha(OUT/selected/"inference.ts"),
                         rule="fewest bound/motor violations then lowest mean validation J; captured before test")
    write("selected.json",selected_record)
    test=[]
    for name,net in policies.items():
        test.extend(evaluate(net,ids["test"],meta,raw,stage,"test",name))
    action_rmse={}
    with torch.no_grad():
        ti=ix["test"]
        for name,net in policies.items():
            if net is not None:
                action_rmse[name]=(net(d["x"][ti],d["ref"][ti,None])-d["u"][ti]).square().mean(0).sqrt().tolist()
    write("report.json", dict(seed=SEED,dt_s=DT,expert_horizon=50,pgac_horizon=PG_HORIZON,pgac_updates=PG_UPDATES,
          torch_version=torch.__version__,python_version=platform.python_version(),numpy_version=np.__version__,
          imitation=bc,perception=per_fit,gradient_probe=probe,pgac=pg,fault_checks=checks,selected=selected_record,
          heldout_action_RMSE_on_teacher_state=action_rmse,validation=validation,test=test))
    print("selected",selected_record,"action RMSE",action_rmse,flush=True)


def fault_checks(raw,d,meta,nom,actor,perc,anchor,stage):
    # Deliberately break otherwise-valid archived rows and require explicit detection.
    ep = np.flatnonzero(raw["eid"]==0)
    args=[raw[k][ep].copy() for k in ("x","xn","u","previous","t","delay","delay_next")]
    checks={}
    for key,slot in (("one_step_action_shift",3),("wrong_dt",4),("rpm_heater_swapped",2),("lost_delay",5)):
        a=[v.copy() for v in args]
        if slot==3: a[slot][1:]=a[2][1:]
        if slot==4: a[slot]*=.2
        if slot==2:
            a[slot][:,[0,1]]=a[slot][:,[1,0]]
            a[3][1:]=a[2][:-1]
        if slot==5: a[slot][10]=0
        try:
            check_episode(*a)
        except ValueError as err:
            checks[key]=str(err)
        else:
            raise ValueError(f"Injected fault missed: {key}")
    with torch.no_grad():
        rows=anchor
        terms,_=stage(raw["x"][rows],raw["u"][rows],raw["xn"][rows],raw["ref"][rows],raw["vk"][rows],raw["vk1"][rows])
        exact=expert_cost(nom,d["x"][rows],d["u"][rows],d["xn"][rows],d["ref"][rows],env_for(d,rows)).numpy()
        np.testing.assert_allclose(exact,terms,rtol=.0002,atol=.0002)
        checks["source_stage_vs_torch_terms_max_abs"]=np.max(np.abs(terms-exact),axis=0).tolist()
        j=rollout(actor,None,nom,d,rows)
        checks["reset_history_delta_J"]=float(rollout(actor,None,nom,d,rows,reset_history=True)-j)
        checks["frozen_speed_time_delta_J"]=float(rollout(actor,None,nom,d,rows,hold_time=True)-j)
        p=power(raw["u"][rows])
        wrong=raw["u"][rows,1]+4.6331*raw["u"][rows,0]-293.2
        checks["stock_minus_expert_power_max_abs_W"]=float(np.max(np.abs(p-wrong)))
        reload=torch.jit.load(str(OUT/"imitation"/"inference.ts"))
        checks["imitation_pack_reload_max_abs"]=float((reload(d["x"][rows],d["ref"][rows,None])-actor(d["x"][rows],d["ref"][rows,None])).abs().max())
    if not all(checks[k]!=0 for k in ("reset_history_delta_J","frozen_speed_time_delta_J")):
        raise ValueError("Delay/speed perturbation had no effect")
    if checks["imitation_pack_reload_max_abs"]!=0:
        raise ValueError("Reloaded deployment predictions differ")
    checks["original_MPCCritic"]=stock_probe(nom,raw,actor)
    return checks


def stock_probe(nom,raw,policy):
    """Run the unmodified archive critic and instrument its actual per-step input."""
    class Guidance(nn.Module):
        def forward(self,x,r):
            z=(torch.cat((x,r),1)-policy.center)/policy.scale
            return policy.actor(z[:,:3],z[:,3:])

    start=1
    refs={0:raw["ref"][:20]}
    speed={0:raw["vk"][:20]}
    seen,called=[],[]
    original_step=nom.step_with_delay
    def logged(x,u,env,*args,**kwargs):
        seen.append([float(env["Vk"][0]),float(env["Vkp1"][0])])
        return original_step(x,u,env,*args,**kwargs)
    perception=core.PerceptionNet()
    handle=perception.register_forward_hook(lambda *_:called.append(1))
    critic=core.MPCCritic(nom,Q,R,S,3,dt=DT)
    def value():
        return float(critic.rollout_J(tensor(raw["x"][start:start+1]),torch.tensor([start]),torch.tensor([0]),
                                      Guidance(),perception,refs,speed).item())
    nom.step_with_delay=logged
    try:
        with torch.no_grad():
            before=value()
            captured=seen.copy()
            for p in perception.parameters():
                p.add_(3)
            after=value()
    finally:
        nom.step_with_delay=original_step
        handle.remove()
    expected=[[float(a),float(b)] for a,b in zip(raw["vk"][start:start+3],raw["vk1"][start:start+3])]
    if before!=after or called or len(captured)!=3 or len(set(map(tuple,captured)))!=1 or expected[0]==expected[-1]:
        raise ValueError("The archived critic no longer exhibits the recorded behavior")
    return dict(original_J_before=before,original_J_after_perception_mutation=after,actual_perception_calls=len(called),
                original_velocity_pairs_mps=captured,advancing_velocity_pairs_mps=expected)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",action="store_true",help="re-import the original expert archive before training")
    args=parser.parse_args()
    if args.collect:
        from heat_data import collect
        collect()
    run()
