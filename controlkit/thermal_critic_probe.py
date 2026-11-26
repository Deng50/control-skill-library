"""Measure whether the archived heat critic calls Perception during a real forward pass."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from .io import digest, runtime, write


def run(folder, output):
    folder=Path(folder).resolve()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(folder))
    import pgac_core as core
    core.device=torch.device("cpu")
    torch.manual_seed(145)
    model=core.load_thermal_model_class(folder/"thermal-management-model.py")()
    nominal=core.ThermalNominalModelTorchAlignDelay(model,dt=5)
    critic=core.MPCCritic(nominal,1/16,1/1600000000,300000000,3)
    references=core.ReferenceBank(145,np.array([293.15]),3,5,6)
    episode=references.new_episode(8,293.15)
    speeds={episode:np.full(8,10/3.6)}
    actor=core.ActorNet()
    perception=core.PerceptionNet()
    calls=[]
    hook=perception.register_forward_hook(lambda *args:calls.append(1))
    def cost():
        return critic.rollout_J(torch.tensor([[292.3,300.5,.57]]),torch.tensor([0]),
                                torch.tensor([episode]),actor,perception,references,speeds).squeeze()
    before=cost()
    with torch.no_grad():
        for p in perception.parameters():
            p.add_(3)
    after=cost()
    after.backward()
    hook.remove()
    gradient=sum(float(p.grad.norm()) for p in actor.parameters() if p.grad is not None)
    ok=bool(torch.isfinite(after) and before.item()==after.item() and not calls and gradient>0)
    write(output,dict(runtime=runtime(145),kernel_sha256=digest(folder/"pgac_core.py"),
          plant_sha256=digest(folder/"thermal-management-model.py"),stage="original heat critic only",
          steps=3,dt_s=5,initial_state=[292.3,300.5,.57],reference_K=293.15,speed_mps=10/3.6,
          cost_before=before.item(),cost_after=after.item(),perception_forward_calls=len(calls),
          sum_actor_gradient_norms=gradient,passed=ok))
    if not ok:
        raise ValueError("Observed heat critic behavior differs from the documented baseline")
    print("Archived heat critic J",after.item(),"perception calls",len(calls))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel",default="../素材/pgac/pgac_auto_train_agent_v1/kernels")
    p.add_argument("--out",default="runs/thermal-critic/report.json")
    a=p.parse_args()
    run(a.kernel,a.out)
