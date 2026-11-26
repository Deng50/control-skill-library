"""Standalone CPU inference from a heat-policy export. Requires only PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def load(folder):
    interface = json.loads((folder/"interface.json").read_text(encoding="utf-8"))
    exe = folder/"inference.ts"
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=interface["inference_sha256"]:
        raise ValueError("TorchScript file differs from the declared inference package")
    if interface["observations"]!=["Tb_K","Tm_K","SOC","Tb_ref_K"]:
        raise ValueError("Four-input heat policy required")
    if interface["actions"]!=["pump_rpm","heater_W","bypass_fraction"] or interface["dt_s"]!=5:
        raise ValueError("Action order or sample period mismatch")
    return torch.jit.load(str(exe), map_location="cpu").eval()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack",type=Path)
    p.add_argument("inputs",nargs=4,type=float,metavar="X",help="Tb_K Tm_K SOC Tb_ref_K")
    a=p.parse_args()
    with torch.inference_mode():
        z=torch.tensor([a.inputs],dtype=torch.float32)
        if not torch.isfinite(z).all():
            raise ValueError("All inputs must be finite")
        u=load(a.pack)(z[:,:3],z[:,3:]).tolist()[0]
    print(json.dumps(dict(zip(("pump_rpm","heater_W","bypass_fraction"),u)),indent=2))
