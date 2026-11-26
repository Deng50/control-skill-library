"""Replay a complete engineering request from the project root."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parent
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ARCHIVE="../素材/pgac/spring-damper-ppt-20261030/artifacts/expert_dataset-506f49fac84b/expert.npz"
SCRATCH=ROOT/"runs"/"checks"


def execute(args, expected=0):
    print(">",subprocess.list2cmdline([str(x) for x in args]),flush=True)
    env=dict(os.environ,PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1",TMPDIR=str(SCRATCH))
    result=subprocess.run(args,cwd=ROOT,env=env)
    if result.returncode!=expected:
        raise SystemExit(f"Expected exit {expected}, received {result.returncode} for {args}")


def module(name,*args,expected=0):
    execute([sys.executable,"-m","controlkit."+name,*args],expected)


def python(file,*args):
    execute([sys.executable,file,*args])


def mpc():
    module("lookup","请核验二阶对象预测矩阵与逐步递推并定位首个出错通道","--limit","1")
    cfg="cases/prediction/input.json"
    module("prediction",cfg,"--out","runs/prediction/pass.json")
    module("prediction",cfg,"--matrix","cases/prediction/legacy-matrix.json","--out","runs/prediction/legacy.json",expected=1)
    python("cases/prediction/independent_check.py")
    module("lookup","给二阶对象设计 MPC 并调优跟踪权重","--limit","1")
    module("spring","--config","cases/spring/controller.json","--controller-factory","controlkit.spring:Controller","--out","runs/mpc")


def pinn():
    module("lookup","用恒功率温度轨迹训练 PINN 并检验物理残差","--limit","1")
    csv="runs/thermal/data.csv"
    module("thermal","sample","--csv",csv)
    module("thermal","train","--csv",csv,"--physics","cases/thermal/physics.json","--out","runs/thermal/paired")
    module("thermal","train","--csv",csv,"--out","runs/thermal/data-only")
    module("thermal","infer","runs/thermal/paired/pinn","20","298","15")
    python("cases/thermal/independent_check.py","--scratch-parent",str(SCRATCH))
    module("table","audit",csv,"cases/thermal/table.json","--out","runs/table/audit.json")
    module("table","train",csv,"cases/thermal/table.json","--out","runs/table")
    module("table","infer","runs/table","296","10","295")


def pgac(with_expert=True):
    module("lookup","用同一对象模仿权重暖启动 PGAC 并核验感知残差对 Critic 的影响","--limit","1")
    if with_expert:
        module("spring","--config","cases/spring/controller.json","--controller-factory","controlkit.spring:Controller","--out","runs/mpc")
    module("sampling_plan")
    for action in ("collect","imitate","optimize"):
        module("policy",action)
    module("policy","infer","runs/pgac/selected","0.1","0","0.4","0.12")
    python("cases/spring/independent_check.py","--scratch-parent",str(SCRATCH))


def archive():
    module("identify","../素材/55-系统辨识MPC-ADRC-PID")
    module("linearize","../素材/pgac/thermal-management-model.py")
    module("thermal_critic_probe")
    module("expert_import",ARCHIVE,"--split","cases/spring/split.json","--dt","0.2","--force","4","--initial","0","--out","runs/archive-import/data.json")
    module("policy","imitate","--data",ARCHIVE,"--manifest","runs/archive-import/data.json","--out","runs/archive-import/imitation")
    module("provenance")
    execute(["node","adapters/inspect_harness.mjs"])


def matlab():
    execute(["matlab","-batch","addpath(fullfile(pwd,'cases','matlab')); replay_second_order('../素材','runs/matlab')"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=["pinn","mpc","pgac","archive","matlab","all","verify"])
    a=p.parse_args()
    SCRATCH.mkdir(parents=True,exist_ok=True)
    if a.task=="all":
        mpc()
        pinn()
        pgac(with_expert=False)
        archive()
        matlab()
        module("verify")
    elif a.task=="verify":
        module("verify")
    else:
        {"pinn":pinn,"mpc":mpc,"pgac":pgac,"archive":archive,"matlab":matlab}[a.task]()
