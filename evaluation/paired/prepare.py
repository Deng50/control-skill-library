"""Prepare identical fresh participant workspaces, with a library snapshot only in treatment."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COMMON=("brief.md","actuator.py","teacher.py")
# Order of submission alternates to avoid putting all treatment interviews last.
ORDER={"r01":False,"r02":True,"r03":True,"r04":False,"r05":False,"r06":True,
       "s01":True,"s02":True,"s03":True}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base,target):
    dest=Path(base).resolve()/target
    dest.mkdir(parents=True,exist_ok=False)
    for filename in COMMON:
        shutil.copy2(HERE/filename,dest/filename)
    if ORDER[target]:
        library=dest/"library"
        snapshot=Path(base).resolve()/"library-snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in (".agents","knowledge","cases","controlkit"):
                shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            for name in ("index.json","README.md","replay.py","requirements.txt","sources.json"):
                shutil.copy2(ROOT/name,snapshot/name)
            # Ship numerical reports for inspection. No paired tasks, judge or participant evidence.
            for name in ("thermal","pgac","mpc","object-plant","native-mpc","heat-policy","plant-transfer","prediction"):
                directory=ROOT/"runs"/name
                for report in directory.rglob("*.json"):
                    relative=report.relative_to(ROOT)
                    (snapshot/relative).parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(report,snapshot/relative)
        shutil.copytree(snapshot,library)
    return dest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id",choices=[*ORDER,"freeze"])
    parser.add_argument("--base",type=Path,default=ROOT.parent/"control-eval-nov26")
    args=parser.parse_args()
    if args.id=="freeze":
        names=[*COMMON,"judge.py","protocol.md"]
        path=HERE/"frozen.json"
        checks={n:sha(HERE/n) for n in names}
        if path.exists():
            assert checks==json.loads(path.read_text(encoding="utf-8")), "Frozen acceptance changed"
        else:
            path.write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    else:
        print(prepare(args.base,args.id))
