"""Set up and preserve the complete controller-delivery paired experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT.parent/"control-eval-nov26"
IDS={"v01":False,"v02":True,"v03":True,"v04":False}
FILES=("brief.md","process.py","settings.json")
OUT=ROOT/"runs/valve-evaluation"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    hashes={k:digest(HERE/k) for k in (*FILES,"check.py","protocol.md")}
    frozen=HERE/"frozen.json"
    if frozen.exists():
        assert hashes==json.loads(frozen.read_text())
    else:
        frozen.write_text(json.dumps(hashes,indent=2)+"\n")
    for name,treatment in IDS.items():
        dest=BASE/name
        dest.mkdir(exist_ok=False)
        for file in FILES:
            shutil.copy2(HERE/file,dest/file)
        if treatment:
            shutil.copytree(BASE/"library-snapshot",dest/"library")


def score():
    hashes=json.loads((HERE/"frozen.json").read_text())
    assert all(digest(HERE/k)==v for k,v in hashes.items())
    ledger=OUT/"results.json"
    previous=json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {}
    results={}
    for name,treatment in IDS.items():
        origin=BASE/name
        folder=OUT/name/"first"
        if not folder.exists():
            if not (origin/"receipt.json").exists():
                raise ValueError("Missing first submission: "+name)
            shutil.copytree(origin,folder,ignore=shutil.ignore_patterns("library","__pycache__","*.pyc"))
        assert all(digest(folder/k)==hashes[k] for k in FILES)
        cmd=[sys.executable,"-B",str(HERE/"check.py"),str(folder)]
        proc=subprocess.run(cmd,cwd=folder,encoding="utf-8",capture_output=True,check=True)
        metrics=json.loads(proc.stdout.splitlines()[-1])
        receipt=json.loads((folder/"receipt.json").read_text(encoding="utf-8-sig"))
        source_hash=digest(folder/"solution.py")
        if name in previous and previous[name]["source_sha256"]!=source_hash:
            raise ValueError("Archived source changed: "+name)
        results[name]=dict(group="explicit" if treatment else "absent",first=metrics,used=receipt["used"],
                           source_sha256=source_hash)
    OUT.mkdir(exist_ok=True,parents=True)
    ledger.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print({arm:[r["first"]["passed"] for r in results.values() if r["group"]==arm] for arm in ("absent","explicit")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("setup","score"))
    args=parser.parse_args()
    (setup if args.mode=="setup" else score)()
