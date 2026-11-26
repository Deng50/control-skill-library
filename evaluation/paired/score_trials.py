"""Archive first submissions and grade with the frozen numerical oracle in fresh processes."""
import argparse
from collections import Counter
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from prepare import HERE, ROOT, ORDER, COMMON, sha


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def score(base,out):
    frozen=json.loads((HERE/"frozen.json").read_text(encoding="utf-8"))
    assert all(sha(HERE/k)==v for k,v in frozen.items())
    results={}
    archive=out/"library-snapshot.zip"
    if not archive.exists():
        shutil.make_archive(str(archive.with_suffix("")),"zip",base/"library-snapshot")
    for name,with_library in ORDER.items():
        src=base/name
        if not (src/"receipt.json").exists():
            continue
        assert all(sha(src/k)==frozen[k] for k in COMMON)
        snapshot=out/name/"first"
        if not snapshot.exists():
            shutil.copytree(src,snapshot,ignore=shutil.ignore_patterns("library","__pycache__","*.pyc"))
        if sha(snapshot/"solution.py")!=sha(src/"solution.py"):
            raise ValueError("Revisions must be scored separately from first submissions")
        command=[sys.executable,"-B",str(HERE/"judge.py"),str(src)]
        process=subprocess.run(command,cwd=src,capture_output=True,encoding="utf-8",check=True)
        scores=json.loads(process.stdout)
        receipt=json.loads((snapshot/"receipt.json").read_text(encoding="utf-8-sig"))
        paths=receipt["used"]
        use=[p for p in paths if p.replace("\\","/").lower().endswith("/skill.md")]
        identity=name[0]
        group="explicit" if identity=="s" else ("available" if with_library else "absent")
        if group=="explicit" and len(use)<1:
            raise ValueError("Explicit trial did not use a skill: "+name)
        results[name]=dict(group=group,source_sha256=sha(snapshot/"solution.py"),read_skills=use,used=paths,
                           first_submission=scores,repair_rounds=0)
    groups={}
    for group in ("absent","available","explicit"):
        trials=[row for row in results.values() if row["group"]==group]
        counts=Counter(task for row in trials for task,s in row["first_submission"].items() if s["pass"])
        groups[group]=dict(sessions=len(trials),successes=sum(counts.values()),tasks=3*len(trials),
             all_three=sum(all(s["pass"] for s in row["first_submission"].values()) for row in trials),
             passed_by_role={task:counts[task] for task in ("model","mpc","pgac")})
    save(out/"results.json",dict(protocol_sha256=frozen["protocol.md"],activation_sha256=sha(HERE/"activation.md"),
         snapshot_sha256=sha(archive),session_setup="fresh no-history agents; same inherited model and tools, instructed working directory",
         budget="6 minutes and 12 top-level calls per session; internal self-tests allowed before first submission",
         unit="three task outcomes clustered within each independent session; prompts repeated across sessions",
         groups=groups,trials=results))
    print(groups)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=ROOT.parent/"control-eval-nov26")
    parser.add_argument("--out",type=Path,default=ROOT/"runs/paired")
    a=parser.parse_args()
    score(a.base,a.out)
