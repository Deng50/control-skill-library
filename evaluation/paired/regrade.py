"""Replay numerical grading from the delivered trial sources, without participant workspaces."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from prepare import HERE, ROOT, COMMON, ORDER, sha


def replay():
    out=ROOT/"runs/paired"
    results=json.loads((out/"results.json").read_text(encoding="utf-8"))
    frozen=json.loads((HERE/"frozen.json").read_text(encoding="utf-8"))
    if set(results["trials"])!=set(ORDER):
        raise ValueError("Missing or unexpected blind trial")
    if any(sha(HERE/f)!=v for f,v in frozen.items()):
        raise ValueError("Inputs or hidden judge changed")
    zipfile=out/"library-snapshot.zip"
    if sha(zipfile)!=results["snapshot_sha256"]:
        raise ValueError("Snapshot checksum changed")
    scratch=ROOT/"runs/checks"
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as work:
        base=Path(work)
        for name,record in results["trials"].items():
            src=out/name/"first"
            if sha(src/"solution.py")!=record["source_sha256"] or any(sha(src/f)!=frozen[f] for f in COMMON):
                raise ValueError("Trial contents changed: "+name)
            dest=base/name
            shutil.copytree(src,dest)
            if record["group"]!="absent":
                with ZipFile(zipfile) as archive:
                    archive.extractall(dest/"library")
            proc=subprocess.run([sys.executable,"-B",str(HERE/"judge.py"),str(dest)],cwd=dest,check=True,
                                capture_output=True,encoding="utf-8")
            if json.loads(proc.stdout)!=record["first_submission"]:
                raise ValueError("Different hidden scores: "+name)
    print(len(results["trials"]),"first submissions independently regraded")


if __name__=="__main__":
    replay()
