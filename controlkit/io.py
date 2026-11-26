from pathlib import Path
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                               default=lambda value: value.tolist()) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(seed=None):
    packages = {}
    for name in ["numpy", "scipy", "torch"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return dict(python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(),
                packages=packages, seed=seed, argv=sys.argv, evidence="executed")
