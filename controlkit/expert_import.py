"""Bind an existing spring expert NPZ to an explicit interface and trajectory split."""
import argparse
from pathlib import Path

from .io import digest, read, write
from .policy import FEATURES, dataset


def run(source, split, dt, force, initial, out):
    from tempfile import TemporaryDirectory
    meta = dict(features=FEATURES, target="force_N", force_limit=force, sample_time_s=dt,
                dataset_sha256=digest(source), initial_previous_N=initial, splits=read(split),
                provenance="user-supplied existing expert trajectories")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        test = Path(tmp)/"candidate.json"
        write(test, meta)
        rows, _, _, _ = dataset(source, test)
    meta.update(rows=len(rows["x"]), has_timestamps="t" in rows, has_next_states="next_x" in rows)
    write(out, meta)
    print(f"Validated {meta['rows']} existing expert observations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--split", required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--force", type=float, required=True)
    parser.add_argument("--initial", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.source, args.split, args.dt, args.force, args.initial, args.out)
