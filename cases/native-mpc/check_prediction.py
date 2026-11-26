"""Exercise every move column of the actual captured chart matrix with the library verifier."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controlkit.prediction import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    input_case = json.loads((args.out/"archived-prediction-input.json").read_text("utf-8-sig"))
    matrix = json.loads((args.out/"archived-matrix.json").read_text("utf-8-sig"))
    evidence = []
    horizon = len(input_case["delta_u"])
    for index in range(horizon):
        probe = deepcopy(input_case)
        du = np.zeros((horizon, 1)); du[index, 0] = 1
        probe["delta_u"] = du.tolist()
        actual = check(probe, matrix)
        repaired = check(probe)
        assert repaired["passed"]
        evidence.append({"input_increment_1based":index+1, "archive_passed":actual["passed"],
                         "archive_first_error":actual["first"], "archive_max_abs":actual["max_abs_error"],
                         "full_lifted_max_abs":repaired["max_abs_error"]})
    assert [row["input_increment_1based"] for row in evidence if row["archive_passed"]] == [1,horizon]
    assert evidence[1]["archive_first_error"]["step"] == 3
    (args.out/"prediction-probes.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    print(f"{sum(not row['archive_passed'] for row in evidence)}/{horizon} archived move columns failed")


if __name__=="__main__":
    main()
