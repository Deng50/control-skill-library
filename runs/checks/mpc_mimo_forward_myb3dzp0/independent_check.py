"""Independent CLI acceptance test for a three-step MIMO predictor with direct feedthrough."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "runs/prediction/independent.json")
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="mpc_mimo_forward_"))
    shutil.copyfile(__file__, scratch / Path(__file__).name)
    A = np.array([[0.7, 0.2], [0.0, 0.5]])
    B = np.array([[0.4, -0.2], [0.3, 0.5]])
    C = np.array([[1.0, -0.3], [0.2, 0.8]])
    D = np.array([[0.10, 0.25], [-0.35, 0.15]])
    E = np.array([[0.2], [-0.1]])
    delta_u = np.array([[0.15, 0.2], [0.25, -0.5], [-0.1, 0.4]])
    disturbance = np.array([[0.2], [-0.3], [0.1]])
    x0 = np.array([0.5, -0.1])
    u0 = np.array([0.05, 0.2])
    tol = 1e-12

    # An input increment persists. Write out the three step-response blocks directly.
    first = C @ B + D
    second = first + C @ A @ B
    third = second + C @ A @ A @ B
    zero = np.zeros_like(first)
    correct = np.block([[first, zero, zero], [second, first, zero], [third, second, first]])
    wrong = correct.copy()
    wrong[5, 3] += 0.2  # step 3, output 2, increment at step 2 of actuator 2
    save(scratch / "good.json", correct.tolist())
    save(scratch / "bad.json", wrong.tolist())

    def run(name, changes, candidate):
        case = {k: value.tolist() for k, value in
                dict(A=A, B=B, C=C, D=D, E=E, x0=x0, previous_u=u0,
                     disturbances=disturbance, delta_u=changes).items()}
        save(scratch / f"{name}-input.json", dict(**case, abs_tol=tol))
        command = [sys.executable, "-B", "-m", "controlkit.prediction",
                   str(scratch / f"{name}-input.json"), "--matrix", str(scratch / candidate),
                   "--out", str(scratch / f"{name}-result.json")]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        target = scratch / f"{name}-result.json"
        if not target.is_file():
            raise RuntimeError(f"{name}: CLI exit {result.returncode}, stderr={result.stderr}")
        report = json.loads(target.read_text(encoding="utf-8"))

        # Independent expected outputs include nonzero initial conditions and changing disturbances.
        x, u = x0.copy(), u0.copy()
        trajectory = []
        for du, d in zip(changes, disturbance):
            u += du
            x = A @ x + B @ u + E @ d
            trajectory.append(C @ x + D @ u)
        np.testing.assert_allclose(report["recursive"], trajectory, rtol=0, atol=tol)
        return result.returncode, report

    probes = list(np.eye(6).reshape(6, 3, 2)) + [delta_u]
    positive_errors = []
    for index, probe in enumerate(probes):
        code, result = run(f"good-{index}", probe, "good.json")
        if code != 0 or not result["passed"] or result["first"] is not None:
            raise AssertionError(f"Valid matrix rejected for probe {index}")
        positive_errors.append(result["max_abs_error"])

    code, result = run("perturbed", delta_u, "bad.json")
    mismatch = result["first"]
    if code != 1 or result["passed"] or (mismatch["step"], mismatch["channel"]) != (3, 2):
        raise AssertionError(f"Wrong defect location: {result}")
    expected_error = 0.2 * delta_u[1, 1]
    np.testing.assert_allclose(mismatch["signed_error"], expected_error, rtol=0, atol=tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, dict(evidence="executed", status="passed", rng="no randomness", python=sys.version,
                        numpy=np.__version__, platform=platform.platform(), scratch=str(scratch),
                        command="python cases/prediction/independent_check.py", abs_tol=tol,
                        nx=2, nu=2, ny=2, nd=1, steps=3,
                        positive_probes="six basis increments and one composite profile",
                        positive_max_errors=positive_errors, negative_exit_code=code,
                        expected=dict(step=3, channel=2, signed_error=float(expected_error)),
                        observed=mismatch))
    print(args.out)


if __name__ == "__main__":
    main()
