"""Check a lifted incremental-input model against an independent state recursion."""
import argparse
import numpy as np

from .io import read, runtime, write


def lifted(A, B, C, D, horizon):
    """Return O, T with Y=O x0 + T U for y[k+1]=C x[k+1]+D u[k]."""
    nx, nu, ny = len(A), B.shape[1], len(C)
    powers = [np.eye(nx)]
    for _ in range(horizon):
        powers.append(powers[-1] @ A)
    O = np.vstack([C @ power for power in powers[1:]])
    T = np.zeros((horizon * ny, horizon * nu))
    for k in range(horizon):
        for j in range(k + 1):
            block = C @ powers[k-j] @ B
            if j == k:
                block = block + D
            T[k*ny:(k+1)*ny, j*nu:(j+1)*nu] = block
    return O, T


def check(case, candidate=None):
    A, B, C, D, E = [np.asarray(case[name], dtype=float) for name in "ABCDE"]
    x, previous, du, disturbances = [np.asarray(case[name], dtype=float)
                                     for name in ("x0", "previous_u", "delta_u", "disturbances")]
    h, nu = du.shape
    ny, nx = C.shape
    if (h < 1 or A.shape != (nx, nx) or B.shape != (nx, nu) or D.shape != (ny, nu)
            or x.shape != (nx,) or previous.shape != (nu,) or E.shape[0] != nx
            or disturbances.shape != (h, E.shape[1])):
        raise ValueError("Inconsistent model, horizon or sequence dimensions")
    if not all(np.isfinite(v).all() for v in (A,B,C,D,E,x,previous,du,disturbances)):
        raise ValueError("Non-finite model or input")
    O, T = lifted(A, B, C, D, h)
    _, Td = lifted(A, E, C, np.zeros((ny,E.shape[1])), h)
    accum = np.kron(np.tril(np.ones((h,h))), np.eye(nu))
    phi = T @ accum if candidate is None else np.asarray(candidate, dtype=float)
    if phi.shape != (h*ny,h*nu) or not np.isfinite(phi).all():
        raise ValueError("Candidate matrix shape or entries invalid")
    flat = O @ x + T @ np.tile(previous,h) + Td @ disturbances.ravel() + phi @ du.ravel()
    u = previous.copy()
    reference = []
    for change, disturbance in zip(du, disturbances):
        u = u + change
        x = A @ x + B @ u + E @ disturbance
        reference.append(C @ x + D @ u)
    prediction = flat.reshape(h,ny)
    err = prediction - np.asarray(reference)
    tol = float(case.get("abs_tol",1e-10))
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("Invalid absolute tolerance")
    failing = np.argwhere(np.abs(err) > tol)
    first = None
    if len(failing):
        k, c = failing[0]
        first = dict(step=int(k+1), channel=int(c+1), signed_error=float(err[k,c]),
                     predicted=float(prediction[k,c]), recursive=float(reference[k][c]))
    return dict(passed=first is None, max_abs_error=float(np.max(np.abs(err))), first=first,
                predicted=prediction, recursive=np.array(reference), absolute_tolerance=tol)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case")
    p.add_argument("--matrix", help="JSON array: output-by-increment lifted matrix")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = check(read(a.case), read(a.matrix) if a.matrix else None)
    write(a.out, dict(**result, runtime=runtime()))
    print("PASS" if result["passed"] else result["first"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
