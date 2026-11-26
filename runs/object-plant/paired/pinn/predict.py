"""Standalone CPU inference; place this file next to weights.npz and interface.json."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class Plant:
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.info = json.loads((bundle / "interface.json").read_text(encoding="utf-8"))
        if self.info["schema"] != "second-order-interval-flow-v1":
            raise ValueError("Unsupported bundle")
        for name, key in (("weights.npz", "weights_sha256"), ("scaler.json", "scaler_sha256")):
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != self.info[key]:
                raise ValueError(f"{name} integrity check failed")
        scale = json.loads((bundle / "scaler.json").read_text(encoding="utf-8"))
        self.mean = np.array(scale["input_mean"])
        self.std = np.array(scale["input_std"])
        self.acc_scale = scale["acceleration_scale"]
        self.h = self.info["max_hold_s"]
        with np.load(bundle / "weights.npz", allow_pickle=False) as data:
            self.w = dict(data)

    def coefficients(self, state, u):
        state = np.asarray(state, dtype=float)
        if state.ndim < 1 or state.shape[-1] != 2:
            raise ValueError("Expected final state axis [y,v]")
        u = np.broadcast_to(np.asarray(u, dtype=float), state.shape[:-1])
        features = np.concatenate((state, u[..., None]), axis=-1)
        if not np.isfinite(features).all():
            raise ValueError("State and control must be finite")
        z = (features-self.mean) / self.std
        w = self.w
        hidden = np.tanh(z @ w["hidden.weight"].T + w["hidden.bias"])
        return self.acc_scale * (hidden @ w["head.weight"].T + z @ w["linear.weight"].T + w["linear.bias"])

    def at(self, state, u, tau):
        """Return y, dy/dt, d²y/dt² within one ZOH hold, tau in [0,h]."""
        tau = float(tau)
        if not 0 <= tau <= self.h:
            raise ValueError(f"Expected a within-hold time in [0,{self.h}] seconds")
        state = np.asarray(state, dtype=float)
        a0, a1, a2 = np.moveaxis(self.coefficients(state, u), -1, 0)
        y0, v0 = np.moveaxis(state, -1, 0)
        s = tau / self.h
        y = y0 + tau*v0 + tau*tau*(a0/2 + s*a1/6 + s*s*a2/12)
        v = v0 + tau*(a0 + s*a1/2 + s*s*a2/3)
        return y, v, a0 + s*a1 + s*s*a2

    def step(self, state, u, dt=.1):
        if not self.info["min_step_s"] <= dt <= self.h:
            raise ValueError("Step duration outside the declared interval")
        return np.stack(self.at(state, u, dt)[:2], axis=-1)

    def rollout(self, initial_state, controls, dt=.1):
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 1 or not np.isfinite(controls).all():
            raise ValueError("Expected a finite one-dimensional command sequence")
        initial_state = np.asarray(initial_state, dtype=float)
        if initial_state.shape != (2,) or not np.isfinite(initial_state).all():
            raise ValueError("Expected a finite initial [y,v] pair")
        trajectory = np.empty((len(controls)+1, 2))
        trajectory[0] = initial_state
        for k, command in enumerate(controls):
            trajectory[k+1] = self.step(trajectory[k], command, dt)
        return trajectory


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--state", type=float, nargs=2, required=True, metavar=("Y", "V"))
    p.add_argument("--u", type=float, nargs="+", required=True)
    p.add_argument("--dt", type=float, default=.1)
    args = p.parse_args()
    print(json.dumps(Plant(args.bundle).rollout(args.state, args.u, args.dt).tolist()))
