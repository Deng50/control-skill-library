"""Thermal training, actuator prediction, and differentiable control costs."""

import numpy as np
import torch


def heat(net, zp, spec):
    """Return the heat-balance residual in watts at each (time, power) pair."""
    if spec.get("C") is None or spec.get("G") is None:
        return None
    with torch.enable_grad():
        inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
        y = net(inputs)
        # Keep a zero-valued input dependency for constant-temperature networks.
        y = y + 0.0 * inputs[:, :1]
        dy = torch.autograd.grad(y, inputs, torch.ones_like(y), create_graph=True)[0]
        dtemp_dt = dy[:, :1] * spec["y_scale_K"] / (60.0 * spec["time_scale_min"])
        temp = spec["y_shift_C"] + spec["y_scale_K"] * y
        return spec["C"] * dtemp_dt + spec["G"] * (temp - spec["Ta_C"]) - 1000.0 * inputs[:, 1:2]


def lift(spec, N):
    """Map initial augmented state and command increments to step outputs."""
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    transition = np.array([[a, b, 0], [0, g, 1-g], [0, 0, 1]], dtype=float)
    control = np.array([0, 1-g, 1], dtype=float)
    output = np.array([c*a, c*b, d], dtype=float)
    F = np.empty((N, 3))
    P = np.zeros((N, N))
    impulse = np.empty(N)
    impulse[0] = d
    for i in range(N):
        F[i] = output
        if i + 1 < N:
            impulse[i+1] = output @ control
        P[i, :i+1] = impulse[i::-1]
        output = output @ transition
    return F, P


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Evaluate the batch mean of the actor's discounted rollout cost."""
    total = torch.zeros_like(x)
    weight = 1.0
    for _ in range(cfg["N"]):
        error = (x - reference - cfg["center"]) / cfg["scale"]
        features = torch.cat((error, last / cfg["limit"]), dim=1)
        command = actor(features).reshape_as(last).clamp(-1, 1) * cfg["limit"]
        applied = cfg["beta"] * last + (1.0 - cfg["beta"]) * command
        next_x = cfg["a"] * x + cfg["b"] * applied
        if coupled:
            next_x = next_x + perception(torch.cat((x, applied), dim=1))
        cost = (cfg["q"] * (next_x - reference).square()
                + cfg["r"] * applied.square()
                + cfg["s"] * (applied - last).square())
        total = total + weight * cost
        weight *= cfg["discount"]
        x, last = next_x, applied
    return total.mean()
