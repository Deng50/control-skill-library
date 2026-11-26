"""Thermal training residual, actuator prediction, and differentiable policy cost."""

import numpy as np
import torch


def heat(net, zp, spec):
    """Return the pointwise heat balance in watts."""
    if spec.get("C") is None or spec.get("G") is None:
        return None

    inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
    temperature = spec["y_shift_C"] + spec["y_scale_K"] * net(inputs)
    # The zero term also defines the derivative of an input-independent field.
    field = temperature + 0 * inputs[:, :1]
    gradient, = torch.autograd.grad(field.sum(), inputs, create_graph=True)
    seconds_per_z = 60 * spec["time_scale_min"]
    storage = spec["C"] * gradient[:, :1] / seconds_per_z
    loss = spec["G"] * (temperature - spec["Ta_C"])
    return storage + loss - 1000 * inputs[:, 1:2]


def lift(spec, N):
    """Predict post-step sensor outputs from [x, memory, previous command] and du."""
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    transition = np.array([[a, b, 0], [0, g, 1 - g], [0, 0, 1]])
    increment_gain = np.array([0, 1 - g, 1])
    F = np.empty((N, 3))
    F[0] = [c * a, c * b, d]
    for k in range(1, N):
        F[k] = F[k - 1] @ transition

    impulse = np.empty(N)
    impulse[0] = d
    impulse[1:] = F[:-1] @ increment_gain
    P = np.zeros((N, N))
    for k in range(N):
        P[k, :k + 1] = impulse[k::-1]
    return F, P


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Roll out the actor and average the discounted physical cost across trajectories."""
    total = torch.zeros_like(x)
    limit = cfg["limit"]
    for k in range(cfg["N"]):
        error = (x - reference - cfg["center"]) / cfg["scale"]
        features = torch.cat((error, last / limit), dim=1)
        command = limit * actor(features).reshape_as(x).clamp(-1, 1)
        applied = cfg["beta"] * last + (1 - cfg["beta"]) * command
        next_x = cfg["a"] * x + cfg["b"] * applied
        if coupled:
            next_x = next_x + perception(torch.cat((x, applied), dim=1))
        stage = (cfg["q"] * (next_x - reference).square()
                 + cfg["r"] * applied.square()
                 + cfg["s"] * (applied - last).square())
        total = total + cfg["discount"] ** k * stage
        x, last = next_x, applied
    return total.mean()
