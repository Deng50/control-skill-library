"""Differentiable thermal residual, lifted actuator model and policy loss."""

import numpy as np
import torch


def heat(net, zp, spec):
    """Return the thermal balance in watts for each time/power pair."""
    if spec.get("C") is None or spec.get("G") is None:
        return None
    with torch.enable_grad():
        inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
        normalized = net(inputs)
        slope = None
        if normalized.requires_grad:
            slope = torch.autograd.grad(
                normalized, inputs, grad_outputs=torch.ones_like(normalized),
                create_graph=True, allow_unused=True,
            )[0]
        dy_dz = torch.zeros_like(normalized) if slope is None else slope[:, :1]
        kelvin_per_second = dy_dz * spec["y_scale_K"] / (60 * spec["time_scale_min"])
        temperature = spec["y_shift_C"] + spec["y_scale_K"] * normalized
        return (spec["C"] * kelvin_per_second
                + spec["G"] * (temperature - spec["Ta_C"])
                - 1000 * inputs[:, 1:2])


def lift(spec, N):
    """Predict post-step sensor readings from initial state and command increments."""
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    # State at the start of a step is [x, memory, previous_command].
    transition = np.array([[a, b, 0], [0, g, 1 - g], [0, 0, 1]], dtype=float)
    increment = np.array([0, 1 - g, 1])
    sensor = np.array([c * a, c * b, d])
    F = np.empty((N, 3))
    response = np.empty(N)
    response[0] = d
    for t in range(N):
        F[t] = sensor
        if t + 1 < N:
            response[t + 1] = sensor @ increment
            sensor = sensor @ transition
    P = np.zeros((N, N))
    for t in range(N):
        P[t, :t + 1] = response[t::-1]
    return F, P


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Mean discounted physical stage cost, differentiable through the rollout."""
    cost = torch.zeros_like(x)
    for t in range(cfg["N"]):
        features = torch.cat((
            (x - reference - cfg["center"]) / cfg["scale"],
            last / cfg["limit"],
        ), dim=1)
        command = actor(features).clamp(-1, 1) * cfg["limit"]
        applied = cfg["beta"] * last + (1 - cfg["beta"]) * command
        next_x = cfg["a"] * x + cfg["b"] * applied
        if coupled:
            next_x = next_x + perception(torch.cat((x, applied), dim=1))
        stage = (cfg["q"] * (next_x - reference).square()
                 + cfg["r"] * applied.square()
                 + cfg["s"] * (applied - last).square())
        cost = cost + cfg["discount"] ** t * stage
        x, last = next_x, applied
    return cost.mean()
