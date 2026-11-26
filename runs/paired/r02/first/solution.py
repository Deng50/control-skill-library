"""Engineering functions for the thermal, actuator, and policy models."""

import numpy as np
import torch


def heat(net, zp, spec):
    """Return storage plus heat loss minus applied power, in watts."""
    if spec.get("C") is None or spec.get("G") is None:
        return None

    inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
    temperature = spec["y_shift_C"] + spec["y_scale_K"] * net(inputs)
    grad = None
    if temperature.requires_grad:
        grad, = torch.autograd.grad(
            temperature, inputs, grad_outputs=torch.ones_like(temperature),
            create_graph=True, allow_unused=True,
        )
    slope = torch.zeros_like(inputs[:, :1]) if grad is None else grad[:, :1]
    kelvin_per_second = slope / (60.0 * spec["time_scale_min"])
    return (spec["C"] * kelvin_per_second
            + spec["G"] * (temperature - spec["Ta_C"])
            - 1000.0 * inputs[:, 1:2])


def lift(spec, N):
    """Map initial [x, memory, previous command] and command increments to readings."""
    a, b, c, d, g = (spec[k] for k in ("a", "b", "c", "d", "g"))
    transition = np.array([[a, b, 0.], [0., g, 1. - g], [0., 0., 1.]])
    increment = np.array([0., 1. - g, 1.])
    reading = np.array([c * a, c * b, d])

    F = np.empty((N, 3))
    for step in range(N):
        F[step] = reading
        reading = reading @ transition

    impulse = np.empty(N)
    impulse[0] = d
    impulse[1:] = F[:-1] @ increment
    P = np.zeros((N, N))
    for step in range(N):
        P[step, :step + 1] = impulse[step::-1]
    return F, P


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Compute the discounted mean rollout cost with derivatives through the actions."""
    total = torch.zeros_like(x)
    for step in range(cfg["N"]):
        features = torch.cat(((x - reference - cfg["center"]) / cfg["scale"],
                              last / cfg["limit"]), dim=1)
        command = actor(features).reshape_as(x).clamp(-1., 1.) * cfg["limit"]
        applied = cfg["beta"] * last + (1. - cfg["beta"]) * command
        next_x = cfg["a"] * x + cfg["b"] * applied
        if coupled:
            next_x = next_x + perception(torch.cat((x, applied), dim=1))
        stage = (cfg["q"] * (next_x - reference).square()
                 + cfg["r"] * applied.square() + cfg["s"] * (applied - last).square())
        total = total + cfg["discount"] ** step * stage
        x, last = next_x, applied
    return total.mean()
