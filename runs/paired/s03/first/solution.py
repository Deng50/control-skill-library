"""Differentiable process and policy models, and lifted actuator predictions."""
import numpy as np
import torch


def heat(net, zp, spec):
    if spec.get("C") is None or spec.get("G") is None:
        return None
    inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
    temperature = spec["y_shift_C"] + spec["y_scale_K"] * net(inputs)
    # A constant-output network has a well-defined zero temperature slope.
    temperature = temperature + 0.0 * inputs[:, :1]
    gradient = torch.autograd.grad(
        temperature, inputs, grad_outputs=torch.ones_like(temperature), create_graph=True
    )[0]
    slope = gradient[:, :1] / (60.0 * spec["time_scale_min"])
    loss_to_ambient = spec["G"] * (temperature - spec["Ta_C"])
    return spec["C"] * slope + loss_to_ambient - 1000.0 * inputs[:, 1:2]


def lift(spec, N):
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    # Augmented state before a step: x, actuator memory, previous command.
    transition = np.array([[a, b, 0.0], [0.0, g, 1.0-g], [0.0, 0.0, 1.0]])
    increment = np.array([0.0, 1.0-g, 1.0])
    measurement = np.array([c*a, c*b, d])
    F = np.empty((N, 3))
    P = np.zeros((N, N))
    for k in range(N):
        F[k] = measurement
        P[k, k] = d
        if k:
            P[k, 0] = F[k-1] @ increment
            P[k, 1:] = P[k-1, :-1]
        measurement = measurement @ transition
    return F, P


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    limit = cfg["limit"]
    total = torch.zeros_like(x)
    for k in range(cfg["N"]):
        features = torch.cat(
            ((x - reference - cfg["center"]) / cfg["scale"], last / limit), dim=1
        )
        command = limit * actor(features).reshape_as(x).clamp(-1.0, 1.0)
        applied = cfg["beta"] * last + (1.0 - cfg["beta"]) * command
        next_x = cfg["a"] * x + cfg["b"] * applied
        if coupled:
            next_x = next_x + perception(torch.cat((x, applied), dim=1))
        stage = (cfg["q"] * (next_x - reference).square()
                 + cfg["r"] * applied.square()
                 + cfg["s"] * (applied - last).square())
        total = total + cfg["discount"] ** k * stage
        x, last = next_x, applied
    return total.mean()
