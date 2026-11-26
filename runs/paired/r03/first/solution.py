import numpy as np
import torch


def heat(net, zp, spec):
    """Evaluate the storage, loss and heating power balance in watts."""
    if spec.get("C") is None or spec.get("G") is None:
        return None
    with torch.enable_grad():
        inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
        temperature = spec["y_shift_C"] + spec["y_scale_K"] * net(inputs)
        # The zero term also gives an input-independent network a zero slope.
        temperature = temperature + 0 * inputs[:, :1]
        gradient, = torch.autograd.grad(
            temperature, inputs, torch.ones_like(temperature), create_graph=True
        )
        slope = gradient[:, :1] / (60 * spec["time_scale_min"])
        return (
            spec["C"] * slope
            + spec["G"] * (temperature - spec["Ta_C"])
            - 1000 * inputs[:, 1:2]
        )


def lift(spec, N):
    """Predict each post-step measurement from the initial state and moves."""
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    transition = np.array([[a, b, 0], [0, g, 1 - g], [0, 0, 1]])
    move = np.array([0, 1 - g, 1])
    row = np.array([c * a, c * b, d])
    F = np.empty((N, 3))
    P = np.zeros((N, N))
    response = np.empty(N)
    response[0] = d
    for k in range(N):
        F[k] = row
        P[k, :k + 1] = response[k::-1]
        if k + 1 < N:
            response[k + 1] = row @ move
            row = row @ transition
    return F, P


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Compute the batch mean of the discounted expert stage costs."""
    cost = torch.zeros_like(x)
    for k in range(cfg["N"]):
        features = torch.cat(
            ((x - reference - cfg["center"]) / cfg["scale"], last / cfg["limit"]),
            dim=1,
        )
        raw = actor(features).reshape_as(x)
        command = cfg["limit"] * raw.clamp(-1, 1)
        applied = cfg["beta"] * last + (1 - cfg["beta"]) * command
        next_x = cfg["a"] * x + cfg["b"] * applied
        if coupled:
            next_x = next_x + perception(torch.cat((x, applied), dim=1))
        stage = (
            cfg["q"] * (next_x - reference).square()
            + cfg["r"] * applied.square()
            + cfg["s"] * (applied - last).square()
        )
        cost = cost + cfg["discount"] ** k * stage
        x, last = next_x, applied
    return cost.mean()
