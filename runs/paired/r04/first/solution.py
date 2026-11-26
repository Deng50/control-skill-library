import numpy as np
import torch


def heat(net, zp, spec):
    """Compute the dimensional heat balance at each time/condition pair."""
    if spec.get("C") is None or spec.get("G") is None:
        return None

    inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
    prediction = net(inputs)
    gradient = None
    if prediction.requires_grad:
        gradient = torch.autograd.grad(
            prediction, inputs, torch.ones_like(prediction),
            create_graph=True, allow_unused=True,
        )[0]
    dy_dz = torch.zeros_like(prediction) if gradient is None else gradient[:, :1]
    rate = dy_dz * spec["y_scale_K"] / (60.0 * spec["time_scale_min"])
    temperature = spec["y_shift_C"] + spec["y_scale_K"] * prediction
    power = 1000.0 * inputs[:, 1:2]
    return spec["C"] * rate + spec["G"] * (temperature - spec["Ta_C"]) - power


def lift(spec, N):
    """Propagate coefficients of the initial state and command increments."""
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    rows = np.empty((N, 3 + N))
    state, memory, command = np.eye(3, 3 + N)
    for k in range(N):
        command[3 + k] = 1.0
        state = a * state + b * memory
        rows[k] = c * state + d * command
        memory = g * memory + (1.0 - g) * command
    return rows[:, :3], rows[:, 3:]


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Roll out physical actions and return the mean discounted trajectory cost."""
    state, previous = x, last
    limit = cfg["limit"]
    total = torch.zeros_like(x)
    for k in range(cfg["N"]):
        error = (state - reference - cfg["center"]) / cfg["scale"]
        actor_input = torch.cat((error, previous / limit), dim=1)
        signed_command = actor(actor_input).reshape_as(previous)
        command = limit * signed_command.clamp(-1.0, 1.0)
        applied = cfg["beta"] * previous + (1.0 - cfg["beta"]) * command
        next_state = cfg["a"] * state + cfg["b"] * applied
        if coupled:
            next_state = next_state + perception(torch.cat((state, applied), dim=1))
        cost = (
            cfg["q"] * (next_state - reference).square()
            + cfg["r"] * applied.square()
            + cfg["s"] * (applied - previous).square()
        )
        total = total + cfg["discount"] ** k * cost
        state, previous = next_state, applied
    return total.mean()
