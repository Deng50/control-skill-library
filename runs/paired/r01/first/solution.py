import numpy as np
import torch


def heat(net, zp, spec):
    """Return the samplewise heat balance in watts."""
    if spec.get("C") is None or spec.get("G") is None:
        return None
    inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
    normalized = net(inputs)
    gradient = None
    if normalized.requires_grad:
        gradient = torch.autograd.grad(
            normalized, inputs, grad_outputs=torch.ones_like(normalized),
            create_graph=True, allow_unused=True,
        )[0]
    slope = torch.zeros_like(normalized) if gradient is None else gradient[:, :1]
    rate = spec["y_scale_K"] * slope / (60 * spec["time_scale_min"])
    temperature = spec["y_shift_C"] + spec["y_scale_K"] * normalized
    return spec["C"] * rate + spec["G"] * (temperature - spec["Ta_C"]) - 1000 * inputs[:, 1:2]


def lift(spec, N):
    """Map initial conditions and command increments to successive sensor values."""
    x, memory, previous = np.eye(3, N + 3)
    rows = np.empty((N, N + 3))
    for k in range(N):
        command = previous.copy()
        command[3 + k] += 1
        x = spec["a"] * x + spec["b"] * memory
        rows[k] = spec["c"] * x + spec["d"] * command
        memory = spec["g"] * memory + (1 - spec["g"]) * command
        previous = command
    return rows[:, :3], rows[:, 3:]


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Evaluate a discounted rollout in physical units with autograd enabled."""
    state, previous = x, last
    cost = torch.zeros_like(state)
    for k in range(cfg["N"]):
        error = (state - reference - cfg["center"]) / cfg["scale"]
        features = torch.cat((error, previous / cfg["limit"]), dim=1)
        raw = actor(features).reshape_as(state)
        command = cfg["limit"] * raw.clamp(-1, 1)
        applied = cfg["beta"] * previous + (1 - cfg["beta"]) * command
        next_state = cfg["a"] * state + cfg["b"] * applied
        if coupled:
            correction = perception(torch.cat((state, applied), dim=1))
            next_state = next_state + correction.reshape_as(state)
        stage = (cfg["q"] * (next_state - reference).square()
                 + cfg["r"] * applied.square()
                 + cfg["s"] * (applied - previous).square())
        cost = cost + cfg["discount"] ** k * stage
        state, previous = next_state, applied
    return cost.mean()
