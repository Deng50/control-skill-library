import numpy as np
import torch


def heat(net, zp, spec):
    """Thermal power imbalance for each time and constant-power condition."""
    if "C" not in spec or "G" not in spec:
        return None

    inputs = zp if zp.requires_grad else zp.detach().requires_grad_(True)
    temperature = spec["y_shift_C"] + spec["y_scale_K"] * net(inputs)
    derivative = None
    if temperature.requires_grad:
        derivative, = torch.autograd.grad(
            temperature, inputs, grad_outputs=torch.ones_like(temperature),
            create_graph=True, allow_unused=True,
        )
    dtdz = torch.zeros_like(inputs[:, :1]) if derivative is None else derivative[:, :1]
    storage = spec["C"] * dtdz / (60 * spec["time_scale_min"])
    loss = spec["G"] * (temperature - spec["Ta_C"])
    return storage + loss - 1000 * inputs[:, 1:2]


def lift(spec, N):
    """Map initial conditions and command increments to future sensor values."""
    a, b, c, d, g = (spec[key] for key in ("a", "b", "c", "d", "g"))
    # Each row carries coefficients of [x0, memory0, u_prev, du0, ...].
    state, memory, command = np.eye(3, 3 + N)
    outputs = np.empty((N, 3 + N))
    for step in range(N):
        command[3 + step] = 1
        state = a * state + b * memory
        outputs[step] = c * state + d * command
        memory = g * memory + (1 - g) * command
    return outputs[:, :3], outputs[:, 3:]


def policy_cost(actor, perception, x, reference, last, cfg, coupled):
    """Mean discounted closed-loop cost, differentiable through the rollout."""
    state, previous = x, last
    total = torch.zeros_like(x)
    for step in range(cfg["N"]):
        error = (state - reference - cfg["center"]) / cfg["scale"]
        features = torch.cat((error, previous / cfg["limit"]), dim=1)
        command = cfg["limit"] * actor(features).reshape_as(previous).clamp(-1, 1)
        applied = cfg["beta"] * previous + (1 - cfg["beta"]) * command
        next_state = cfg["a"] * state + cfg["b"] * applied
        if coupled:
            next_state = next_state + perception(torch.cat((state, applied), dim=1))
        stage = (cfg["q"] * (next_state - reference).square()
                 + cfg["r"] * applied.square()
                 + cfg["s"] * (applied - previous).square())
        total = total + cfg["discount"] ** step * stage
        state, previous = next_state, applied
    return total.mean()
