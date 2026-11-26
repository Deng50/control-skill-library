"""Scalar expert convention, usable with Python floats. raw_actor returns a signed normalized command."""
def stage(raw_actor, state, reference, previous, cfg):
    normalized_command = raw_actor([(state-reference-cfg["center"])/cfg["scale"],previous/cfg["limit"]])
    command = cfg["limit"] * min(1.,max(-1.,normalized_command))
    applied = cfg["beta"] * previous + (1.-cfg["beta"])*command
    state_next = cfg["a"]*state + cfg["b"]*applied
    return state_next, applied

def stage_cost(next_state, reference, applied, previous, cfg):
    return cfg["q"]*(next_state-reference)**2 + cfg["r"]*applied**2 + cfg["s"]*(applied-previous)**2
