"""Discrete sampled process specification supplied with the engineering request."""
import numpy as np


def step(state,command,disturbance,cfg):
    flow,opening=state
    dt=cfg["dt"]
    retention=float(np.exp(-dt/.75))
    next_flow=flow + dt*(-.8*flow+1.1*opening+.3*disturbance+.09)
    next_opening=retention*opening+(1-retention)*command
    sensor=next_flow+.12*opening+.05*command
    return np.array([next_flow,next_opening]),float(sensor)


def baseline(state,reference,last,cfg):
    desired=1.4*(reference-state[0])
    du=cfg["rate"]*cfg["dt"]
    return float(np.clip(np.clip(desired,last-du,last+du),cfg["lo"],cfg["hi"]))


def scenario(name,t):
    if name=="design":
        return np.where(t<1.,.15,np.where(t<4.,.75,-.3)),.14*np.sin(2*t)
    if name=="validation":
        return np.where(t<.8,-.12,np.where(t<3.6,-.44,.51)),.15*np.cos(t)
    raise ValueError("Unknown public operating condition")
