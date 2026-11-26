"""Extracted interface for the held-out single-input airflow benchmark."""
class Actuator:
    def __init__(self, spec, x, memory):
        self.spec = spec
        self.x = x
        self.memory = memory

    def step(self, command):
        p = self.spec
        self.x = p["a"]*self.x + p["b"]*self.memory
        output = p["c"]*self.x + p["d"]*command
        self.memory = p["g"]*self.memory + (1-p["g"])*command
        return output
