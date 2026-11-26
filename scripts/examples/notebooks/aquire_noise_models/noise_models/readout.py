"""Readout error: each qudit's outcome is replaced by a random other value with probability
p = strength x 1%."""
from .simulator import NoiseModel


class Readout(NoiseModel):
    name = "readout"

    def __init__(self, strength=1.0, d=3):
        self.p = strength * 0.01

    def measure(self, sim, pdf, circuit):
        T = pdf.reshape(sim.dims)
        for q, d in enumerate(sim.dims):
            T = (1 - self.p) * T + self.p * (T.sum(axis=q, keepdims=True) - T) / (d - 1)
        return T.ravel()

    def describe(self):
        return f"readout: p={self.p:.4f}"
