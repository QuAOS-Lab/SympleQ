"""Uniform noise: AQUIRE's own error model (Eq. F1 of arXiv:2605.00682).

With probability xi(C) = 1 - (1 - p_local)^n_local (1 - p_entangling)^n_entangling a shot of
circuit C returns a uniformly random outcome. Global and state-independent, so AQUIRE's
diagnostics measure exactly the xi that biases the estimate.
"""
from .simulator import P_ENTANGLING, P_LOCAL, NoiseModel


class Uniform(NoiseModel):
    name = "uniform"

    def __init__(self, strength=1.0, d=3):
        self.p_local, self.p_entangling = strength * P_LOCAL, strength * P_ENTANGLING

    def measure(self, sim, pdf, circuit):
        n_ent = sum(g.n_qudits >= 2 for g in circuit.gates)
        xi = 1 - (1 - self.p_entangling) ** n_ent * (1 - self.p_local) ** (len(circuit.gates) - n_ent)
        return (1 - xi) * pdf + xi / sim.D

    def describe(self):
        return f"uniform: p_local={self.p_local:.5f}, p_entangling={self.p_entangling:.4f}"
