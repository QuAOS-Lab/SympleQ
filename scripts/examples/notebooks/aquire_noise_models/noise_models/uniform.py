"""Uniform noise: AQUIRE's own error model (Eq. F1 of arXiv:2605.00682).

With probability xi(C) = 1 - (1 - p_local)^n_local (1 - p_entangling)^n_entangling a shot of
circuit C returns a uniformly random outcome. It acts on the outcome probabilities, not after
each gate: pdf -> (1 - xi) pdf + xi / D.
"""
from .simulator import P_ENTANGLING, P_LOCAL, NoiseModel


class Uniform(NoiseModel):
    name = "uniform"

    def __init__(self, strength=1.0, d=3):
        self.p = {"local": strength * P_LOCAL, "entangling": strength * P_ENTANGLING}

    def measure(self, sim, pdf, circuit):
        n_ent = sum(g.n_qudits >= 2 for g in circuit.gates)
        xi = 1 - (1 - self.p["entangling"]) ** n_ent * (1 - self.p["local"]) ** (len(circuit.gates) - n_ent)
        return (1 - xi) * pdf + xi / sim.D

    def describe(self):
        return f"uniform: p_local={self.p['local']:.5f}, p_entangling={self.p['entangling']:.4f}"
