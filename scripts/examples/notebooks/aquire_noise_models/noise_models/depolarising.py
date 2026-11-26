"""Local depolarising noise: after every gate, with probability p the gate's qudits S are
replaced by the maximally mixed state, rho -> (1 - p) rho + p Tr_S(rho) (x) 1_S / d_S.
The reference model: p = strength x the paper's fitted rates.
"""
from .simulator import P_ENTANGLING, P_LOCAL, NoiseModel


class Depolarising(NoiseModel):
    name = "depolarising"

    def __init__(self, strength=1.0, d=3, fraction=1.0):
        self.p = {"local": fraction * strength * P_LOCAL, "entangling": fraction * strength * P_ENTANGLING}

    def after_gate(self, sim, rho, qudits, kind):
        p = self.p[kind]
        return (1 - p) * rho + p * sim.replace(rho, qudits) if p else rho

    def describe(self):
        return f"depolarising: p_local={self.p['local']:.5f}, p_entangling={self.p['entangling']:.4f}"
