"""Dephasing: after every gate, with probability p the coherences of the gate's qudits are
destroyed, rho -> (1 - p) rho + p sum_k |k><k| rho |k><k|. Computational-basis populations
are unchanged, so circuits acting on |0...0> (AQUIRE's diagnostics) never show it.
p gives the matched process infidelity r: p = r / (1 - 1/d_S).
"""
from .simulator import NoiseModel, dephasing_probability, matched_infidelity


class Dephasing(NoiseModel):
    name = "dephasing"

    def __init__(self, strength=1.0, d=3, fraction=1.0):
        self.p = {"local": dephasing_probability(fraction * matched_infidelity(strength, d, "local"), d, 1),
                  "entangling": dephasing_probability(fraction * matched_infidelity(strength, d, "entangling"), d, 2)}

    def after_gate(self, sim, rho, qudits, kind):
        p = self.p[kind]
        return (1 - p) * rho + p * sim.dephase(rho, qudits) if p else rho

    def describe(self):
        return f"dephasing: p_local={self.p['local']:.5f}, p_entangling={self.p['entangling']:.4f}"
