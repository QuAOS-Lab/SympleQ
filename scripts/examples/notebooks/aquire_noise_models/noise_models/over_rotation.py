"""Coherent over-rotation: after every gate, each of the gate's qudits is rotated by
exp(-i theta (X + X^dagger)/2), with theta giving the matched process infidelity. theta is
the same at every gate and every shot, one value for local gates and one for entangling
gates. The rotation moves population between computational basis states.
"""
from .simulator import NoiseModel, coherent_angle, coherent_unitary, matched_infidelity


class OverRotation(NoiseModel):
    name = "over_rotation"
    axis = "x"

    def __init__(self, strength=1.0, d=3):
        self.theta = {kind: coherent_angle(matched_infidelity(strength, d, kind), d, n, self.axis)
                      for kind, n in (("local", 1), ("entangling", 2))}
        self._unitaries = {}

    def after_gate(self, sim, rho, qudits, kind):
        theta = self.theta[kind]
        if not theta:
            return rho
        key = (tuple(qudits), kind, id(sim))
        if key not in self._unitaries:
            self._unitaries[key] = sim.local_unitary(qudits, lambda d: coherent_unitary(d, self.axis, theta))
        U = self._unitaries[key]
        return U @ rho @ U.conj().T

    def describe(self):
        return f"{self.name}: theta_local={self.theta['local']:.4f}, theta_entangling={self.theta['entangling']:.3f}"
