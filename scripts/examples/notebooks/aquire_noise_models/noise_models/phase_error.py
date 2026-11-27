"""Coherent phase error: as the over-rotation, with exp(-i theta diag(k - (d-1)/2)), a qudit Z
rotation (detuning, AC Stark shift). It only changes phases, which do nothing to a computational
basis state, so AQUIRE's diagnostics (the measurement circuit run on a computational basis state,
with every H replaced by H^4) only see it inside their H^4 groups, not after the circuit's S and CX gates.
"""
from .over_rotation import OverRotation


class PhaseError(OverRotation):
    name = "phase_error"
    axis = "z"
