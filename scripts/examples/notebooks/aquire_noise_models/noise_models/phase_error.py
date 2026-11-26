"""Coherent phase error: as the over-rotation, with exp(-i theta diag(k - (d-1)/2)), a qudit Z
rotation (detuning, AC Stark shift). It only changes phases, so AQUIRE's diagnostics, which
start from |0...0>, cannot see it.
"""
from .over_rotation import OverRotation


class PhaseError(OverRotation):
    name = "phase_error"
    axis = "z"
