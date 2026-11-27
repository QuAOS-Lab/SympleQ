"""Gate-level noise models for testing AQUIRE, one per file, and the simulator they plug into."""
from .dephasing import Dephasing
from .depolarising import Depolarising
from .over_rotation import OverRotation
from .phase_error import PhaseError
from .simulator import GateNoiseSimulator, NoiseModel, matched_infidelity, process_infidelity
from .uniform import Uniform

# the models of the figure, in its column order
MODELS = {"uniform": Uniform, "depolarising": Depolarising, "over_rotation": OverRotation,
          "dephasing": Dephasing, "phase_error": PhaseError}

__all__ = ["GateNoiseSimulator", "NoiseModel", "MODELS", "Uniform", "Depolarising", "Dephasing", "OverRotation",
           "PhaseError", "matched_infidelity", "process_infidelity"]
