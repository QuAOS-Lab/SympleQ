from __future__ import annotations

from os import PathLike
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt

from .p_pauli_string import PPauliString


class PPauliSum(Protocol):

    @property
    def tableau(self) -> npt.NDArray[np.int64]:
        ...

    @property
    def dimensions(self) -> npt.NDArray[np.uint64]:
        ...

    @dimensions.setter
    def dimensions(self, value: Any) -> None:
        ...

    @property
    def lcm(self) -> int:
        ...

    @lcm.setter
    def lcm(self, value: Any) -> None:
        ...

    @property
    def x_exp(self) -> npt.NDArray[np.int64]:
        ...

    @property
    def z_exp(self) -> npt.NDArray[np.int64]:
        ...

    @property
    def phases(self) -> npt.NDArray[np.int64]:
        ...

    @phases.setter
    def phases(self, value: Any) -> None:
        ...

    @property
    def weights(self) -> npt.NDArray[np.complex128]:
        ...

    @weights.setter
    def weights(self, value: Any) -> None:
        ...

    def __init__(
        self,
        tableau: Any,
        dimensions: Any,
        weights: Any,
        phases: Any,
    ) -> None:
        ...

    @classmethod
    def from_tableau(
        cls,
        tableau: Any,
        dimensions: Any | None = None,
        weights: Any | None = None,
        phases: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def from_hilbert_space(cls, matrix: Any, dimensions: Any) -> Self:
        ...

    @classmethod
    def from_pauli_strings(
        cls,
        pauli_strings: Any,
        weights: Any | None = None,
        phases: Any | None = None,
        inherit_phases: bool = False,
    ) -> Self:
        ...

    @classmethod
    def from_pauli_objects(
        cls,
        pauli_objects: Any,
        weights: Any | None = None,
        phases: Any | None = None,
        inherit_weights: bool = False,
        inherit_phases: bool = False,
    ) -> Self:
        ...

    @classmethod
    def from_string(
        cls,
        string: Any,
        dimensions: Any | None = None,
        weights: Any | None = None,
        phases: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def from_random(
        cls,
        n_paulis: int,
        dimensions: Any,
        rand_weights: bool = True,
        rand_phases: bool = False,
        seed: int | None = None,
    ) -> Self:
        ...

    @classmethod
    def from_file(
        cls,
        path: str | PathLike[str],
        dimensions: Any | None = None,
    ) -> Self:
        ...

    def n_qudits(self) -> int:
        ...

    def n_paulis(self) -> int:
        ...

    def shape(self) -> tuple[int, int]:
        ...

    def set_phases(self, value: Any) -> None:
        ...

    def reset_phases(self) -> None:
        ...

    def set_weights(self, value: Any) -> None:
        ...

    def reset_weights(self) -> None:
        ...

    def to_file(self, path: str) -> None:
        ...

    def has_equal_tableau(self, other_pauli: Any, literal: bool = True) -> bool:
        ...

    def is_close(
        self,
        other_pauli: Any,
        threshold: int | None = None,
        literal: bool | None = None,
    ) -> bool:
        ...

    def H(self) -> Self:
        ...

    def is_hermitian(self) -> bool:
        ...

    def copy(self) -> Self:
        ...

    def phase_to_weight(self) -> None:
        ...

    def weight_to_phase(self) -> None:
        ...

    def standardize(self) -> None:
        ...

    def to_standard_form(self) -> Self:
        ...

    def to_hilbert_space(self, pauli_string_index: int | None = None) -> Any:
        ...

    def combine_equivalent_paulis(self) -> None:
        ...

    def remove_trivial_qudits(self) -> None:
        ...

    def remove_trivial_paulis(self) -> None:
        ...

    def remove_zero_weight_paulis(self, threshold: float | None = None) -> None:
        ...

    def is_identity(self) -> bool:
        ...

    def is_x(self) -> bool:
        ...

    def is_z(self) -> bool:
        ...

    def is_commuting(self, pauli_string_indexes: Any | None = None) -> bool:
        ...

    def is_quditwise_commuting(
        self, pauli_string_indexes: Any | None = None
    ) -> bool:
        ...

    def select_pauli_string(self, index: int) -> PPauliString:
        ...

    def delete_paulis(self, pauli_indices: Any) -> None:
        ...

    def delete_qudits(self, qudit_indexes: Any) -> None:
        ...

    def symplectic_product_matrix(self) -> npt.NDArray[np.int64]:
        ...

    def symplectic_residue_tensor(self) -> npt.NDArray[np.int64]:
        ...

    def quditwise_symplectic_product_matrix(self) -> npt.NDArray[np.int64]:
        ...

    def get_subspace(
        self,
        qudit_indexes: Any,
        pauli_indexes: Any | None = None,
    ) -> Self:
        ...

    def acquire_phase(
        self,
        phases: Any,
        pauli_indexes: Any | None = None,
    ) -> None:
        ...

    def ordered_eigenspectrum(
        self, num_eigens: int | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        ...

    def stabilizer_to_hilbert_space(self) -> Any:
        ...

    def make_hermitian(self, in_place: bool = False) -> Self:
        ...

    def XZ_to_Y(self, in_place: bool = False) -> Self:
        ...

    def reorder(self, order: Any) -> None:
        ...

    def swap_paulis(self, index_1: int, index_2: int) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __gt__(self, other: PPauliSum) -> bool:
        ...

    def __lt__(self, other: PPauliSum) -> bool:
        ...

    def __getitem__(self, key: Any) -> Self:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def __add__(self, other: Any) -> Self:
        ...

    def __radd__(self, other: Any) -> Self:
        ...

    def __sub__(self, other: Any) -> Self:
        ...

    def __rsub__(self, other: Any) -> Self:
        ...

    def __mul__(self, other: Any) -> Self:
        ...

    def __rmul__(self, other: Any) -> Self:
        ...

    def __matmul__(self, other: Any) -> Self:
        ...

    def __rmatmul__(self, other: Any) -> Self:
        ...

    def __truediv__(self, other: Any) -> Self:
        ...

    def __pow__(self, exponent: int, modulo: int | None = None) -> Self:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...
