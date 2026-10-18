from __future__ import annotations

from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt


class PPauliString(Protocol):

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
    def from_exponents(
        cls,
        x_exp: Any,
        z_exp: Any,
        dimensions: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def from_tableau(
        cls,
        tableau: Any,
        dimensions: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def from_string(
        cls,
        string: str,
        dimensions: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def from_random(
        cls,
        dimensions: Any,
        seed: int | None = None,
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

    def to_standard_form(self) -> Self:
        ...

    def has_equal_tableau(self, other_pauli: Any, literal: bool = True) -> bool:
        ...

    def get_subspace(self, qudit_indexes: Any) -> Self:
        ...

    def commute(self, other_pauli: PPauliString) -> bool:
        ...

    def standardize(self) -> None:
        ...

    def n_identities(self) -> int:
        ...

    def symplectic_residues(
        self, other: PPauliString
    ) -> npt.NDArray[np.int64]:
        ...

    def symplectic_product(
        self,
        other: PPauliString,
        *,
        as_scalar: bool = True,
    ) -> npt.NDArray[np.int64] | int:
        ...

    def quditwise_product(self, other: PPauliString) -> int:
        ...

    def delete_qudits(self, qudit_indexes: Any) -> None:
        ...

    def amend(self, qudit_index: int, new_x: int, new_z: int) -> Self:
        ...

    def acquired_phase(self, other: PPauliString) -> int:
        ...

    def is_identity(self) -> bool:
        ...

    def is_x(self) -> bool:
        ...

    def is_z(self) -> bool:
        ...

    def to_hilbert_space(self) -> Any:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __gt__(self, other: PPauliString) -> bool:
        ...

    def __lt__(self, other: PPauliString) -> bool:
        ...

    def __getitem__(self, key: Any) -> Self:
        ...

    def __setitem__(self, key: Any, value: PPauliString) -> None:
        ...

    def __add__(self, other: PPauliString) -> Self:
        ...

    def __radd__(self, other: PPauliString) -> Self:
        ...

    def __sub__(self, other: PPauliString) -> Self:
        ...

    def __rsub__(self, other: PPauliString) -> Self:
        ...

    def __mul__(self, other: Any) -> Self:
        ...

    def __rmul__(self, other: Any) -> Self:
        ...

    def __matmul__(self, other: Any) -> Self:
        ...

    def __rmatmul__(self, other: Any) -> Self:
        ...

    def __pow__(self, exponent: int, modulo: int | None = None) -> Self:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...
