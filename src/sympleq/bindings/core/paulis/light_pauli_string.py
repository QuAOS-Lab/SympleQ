from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Complex as ComplexNumber
import re
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from sympleq_light.light._light import PauliString as _NativePauliString

from sympleq.core.paulis.protocols.p_pauli_string import PPauliString
from sympleq_light.light.protocols import SymplecticAlgebraProtocol


class PauliString(PPauliString, SymplecticAlgebraProtocol):
    __slots__ = ("_native", "_phases", "_weights")

    def __new__(cls) -> Self:
        return object.__new__(cls)

    @classmethod
    def _from_native(
        cls,
        native: _NativePauliString,
        *,
        phases: Any = None,
        weights: Any = None,
    ) -> Self:
        instance = object.__new__(cls)
        instance._native = native
        phase_values = np.zeros(1, dtype=np.int64) if phases is None else np.asarray(
            phases, dtype=np.int64
        ).reshape(-1)
        weight_values = np.ones(1, dtype=np.complex128) if weights is None else np.asarray(
            weights, dtype=np.complex128
        ).reshape(-1)
        if phase_values.size != 1 or weight_values.size != 1:
            raise ValueError("PauliString phases and weights must contain one value")
        instance._phases = phase_values % native.phase_modulus
        instance._weights = weight_values
        return instance

    @classmethod
    def from_string(
        cls, string: str, dimensions: int | Sequence[int] | None = None
    ) -> Self:
        tokens = string.split()
        if dimensions is not None:
            if isinstance(dimensions, Iterable) and not isinstance(
                dimensions, (str, bytes)
            ):
                dimension_values = list(dimensions)
            else:
                dimension_values = [dimensions] * len(tokens)
            if len(dimension_values) != len(tokens):
                raise ValueError(
                    "dimensions must contain one value per qudit "
                    f"({len(tokens)} expected, got {len(dimension_values)})"
                )
        else:
            dimension_values = [None] * len(tokens)
        normalized_tokens = []
        for token, dimension in zip(tokens, dimension_values):
            match = re.fullmatch(r"x(-?\d+)z(-?\d+)(?:/(\d+))?", token)
            if match is None:
                normalized_tokens.append(token)
                continue
            x_exp, z_exp = int(match.group(1)), int(match.group(2))
            token_dimension = (
                dimension
                if dimension is not None
                else int(match.group(3)) if match.group(3) else None
            )
            if token_dimension is not None:
                x_exp %= int(token_dimension)
                z_exp %= int(token_dimension)
                normalized_tokens.append(f"x{x_exp}z{z_exp}/{token_dimension}")
            else:
                normalized_tokens.append(f"x{x_exp}z{z_exp}")
        string = " ".join(normalized_tokens)
        return cls._from_native(_NativePauliString.from_string(string))

    @classmethod
    def from_exponents(
        cls,
        x_exp: int | Sequence[int],
        z_exp: int | Sequence[int],
        dimensions: int | Sequence[int] | None = None,
    ) -> Self:
        x_values = [x_exp] if np.isscalar(x_exp) else list(x_exp)
        z_values = [z_exp] if np.isscalar(z_exp) else list(z_exp)
        dimension_values = (
            [dimensions]
            if dimensions is not None and np.isscalar(dimensions)
            else list(dimensions)
            if dimensions is not None
            else None
        )
        if dimension_values is not None:
            if len(dimension_values) == 1 and len(x_values) != 1:
                dimension_values *= len(x_values)
            if len(x_values) != len(z_values) or len(x_values) != len(dimension_values):
                raise ValueError("x_exp, z_exp, and dimensions must have equal lengths")
            x_values = [int(x) % int(d) for x, d in zip(x_values, dimension_values)]
            z_values = [int(z) % int(d) for z, d in zip(z_values, dimension_values)]
        elif len(x_values) == len(z_values):
            x_values = [int(x) % 2 for x in x_values]
            z_values = [int(z) % 2 for z in z_values]
        native = _NativePauliString.from_exponents(
            x_values,
            z_values,
            dimension_values,
        )
        return cls._from_native(native)

    @property
    def tableau(self) -> npt.NDArray[np.int64]:
        return np.asarray(self._native.tableau, dtype=np.int64).reshape(
            self.n_pauli_strings(), 2 * self.n_qudits()
        )

    @property
    def dimensions(self) -> npt.NDArray[np.uint64]:
        dimensions = np.asarray(self._native.dimensions, dtype=np.uint64)
        dimensions.setflags(write=False)
        return dimensions

    @dimensions.setter
    def dimensions(self, value: Any) -> None:
        raise ValueError("dimensions are immutable; construct a new PauliString instead")

    @property
    def lcm(self) -> int:
        return self._native.lcm

    @property
    def phase_modulus(self) -> int:
        return self._native.phase_modulus

    @lcm.setter
    def lcm(self, value: Any) -> None:
        raise ValueError("lcm is determined by the qudit dimensions and cannot be set")

    @property
    def x_exp(self) -> npt.NDArray[np.int64]:
        return np.asarray(self._native.x_exponents, dtype=np.int64)

    @property
    def z_exp(self) -> npt.NDArray[np.int64]:
        return np.asarray(self._native.z_exponents, dtype=np.int64)

    @property
    def phases(self) -> npt.NDArray[np.int64]:
        return self._phases

    @phases.setter
    def phases(self, value: Any) -> None:
        phases = np.asarray(value, dtype=np.int64).reshape(-1)
        if phases.size != 1:
            raise ValueError("PauliString phases must contain exactly one value")
        self._phases[:] = phases % self.phase_modulus

    @property
    def weights(self) -> npt.NDArray[np.complex128]:
        return self._weights

    @weights.setter
    def weights(self, value: Any) -> None:
        weights = np.asarray(value, dtype=np.complex128).reshape(-1)
        if weights.size != 1:
            raise ValueError("PauliString weights must contain exactly one value")
        self._weights[:] = weights

    def __init__(
        self, tableau: Any = None, dimensions: Any = None,
        weights: Any = None, phases: Any = None,
    ) -> None:
        if tableau is None:
            raise TypeError(
                "use PauliString.from_string() or PauliString.from_exponents()"
            )
        native = self.from_tableau(tableau, dimensions)
        self._native = native._native
        self._phases = np.zeros(1, dtype=np.int64)
        self._weights = np.ones(1, dtype=np.complex128)
        if phases is not None:
            self.phases = phases
        if weights is not None:
            self.weights = weights

    @classmethod
    def from_tableau(cls, tableau: Any, dimensions: Any | None = None) -> Self:
        values = np.asarray(tableau, dtype=np.int64)
        if values.ndim == 2:
            if values.shape[0] != 1:
                raise ValueError("PauliString tableau must contain exactly one row")
            values = values[0]
        if values.ndim != 1 or values.size % 2:
            raise ValueError("PauliString tableau must have an even number of columns")
        n_qudits = values.size // 2
        dimension_values = (
            np.full(n_qudits, 2, dtype=np.int64)
            if dimensions is None
            else np.asarray(dimensions, dtype=np.int64).reshape(-1)
        )
        if dimension_values.size == 1 and n_qudits != 1:
            dimension_values = np.repeat(dimension_values, n_qudits)
        if dimension_values.size != n_qudits:
            raise ValueError("dimensions must contain one value per qudit")
        return cls.from_exponents(
            values[:n_qudits].tolist(),
            values[n_qudits:].tolist(),
            dimension_values.tolist(),
        )

    @classmethod
    def from_random(cls, dimensions: Any, seed: int | None = None) -> Self:
        dimension_values = np.asarray(dimensions, dtype=np.int64).reshape(-1)
        if dimension_values.size == 0:
            raise ValueError("dimensions must contain at least one qudit")
        rng = np.random.default_rng(seed)
        return cls.from_exponents(
            [int(rng.integers(dimension)) for dimension in dimension_values],
            [int(rng.integers(dimension)) for dimension in dimension_values],
            dimension_values.tolist(),
        )

    def n_qudits(self) -> int:
        return self._native.n_qudits()

    def n_paulis(self) -> int:
        return 1

    def n_pauli_strings(self) -> int:
        return self._native.n_pauli_strings()

    def first_anticommuting_pair(self) -> tuple[int, int] | None:
        return self._native.first_anticommuting_pair()

    def is_mutually_commuting(self) -> bool:
        return self._native.is_mutually_commuting()

    def assert_same_dimensions(self, dimensions: Sequence[int]) -> None:
        self._native.assert_same_dimensions(list(dimensions))

    def shape(self) -> tuple[int, int]:
        return self.n_pauli_strings(), self.n_qudits()

    def set_phases(self, value: Any) -> None:
        self.phases = value

    def reset_phases(self) -> None:
        self._phases[:] = 0

    def set_weights(self, value: Any) -> None:
        self.weights = value

    def reset_weights(self) -> None:
        self._weights[:] = 1

    def is_close(
        self,
        other_pauli: Any,
        threshold: int | None = None,
        literal: bool | None = None,
    ) -> bool:
        if not isinstance(other_pauli, (PauliString, _NativePauliString)):
            return False
        other = (
            other_pauli
            if isinstance(other_pauli, PauliString)
            else self._from_native(other_pauli)
        )
        if list(self.dimensions) != list(other.dimensions):
            return False
        if not np.allclose(self.tableau, other.tableau):
            return False
        if literal is not False and not np.array_equal(self.phases, other.phases):
            return False
        return bool(np.allclose(self.weights, other.weights, atol=threshold or 1e-8))

    def H(self) -> Self:
        dimensions = self.dimensions.astype(np.int64)
        x = (-self.x_exp) % dimensions
        z = (-self.z_exp) % dimensions
        result = self.from_exponents(x.tolist(), z.tolist(), dimensions.tolist())
        acquired = int(
            2 * np.sum(self.x_exp * self.z_exp * (self.lcm // dimensions))
        )
        result._phases[:] = (-self._phases + acquired) % self.phase_modulus
        result._weights[:] = np.conjugate(self._weights)
        return result

    def is_hermitian(self) -> bool:
        matrix = self.to_hilbert_space().toarray()
        return bool(np.allclose(matrix, matrix.conj().T))

    def copy(self) -> Self:
        return self._from_native(
            self._native.copy(), phases=self._phases, weights=self._weights
        )

    def phase_to_weight(self) -> None:
        self._weights[:] *= np.exp(
            1j * np.pi * self._phases.astype(np.float64) / self.lcm
        )
        self.reset_phases()

    def to_standard_form(self) -> Self:
        return self.copy()

    def has_equal_tableau(self, other_pauli: Any, literal: bool = True) -> bool:
        if not isinstance(other_pauli, (PauliString, _NativePauliString)):
            return False
        other = (
            other_pauli
            if isinstance(other_pauli, PauliString)
            else self._from_native(other_pauli)
        )
        return (
            list(self.dimensions) == list(other.dimensions)
            and np.array_equal(self.tableau, other.tableau)
        )

    def get_subspace(self, qudit_indexes: Any) -> Self:
        indexes = np.asarray(qudit_indexes, dtype=np.int64).reshape(-1)
        if np.any(indexes < 0) or np.any(indexes >= self.n_qudits()):
            raise IndexError("qudit index out of range")
        result = self.from_exponents(
            self.x_exp[indexes].tolist(),
            self.z_exp[indexes].tolist(),
            self.dimensions[indexes].astype(np.int64).tolist(),
        )
        result._weights[:] = self._weights
        result._phases[:] = self._phases
        return result

    def commute(self, other_pauli: PPauliString) -> bool:
        return self._native.commutes_with(self._other_native(other_pauli))

    def standardize(self) -> None:
        return None

    def n_identities(self) -> int:
        return int(np.count_nonzero((self.x_exp == 0) & (self.z_exp == 0)))

    def symplectic_residues(
        self, other: PPauliString
    ) -> npt.NDArray[np.int64]:
        other_native = self._other_native(other)
        self._native.assert_same_dimensions(other_native.dimensions)
        dimensions = np.asarray(self.dimensions, dtype=np.int64)
        return (
            (self.x_exp * np.asarray(other_native.z_exponents, dtype=np.int64))
            - (self.z_exp * np.asarray(other_native.x_exponents, dtype=np.int64))
        ) % dimensions

    def symplectic_product(
        self,
        other: PPauliString,
        *,
        as_scalar: bool = True,
    ) -> npt.NDArray[np.int64] | int:
        residues = self.symplectic_residues(other)
        lifted = residues * (self.lcm // self.dimensions.astype(np.int64))
        product = int(np.sum(lifted) % self.lcm)
        return product if as_scalar else np.asarray([product], dtype=np.int64)

    def quditwise_product(self, other: PPauliString) -> int:
        return int(np.count_nonzero(self.symplectic_residues(other)))

    def delete_qudits(self, qudit_indexes: Any) -> None:
        indexes = np.asarray(qudit_indexes, dtype=np.int64).reshape(-1)
        if np.any(indexes < 0) or np.any(indexes >= self.n_qudits()):
            raise IndexError("qudit index out of range")
        keep = np.ones(self.n_qudits(), dtype=bool)
        keep[indexes] = False
        self._native = self.from_exponents(
            self.x_exp[keep].tolist(),
            self.z_exp[keep].tolist(),
            self.dimensions[keep].astype(np.int64).tolist(),
        )._native
        self._phases %= self.phase_modulus

    def amend(self, qudit_index: int, new_x: int, new_z: int) -> Self:
        if not 0 <= qudit_index < self.n_qudits():
            raise IndexError("qudit index out of range")
        x = self.x_exp.copy()
        z = self.z_exp.copy()
        dimensions = self.dimensions.astype(np.int64)
        x[qudit_index] = new_x % dimensions[qudit_index]
        z[qudit_index] = new_z % dimensions[qudit_index]
        return self.from_exponents(x.tolist(), z.tolist(), dimensions.tolist())

    def is_x(self) -> bool:
        return bool(np.all(self.z_exp == 0) and np.any(self.x_exp != 0))

    def is_z(self) -> bool:
        return bool(np.all(self.x_exp == 0) and np.any(self.z_exp != 0))

    def to_hilbert_space(self) -> Any:
        matrix = np.array([[1 + 0j]])
        for dimension, x, z in zip(
            self.dimensions, self.x_exp, self.z_exp
        ):
            d = int(dimension)
            shift = np.roll(np.eye(d, dtype=np.complex128), int(x), axis=0)
            phase = np.diag(np.exp(2j * np.pi * int(z) * np.arange(d) / d))
            matrix = np.kron(matrix, shift @ phase)
        scalar = self._weights[0] * np.exp(
            1j * np.pi * int(self._phases[0]) / self.lcm
        )
        return sp.csr_matrix(scalar * matrix)

    @property
    def x_exponents(self) -> npt.NDArray[np.int64]:
        return self.x_exp

    @property
    def z_exponents(self) -> npt.NDArray[np.int64]:
        return self.z_exp

    def is_identity(self) -> bool:
        return self._native.is_identity()

    def commutes_with(self, other: PauliString) -> bool:
        if not isinstance(other, PauliString):
            raise TypeError("other must be a PauliString")
        return self._native.commutes_with(other._native)

    def inverse(self) -> Self:
        native_inverse = self._native.inverse()
        inverse_phase = (
            -int(self._phases[0])
            - self._acquired_phase_native(self._native, native_inverse)
        )
        return self._from_native(
            native_inverse,
            phases=[inverse_phase],
            weights=[1 / self._weights[0]],
        )

    @staticmethod
    def _acquired_phase_native(
        left: _NativePauliString, right: _NativePauliString
    ) -> int:
        dimensions = list(left.dimensions)
        if dimensions != list(right.dimensions):
            raise ValueError("Pauli strings must have matching qudit dimensions")
        lcm = int(left.lcm)
        phase_modulus = 2 * lcm
        phase = sum(
            int(z_exp) * int(x_other) * (lcm // dimension)
            for z_exp, x_other, dimension in zip(
                left.z_exponents, right.x_exponents, dimensions
            )
        )
        return (2 * phase) % phase_modulus

    def acquired_phase(self, other: PPauliString) -> int:
        if isinstance(other, PauliString):
            native_other = other._native
        elif isinstance(other, _NativePauliString):
            native_other = other
        else:
            raise TypeError("other must be a PauliString")
        return self._acquired_phase_native(self._native, native_other)

    def __eq__(self, other: object) -> Any:
        if isinstance(other, PauliString):
            return (
                self._native.equals(other._native)
                and np.array_equal(self.phases, other.phases)
                and np.array_equal(self.weights, other.weights)
            )
        if isinstance(other, _NativePauliString):
            return self._native.equals(other) and bool(
                np.array_equal(self.phases, np.zeros(1, dtype=np.int64))
                and np.array_equal(self.weights, np.ones(1, dtype=np.complex128))
            )
        return NotImplemented

    def __ne__(self, other: object) -> Any:
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __gt__(self, other: PPauliString) -> Any:
        other_native = self._other_native(other)
        self._assert_orderable(other_native)
        return tuple(self.tableau.ravel()) < tuple(
            np.asarray(other_native.tableau, dtype=np.int64).ravel()
        )

    def __lt__(self, other: PPauliString) -> Any:
        other_native = self._other_native(other)
        self._assert_orderable(other_native)
        return tuple(self.tableau.ravel()) > tuple(
            np.asarray(other_native.tableau, dtype=np.int64).ravel()
        )

    def __ge__(self, other: PPauliString) -> Any:
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else bool(equal or self > other)

    def __le__(self, other: PPauliString) -> Any:
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else bool(equal or self < other)

    def __getitem__(self, key: Any) -> Self:
        if not isinstance(key, (int, np.integer, slice, list, np.ndarray)):
            raise ValueError("key must be an integer, slice, list, or integer array")
        indexes = np.arange(self.n_qudits())[key]
        return self.get_subspace(np.asarray(indexes).reshape(-1))

    def __setitem__(self, key: Any, value: PPauliString) -> None:
        if not isinstance(value, (PauliString, _NativePauliString)):
            raise TypeError("assigned value must be a PauliString")
        if not isinstance(key, (int, np.integer, slice, list, np.ndarray)):
            raise ValueError("key must be an integer, slice, list, or integer array")
        native = value._native if isinstance(value, PauliString) else value
        indexes = np.asarray(np.arange(self.n_qudits())[key]).reshape(-1)
        if len(indexes) != len(native.dimensions):
            raise ValueError("assigned PauliString must match the selected qudits")
        if not np.array_equal(self.dimensions[indexes], native.dimensions):
            raise ValueError("assigned PauliString must have matching qudit dimensions")
        x = self.x_exp.copy()
        z = self.z_exp.copy()
        x[indexes] = native.x_exponents
        z[indexes] = native.z_exponents
        self._native = self.from_exponents(
            x.tolist(), z.tolist(), self.dimensions.astype(np.int64).tolist()
        )._native

    def __add__(self, other: Any) -> Any:
        if isinstance(other, (PauliString, _NativePauliString)):
            from .light_pauli_sum import PauliSum

            other_native = other._native if isinstance(other, PauliString) else other
            if list(self._native.dimensions) != list(other_native.dimensions):
                raise ValueError(
                    "Pauli strings must have matching qudit dimensions"
                )
            return PauliSum.from_pauli_strings(
                [self, other], inherit_weights=True, inherit_phases=True
            )
        if other == 0:
            return self.copy()
        return NotImplemented

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, (PauliString, _NativePauliString)):
            from .light_pauli_sum import PauliSum

            other_native = other._native if isinstance(other, PauliString) else other
            if list(self.dimensions) != list(other_native.dimensions):
                raise ValueError(
                    "Pauli strings must have matching qudit dimensions"
                )
            left = self.copy()
            right = other.copy() if isinstance(other, PauliString) else self._from_native(other)
            right._weights *= -1
            return PauliSum.from_pauli_strings(
                [left, right], inherit_weights=True, inherit_phases=True
            )
        return NotImplemented

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, (PauliString, _NativePauliString)):
            return self._from_native(other) - self
        return NotImplemented

    def __mul__(self, other: Any) -> Self:
        if isinstance(other, ComplexNumber):
            result = self.copy()
            result._weights[0] *= complex(other)
            return result
        if isinstance(other, PauliString):
            native = self._native.multiply(other._native)
            phases = (
                int(self._phases[0])
                + int(other._phases[0])
                + self.acquired_phase(other)
            )
            return self._from_native(
                native,
                phases=[phases],
                weights=[self._weights[0] * other._weights[0]],
            )
        if isinstance(other, _NativePauliString):
            native = self._native.multiply(other)
            return self._from_native(
                native,
                phases=[
                    int(self._phases[0])
                    + self.acquired_phase(self._from_native(other))
                ],
                weights=self._weights,
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    def __matmul__(self, other: Any) -> Self:
        if not isinstance(other, (PauliString, _NativePauliString)):
            return NotImplemented
        from .light_pauli_sum import PauliSum

        native = other._native if isinstance(other, PauliString) else other
        if isinstance(other, PauliString):
            other_phase = other._phases
            other_weight = other._weights
        else:
            other_phase = np.zeros(1, dtype=np.int64)
            other_weight = np.ones(1, dtype=np.complex128)
        dimensions = np.concatenate((self.dimensions, native.dimensions))
        x = np.concatenate((self.x_exp, native.x_exponents))
        z = np.concatenate((self.z_exp, native.z_exponents))
        combined = self.from_exponents(
            x.tolist(), z.tolist(), dimensions.astype(np.int64).tolist()
        )
        combined_lcm = combined.lcm
        combined._phases[:] = (
            self._phases * (combined_lcm // self.lcm)
            + other_phase * (combined_lcm // int(native.lcm))
        )
        combined._weights[:] = self._weights * other_weight
        return combined

    def __rmatmul__(self, other: Any) -> Self:
        if isinstance(other, (PauliString, _NativePauliString)):
            return self._from_native(other) @ self
        return NotImplemented

    def __pow__(self, exponent: int, modulo: int | None = None) -> Self:
        if modulo is not None:
            raise TypeError("modular exponentiation is not defined for PauliString")
        if not isinstance(exponent, int):
            return NotImplemented

        dimensions = list(self._native.dimensions)
        n_qudits = self.n_qudits()
        result = self._from_native(
            _NativePauliString.from_exponents(
                [0] * n_qudits,
                [0] * n_qudits,
                dimensions,
            )
        )
        base = self if exponent >= 0 else self.inverse()
        power = abs(exponent)
        while power:
            if power & 1:
                result = result * base
            power >>= 1
            if power:
                base = base * base
        return result

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.dimensions),
                tuple(self.x_exp),
                tuple(self.z_exp),
                tuple(self.phases),
                tuple(self.weights),
            )
        )

    def __str__(self) -> str:
        return str(self._native)

    def __repr__(self) -> str:
        return f"PauliString.from_string({str(self)!r})"

    @staticmethod
    def _other_native(other: Any) -> _NativePauliString:
        if isinstance(other, PauliString):
            return other._native
        if isinstance(other, _NativePauliString):
            return other
        raise TypeError("other must be a PauliString")

    def _assert_orderable(self, other: _NativePauliString) -> None:
        if self.n_paulis() != 1 or not np.array_equal(self.dimensions, other.dimensions):
            raise ValueError("Pauli strings must have one row and matching dimensions")
