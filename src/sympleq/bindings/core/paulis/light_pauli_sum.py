from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Never, Self
import math

import numpy as np
import numpy.typing as npt

from sympleq_light import PauliString as _NativePauliString
from sympleq_light import PauliSum as _NativePauliSum

from sympleq.core.paulis.protocols.p_pauli_string import PPauliString
from sympleq.core.paulis.protocols.p_pauli_sum import PPauliSum
from sympleq_light.light.protocols import SymplecticPhaseProtocol

from .light_pauli_string import PauliString


class PauliSum(PPauliSum, SymplecticPhaseProtocol):
    __slots__ = ("_native", "_tableau", "_dimensions", "_weights", "_phases")

    def __new__(cls) -> Never:
        raise TypeError("use PauliSum.from_string() or PauliSum.zero()")

    @classmethod
    def _from_native(cls, native: _NativePauliSum) -> Self:
        instance = object.__new__(cls)
        instance._native = native
        n_qudits = native.n_qudits()
        instance._tableau = np.asarray(native.tableau, dtype=np.int64).reshape(
            native.n_pauli_strings(), 2 * n_qudits
        )
        instance._dimensions = np.asarray(native.dimensions, dtype=np.int64)
        instance._weights = np.asarray(
            [complex(real, imaginary) for real, imaginary in native.weights],
            dtype=np.complex128,
        )
        instance._phases = np.asarray(native.phases, dtype=np.int64)
        return instance

    @classmethod
    def _from_components(
        cls, tableau: Any, dimensions: Any, weights: Any = None, phases: Any = None
    ) -> Self:
        dimensions = np.asarray(dimensions, dtype=np.int64).reshape(-1)
        tableau = np.asarray(tableau, dtype=np.int64)
        if tableau.ndim == 1:
            tableau = tableau.reshape(1, -1)
        if tableau.ndim != 2 or tableau.shape[1] != 2 * len(dimensions):
            raise ValueError("tableau must have 2 * n_qudits columns")
        n_rows = tableau.shape[0]
        weight_values = (
            np.ones(n_rows, dtype=np.complex128)
            if weights is None
            else np.asarray(weights, dtype=np.complex128).reshape(-1)
        )
        phase_values = (
            np.zeros(n_rows, dtype=np.int64)
            if phases is None
            else np.asarray(phases, dtype=np.int64).reshape(-1)
        )
        if len(weight_values) != n_rows or len(phase_values) != n_rows:
            raise ValueError("weights and phases must contain one value per Pauli string")
        if np.any(dimensions < 2):
            raise ValueError("qudit dimensions must be at least 2")
        tableau = tableau % np.tile(dimensions, 2)
        lcm = math.lcm(*(int(d) for d in dimensions)) if len(dimensions) else 1
        phase_values %= 2 * lcm
        if n_rows:
            rows = []
            n_qudits = len(dimensions)
            for row, weight, phase in zip(tableau, weight_values, phase_values):
                pauli = " ".join(
                    f"x{int(row[i])}z{int(row[n_qudits + i])}/{int(d)}"
                    for i, d in enumerate(dimensions)
                )
                rows.append(
                    f"{float(weight.real)!r} {float(weight.imag)!r} | {pauli} | {int(phase)}"
                )
            native = _NativePauliSum.from_string("\n".join(rows))
        else:
            native = _NativePauliSum.zero(dimensions.tolist())
        instance = object.__new__(cls)
        instance._native = native
        instance._tableau = tableau.copy()
        instance._dimensions = dimensions.copy()
        instance._weights = weight_values.copy()
        instance._phases = phase_values.copy()
        return instance

    @classmethod
    def from_string(
        cls,
        string: Any,
        dimensions: Any | None = None,
        weights: Any | None = None,
        phases: Any | None = None,
    ) -> Self:
        if isinstance(string, str):
            source_rows = [line.strip() for line in string.splitlines() if line.strip()]
        else:
            source_rows = list(string)

        rows: list[tuple[str, str | None, str | None]] = []
        for row in source_rows:
            row = str(row).strip()
            if "|" in row:
                fields = [field.strip() for field in row.split("|")]
                if len(fields) != 3:
                    raise ValueError("Each PauliSum row must contain weight, Pauli, and phase")
                rows.append((fields[1], fields[0], fields[2]))
            elif row:
                rows.append((row, None, None))

        if not rows:
            if weights is not None or phases is not None:
                raise ValueError("weights and phases require at least one Pauli string")
            register = dimensions
            if register is None:
                register = " ".join(source_rows)
            elif isinstance(register, Iterable) and not isinstance(register, (str, bytes)):
                register = " ".join(str(dimension) for dimension in register)
            else:
                register = str(register)
            return cls._from_native(_NativePauliSum.from_string(register))

        dimension_values = None
        scalar_dimension = False
        if dimensions is not None:
            if isinstance(dimensions, Iterable) and not isinstance(dimensions, (str, bytes)):
                dimension_values = list(dimensions)
            else:
                dimension_values = [dimensions]
                scalar_dimension = True

        def values_for(value: Any, name: str) -> list[Any] | None:
            if value is None:
                return None
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                values = list(value)
                if len(values) != len(rows):
                    raise ValueError(
                        f"{name} must contain one value per Pauli string "
                        f"({len(rows)} expected, got {len(values)})"
                    )
                return values
            else:
                return [value] * len(rows)

        weight_values = values_for(weights, "weights")
        phase_values = values_for(phases, "phases")
        parsed_strings: list[PauliString] = []
        parsed_weights: list[complex] = []
        parsed_phases: list[int] = []
        for index, (pauli, existing_weight, existing_phase) in enumerate(rows):
            if dimension_values is not None:
                tokens = pauli.split()
                row_dimensions = (
                    dimension_values * len(tokens)
                    if scalar_dimension
                    else dimension_values
                )
                if len(row_dimensions) != len(tokens):
                    raise ValueError(
                        "dimensions must contain one value per qudit "
                        f"({len(tokens)} expected, got {len(row_dimensions)})"
                    )
                pauli = " ".join(
                    f"{token.split('/', 1)[0]}/{dimension}"
                    for token, dimension in zip(tokens, row_dimensions)
                )

            weight = (
                weight_values[index]
                if weight_values is not None
                else existing_weight if existing_weight is not None else 1
            )
            if isinstance(weight, tuple) and len(weight) == 2:
                weight = complex(float(weight[0]), float(weight[1]))
            elif isinstance(weight, str) and len(weight.split()) == 2:
                real, imaginary = weight.split()
                weight = complex(float(real), float(imaginary))
            else:
                weight = complex(str(weight))
            phase = (
                phase_values[index]
                if phase_values is not None
                else existing_phase if existing_phase is not None else 0
            )
            parsed_strings.append(PauliString.from_string(pauli))
            parsed_weights.append(weight)
            parsed_phases.append(int(phase))
        if any(
            not np.array_equal(pauli.dimensions, parsed_strings[0].dimensions)
            for pauli in parsed_strings[1:]
        ):
            raise ValueError("all Pauli strings must have matching dimensions")
        return cls._from_components(
            np.vstack([pauli.tableau[0] for pauli in parsed_strings]),
            parsed_strings[0].dimensions,
            parsed_weights,
            parsed_phases,
        )

    def __init__(
        self, tableau: Any = None, dimensions: Any = None,
        weights: Any = None, phases: Any = None,
    ) -> None:
        raise TypeError("use PauliSum.from_string() or PauliSum.zero()")

    @classmethod
    def zero(cls, dimensions: Sequence[int]) -> Self:
        return cls._from_components(np.empty((0, 2 * len(dimensions)), dtype=np.int64), dimensions)

    @classmethod
    def from_tableau(
        cls,
        tableau: Any,
        dimensions: Any | None = None,
        weights: Any | None = None,
        phases: Any | None = None,
    ) -> Self:
        values = np.asarray(tableau, dtype=np.int64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] % 2:
            raise ValueError("tableau must be a two-dimensional array with an even number of columns")
        n_qudits = values.shape[1] // 2
        dimension_values = (
            np.full(n_qudits, 2, dtype=np.int64)
            if dimensions is None
            else np.asarray(dimensions, dtype=np.int64).reshape(-1)
        )
        if dimension_values.size == 1 and n_qudits != 1:
            dimension_values = np.repeat(dimension_values, n_qudits)
        if len(dimension_values) != n_qudits:
            raise ValueError("dimensions must contain one value per qudit")
        return cls._from_components(values, dimension_values, weights, phases)

    @classmethod
    def from_hilbert_space(
        cls, matrix: Any, dimensions: Any, threshold: int = 12
    ) -> Self:
        dimension_values = np.asarray(dimensions, dtype=np.int64).reshape(-1)
        if dimension_values.size == 1 and np.asarray(matrix).ndim == 2:
            dimension_values = np.repeat(dimension_values, int(
                round(np.log(np.asarray(matrix).shape[0]) / np.log(dimension_values[0]))))
        matrix = np.asarray(matrix, dtype=np.complex128)
        dimension = int(np.prod(dimension_values))
        if matrix.shape != (dimension, dimension):
            raise ValueError("matrix shape must match the product of dimensions")
        rows, weights = [], []
        tolerance = 10.0 ** (-threshold)
        for encoded in range(dimension * dimension):
            x_digits = []
            z_digits = []
            remaining = encoded
            for d in dimension_values:
                pair = remaining % (int(d) ** 2)
                remaining //= int(d) ** 2
                x_digits.append(pair // int(d))
                z_digits.append(pair % int(d))
            pauli = PauliString.from_exponents(x_digits, z_digits, dimension_values)
            pauli_matrix = pauli.to_hilbert_space().toarray()
            weight = np.trace(pauli_matrix.conj().T @ matrix) / dimension
            if abs(weight) > tolerance:
                rows.append(pauli.tableau[0])
                weights.append(weight)
        return cls._from_components(np.asarray(rows), dimension_values, weights)

    @classmethod
    def from_pauli_strings(
        cls,
        pauli_strings: Any,
        weights: Any | None = None,
        phases: Any | None = None,
        inherit_phases: bool = False,
        inherit_weights: bool = False,
    ) -> Self:
        if isinstance(pauli_strings, (PauliString, _NativePauliString)):
            strings = [pauli_strings]
        else:
            strings = list(pauli_strings)
        if not strings:
            raise ValueError("at least one PauliString is required")
        if not all(isinstance(pauli, (PauliString, _NativePauliString)) for pauli in strings):
            raise TypeError("pauli_strings must contain only PauliString instances")
        dimensions = np.asarray(
            strings[0].dimensions if isinstance(strings[0], PauliString) else strings[0].dimensions,
            dtype=np.int64,
        )
        native_strings = [
            pauli if isinstance(pauli, PauliString) else PauliString._from_native(pauli)
            for pauli in strings
        ]
        if any(not np.array_equal(pauli.dimensions, dimensions) for pauli in native_strings):
            raise ValueError("all Pauli strings must have matching dimensions")
        if inherit_phases:
            phases = np.concatenate([pauli.phases for pauli in native_strings])
        if inherit_weights:
            weights = np.concatenate([pauli.weights for pauli in native_strings])
        if weights is None:
            weight_values = np.ones(len(native_strings), dtype=np.complex128)
        elif np.isscalar(weights):
            weight_values = np.full(len(native_strings), weights, dtype=np.complex128)
        else:
            weight_values = np.asarray(weights, dtype=np.complex128).reshape(-1)
        if len(weight_values) != len(native_strings):
            raise ValueError("weights must contain one value per Pauli string")
        if phases is None:
            phase_values = np.zeros(len(native_strings), dtype=np.int64)
        elif np.isscalar(phases):
            phase_values = np.full(len(native_strings), phases, dtype=np.int64)
        else:
            phase_values = np.asarray(phases, dtype=np.int64).reshape(-1)
        if len(phase_values) != len(native_strings):
            raise ValueError("phases must contain one value per Pauli string")
        if inherit_phases and phases is not None:
            # Explicit phases are intentionally superseded by inherited phases.
            phase_values = np.concatenate([pauli.phases for pauli in native_strings])
        return cls._from_components(
            np.vstack([pauli.tableau[0] for pauli in native_strings]),
            dimensions,
            weight_values,
            phase_values,
        )

    @classmethod
    def from_pauli_objects(
        cls,
        pauli_objects: Any,
        weights: Any | None = None,
        phases: Any | None = None,
        inherit_weights: bool = False,
        inherit_phases: bool = False,
    ) -> Self:
        if isinstance(pauli_objects, (PauliString, PauliSum, _NativePauliString, _NativePauliSum)):
            objects = [pauli_objects]
        else:
            objects = list(pauli_objects)
        if not objects:
            raise ValueError("at least one Pauli object is required")
        rows = []
        object_weights = []
        object_phases = []
        dimensions = None
        for item in objects:
            if isinstance(item, PauliString):
                tableau = item.tableau
                item_dimensions = item.dimensions
                weights_for_item = item.weights
                phases_for_item = item.phases
            elif isinstance(item, _NativePauliString):
                wrapped = PauliString._from_native(item)
                tableau, item_dimensions = wrapped.tableau, wrapped.dimensions
                weights_for_item, phases_for_item = wrapped.weights, wrapped.phases
            elif isinstance(item, PauliSum):
                tableau, item_dimensions = item.tableau, item.dimensions
                weights_for_item, phases_for_item = item.weights, item.phases
            elif isinstance(item, _NativePauliSum):
                wrapped = cls._from_native(item)
                tableau, item_dimensions = wrapped.tableau, wrapped.dimensions
                weights_for_item, phases_for_item = wrapped.weights, wrapped.phases
            else:
                raise TypeError("pauli_objects must contain PauliString or PauliSum instances")
            if dimensions is None:
                dimensions = np.asarray(item_dimensions, dtype=np.int64)
            elif not np.array_equal(dimensions, item_dimensions):
                raise ValueError("all Pauli objects must have matching dimensions")
            rows.extend(tableau)
            object_weights.extend(weights_for_item)
            object_phases.extend(phases_for_item)
        if inherit_weights:
            weights = object_weights
        if inherit_phases:
            phases = object_phases
        n_rows = len(rows)
        if weights is None:
            weights = np.ones(n_rows, dtype=np.complex128)
        elif np.isscalar(weights):
            weights = np.full(n_rows, weights, dtype=np.complex128)
        if phases is None:
            phases = np.zeros(n_rows, dtype=np.int64)
        elif np.isscalar(phases):
            phases = np.full(n_rows, phases, dtype=np.int64)
        return cls._from_components(rows, dimensions, weights, phases)

    @classmethod
    def from_random(
        cls,
        n_paulis: int,
        dimensions: Any,
        rand_weights: bool = True,
        rand_phases: bool = False,
        seed: int | None = None,
    ) -> Self:
        dimension_values = np.asarray(dimensions, dtype=np.int64).reshape(-1)
        if dimension_values.size == 1 and not isinstance(dimensions, Iterable):
            dimension_values = dimension_values.reshape(1)
        if n_paulis < 0:
            raise ValueError("n_paulis must be non-negative")
        rng = np.random.default_rng(seed)
        tableau = np.empty((n_paulis, 2 * len(dimension_values)), dtype=np.int64)
        for i, dimension in enumerate(dimension_values):
            tableau[:, i] = rng.integers(int(dimension), size=n_paulis)
            tableau[:, len(dimension_values) + i] = rng.integers(int(dimension), size=n_paulis)
        weights = rng.uniform(-1, 1, size=n_paulis) if rand_weights else np.ones(n_paulis)
        lcm = math.lcm(*(int(d) for d in dimension_values)) if len(dimension_values) else 1
        phases = rng.integers(0, 2 * lcm, size=n_paulis) if rand_phases else np.zeros(n_paulis, dtype=np.int64)
        return cls._from_components(tableau, dimension_values, weights, phases)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dimensions: Any | None = None,
    ) -> Self:
        return cls.from_string(Path(path).read_text(), dimensions=dimensions)

    @property
    def tableau(self) -> npt.NDArray[np.int64]:
        return self._tableau

    @property
    def dimensions(self) -> npt.NDArray[np.uint64]:
        dimensions = self._dimensions.astype(np.uint64, copy=True)
        dimensions.setflags(write=False)
        return dimensions

    @dimensions.setter
    def dimensions(self, value: Any) -> None:
        raise ValueError("dimensions are determined by the qudit tableau and cannot be set")

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
    def x_exp(self) -> Any:
        return self._tableau[:, : self.n_qudits()]

    @property
    def z_exp(self) -> Any:
        return self._tableau[:, self.n_qudits() :]

    @property
    def phases(self) -> npt.NDArray[np.int64]:
        return self._phases

    @phases.setter
    def phases(self, value: Any) -> None:
        values = np.asarray(value, dtype=np.int64).reshape(-1)
        if values.size != self.n_paulis():
            raise ValueError("phases must contain one value per Pauli string")
        self._phases[:] = values % self.phase_modulus
        self._refresh_native()

    @property
    def n_terms(self) -> int:
        return self.n_paulis()

    def n_qudits(self) -> int:
        return len(self._dimensions)

    def n_paulis(self) -> int:
        return self._tableau.shape[0]

    def n_pauli_strings(self) -> int:
        return self.n_paulis()

    def first_anticommuting_pair(self) -> tuple[int, int] | None:
        for i in range(self.n_paulis()):
            left = self.select_pauli_string(i)
            for j in range(i + 1, self.n_paulis()):
                if left.symplectic_product(self.select_pauli_string(j)) != 0:
                    return i, j
        return None

    def is_mutually_commuting(self) -> bool:
        return self.first_anticommuting_pair() is None

    def assert_same_dimensions(self, dimensions: Sequence[int]) -> None:
        self._native.assert_same_dimensions(list(dimensions))

    def phase(self, row: int) -> int | None:
        if row < 0 or row >= self.n_paulis():
            return None
        return int(self._phases[row])

    def reduce_phase(self, phase: int) -> int:
        return int(phase) % self.phase_modulus

    def conjugate_phase(self, row: int) -> int | None:
        if row < 0 or row >= self.n_paulis():
            return None
        term = self.select_pauli_string(row)
        return int(term.H().phases[0])

    def conjugate_phases(self) -> npt.NDArray[np.int64]:
        return self.H().phases.copy()

    def hermitian_conjugate(self) -> Self:
        return self.H()

    def shape(self) -> tuple[int, int]:
        return self.n_pauli_strings(), self.n_qudits()

    def set_phases(self, value: Any) -> None:
        self.phases = value

    def reset_phases(self) -> None:
        self._phases[:] = 0
        self._refresh_native()

    def set_weights(self, value: Any) -> None:
        self.weights = value

    def reset_weights(self) -> None:
        self._weights[:] = 1
        self._refresh_native()

    def to_file(self, path: str) -> None:
        Path(path).write_text(str(self))

    def has_equal_tableau(self, other_pauli: Any, literal: bool = True) -> bool:
        if not isinstance(other_pauli, (PauliSum, _NativePauliSum, PauliString, _NativePauliString)):
            return False
        other = self._as_sum(other_pauli)
        if not np.array_equal(self.dimensions, other.dimensions):
            return False
        if literal:
            return bool(np.array_equal(self.tableau, other.tableau))
        return sorted(map(tuple, self.tableau.tolist())) == sorted(map(tuple, other.tableau.tolist()))

    def is_close(
        self,
        other_pauli: Any,
        threshold: int | None = None,
        literal: bool | None = None,
    ) -> bool:
        if not isinstance(other_pauli, (PauliSum, _NativePauliSum, PauliString, _NativePauliString)):
            return False
        other = self._as_sum(other_pauli)
        if not np.array_equal(self.dimensions, other.dimensions):
            return False
        atol = 10.0 ** (-(threshold if threshold is not None else 10))
        if literal is not False:
            return bool(
                np.array_equal(self.tableau, other.tableau)
                and np.array_equal(self.phases, other.phases)
                and np.allclose(self.weights, other.weights, atol=atol, rtol=0)
            )
        return self._coefficients_close(other, atol)

    def H(self) -> Self:
        result = self.copy()
        dimensions = self._dimensions
        n_qudits = self.n_qudits()
        result._tableau = (-self._tableau) % np.tile(dimensions, 2)
        phase_adjustment = 2 * np.sum(
            self.x_exp * self.z_exp * (self.lcm // dimensions), axis=1
        )
        result._phases = (-self._phases + phase_adjustment) % self.phase_modulus
        result._weights = self._weights.conjugate()
        result._refresh_native()
        return result

    def is_hermitian(self) -> bool:
        return self._coefficients_close(self.H(), 1e-10)

    def copy(self) -> Self:
        return self._from_components(
            self._tableau, self._dimensions, self._weights, self._phases
        )

    def phase_to_weight(self) -> None:
        self._weights[:] *= np.exp(1j * np.pi * self._phases / self.lcm)
        self._phases[:] = 0
        self._refresh_native()

    def weight_to_phase(self) -> None:
        if not self.n_paulis():
            return
        phases = self._phases.copy()
        weights = self._weights.copy()
        for i, weight in enumerate(weights):
            if abs(weight) <= 1e-14:
                continue
            target = int(round(np.angle(weight) * self.lcm / np.pi)) % self.phase_modulus
            phases[i] = (phases[i] + target) % self.phase_modulus
            weights[i] *= np.exp(-1j * np.pi * target / self.lcm)
        self._phases = phases
        self._weights = weights
        self._refresh_native()

    def standardize(self) -> None:
        self.combine_equivalent_paulis()

    def to_standard_form(self) -> Self:
        result = self.copy()
        result.combine_equivalent_paulis()
        if result.n_paulis():
            def order_key(i: int) -> tuple[int, tuple[int, ...]]:
                row = result.tableau[i]
                nonzero = np.flatnonzero(row)
                return (int(nonzero[0]) if nonzero.size else -1, tuple(int(v) for v in row))

            order = sorted(range(result.n_paulis()), key=order_key)
            result.reorder(order)
        return result

    def to_hilbert_space(self, pauli_string_index: int | None = None) -> Any:
        from scipy.sparse import csr_matrix

        if pauli_string_index is not None:
            return self.select_pauli_string(pauli_string_index).to_hilbert_space()
        result = csr_matrix((int(np.prod(self.dimensions)),) * 2, dtype=np.complex128)
        for index in range(self.n_paulis()):
            result += self.select_pauli_string(index).to_hilbert_space()
        return result.tocsr()

    def combine_equivalent_paulis(self) -> None:
        if not self.n_paulis():
            return
        self.phase_to_weight()
        groups: dict[tuple[int, ...], complex] = {}
        for row, weight in zip(self._tableau, self._weights):
            key = tuple(int(v) for v in row)
            groups[key] = groups.get(key, 0j) + weight
        rows, weights = [], []
        for key in sorted(groups):
            if abs(groups[key]) > 1e-14:
                rows.append(key)
                weights.append(groups[key])
        self._tableau = np.asarray(rows, dtype=np.int64).reshape(-1, self._tableau.shape[1])
        self._weights = np.asarray(weights, dtype=np.complex128)
        self._phases = np.zeros(len(rows), dtype=np.int64)
        self._refresh_native()

    def remove_trivial_qudits(self) -> None:
        if not self.n_qudits():
            return
        keep = np.any(self.x_exp != 0, axis=0) | np.any(self.z_exp != 0, axis=0)
        self._tableau = np.concatenate((self.x_exp[:, keep], self.z_exp[:, keep]), axis=1)
        self._dimensions = self._dimensions[keep]
        self._refresh_native()

    def remove_trivial_paulis(self) -> None:
        keep = np.any(self._tableau != 0, axis=1)
        self._filter_rows(keep)

    def remove_zero_weight_paulis(self, threshold: float | None = None) -> None:
        cutoff = 1e-14 if threshold is None else threshold
        self._filter_rows(np.abs(self._weights) > cutoff)

    def is_identity(self) -> bool:
        if not self.n_paulis():
            return True
        result = self.to_standard_form()
        return bool(
            np.all(result._tableau == 0)
            and np.all(result._phases == 0)
            and np.all(result._weights == 1)
        )

    def is_x(self) -> bool:
        return not bool(np.any(self.z_exp))

    def is_z(self) -> bool:
        return not bool(np.any(self.x_exp))

    def is_commuting(self, pauli_string_indexes: Any | None = None) -> bool:
        indexes = np.arange(self.n_paulis()) if pauli_string_indexes is None else np.asarray(
            pauli_string_indexes, dtype=int).reshape(-1)
        for pos, i in enumerate(indexes):
            for j in indexes[:pos]:
                if self.select_pauli_string(int(i)).symplectic_product(
                    self.select_pauli_string(int(j))
                ) != 0:
                    return False
        return True

    def is_quditwise_commuting(self, pauli_string_indexes: Any | None = None) -> bool:
        indexes = np.arange(self.n_paulis()) if pauli_string_indexes is None else np.asarray(
            pauli_string_indexes, dtype=int).reshape(-1)
        for pos, i in enumerate(indexes):
            for j in indexes[:pos]:
                if self.select_pauli_string(int(i)).quditwise_product(
                    self.select_pauli_string(int(j))
                ):
                    return False
        return True

    def select_pauli_string(self, index: int) -> PPauliString:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")
        index = int(index)
        if index < 0:
            index += self.n_paulis()
        if index < 0 or index >= self.n_paulis():
            raise IndexError("Pauli string index out of range")
        native = _NativePauliString.from_exponents(
            self.x_exp[index].tolist(), self.z_exp[index].tolist(), self._dimensions.tolist()
        )
        return PauliString._from_native(
            native, phases=[self._phases[index]], weights=[self._weights[index]]
        )

    def delete_paulis(self, pauli_indices: Any) -> None:
        indexes = np.asarray(pauli_indices, dtype=int).reshape(-1)
        keep = np.ones(self.n_paulis(), dtype=bool)
        keep[indexes] = False
        self._filter_rows(keep)

    def delete_qudits(self, qudit_indexes: Any) -> None:
        indexes = np.asarray(qudit_indexes, dtype=int).reshape(-1)
        keep = np.ones(self.n_qudits(), dtype=bool)
        keep[indexes] = False
        self._tableau = np.concatenate((self.x_exp[:, keep], self.z_exp[:, keep]), axis=1)
        self._dimensions = self._dimensions[keep]
        self._refresh_native()

    def _delete_qudits(self, qudit_indexes: Any) -> None:
        self.delete_qudits(qudit_indexes)

    def symplectic_product_matrix(self) -> npt.NDArray[np.int64]:
        n, lcm = self.n_paulis(), self.lcm
        matrix = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i):
                value = self.select_pauli_string(i).symplectic_product(
                    self.select_pauli_string(j)
                )
                matrix[i, j] = matrix[j, i] = value
        return matrix % lcm

    def symplectic_residue_tensor(self) -> npt.NDArray[np.int64]:
        n, n_qudits = self.n_paulis(), self.n_qudits()
        tensor = np.zeros((n, n, n_qudits), dtype=np.int64)
        for i in range(n):
            for j in range(i):
                residues = self.select_pauli_string(i).symplectic_residues(
                    self.select_pauli_string(j)
                )
                tensor[i, j] = residues
                tensor[j, i] = residues
        return tensor

    def quditwise_symplectic_product_matrix(self) -> npt.NDArray[np.int64]:
        n = self.n_paulis()
        matrix = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                value = self.select_pauli_string(i).quditwise_product(
                    self.select_pauli_string(j)
                )
                matrix[i, j] = matrix[j, i] = value
        return matrix

    def get_subspace(
        self, qudit_indexes: Any, pauli_indexes: Any | None = None
    ) -> Self:
        q_indexes = np.asarray(qudit_indexes, dtype=int).reshape(-1)
        p_indexes = np.arange(self.n_paulis()) if pauli_indexes is None else np.asarray(
            pauli_indexes, dtype=int).reshape(-1)
        selected = self._tableau[p_indexes]
        new_tableau = np.concatenate(
            (selected[:, q_indexes], selected[:, self.n_qudits() + q_indexes]), axis=1
        )
        return self._from_components(
            new_tableau, self._dimensions[q_indexes], self._weights[p_indexes], self._phases[p_indexes]
        )

    def acquire_phase(
        self, phases: Any, pauli_indexes: Any | None = None
    ) -> None:
        values = np.asarray(phases, dtype=np.int64).reshape(-1)
        indexes = np.arange(self.n_paulis()) if pauli_indexes is None else np.asarray(
            pauli_indexes, dtype=int).reshape(-1)
        if values.size == 1 and indexes.size != 1:
            values = np.repeat(values, indexes.size)
        if values.size != indexes.size:
            raise ValueError("phases must match the number of selected Pauli strings")
        self._phases[indexes] = (self._phases[indexes] + values) % self.phase_modulus
        self._refresh_native()

    def ordered_eigenspectrum(self, num_eigens: int | None = None) -> Any:
        if not self.is_hermitian():
            raise ValueError("eigenspectrum is defined only for Hermitian Pauli sums")
        matrix = self.to_hilbert_space().toarray()
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)
        if num_eigens is not None:
            if num_eigens < 0 or num_eigens > len(values):
                raise ValueError("num_eigens must be between 0 and the Hilbert dimension")
            order = order[:num_eigens]
        return values[order], vectors[:, order].T

    def stabilizer_to_hilbert_space(self) -> Any:
        raise NotImplementedError("stabilizer_to_hilbert_space is not exposed by the native binding")

    def make_hermitian(self, in_place: bool = False) -> Self:
        result = self if in_place else self.copy()
        if result.is_hermitian():
            return result
        hermitian = 0.5 * (result + result.H())
        hermitian.combine_equivalent_paulis()
        return hermitian

    def XZ_to_Y(self, in_place: bool = False) -> Self:
        result = self if in_place else self.copy()
        if 2 in result._dimensions:
            selected = (result.x_exp != 0) & (result.z_exp != 0)
            result._phases += selected[:, result._dimensions == 2].sum(axis=1) * (result.lcm // 2)
            result._phases %= result.phase_modulus
            result._refresh_native()
        return result

    def reorder(self, order: Any) -> None:
        indexes = list(np.asarray(order, dtype=int).reshape(-1))
        indexes.extend(i for i in range(self.n_paulis()) if i not in indexes)
        self._tableau = self._tableau[indexes].copy()
        self._weights = self._weights[indexes].copy()
        self._phases = self._phases[indexes].copy()
        self._refresh_native()

    def swap_paulis(self, index_1: int, index_2: int) -> None:
        for values in (self._tableau, self._weights, self._phases):
            values[[index_1, index_2]] = values[[index_2, index_1]]
        self._refresh_native()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (PauliString, _NativePauliString, PauliSum, _NativePauliSum)):
            try:
                rhs = self._as_sum(other)
            except (TypeError, ValueError):
                return False
            return bool(
                np.array_equal(self._dimensions, rhs._dimensions)
                and np.array_equal(self._tableau, rhs._tableau)
                and np.array_equal(self._weights, rhs._weights)
                and np.array_equal(self._phases, rhs._phases)
            )
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __gt__(self, other: PPauliSum) -> bool:
        rhs = self._as_sum(other)
        self._assert_orderable(rhs)
        return tuple(self._tableau[0]) < tuple(rhs._tableau[0])

    def __lt__(self, other: PPauliSum) -> bool:
        return not self > other and not self == other

    def __getitem__(self, key: Any) -> Self:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("tuple indices must contain Pauli and qudit selections")
            pauli_indexes, qudit_indexes = key
            if isinstance(pauli_indexes, (int, np.integer)):
                term = self.select_pauli_string(int(pauli_indexes))
                return term[qudit_indexes]
            p_indexes = np.arange(self.n_paulis())[pauli_indexes]
            q_indexes = np.arange(self.n_qudits())[qudit_indexes]
            return self.get_subspace(q_indexes, p_indexes)
        if isinstance(key, (int, np.integer)):
            return self.select_pauli_string(int(key))
        if isinstance(key, (slice, list, np.ndarray)):
            indexes = np.arange(self.n_paulis())[key]
            indexes = np.asarray(indexes, dtype=int).reshape(-1)
            return self._from_components(
                self._tableau[indexes], self._dimensions,
                self._weights[indexes], self._phases[indexes]
            )
        raise ValueError("key must be an integer, slice, list, array, or 2-tuple")

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, PauliString):
            values = self.from_pauli_strings(value)
        elif isinstance(value, PauliSum):
            values = value
        else:
            raise TypeError("assigned value must be a PauliString or PauliSum")
        if not np.array_equal(self._dimensions, values._dimensions):
            raise ValueError("assigned object must have matching dimensions")
        indexes = np.arange(self.n_paulis())[key]
        indexes = np.asarray(indexes, dtype=int).reshape(-1)
        if indexes.size != values.n_paulis():
            raise ValueError("assigned value must match the number of selected rows")
        self._tableau[indexes] = values._tableau
        self._weights[indexes] = values._weights
        self._phases[indexes] = values._phases
        self._refresh_native()

    def __add__(self, other: Any) -> Self:
        if other == 0 and not isinstance(other, (PauliSum, PauliString)):
            return self.copy()
        rhs = self._as_sum(other)
        self._assert_same_dimensions(rhs)
        return self._from_components(
            np.vstack((self._tableau, rhs._tableau)), self._dimensions,
            np.concatenate((self._weights, rhs._weights)),
            np.concatenate((self._phases, rhs._phases)),
        )

    def __radd__(self, other: Any) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Self:
        rhs = self._as_sum(other)
        self._assert_same_dimensions(rhs)
        return self._from_components(
            np.vstack((self._tableau, rhs._tableau)), self._dimensions,
            np.concatenate((self._weights, -rhs._weights)),
            np.concatenate((self._phases, rhs._phases)),
        )

    def __rsub__(self, other: Any) -> Self:
        return self._as_sum(other).__sub__(self)

    def __mul__(self, other: Any) -> Self:
        if np.isscalar(other) and isinstance(other, (int, float, complex, np.number)):
            return self._from_components(
                self._tableau, self._dimensions, self._weights * complex(other), self._phases
            )
        rhs = self._as_sum(other)
        self._assert_same_dimensions(rhs)
        if self.n_paulis() == 0 or rhs.n_paulis() == 0:
            return self.zero(self._dimensions)
        rows, weights, phases = [], [], []
        for i, row_a in enumerate(self._tableau):
            for j, row_b in enumerate(rhs._tableau):
                rows.append((row_a + row_b) % np.tile(self._dimensions, 2))
                acquired = 2 * sum(
                    int(row_a[self.n_qudits() + k]) * int(row_b[k])
                    * (self.lcm // int(d))
                    for k, d in enumerate(self._dimensions)
                )
                weights.append(self._weights[i] * rhs._weights[j])
                phases.append(self._phases[i] + rhs._phases[j] + acquired)
        return self._from_components(rows, self._dimensions, weights, phases)

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    def __matmul__(self, other: Any) -> Self:
        rhs = self._as_sum(other)
        dimensions = np.concatenate((self._dimensions, rhs._dimensions))
        new_lcm = math.lcm(*(int(d) for d in dimensions)) if len(dimensions) else 1
        rows, weights, phases = [], [], []
        for i, row_a in enumerate(self._tableau):
            for j, row_b in enumerate(rhs._tableau):
                n1, n2 = self.n_qudits(), rhs.n_qudits()
                rows.append(np.concatenate((row_a[:n1], row_b[:n2], row_a[n1:], row_b[n2:])))
                weights.append(self._weights[i] * rhs._weights[j])
                phases.append(
                    self._phases[i] * (new_lcm // self.lcm)
                    + rhs._phases[j] * (new_lcm // rhs.lcm)
                )
        return self._from_components(rows, dimensions, weights, phases)

    def __rmatmul__(self, other: Any) -> Self:
        return self._as_sum(other).__matmul__(self)

    def __truediv__(self, other: Any) -> Self:
        if not isinstance(other, (int, float, complex, np.number)):
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError("division by zero")
        return self * (1 / other)

    def __pow__(self, exponent: int, modulo: int | None = None) -> Self:
        if modulo is not None or not isinstance(exponent, int):
            raise TypeError("PauliSum powers require an integer exponent")
        if exponent < 0:
            raise ValueError("negative powers of a PauliSum are not defined")
        result = self._from_components(
            np.zeros((1, 2 * self.n_qudits()), dtype=np.int64),
            self._dimensions,
            [1],
            [0],
        )
        base, power = self, exponent
        while power:
            if power & 1:
                result = result * base
            power >>= 1
            if power:
                base = base * base
        return result

    def __hash__(self) -> int:
        return hash((tuple(map(tuple, self._tableau)), tuple(self._dimensions),
                     tuple(self._weights), tuple(self._phases)))

    @property
    def weights(self) -> npt.NDArray[np.complex128]:
        return self._weights

    @weights.setter
    def weights(self, value: Any) -> None:
        values = np.asarray(value, dtype=np.complex128).reshape(-1)
        if values.size != self.n_paulis():
            raise ValueError("weights must contain one value per Pauli string")
        self._weights[:] = values
        self._refresh_native()

    def __str__(self) -> str:
        return "\n".join(
            f"{float(w.real)!r} {float(w.imag)!r} | "
            + " ".join(
                f"x{int(x)}z{int(z)}/{int(d)}"
                for x, z, d in zip(self.x_exp[i], self.z_exp[i], self._dimensions)
            )
            + f" | {int(self._phases[i])}"
            for i, w in enumerate(self._weights)
        )

    def __repr__(self) -> str:
        return f"PauliSum.from_string({str(self)!r})"

    def __iter__(self):
        for index in range(self.n_paulis()):
            yield self.select_pauli_string(index)

    def standardise(self) -> None:
        self.standardize()

    def _refresh_native(self) -> None:
        refreshed = self._from_components(
            self._tableau, self._dimensions, self._weights, self._phases
        )
        self._native = refreshed._native

    def _filter_rows(self, keep: Any) -> None:
        keep = np.asarray(keep, dtype=bool).reshape(-1)
        self._tableau = self._tableau[keep].copy()
        self._weights = self._weights[keep].copy()
        self._phases = self._phases[keep].copy()
        self._refresh_native()

    @staticmethod
    def _as_sum(value: Any) -> Self:
        if isinstance(value, PauliSum):
            return value
        if isinstance(value, _NativePauliSum):
            return PauliSum._from_native(value)
        if isinstance(value, PauliString):
            return PauliSum.from_pauli_strings(value)
        if isinstance(value, _NativePauliString):
            return PauliSum.from_pauli_strings(PauliString._from_native(value))
        raise TypeError("expected a PauliString or PauliSum")

    def _assert_same_dimensions(self, other: Self) -> None:
        if not np.array_equal(self._dimensions, other._dimensions):
            raise ValueError("Pauli objects must have matching dimensions")

    def _assert_orderable(self, other: Self) -> None:
        if self.n_paulis() != 1 or other.n_paulis() != 1:
            raise ValueError("Pauli sums with more than one term cannot be ordered")
        self._assert_same_dimensions(other)

    def _coefficient_map(self) -> dict[tuple[int, ...], complex]:
        coefficients: dict[tuple[int, ...], complex] = {}
        for row, weight, phase in zip(self._tableau, self._weights, self._phases):
            key = tuple(int(v) for v in row)
            value = complex(weight * np.exp(1j * np.pi * int(phase) / self.lcm))
            coefficients[key] = coefficients.get(key, 0j) + value
        return coefficients

    def _coefficients_close(self, other: Self, tolerance: float) -> bool:
        if not np.array_equal(self._dimensions, other._dimensions):
            return False
        left, right = self._coefficient_map(), other._coefficient_map()
        return all(
            np.isclose(left.get(key, 0j), right.get(key, 0j), atol=tolerance, rtol=0)
            for key in left.keys() | right.keys()
        )
