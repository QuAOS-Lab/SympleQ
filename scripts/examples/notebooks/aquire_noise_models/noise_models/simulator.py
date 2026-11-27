"""Exact density-matrix simulation of qudit circuits with gate-level noise models.

A noise model (NoiseModel subclass) has up to two hooks:

    after_gate(sim, rho, qudits, kind) noise after every gate, on that gate's qudits;
                                       kind is "local" (one qudit) or "entangling" (two or more)
    measure(sim, pdf, circuit)         noise on the outcome probabilities (AQUIRE's uniform model)

GateNoiseSimulator(dims, [models]) returns the exact outcome probabilities of a circuit. It
is called as AQUIRE's outcome_distribution_function: sim(circuit, state, diagnostic=False).

Model strengths are set relative to the paper's fitted error probabilities (P_LOCAL per
single-qudit gate, P_ENTANGLING per entangling gate): at strength s every gate has the
process infidelity of local depolarising noise with probability s * P, so the models differ
in the type of error only.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from sympleq.core.circuits.utils import embed_unitary

P_LOCAL = 0.0041        # the paper's fitted error probability per single-qudit gate (arXiv:2605.00682)
P_ENTANGLING = 0.079    # ... per entangling gate


class NoiseModel:
    """No noise. Subclasses override the hooks they need."""
    name = "noiseless"

    def after_gate(self, sim, rho, qudits, kind):
        return rho

    def measure(self, sim, pdf, circuit):
        return pdf

    def describe(self):
        return self.name


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


class GateNoiseSimulator:
    def __init__(self, dims, models=()):
        self.dims = [int(d) for d in dims]
        self.n = len(self.dims)
        self.D = int(np.prod(self.dims))
        self.models = list(models)
        self._gates = {}

    def __call__(self, circuit, state, diagnostic=False):
        """AQUIRE's outcome_distribution_function. Measurement and diagnostic circuits are simulated alike,
        from the given state, so diagnostic is not used."""
        return self.outcome_distribution(circuit, state)

    def outcome_distribution(self, circuit, state):
        psi = np.asarray(state, dtype=complex).ravel()
        rho = np.outer(psi, psi.conj())
        for gate, qudits in zip(circuit.gates, circuit.qudit_indices):
            qudits = [int(q) for q in qudits]
            key = (gate.name, tuple(qudits))
            if key not in self._gates:
                self._gates[key] = _dense(embed_unitary(gate.local_unitary(self.dims[qudits[0]]), qudits, self.dims))
            U = self._gates[key]
            rho = U @ rho @ U.conj().T
            kind = "entangling" if gate.n_qudits >= 2 else "local"
            for m in self.models:
                rho = m.after_gate(self, rho, qudits, kind)
        pdf = np.clip(np.real(np.diag(rho)), 0.0, None)
        for m in self.models:
            pdf = m.measure(self, pdf, circuit)
        return pdf / pdf.sum()

    # building blocks for the models
    def replace(self, rho, subset):
        """Tr_S(rho) (x) 1_S / d_S: the qudits in subset replaced by the maximally mixed state."""
        n, dims = self.n, self.dims
        rows, cols = list(range(n)), list(range(n, 2 * n))
        keep = [q for q in range(n) if q not in subset]
        traced = [2 * n + q for q in range(n)]
        idx = [traced[q] if q in subset else rows[q] for q in range(n)] + \
              [traced[q] if q in subset else cols[q] for q in range(n)]
        reduced = np.einsum(rho.reshape(dims + dims), idx, [rows[q] for q in keep] + [cols[q] for q in keep])
        dS = int(np.prod([dims[q] for q in subset]))
        mixed = (np.eye(dS) / dS).reshape([dims[q] for q in subset] * 2)
        out = np.einsum(reduced, [rows[q] for q in keep] + [cols[q] for q in keep],
                        mixed, [rows[q] for q in subset] + [cols[q] for q in subset], rows + cols)
        return out.reshape(self.D, self.D)

    def dephase(self, rho, subset):
        """Remove the coherences of the qudits in subset (keep their computational-basis diagonal)."""
        n, dims = self.n, self.dims
        T = rho.reshape(dims + dims).copy()
        for q in subset:
            shape = [1] * (2 * n)
            shape[q] = shape[n + q] = dims[q]
            T = T * np.eye(dims[q]).reshape(shape)
        return T.reshape(self.D, self.D)

    def local_unitary(self, qudits, u_of_d):
        U = np.eye(self.D, dtype=complex)
        for q in qudits:
            U = _dense(embed_unitary(u_of_d(self.dims[q]), [q], self.dims)) @ U
        return U


# ---------------------------------------------------------------------- matching error rates
def matched_infidelity(strength, d, kind):
    """Process infidelity per gate at a strength: that of local depolarising noise with probability strength * P."""
    p, n = (strength * P_LOCAL, 1) if kind == "local" else (strength * P_ENTANGLING, 2)
    return p * (1 - 1 / d ** (2 * n))


def dephasing_probability(r, d, n_qudits):
    """Probability of full dephasing with process infidelity r (full dephasing has fidelity 1/d^n)."""
    return r / (1 - 1 / d ** n_qudits)


def rotation_generator(d, axis):
    """(X + X^dagger)/2 for axis "x" (over-rotation), diag(k) - (d-1)/2 for axis "z" (phase error)."""
    if axis == "x":
        X = np.roll(np.eye(d), 1, axis=0)
        return (X + X.T) / 2
    return np.diag(np.arange(d) - (d - 1) / 2)


def coherent_angle(r, d, n_qudits, axis):
    """theta with process infidelity r for exp(-i theta G) on each of n_qudits qudits."""
    if r <= 0:
        return 0.0
    g = np.linalg.eigvalsh(rotation_generator(d, axis))
    return float(brentq(lambda t: 1 - (abs(np.exp(-1j * t * g).sum()) ** 2 / d ** 2) ** n_qudits - r,
                        0.0, np.pi / (2 * abs(g).max())))


def coherent_unitary(d, axis, theta):
    return expm(-1j * theta * rotation_generator(d, axis))


def process_infidelity(model, d, kind):
    """1 - <Phi|(E x id)(Phi)|Phi> of a model's per-gate channel, computed numerically (a check)."""
    n = 1 if kind == "local" else 2
    sim = GateNoiseSimulator([d] * (2 * n))
    D = d ** n
    phi = np.zeros(D * D)
    phi[[k * D + k for k in range(D)]] = 1 / np.sqrt(D)
    rho = model.after_gate(sim, np.outer(phi, phi).astype(complex), list(range(n)), kind)
    return 1 - float(np.real(phi @ rho @ phi))
