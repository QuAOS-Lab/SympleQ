"""The test system of arXiv:2605.00682: the periodic pure-gauge plaquette of a 2D U(1) lattice
gauge theory, on three qudits (App. C2 of Meth et al., arXiv:2310.12110, electric representation):

    H(g^2) = g^2 * 2 [R1^2 + R2^2 + R3^2 - R2 (R1 + R3)]  -  1/(2 g^2) (P1 + P2 + P3 + P1 P2 P3 + h.c.)

Each rotator is truncated to r = -l..l (one qudit of dimension d = 2l + 1); P lowers r by one
and annihilates r = -l. For qutrits the PauliSum has 86 strings besides the identity, as in
the paper. The identity is left out: AQUIRE does not measure it.
"""
import numpy as np
from sympleq.core.paulis import PauliSum


def plaquette_matrix(d, g2):
    r_max = (d - 1) // 2
    R = np.diag(r_max - np.arange(d)).astype(float)   # basis index k <-> r = r_max - k
    P = np.diag(np.ones(d - 1), -1)                   # P|r> = |r - 1>, P|-l> = 0
    identity = np.eye(d)

    def on(op, q):
        ops = [identity] * 3
        ops[q] = op
        return np.kron(np.kron(ops[0], ops[1]), ops[2])
    R1, R2, R3 = (on(R, q) for q in range(3))
    P1, P2, P3 = (on(P, q) for q in range(3))
    HE = 2 * (R1 @ R1 + R2 @ R2 + R3 @ R3 - R2 @ (R1 + R3))
    B = P1 + P2 + P3 + P1 @ P2 @ P3
    return g2 * HE - 0.5 * (B + B.T) / g2


def plaquette_hamiltonian(d=3, g2=0.01):
    """The plaquette as a PauliSum, without its identity string."""
    H = PauliSum.from_hilbert_space(plaquette_matrix(d, g2), [d, d, d], threshold=10)
    keep = [i for i in range(H.n_paulis()) if np.any(np.asarray(H[i].x_exp)) or np.any(np.asarray(H[i].z_exp))]
    return H[keep]


def product_state(d=3):
    """(|0> + ... + |d-1>)^(x3), normalised: the hardware state of the paper."""
    v = np.ones(d) / np.sqrt(d)
    return np.kron(np.kron(v, v), v)
