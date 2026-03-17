import numpy as np

# Fixed seed for reproducibility across experiments
random_seed = 1111
np.random.seed(random_seed)

# Storage for generalization metrics (used if aggregating multiple runs)
gen_errors_loss = []
gen_errors_accuracies = []
W_norms = []

from scipy.sparse import kron, csr_matrix
from sklearn.utils import shuffle


# ───────────────────────────────────────────────
#          Pauli matrices & Kronecker helpers
# ───────────────────────────────────────────────

def pauli_matrices():
    """Single-qubit Pauli X, Z, and identity (sparse format)."""
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    I = csr_matrix(np.eye(2, dtype=complex))
    return X, Z, I


def kron_N(ops):
    """Sequential Kronecker product of operators (CSR format)."""
    res = ops[0]
    for op in ops[1:]:
        res = kron(res, op, format='csr')
    return res


# ───────────────────────────────────────────────
#     Periodic generalized cluster Hamiltonian
# ───────────────────────────────────────────────

def cluster_hamiltonian(N, J1, J2):
    """
    Periodic cluster Hamiltonian:
        H = ∑ⱼ Zⱼ − J₁ Xⱼ X_{j+1} − J₂ X_{j−1} Zⱼ X_{j+1}
    """
    X, Z, I = pauli_matrices()
    H = csr_matrix((2**N, 2**N), dtype=complex)

    for j in range(N):
        # Local Z term
        ops = [I] * N
        ops[j] = Z
        H += kron_N(ops)

        # Nearest-neighbor XX coupling
        ops = [I] * N
        ops[j] = X
        ops[(j + 1) % N] = X
        H -= J1 * kron_N(ops)

        # Three-body cluster interaction
        ops = [I] * N
        ops[(j - 1) % N] = X
        ops[j] = Z
        ops[(j + 1) % N] = X
        H -= J2 * kron_N(ops)

    return H


def ground_state(H):
    """Normalized ground state vector (via dense diagonalization)."""
    H_dense = H.toarray()
    vals, vecs = np.linalg.eigh(H_dense)
    gs = vecs[:, 0]
    gs /= np.linalg.norm(gs)
    return gs


# ───────────────────────────────────────────────
#           Dataset generation (phase diagram)
# ───────────────────────────────────────────────

def generate_cluster_dataset(N, n_samples_total, seed):
    """
    Generate dataset with random sample distribution across 4 phases.
    Labels:
      0 : small J1, small J2
      1 : large J1, small J2
      2 : small J1, large J2
      3 : large J1, large J2
    """
    np.random.seed(seed)
    data, labels = [], []

    # Random weights → multinomial allocation
    weights = np.random.rand(4)
    weights /= np.sum(weights)
    n_per_phase = np.random.multinomial(n_samples_total, weights)

    def sample_phase(J1_range, J2_range, n_samples, label):
        for _ in range(n_samples):
            J1 = np.random.uniform(*J1_range)
            J2 = np.random.uniform(*J2_range)
            H = cluster_hamiltonian(N, J1, J2)
            gs = ground_state(H)
            data.append(np.real(gs))
            labels.append(label)

    sample_phase((0.0, 0.5), (0.0, 0.5), n_per_phase[0], 0)     # Phase 0
    sample_phase((1.0, 2.0), (0.0, 0.5), n_per_phase[1], 1)     # Phase 1
    sample_phase((0.0, 0.5), (1.0, 2.0), n_per_phase[2], 2)     # Phase 2
    sample_phase((1.0, 2.0), (1.0, 2.0), n_per_phase[3], 3)     # Phase 3

    data = np.array(data)
    labels = np.array(labels)
    data, labels = shuffle(data, labels, random_state=seed)

    return data, labels, n_per_phase


# Target density matrices used for fidelity comparison
dm_labels = [
    np.diag([1., 0., 0., 0.]),
    np.diag([0., 1., 0., 0.]),
    np.diag([0., 0., 1., 0.]),
    np.diag([0., 0., 0., 1.])
]


import pennylane as qml
from pennylane import numpy as np


# ───────────────────────────────────────────────
#            Quantum circuit components
# ───────────────────────────────────────────────

def density_matrix(state):
    """Convert pure state vector to density matrix."""
    return state * np.conj(state).T


def U2(U2_params, wires):
    """Strong entangling two-qubit variational block."""
    qml.Rot(U2_params[0], U2_params[1], U2_params[2], wires=wires[0])
    qml.Rot(U2_params[3], U2_params[4], U2_params[5], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(U2_params[6], wires=wires[0])
    qml.RY(U2_params[7], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(U2_params[8], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.Rot(U2_params[9], U2_params[10], U2_params[11], wires=wires[0])
    qml.Rot(U2_params[12], U2_params[13], U2_params[14], wires=wires[1])


def U1(phi, varphi, wire):
    """Gradient-compatible single-qubit conditional rotation."""
    qml.RZ(-varphi, wires=wire)
    qml.RY(2 * phi, wires=wire)
    qml.RZ(varphi, wires=wire)


def F_prob(theta, phi, varphi, main_wire, ancilla_wire):
    """Conditional single-qubit gate controlled by ancilla measurement."""
    qml.RX(theta, wires=ancilla_wire)
    m = qml.measure(ancilla_wire)

    def apply_u1():
        U1(phi, varphi, main_wire)

    qml.cond(m, apply_u1)()


# ───────────────────────────────────────────────
#                   Main QCNN circuit
# ───────────────────────────────────────────────

dev = qml.device("default.qubit", wires=18)


@qml.qnode(dev)
def QCNN(params_F1, params_U1, params_F2, params_U2, params_convolutions, x, y):
    """
    QCNN with amplitude embedding + two layers of conditional F blocks
    + final convolution layer before measurement.
    """
    # Embed real part of ground state vector
    qml.AmplitudeEmbedding(x, wires=range(6))

    # Layer 1 & 2: conditional F blocks on each data qubit
    for i in range(6):
        # First conditional layer
        F_prob(params_F1[i], params_U1[i][0], params_U1[i][1],
               main_wire=i, ancilla_wire=6 + i)
        # Second conditional layer
        F_prob(params_F2[i], params_U2[i][0], params_U2[i][1],
               main_wire=i, ancilla_wire=12 + i)

    # Final convolution layer (partial – only 5 applications here)
    U2(params_convolutions, wires=[0,1])
    U2(params_convolutions, wires=[2,3])
    U2(params_convolutions, wires=[4,5])
    U2(params_convolutions, wires=[1,2])
    U2(params_convolutions, wires=[3,4])

    # Fidelity measurement on logical output qubits
    return qml.expval(qml.Hermitian(y, wires=[2, 3]))


# ───────────────────────────────────────────────
#              Evaluation & training helpers
# ───────────────────────────────────────────────

def test(params_F1, params_U1, params_F2, params_U2, params_convolutions, x, y, state_labels=None):
    """Predict class by maximum fidelity with target states."""
    dm_labels = [density_matrix(s) for s in state_labels] if state_labels is not None else dm_labels
    predicted = []
    fidelity_values = []

    for xi in x:
        fidel_fn = lambda dm: QCNN(params_F1, params_U1, params_F2, params_U2, params_convolutions, xi, dm)
        fidelities = [fidel_fn(dm) for dm in dm_labels]
        pred = np.argmax(fidelities)
        predicted.append(pred)
        fidelity_values.append(fidelities)

    return np.array(predicted), np.array(fidelity_values)


def accuracy_score(y_true, y_pred):
    """Classification accuracy."""
    return np.mean(y_true == y_pred)


def iterate_minibatches(inputs, targets, batch_size):
    """Yield consecutive minibatches."""
    for start in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        yield inputs[start:start + batch_size], targets[start:start + batch_size]


# ───────────────────────────────────────────────
#           Kraus operators & process matrix
# ───────────────────────────────────────────────

def U(phi, varphi):
    """Single-qubit phase gate U(φ, ψ)."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [c,             -np.exp(1j * varphi) * s],
        [np.exp(-1j * varphi) * s,  c]
    ], dtype=complex)


def single_qubit_kraus(theta, phi, varphi):
    """Two Kraus operators for probabilistic single-qubit channel."""
    K0 = np.cos(theta / 2) * np.eye(2, dtype=complex)
    K1 = np.sin(theta / 2) * np.conj(U(phi, varphi))
    return [K0, K1]


paulis = [
    np.eye(2, dtype=complex),                  # I
    np.array([[0, 1], [1, 0]], dtype=complex), # X
    np.array([[0, -1j], [1j, 0]], dtype=complex), # Y
    np.array([[1, 0], [0, -1]], dtype=complex) # Z
]


def multi_kron(mats):
    """Kronecker product of multiple matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def pauli_strings_list(n, normalized=False):
    """All n-qubit Pauli tensor products."""
    from itertools import product
    indices = list(product(range(4), repeat=n))
    strings = [multi_kron([paulis[i] for i in idx]) for idx in indices]
    if normalized:
        factor = 1 / np.sqrt(2**n)
        strings = [factor * s for s in strings]
    return strings


def chi_from_kraus(K_list, n, paulis_list):
    """
    Process matrix χ in Pauli basis from Kraus operators
    (autograd/Pennylane compatible implementation).
    """
    d = 2**n
    e_rows = []
    for sigma in paulis_list:
        row = [np.trace(np.conj(sigma.T) @ K) / d for K in K_list]
        e_rows.append(row)
    e = np.array(e_rows, dtype=complex)

    chi = np.zeros((len(paulis_list), len(paulis_list)), dtype=complex)
    for vec in e.T:
        chi += np.outer(vec, np.conj(vec))
    return chi


def PM_reducidas(params_F1, params_U1, params_F2, params_U2, n):
    """
    Compute single-qubit process matrices for each of the n qubits
    after applying both conditional layers (F1 + F2).
    """
    PMs = []
    paulis_1q = pauli_strings_list(1)

    for i in range(n):
        K1_list = single_qubit_kraus(params_F1[i], params_U1[i][0], params_U1[i][1])
        K2_list = single_qubit_kraus(params_F2[i], params_U2[i][0], params_U2[i][1])
        combined_kraus = [K2 @ K1 for K2 in K2_list for K1 in K1_list]
        chi = chi_from_kraus(combined_kraus, n=1, paulis_list=paulis_1q)
        PMs.append(chi)

    return PMs


# ───────────────────────────────────────────────
#                    Cost function
# ───────────────────────────────────────────────

def cost(params_F1, params_U1, params_F2, params_U2, params_convolutions, x, y, reg, state_labels=None):
    """Classification loss + spectral norm regularization on process matrices."""
    dm_labels = [density_matrix(s) for s in state_labels] if state_labels is not None else dm_labels

    loss = 0.0
    for i in range(len(x)):
        f = QCNN(params_F1, params_U1, params_F2, params_U2, params_convolutions, x[i], dm_labels[int(y[i])])
        loss += (1 - f) ** 2

    # Regularization: product of spectral norms of single-qubit channels
    PM_list = PM_reducidas(params_F1, params_U1, params_F2, params_U2, n=6)
    norm_prod = 1.0
    for PM in PM_list:
        norm_prod *= np.linalg.norm(PM)

    return loss / len(x) + reg * norm_prod


# ───────────────────────────────────────────────
#                    Training loop
# ───────────────────────────────────────────────

import csv
import os
from pennylane.optimize import AdamOptimizer

contador = 0
iteration = 0
n = 6

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(base_dir, "DPQC.csv")

# Initialize results file if needed
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "seed", "accuracy_train", "accuracy_test",
            "loss_train", "loss_test", "gen_error_accuracy", "gen_error_loss",
            "beta", "regularization", "n_per_Phase"
        ])


while contador < 500:
    print(f"Iteration: {iteration}")
    iteration += 1

    seed = random_seed + 11 * iteration
    np.random.seed(seed)

    X_train, y_train, n_phase = generate_cluster_dataset(N=n, n_samples_total=8, seed=seed)
    X_test, y_test, _ = generate_cluster_dataset(N=n, n_samples_total=1000, seed=seed + 99)

    learning_rate = 0.005
    epochs = 200
    batch_size = len(X_train)

    from pennylane import numpy as np
    np.random.seed(random_seed + 11 * iteration)

    sigma = 0.5
    reg = np.abs(np.random.normal(0, sigma))

    opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

    # Parameter initialization
    params_F1 = np.random.uniform(size=(6,), requires_grad=True)
    params_F2 = np.random.uniform(size=(6,), requires_grad=True)
    params_U1 = np.random.uniform(size=(6, 2), requires_grad=True)
    params_U2 = np.random.uniform(size=(6, 2), requires_grad=True)
    params_convolutions = np.random.uniform(size=15, requires_grad=True)

    # Initial performance
    predicted_train, _ = test(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_train, y_train)
    accuracy_train = accuracy_score(y_train, predicted_train)

    predicted_test, _ = test(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_test, y_test)
    accuracy_test = accuracy_score(y_test, predicted_test)

    loss = cost(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_train, y_train, reg)

    print(f"Epoch:  0 | Cost: {loss:.3f} | Train acc: {accuracy_train:.3f} | Test acc: {accuracy_test:.3f}")

    # Training
    for _ in range(epochs):
        for Xbatch, ybatch in iterate_minibatches(X_train, y_train, batch_size):
            params_F1, params_U1, params_F2, params_U2, params_convolutions, _, _, _, _ = opt.step(
                cost, params_F1, params_U1, params_F2, params_U2, params_convolutions,
                Xbatch, ybatch, reg
            )

    # Final evaluation
    predicted_train, _ = test(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_train, y_train)
    accuracy_train = accuracy_score(y_train, predicted_train)
    loss_train = cost(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_train, y_train, reg)

    predicted_test, _ = test(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_test, y_test)
    accuracy_test = accuracy_score(y_test, predicted_test)

    print(f"Epoch: {epochs} | Cost: {loss_train:.3f} | Train acc: {accuracy_train:.3f} | Test acc: {accuracy_test:.3f}")

    loss_test = cost(params_F1, params_U1, params_F2, params_U2, params_convolutions, X_test, y_test, reg)

    gen_error_accuracy = np.abs(accuracy_train - accuracy_test)
    gen_error_loss = np.abs(loss_test - loss_train)
    contador += 1

    # Compute regularization-related quantity (adjusted norm)
    PM_list = PM_reducidas(params_F1, params_U1, params_F2, params_U2, n)
    norm_prod = 1.0
    for PM in PM_list:
        norm_prod *= np.linalg.norm(PM)

    # Deviation from the maximally depolarizing
    W_norm = np.sqrt(norm_prod**2 - 1 / (4**n))

    # Save results
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            iteration,
            seed,
            accuracy_train,
            accuracy_test,
            loss_train,
            loss_test,
            gen_error_accuracy,
            gen_error_loss,
            W_norm,
            reg,
            n_phase
        ])