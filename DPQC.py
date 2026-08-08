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


# Target density matrices used for fidelity comparison (2 measurement qubits → 4x4)
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
    state = np.array(state).flatten()
    return np.outer(state, np.conj(state))


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
#          Main 2-Layer DPQC circuit (4 qubits)
# ───────────────────────────────────────────────
# 4 data qubits (0–3)
# Layer 1 ancillas: 4 qubits (4–7)   — one F-block per qubit
# Layer 2 ancillas: 4 qubits (8–11)  — first F-block per qubit
#                   4 qubits (12–15)  — second F-block per qubit
# Total: 16 wires

n = 4  # number of data qubits
dev = qml.device("default.qubit", wires=16)


@qml.qnode(dev)
def DPQC(params_F1, params_U1, params_conv,
         params_F2a, params_U2a, params_F2b, params_U2b,
         x, y):
    """
    2-layer DPQC with amplitude embedding.
    Layer 1: one F-block per qubit + convolution.
    Layer 2: two consecutive F-blocks per qubit.
    """
    # Embed real part of ground state vector (2^4 = 16 amplitudes)
    qml.AmplitudeEmbedding(x, wires=range(n))

    # ─── Layer 1: F-blocks + Convolution ───
    for i in range(n):
        F_prob(params_F1[i], params_U1[i][0], params_U1[i][1],
               main_wire=i, ancilla_wire=4 + i)

    # Convolution on 4 qubits
    U2(params_conv, wires=[0, 1])
    U2(params_conv, wires=[2, 3])
    U2(params_conv, wires=[1, 2])

    # ─── Layer 2: Two consecutive F-blocks ───
    # First F-block sublayer (ancilla wires 8–11)
    for i in range(n):
        F_prob(params_F2a[i], params_U2a[i][0], params_U2a[i][1],
               main_wire=i, ancilla_wire=8 + i)

    # Second F-block sublayer (ancilla wires 12–15)
    for i in range(n):
        F_prob(params_F2b[i], params_U2b[i][0], params_U2b[i][1],
               main_wire=i, ancilla_wire=12 + i)

    # Fidelity measurement on logical output qubits [1, 2]
    return qml.expval(qml.Hermitian(y, wires=[1, 2]))


# ───────────────────────────────────────────────
#              Evaluation & training helpers
# ───────────────────────────────────────────────

def test(params_F1, params_U1, params_conv,
         params_F2a, params_U2a, params_F2b, params_U2b,
         x, y, state_labels=None):
    """Predict class by maximum fidelity with target states."""
    dm_labels_loc = [density_matrix(s) for s in state_labels] if state_labels is not None else dm_labels
    predicted = []
    fidelity_values = []

    for xi in x:
        fidelities = [DPQC(params_F1, params_U1, params_conv,
                           params_F2a, params_U2a, params_F2b, params_U2b,
                           xi, dm) for dm in dm_labels_loc]
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


def PM_layer_single(params_F, params_U, n_qubits):
    """
    Compute single-qubit process matrices for ONE F-block layer.
    Returns list of per-qubit chi matrices (4x4 each).
    """
    PMs = []
    paulis_1q = pauli_strings_list(1)

    for i in range(n_qubits):
        K_list = single_qubit_kraus(params_F[i], params_U[i][0], params_U[i][1])
        chi = chi_from_kraus(K_list, n=1, paulis_list=paulis_1q)
        PMs.append(chi)

    return PMs


def PM_layer_composed(params_Fa, params_Ua, params_Fb, params_Ub, n_qubits):
    """
    Compute single-qubit process matrices for TWO consecutive F-block layers.
    The composed channel per qubit has Kraus operators:
        {K_j^(b) @ K_i^(a)} for all i, j
    Returns list of per-qubit chi matrices (4x4 each).
    """
    PMs = []
    paulis_1q = pauli_strings_list(1)

    for i in range(n_qubits):
        # Kraus operators for first and second F-block
        K_a = single_qubit_kraus(params_Fa[i], params_Ua[i][0], params_Ua[i][1])
        K_b = single_qubit_kraus(params_Fb[i], params_Ub[i][0], params_Ub[i][1])

        # Composed Kraus operators: E_b ∘ E_a
        K_composed = [Kb @ Ka for Kb in K_b for Ka in K_a]

        chi = chi_from_kraus(K_composed, n=1, paulis_list=paulis_1q)
        PMs.append(chi)

    return PMs


def compute_W_F_sq(chi_list, n_qubits):
    """
    Compute ||W||_F^2 for n qubits from per-qubit chi matrices.
    ||chi_full||_F^2 = prod_i ||chi_i||_F^2  (tensor product)
    ||W||_F^2 = ||chi_full||_F^2 - 1/4^n
    """
    chi_F_sq = 1.0
    for chi in chi_list:
        chi_F_sq *= np.linalg.norm(chi) ** 2
    return chi_F_sq - 1.0 / (4 ** n_qubits)


def compute_W_11_norm(chi_list, n_qubits):
    """
    Compute the exact ||W_full||_{1,1} for an n-qubit tensor-product channel.
    W = chi_full - I/4^n, so ||W||_{1,1} = sum_{A,B} |chi_full(A,B) - delta_{A,B}/4^n|.

    For tensor products this decomposes as:
      off-diagonal: prod_i ||chi_i||_{1,1} - 1
      diagonal correction: sum_A |prod_i chi_i(a_i,a_i) - 1/4^n|
    """
    # Off-diagonal contribution: prod_i ||chi_i||_{1,1} - 1
    chi_11_prod = 1.0
    for chi in chi_list:
        chi_11_prod *= np.sum(np.abs(chi))
    off_diag = chi_11_prod - 1.0

    # Diagonal correction: sum_A |prod_i chi_i(a_i,a_i) - 1/4^n|
    diags = [np.real(np.diag(chi)) for chi in chi_list]

    # Kronecker product of diagonals gives all prod_i chi_i(a_i,a_i)
    d_full = diags[0]
    for d in diags[1:]:
        d_full = np.outer(d_full, d).ravel()

    diag_correction = np.sum(np.abs(d_full - 1.0 / (4 ** n_qubits)))

    return off_diag + diag_correction


# ───────────────────────────────────────────────
#                    Cost function
# ───────────────────────────────────────────────

def complexity_term(params_F1, params_U1, params_F2a, params_U2a,
                    params_F2b, params_U2b, n_qubits):
    """
    Complexity term for 2-layer DPQC (used as regularizer during training).

    Layer 1: single F-block → ||W_1||_F^2 (Frobenius invariant under conv)
    Layer 2: two consecutive F-blocks → ||W_2||_F^2 and ||W_2||_{1,1}

    beta_PM = 1 + ||W_2||_{1,1}  (for L=2)
    complexity = beta_PM^2 * (||W_1||_F^2 + ||W_2||_F^2)
    """
    # Layer 1: single F-block per qubit
    PM_L1 = PM_layer_single(params_F1, params_U1, n_qubits)

    # Layer 2: two composed F-blocks per qubit
    PM_L2 = PM_layer_composed(params_F2a, params_U2a, params_F2b, params_U2b, n_qubits)

    W1_F_sq = compute_W_F_sq(PM_L1, n_qubits)
    W2_F_sq = compute_W_F_sq(PM_L2, n_qubits)

    W2_11 = compute_W_11_norm(PM_L2, n_qubits)

    # beta_PM for L=2: ||W_2||_{1,1} + 1
    beta_PM = W2_11 + 1.0

    return beta_PM, W1_F_sq, W2_F_sq


def cost(params_F1, params_U1, params_conv,
         params_F2a, params_U2a, params_F2b, params_U2b,
         x, y, reg, state_labels=None):
    """Classification loss + complexity-based regularization for 2-layer DPQC."""
    dm_labels_loc = [density_matrix(s) for s in state_labels] if state_labels is not None else dm_labels

    loss = 0.0
    for i in range(len(x)):
        f = DPQC(params_F1, params_U1, params_conv,
                 params_F2a, params_U2a, params_F2b, params_U2b,
                 x[i], dm_labels_loc[int(y[i])])
        loss += (1 - f) ** 2

    # Regularization: beta^2 * (||W_1||_F^2 + ||W_2||_F^2)
    beta_PM, W1_F_sq, W2_F_sq = complexity_term(
        params_F1, params_U1, params_F2a, params_U2a,
        params_F2b, params_U2b, n_qubits=n)

    return loss / len(x) + reg * beta_PM ** 2 * (W1_F_sq + W2_F_sq)


# ───────────────────────────────────────────────
#                    Training loop
# ───────────────────────────────────────────────

import csv
import os
from pennylane.optimize import AdamOptimizer

contador = 0
iteration = 0

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(base_dir, "DPQC_2L_results.csv")

# Initialize results file if needed
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "seed", "accuracy_train", "accuracy_test",
            "loss_train", "loss_test", "gen_error_accuracy", "gen_error_loss",
            "beta_PM", "W1_F_sq", "W2_F_sq", "complexity",
            "regularization", "n_per_Phase"
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
    # Layer 1: one F-block per qubit
    params_F1 = np.random.uniform(size=(n,), requires_grad=True)
    params_U1 = np.random.uniform(size=(n, 2), requires_grad=True)
    params_conv = np.random.uniform(size=15, requires_grad=True)

    # Layer 2: two consecutive F-blocks per qubit
    params_F2a = np.random.uniform(size=(n,), requires_grad=True)
    params_U2a = np.random.uniform(size=(n, 2), requires_grad=True)
    params_F2b = np.random.uniform(size=(n,), requires_grad=True)
    params_U2b = np.random.uniform(size=(n, 2), requires_grad=True)

    # Initial performance
    predicted_train, _ = test(params_F1, params_U1, params_conv,
                              params_F2a, params_U2a, params_F2b, params_U2b,
                              X_train, y_train)
    accuracy_train = accuracy_score(y_train, predicted_train)

    predicted_test, _ = test(params_F1, params_U1, params_conv,
                             params_F2a, params_U2a, params_F2b, params_U2b,
                             X_test, y_test)
    accuracy_test = accuracy_score(y_test, predicted_test)

    loss = cost(params_F1, params_U1, params_conv,
                params_F2a, params_U2a, params_F2b, params_U2b,
                X_train, y_train, reg)

    print(f"Epoch:  0 | Cost: {loss:.3f} | Train acc: {accuracy_train:.3f} | Test acc: {accuracy_test:.3f}")

    # Training
    for _ in range(epochs):
        for Xbatch, ybatch in iterate_minibatches(X_train, y_train, batch_size):
            Xbatch = np.array(Xbatch, requires_grad=False)
            ybatch = np.array(ybatch, requires_grad=False)
            (params_F1, params_U1, params_conv,
             params_F2a, params_U2a, params_F2b, params_U2b,
             _, _, _) = opt.step(
                cost, params_F1, params_U1, params_conv,
                params_F2a, params_U2a, params_F2b, params_U2b,
                Xbatch, ybatch, reg
            )

    # Final evaluation
    predicted_train, _ = test(params_F1, params_U1, params_conv,
                              params_F2a, params_U2a, params_F2b, params_U2b,
                              X_train, y_train)
    accuracy_train = accuracy_score(y_train, predicted_train)
    loss_train = cost(params_F1, params_U1, params_conv,
                      params_F2a, params_U2a, params_F2b, params_U2b,
                      X_train, y_train, reg)

    predicted_test, _ = test(params_F1, params_U1, params_conv,
                             params_F2a, params_U2a, params_F2b, params_U2b,
                             X_test, y_test)
    accuracy_test = accuracy_score(y_test, predicted_test)

    print(f"Epoch: {epochs} | Cost: {loss_train:.3f} | Train acc: {accuracy_train:.3f} | Test acc: {accuracy_test:.3f}")

    loss_test = cost(params_F1, params_U1, params_conv,
                     params_F2a, params_U2a, params_F2b, params_U2b,
                     X_test, y_test, reg)

    gen_error_accuracy = np.abs(accuracy_train - accuracy_test)
    gen_error_loss = np.abs(loss_test - loss_train)
    contador += 1

    # Compute complexity term for logging
    beta_PM, W1_F_sq, W2_F_sq = complexity_term(
        params_F1, params_U1, params_F2a, params_U2a,
        params_F2b, params_U2b, n_qubits=n)
    complexity_val = beta_PM ** 2 * (W1_F_sq + W2_F_sq)

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
            beta_PM,
            W1_F_sq,
            W2_F_sq,
            complexity_val,
            reg,
            n_phase
        ])
