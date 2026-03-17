import numpy as np

# Fix random seed for reproducibility
random_seed = 0
np.random.seed(random_seed)

# Lists to store generalization metrics across runs (if used outside this script)
gen_errors_loss = []
gen_errors_accuracies = []
W_norms = []

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from sklearn.utils import shuffle


# ───────────────────────────────────────────────
#          Helper functions for Pauli operators
# ───────────────────────────────────────────────

def pauli_matrices():
    """Return single-qubit Pauli X, Z and identity as sparse matrices."""
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    I = csr_matrix(np.eye(2, dtype=complex))
    return X, Z, I


def kron_N(ops):
    """Compute sequential Kronecker product of a list of operators."""
    res = ops[0]
    for op in ops[1:]:
        res = kron(res, op, format='csr')
    return res


# ───────────────────────────────────────────────
#     Generalized periodic cluster Hamiltonian
# ───────────────────────────────────────────────

def cluster_hamiltonian(N, J1, J2):
    """
    Periodic cluster Hamiltonian:
        H = ∑ⱼ (Zⱼ − J₁ Xⱼ X_{j+1} − J₂ X_{j−1} Zⱼ X_{j+1})
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

        # Three-body cluster term X_{j-1} Z_j X_{j+1}
        ops = [I] * N
        ops[(j - 1) % N] = X
        ops[j] = Z
        ops[(j + 1) % N] = X
        H -= J2 * kron_N(ops)

    return H


def ground_state(H):
    """Compute normalized ground state vector (dense eigensolver)."""
    H_dense = H.toarray()
    vals, vecs = np.linalg.eigh(H_dense)
    gs = vecs[:, 0]
    gs /= np.linalg.norm(gs)
    return gs


# ───────────────────────────────────────────────
#           Dataset generation (4-phase diagram)
# ───────────────────────────────────────────────

def generate_cluster_dataset(N, n_samples_total, seed):
    """
    Generate dataset with random distribution across 4 quantum phases.
    Labels:
        0 : small J1, small J2
        1 : large J1, small J2
        2 : small J1, large J2
        3 : large J1, large J2
    """
    np.random.seed(seed)
    data, labels = [], []

    # Randomly distribute total samples among the 4 phases
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

    # Phase 0: small J1, small J2
    sample_phase((0.0, 0.5), (0.0, 0.5), n_per_phase[0], 0)

    # Phase 1: large J1, small J2
    sample_phase((1.0, 2.0), (0.0, 0.5), n_per_phase[1], 1)

    # Phase 2: small J1, large J2
    sample_phase((0.0, 0.5), (1.0, 2.0), n_per_phase[2], 2)

    # Phase 3: large J1, large J2
    sample_phase((1.0, 2.0), (1.0, 2.0), n_per_phase[3], 3)

    # Convert and shuffle
    data = np.array(data)
    labels = np.array(labels)
    data, labels = shuffle(data, labels, random_state=seed)

    return data, labels, n_per_phase


# Target density matrices (one-hot projectors in 4-dimensional output space)
dm_labels = [
    np.diag([1., 0., 0., 0.]),
    np.diag([0., 1., 0., 0.]),
    np.diag([0., 0., 1., 0.]),
    np.diag([0., 0., 0., 1.])
]


import pennylane as qml
from pennylane import numpy as np


# ───────────────────────────────────────────────
#            Quantum circuit building blocks
# ───────────────────────────────────────────────

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


def Upool(Upool_params, wires):
    """Parameterised controlled rotation for pooling."""
    qml.CRot(Upool_params[0], Upool_params[1], Upool_params[2], wires=[wires[1], wires[0]])


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
def QCNN(params_convolutions, params_poolings, params_F1, params_F2, params_F3, x, y):
    """Quantum Convolutional Neural Network classifier."""

    # Amplitude embedding of real part of ground state vector
    qml.AmplitudeEmbedding(x, wires=range(6))

    # ── Layer 1 ────────────────────────────────────────
    # Convolution layer (6 applications of U2)
    U2(params_convolutions[0], wires=[0,1])
    U2(params_convolutions[0], wires=[2,3])
    U2(params_convolutions[0], wires=[4,5])
    U2(params_convolutions[0], wires=[1,2])
    U2(params_convolutions[0], wires=[3,4])
    U2(params_convolutions[0], wires=[5,0])

    # Measurement-controlled F blocks (one per data qubit)
    F_prob(*params_F1[0], main_wire=0, ancilla_wire=6)
    F_prob(*params_F1[1], main_wire=1, ancilla_wire=7)
    F_prob(*params_F1[2], main_wire=2, ancilla_wire=8)
    F_prob(*params_F1[3], main_wire=3, ancilla_wire=9)
    F_prob(*params_F1[4], main_wire=4, ancilla_wire=10)
    F_prob(*params_F1[5], main_wire=5, ancilla_wire=11)

    # Pooling layer (reduces 6 → 3 qubits)
    Upool(params_poolings[0], wires=[0,1])
    Upool(params_poolings[0], wires=[2,3])
    Upool(params_poolings[0], wires=[4,5])

    # ── Layer 2 ────────────────────────────────────────
    # Convolution on remaining 3 qubits
    U2(params_convolutions[1], wires=[0,2])
    U2(params_convolutions[1], wires=[2,4])
    U2(params_convolutions[1], wires=[4,0])

    # Final F blocks
    F_prob(*params_F2[0], main_wire=0, ancilla_wire=12)
    F_prob(*params_F2[1], main_wire=2, ancilla_wire=13)
    F_prob(*params_F2[2], main_wire=4, ancilla_wire=14)

    F_prob(*params_F3[0], main_wire=0, ancilla_wire=15)
    F_prob(*params_F3[1], main_wire=2, ancilla_wire=16)
    F_prob(*params_F3[2], main_wire=4, ancilla_wire=17)

    # Final pooling
    Upool(params_poolings[1], wires=[0,4])
    Upool(params_poolings[1], wires=[2,4])

    # Fidelity with target density matrix (on logical output qubits 0 & 2)
    return qml.expval(qml.Hermitian(y, wires=[0,2]))


# ───────────────────────────────────────────────
#              Evaluation & training utilities
# ───────────────────────────────────────────────

def test(params_convolutions, params_poolings, params_F1, params_F2, params_F3, x, y, dm_labels):
    """Predict class via maximum fidelity with target density matrices."""
    fidelity_values = []
    predicted = []

    for i in range(len(x)):
        fidel_function = lambda dm: QCNN(params_convolutions, params_poolings, params_F1, params_F2, params_F3, x[i], dm)
        fidelities = [fidel_function(dm) for dm in dm_labels]
        best_idx = np.argmax(fidelities)
        predicted.append(best_idx)
        fidelity_values.append(fidelities)

    return np.array(predicted), np.array(fidelity_values)


def accuracy_score(y_true, y_pred):
    """Compute classification accuracy."""
    score = y_true == y_pred
    return score.sum() / len(y_true)


def iterate_minibatches(inputs, targets, batch_size):
    """Yield minibatches of data."""
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]


# ───────────────────────────────────────────────
#     Equivalent unitary & Kraus representations
# ───────────────────────────────────────────────

n_qubits_out1 = 3
dev2 = qml.device("default.qubit", wires=n_qubits_out1)


def circuito2(params_convolutions):
    """Second-layer convolution circuit for matrix extraction."""
    U2(params_convolutions[1], wires=[0,1])
    U2(params_convolutions[1], wires=[1,2])
    U2(params_convolutions[1], wires=[0,1])


def unitary_equivalente2(params_convolutions):
    """Compute unitary matrix of second-layer convolution."""
    with qml.tape.QuantumTape() as tape:
        circuito2(params_convolutions)
    return qml.matrix(tape, wire_order=range(n_qubits_out1))


# ───────────────────────────────────────────────
#              Kraus operators & channels
# ───────────────────────────────────────────────

def U(phi, varphi):
    """Single-qubit unitary U(φ,ψ)."""
    return np.array([
        [np.cos(phi), -np.exp(1j * varphi) * np.sin(phi)],
        [np.exp(-1j * varphi) * np.sin(phi), np.cos(phi)]
    ], dtype=complex)


def kraus_F(params):
    """Kraus operators for probabilistic F block."""
    theta, phi, varphi = params
    K0 = np.cos(theta / 2) * np.eye(2, dtype=complex)
    K1 = np.sin(theta / 2) * np.conj(U(phi, varphi))
    return [K0, K1]


def kraus_pooling(params):
    """Kraus operators for pooling operation."""
    phi, theta, omega = params
    I2 = np.eye(2, dtype=complex)
    bra0 = np.array([[1, 0]], dtype=complex)
    bra1 = np.array([[0, 1]], dtype=complex)

    R = np.array([
        [np.exp(-1j*(phi+omega)/2)*np.cos(theta/2), -np.exp(1j*(phi-omega)/2)*np.sin(theta/2)],
        [np.exp(-1j*(phi-omega)/2)*np.sin(theta/2), np.exp(1j*(phi+omega)/2)*np.cos(theta/2)]
    ], dtype=complex)

    K0 = np.kron(I2, bra0)
    K1 = np.kron(R, bra1)
    return [K0, K1]


def kraus_pooling_F_block(params_pool, params_F_target, params_F_control):
    """Combined Kraus operators for pooling + two F blocks."""
    K_F_t = kraus_F(params_F_target)
    K_F_c = kraus_F(params_F_control)
    K_pool = kraus_pooling(params_pool)
    total_kraus = [K_p @ np.kron(K_t, K_c) for K_t in K_F_t for K_c in K_F_c for K_p in K_pool]
    return total_kraus


def kraus_F_combined(params_F2, params_F3):
    """Combined Kraus for sequential F2 and F3 on same qubit."""
    K2 = kraus_F(params_F2)
    K3 = kraus_F(params_F3)
    return [k3 @ k2 for k3 in K3 for k2 in K2]


def get_pooling_matrix(params):
    """Unitary matrix of final pooling layer (for PTM calculation)."""
    with qml.tape.QuantumTape() as tape:
        Upool(params, wires=[0,4])
        Upool(params, wires=[2,4])
    return qml.matrix(tape, wire_order=[0,2,4])


# ───────────────────────────────────────────────
#           Pauli basis & process matrices
# ───────────────────────────────────────────────

paulis = [
    np.eye(2),
    np.array([[0,1],[1,0]]),
    np.array([[0,-1j],[1j,0]]),
    np.array([[1,0],[0,-1]])
]  # I, X, Y, Z


def pauli_strings(n, normalized=False):
    """Generate all n-qubit Pauli strings (tensor products)."""
    from itertools import product
    indices = list(product(range(4), repeat=n))
    strings = [multi_kron([paulis[i] for i in idx]) for idx in indices]
    if normalized:
        factor = 1 / np.sqrt(2**n)
        strings = [factor * s for s in strings]
    return strings


def multi_kron(mats):
    """Kronecker product of multiple matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def apply_channel(rho, K_list):
    """Apply quantum channel defined by Kraus operators."""
    result = np.zeros((K_list[0].shape[0], K_list[0].shape[0]), dtype=complex)
    for K in K_list:
        result += K @ rho @ np.conj(K.T)
    return result


# ───────────────────────────────────────────────
#     Process matrix / PTM construction helpers
# ───────────────────────────────────────────────

def W_from_kraus_build(K_list, n_in, n_out, Sigma_A, Sigma_B):
    """Build process matrix W from Kraus operators (PTM-like)."""
    d_in = 2**n_in
    d_out = 2**n_out
    rows = []
    for i, A in enumerate(Sigma_A):
        row = []
        for j, B in enumerate(Sigma_B):
            phi_B = apply_channel(B, K_list)
            val = np.trace(A @ phi_B) / np.sqrt(d_in * d_out)
            if i == 0 and j == 0:
                row.append(val - np.sqrt(d_in / d_out))
            else:
                row.append(val)
        rows.append(row)
    R = np.array(rows, dtype=complex)
    return np.real(R)


def reduced_ptm_from_kraus(K_block):
    """Reduced PTM for a 2→1 qubit channel."""
    d_in = 2**2
    d_out = 2**1
    Sigma_A = pauli_strings(1)
    Sigma_B = pauli_strings(2)
    rows = []
    for i, A in enumerate(Sigma_A):
        row = []
        for j, B in enumerate(Sigma_B):
            phi_B = apply_channel(B, K_block)
            val = np.trace(A @ phi_B) / np.sqrt(d_in * d_out)
            row.append(val)
        rows.append(row)
    R = np.array(rows, dtype=complex)
    return np.real(R)


def PTM_reducidas_layer1(params_pooling, params_F, n):
    """Compute reduced PTMs for each pooling+F block in layer 1."""
    PTMs = []
    for i in range(n//2):
        params_F_t = params_F[2*i]
        params_F_c = params_F[2*i + 1]
        kraus_block = kraus_pooling_F_block(params_pooling, params_F_t, params_F_c)
        PTMs.append(reduced_ptm_from_kraus(kraus_block))
    return PTMs


def compute_ptm_L2(params_convolutions, params_poolings, params_F2, params_F3, include_conv, subtract_id):
    """Compute process matrix / PTM contribution of layer 2."""
    Sigma_A = pauli_strings(2)
    Sigma_B = pauli_strings(3)
    U_pool = get_pooling_matrix(params_poolings[1])
    F_comb = [kraus_F_combined(params_F2[k], params_F3[k]) for k in range(3)]

    if include_conv:
        U_conv = unitary_equivalente2(params_convolutions)
    else:
        U_conv = np.eye(8, dtype=complex)

    d_in = 8
    d_out = 4
    s = np.sqrt(d_in / d_out)
    rows = []

    for i, A in enumerate(Sigma_A):
        row = []
        for j, B in enumerate(Sigma_B):
            rho = np.dot(U_conv, np.dot(B, np.conjugate(U_conv).T)) if include_conv else B

            # Apply F blocks on wires 0,2,4
            kraus_f0 = [np.kron(k, np.eye(4, dtype=complex)) for k in F_comb[0]]
            rho = apply_channel(rho, kraus_f0)

            kraus_f2 = [np.kron(np.eye(2), np.kron(k, np.eye(2))) for k in F_comb[1]]
            rho = apply_channel(rho, kraus_f2)

            kraus_f4 = [np.kron(np.eye(4), k) for k in F_comb[2]]
            rho = apply_channel(rho, kraus_f4)

            rho = np.dot(U_pool, np.dot(rho, np.conjugate(U_pool).T))

            # Partial trace over ancilla qubits
            rho_reshaped = rho.reshape(4, 2, 4, 2)
            rho_out = rho_reshaped[:, 0, :, 0] + rho_reshaped[:, 1, :, 1]

            val = np.trace(np.dot(A, rho_out)) / np.sqrt(d_in * d_out)
            if subtract_id and i == 0 and j == 0:
                row.append(val - s)
            else:
                row.append(val)
        rows.append(row)

    return np.array(rows)


# ───────────────────────────────────────────────
#                    Cost function
# ───────────────────────────────────────────────

def cost(params_convolutions, params_poolings, params_F1, params_F2, params_F3, x, y, reg, dm_labels):
    """Loss + regularization term (generalization bound related)."""
    loss = 0.0
    for i in range(len(x)):
        f = QCNN(params_convolutions, params_poolings, params_F1, params_F2, params_F3, x[i], dm_labels[int(y[i])])
        loss += (1 - f) ** 2

    # Regularization term (β)
    W_L2 = compute_ptm_L2(params_convolutions, params_poolings, params_F2, params_F3,
                          include_conv=True, subtract_id=True)
    beta = np.sqrt(2**2 / 2**6) * np.sum(np.abs(W_L2)) + np.sqrt(2**2 / 2**3)

    return loss / len(x) + reg * beta


# ───────────────────────────────────────────────
#                    Training loop
# ───────────────────────────────────────────────

import csv
import os
from pennylane.optimize import AdamOptimizer

contador = 0
iteration = 0

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(base_dir, "QCNN.csv")

# Create CSV file with header if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "seed", "accuracy_train", "accuracy_test",
            "loss_train", "loss_test", "gen_error_accuracy", "gen_error_loss",
            "beta", "frobenius1", "frobenius2", "bound", "regularization",
            "n_per_Phase"
        ])

while contador < 500:
    print(f"Iteration: {iteration}")
    iteration += 1

    seed = random_seed + 11 * iteration
    np.random.seed(seed)

    # Generate small training and large test set
    X_train, y_train, n_phase = generate_cluster_dataset(N=6, n_samples_total=8, seed=seed)
    X_test, y_test, _ = generate_cluster_dataset(N=6, n_samples_total=1000, seed=seed + 99)

    learning_rate = 0.005
    epochs = 200
    batch_size = len(X_train)

    from pennylane import numpy as np
    np.random.seed(random_seed + 11 * iteration)

    sigma = 4
    reg = np.abs(np.random.normal(0, sigma))

    opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

    # Random parameter initialization
    params_convolutions = np.random.uniform(size=(2, 15), requires_grad=True)
    params_poolings    = np.random.uniform(size=(2, 3), requires_grad=True)
    params_F1          = np.random.uniform(size=(6, 3), requires_grad=True)
    params_F2          = np.random.uniform(size=(3, 3), requires_grad=True)
    params_F3          = np.random.uniform(size=(3, 3), requires_grad=True)

    # Initial evaluation
    predicted_train, fidel_train = test(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_train, y_train, dm_labels)
    accuracy_train = accuracy_score(y_train, predicted_train)

    predicted_test, fidel_test = test(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_test, y_test, dm_labels)
    accuracy_test = accuracy_score(y_test, predicted_test)

    loss = cost(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_train, y_train, reg, dm_labels)

    print(f"Epoch:  0 | Cost: {loss:.3f} | Train acc: {accuracy_train:.3f} | Test acc: {accuracy_test:.3f}")

    # Training loop
    for it in range(epochs):
        for Xbatch, ybatch in iterate_minibatches(X_train, y_train, batch_size):
            params_convolutions, params_poolings, params_F1, params_F2, params_F3, _, _, _, _ = opt.step(
                cost, params_convolutions, params_poolings, params_F1, params_F2, params_F3,
                Xbatch, ybatch, reg, dm_labels
            )

    # Final evaluation
    predicted_train, fidel_train = test(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_train, y_train, dm_labels)
    accuracy_train = accuracy_score(y_train, predicted_train)
    loss_train = cost(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_train, y_train, reg, dm_labels)

    predicted_test, fidel_test = test(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_test, y_test, dm_labels)
    accuracy_test = accuracy_score(y_test, predicted_test)

    print(f"Epoch: {epochs} | Cost: {loss_train:.3f} | Train acc: {accuracy_train:.3f} | Test acc: {accuracy_test:.3f}")

    # Save results
    loss_test = cost(params_convolutions, params_poolings, params_F1, params_F2, params_F3, X_test, y_test, reg, dm_labels)

    gen_error_accuracy = np.abs(accuracy_train - accuracy_test)
    gen_error_loss = np.abs(loss_test - loss_train)
    contador += 1

    # Compute generalization bound quantities
    W_L2 = compute_ptm_L2(params_convolutions, params_poolings, params_F2, params_F3, include_conv=True, subtract_id=True)
    beta = np.sqrt(2**2 / 2**6) * np.sum(np.abs(W_L2)) + np.sqrt(2**2 / 2**3)

    PTM_list_1 = PTM_reducidas_layer1(params_poolings[0], params_F1, n=6)
    norm_L1 = 1.0
    for PTM in PTM_list_1:
        norm_L1 *= np.linalg.norm(PTM)
    norm_L1 = np.sqrt(norm_L1**2 - (2**6 / 2**3))

    R_L2 = compute_ptm_L2(params_convolutions, params_poolings, params_F2, params_F3, include_conv=False, subtract_id=False)
    norm_L2 = np.sqrt(np.linalg.norm(R_L2)**2 - (2**3 / 2**2))

    bound = beta * np.sqrt(norm_L1**2 + norm_L2**2)

    # Write to CSV
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
            beta,
            norm_L1,
            norm_L2,
            bound,
            reg,
            n_phase
        ])