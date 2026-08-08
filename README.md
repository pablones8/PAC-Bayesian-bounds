# A PAC-Bayesian approach to generalization for quantum models

This repository contains the implementation and numerical results for the experiments presented in the article "A PAC-Bayesian approach to generalization in quantum models", available on [arXiv](https://arxiv.org/pdf/2603.22964).

The code computes generalization errors of two quantum machine learning architectures for classifying ground states of a generalized cluster Hamiltonian into four distinct quantum phases:

- **QCNN** — Quantum Convolutional Neural Network: Operates on six data qubits. The first layer applies shared-parameter two-qubit unitaries, one dynamic block per qubit, and a pooling operation (controlled rotation followed by partial trace), reducing the register from six to three qubits. The second layer repeats the same structure: shared two-qubit unitaries, two dynamic blocks per qubit, and pooling of one qubit, reducing from three to two qubits. The final two qubits are measured to determine the label of the four-class classification task.
- **DPQC** — Dynamic Parameterized Quantum Circuit: Operates on four data qubits. The first layer applies one dynamic block per qubit followed by a convolution of parameterized two-qubit unitaries. The second layer applies two consecutive dynamic blocks per qubit. Two of the four qubits are measured to assign the class label.

Both models are trained to distinguish phases characterized by different regimes of the coupling parameters J₁ and J₂.

## Files

- **`QCNN.py`**  
  Main script that implements and trains the QCNN architecture.  
  Performs 500 independent training runs with different random seeds.  
  Saves training/test accuracy, loss, generalization gap and regularization metrics to `QCNN.csv`.

- **`DPQC.py`**  
  Main script that implements and trains the DPQC architecture.  
  Similarly runs 500 independent experiments and saves results to `DPQC.csv`.

- **`QCNN.csv`** & **`DPQC.csv`**  
  Results of the X training runs for each architecture. All points have training accuracy $\geq$0.75.
  Columns include: iteration, seed, train/test accuracy & loss, generalization errors, regularization terms (β or norm-based), and phase sample distribution.


## Requirements

- Python 3.8+
- PennyLane (with default.qubit simulator)
- NumPy, SciPy, scikit-learn

