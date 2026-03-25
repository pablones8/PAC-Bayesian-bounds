# A PAC-Bayesian approach to generalization for quantum models

This repository contains the implementation and numerical results for the experiments presented in the article "A PAC-Bayesian approach to generalization in quantum models", available on [arXiv](https://arxiv.org/pdf/2603.22964).

The code computes generalization errors of two quantum machine learning architectures for classifying ground states of a generalized cluster Hamiltonian into four distinct quantum phases:

- **QCNN** — Quantum Convolutional Neural Network (with amplitude embedding + convolution + pooling layers)
- **DPQC** — Dynamic Parameterized Quantum Circuit (sequential conditional single-qubit operations controlled by measurements + final convolution)

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
  Results of the X training runs for each architecture.  
  Columns include: iteration, seed, train/test accuracy & loss, generalization errors, regularization terms (β or norm-based), and phase sample distribution.


## Requirements

- Python 3.8+
- PennyLane (with default.qubit simulator)
- NumPy, SciPy, scikit-learn

