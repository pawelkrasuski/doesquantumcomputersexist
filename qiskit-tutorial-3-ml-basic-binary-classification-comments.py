from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import numpy as np

# Create a simple binary classification dataset
# This dataset represents the XOR problem: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Define the number of qubits based on the input features
num_qubits = 2

# ZZFeatureMap is used for encoding classical data into quantum states
# It applies Hadamard gates followed by ZZ entangling gates
feature_map = ZZFeatureMap(num_qubits)

# RealAmplitudes ansatz applies rotation gates (RY) and entangling gates (CX)
# 'reps=1' means one repetition of the ansatz circuit
ansatz = RealAmplitudes(num_qubits, reps=1)

# Create a sampler for measuring quantum states
# This replaces the older 'backend' approach in Qiskit
sampler = Sampler()

# Create the Variational Quantum Classifier (VQC)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    # SPSA (Simultaneous Perturbation Stochastic Approximation) is used for optimization
    # It's particularly useful for noisy functions, like those in quantum circuits
    optimizer=SPSA(maxiter=100),
    sampler=sampler
)

# Train the VQC model using the XOR dataset
vqc.fit(X, y)

# Use the trained model to make predictions on the same dataset
# In a real scenario, you'd typically use a separate test set
predictions = vqc.predict(X)
print("Predictions:", predictions)
# Ideally, predictions should match the original y values for perfect classification