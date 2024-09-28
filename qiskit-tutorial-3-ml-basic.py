from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import numpy as np

# Create a simple binary classification dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

num_qubits = 2
feature_map = ZZFeatureMap(num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

# Create a sampler
sampler = StatevectorSampler()

# Create the VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=SPSA(maxiter=100),
    sampler=sampler
)

# Fit the model
vqc.fit(X, y)

# Make predictions
predictions = vqc.predict(X)
print("Predictions:", predictions)