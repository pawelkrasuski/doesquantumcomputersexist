from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

# Set up the problem parameters
feature_dim = 2  # Dimension of each data point
training_size = 20  # Number of training samples
test_size = 10  # Number of test samples

# Generate ad hoc dataset for binary classification
# The dataset is created with a specified gap between classes for easier separation
training_features, training_labels, test_features, test_labels = ad_hoc_data(
    training_size=training_size, test_size=test_size, n=feature_dim, gap=0.3
)

# Define the feature map
# ZZFeatureMap is used to encode classical data into quantum states
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear")

# Define the variational form (ansatz)
# TwoLocal creates a circuit of alternating rotation and entanglement layers
ansatz = TwoLocal(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3)

# Create the Variational Quantum Classifier (VQC)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=SPSA(maxiter=100),  # SPSA is a gradient-free optimizer suitable for noisy functions
)

# Train the VQC model
vqc.fit(training_features, training_labels)

# Evaluate the model on the test set
score = vqc.score(test_features, test_labels)
print(f"Testing accuracy: {score:0.2f}")