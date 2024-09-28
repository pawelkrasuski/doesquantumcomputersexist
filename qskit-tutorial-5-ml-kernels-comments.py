import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate ad hoc dataset
num_features = 2  # Number of features in the dataset
train_size = 20   # Number of training samples
test_size = 5     # Number of test samples
train_features, train_labels, test_features, test_labels = ad_hoc_data(
    training_size=train_size,
    test_size=test_size,
    n=num_features,
    gap=0.3  # Separation between classes
)

# Create quantum feature map
# ZZFeatureMap applies rotations and entangling operations to encode classical data into quantum states
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")

# Create quantum kernel
# Sampler is used to run quantum circuits and collect measurement results
sampler = Sampler()
# ComputeUncompute calculates fidelity between quantum states
fidelity = ComputeUncompute(sampler=sampler)
# FidelityQuantumKernel uses quantum circuits to compute kernel values
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# Create and train SVC (Support Vector Classifier) with quantum kernel
# The quantum kernel is used as the kernel function for the SVC
svc = SVC(kernel=quantum_kernel.evaluate)
svc.fit(train_features, train_labels[:, 0])

# Make predictions on test set
predictions = svc.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(test_labels[:, 0], predictions)

print(f"Test accuracy: {accuracy}")

# Print individual predictions
for i, (true_label, predicted_label) in enumerate(zip(test_labels[:, 0], predictions)):
    print(f"Sample {i+1}: True label = {true_label}, Predicted label = {predicted_label}")

