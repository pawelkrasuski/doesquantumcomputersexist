import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
#from qiskit.utils import algorithm_globals
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
#algorithm_globals.random_seed = 12345

# Generate ad hoc dataset
num_features = 2
train_size = 20
test_size = 5
train_features, train_labels, test_features, test_labels = ad_hoc_data(
    training_size=train_size,
    test_size=test_size,
    n=num_features,
    gap=0.3
)



# Create quantum feature map
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")

# Create quantum kernel
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)


# Create and train SVC with quantum kernel
svc = SVC(kernel=quantum_kernel.evaluate)
svc.fit(train_features, train_labels[:, 0])

# Make predictions on test set
predictions = svc.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(test_labels[:, 0], predictions)

print(f"Test accuracy: {accuracy}")

# Optionally, print individual predictions
for i, (true_label, predicted_label) in enumerate(zip(test_labels[:, 0], predictions)):
    print(f"Sample {i+1}: True label = {true_label}, Predicted label = {predicted_label}")