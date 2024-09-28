from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with 2 qubits (photons) and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply Hadamard gates to put both qubits in superposition
qc.h(0)
qc.h(1)

# Apply CNOT gates to simulate the Plinko board interactions
qc.cx(0, 1)
qc.cx(1, 0)

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Draw the circuit
qc.draw(output='mpl')
plt.show()

# Create a simulator
simulator = AerSimulator()

# Transpile the circuit for the simulator
from qiskit import transpile
transpiled_qc = transpile(qc, simulator)

# Run the circuit on the simulator
job = simulator.run(transpiled_qc, shots=1000)

# Get the results
result = job.result()
counts = result.get_counts()

# Plot the results
plot_histogram(counts)
plt.title("Quantum Plinko Distribution")
plt.xlabel("Bin")
plt.ylabel("Photon Count")
plt.show()