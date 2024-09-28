from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with 3 qubits (representing more levels) and 3 classical bits
qc_advanced = QuantumCircuit(3, 3)

# Apply Hadamard gates to all qubits
for i in range(3):
    qc_advanced.h(i)

# Apply CNOT gates to simulate more complex Plinko board interactions
qc_advanced.cx(0, 1)
qc_advanced.cx(1, 2)
qc_advanced.cx(2, 0)
qc_advanced.cx(1, 0)


# Measure both qubits
qc_advanced.measure([0, 1, 2], [0, 1, 2])

# Draw the circuit
qc_advanced.draw(output='mpl')
plt.show()

# Create a simulator
simulator = AerSimulator()

# Transpile the circuit for the simulator
from qiskit import transpile
transpiled_qc = transpile(qc_advanced, simulator)

# Run the circuit on the simulator
job = simulator.run(transpiled_qc, shots=1000)

# Get the results
result = job.result()
counts = result.get_counts()

# Plot the results
plot_histogram(counts)
plt.title("Advanced Quantum Plinko Distribution")
plt.xlabel("Bin")
plt.ylabel("Photon Count")
plt.show()