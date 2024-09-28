from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with 3 qubits and 3 classical bits
# This represents a more complex Plinko board with 8 possible outcomes (2^3)
qc_advanced = QuantumCircuit(3, 3)

# Apply Hadamard gates to all qubits
# This puts each qubit in a superposition state, simulating the initial drop of the ball
for i in range(3):
    qc_advanced.h(i)

# Apply CNOT gates to simulate more complex Plinko board interactions
# These gates create entanglement between qubits, representing the interdependence of paths
qc_advanced.cx(0, 1)  # Control: qubit 0, Target: qubit 1
qc_advanced.cx(1, 2)  # Control: qubit 1, Target: qubit 2
qc_advanced.cx(2, 0)  # Control: qubit 2, Target: qubit 0 (creating a cycle)
qc_advanced.cx(1, 0)  # Additional interaction between qubits 1 and 0

# Measure all qubits
# This collapses the superposition and determines the final bin
qc_advanced.measure([0, 1, 2], [0, 1, 2])

# Draw the circuit for visualization
qc_advanced.draw(output='mpl')
plt.show()

# Create a simulator to run the quantum circuit
simulator = AerSimulator()

# Transpile the circuit for the simulator
# This optimizes the circuit for the specific backend
from qiskit import transpile
transpiled_qc = transpile(qc_advanced, simulator)

# Run the circuit on the simulator
# 1000 shots simulate dropping 1000 balls through the Plinko board
job = simulator.run(transpiled_qc, shots=1000)

# Get the results of the simulation
result = job.result()
counts = result.get_counts()

# Plot the results as a histogram
# This shows the distribution of outcomes, analogous to the final bin positions in Plinko
plot_histogram(counts)
plt.title("Advanced Quantum Plinko Distribution")
plt.xlabel("Bin (Final Qubit State)")
plt.ylabel("Photon Count (Number of Balls)")
plt.show()