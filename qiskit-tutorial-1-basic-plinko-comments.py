from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with 2 qubits (photons) and 2 classical bits
# This represents our quantum Plinko board with two levels
qc = QuantumCircuit(2, 2)

# Apply Hadamard gates to put both qubits in superposition
# This simulates the initial random direction of the photons
qc.h(0)
qc.h(1)

# Apply CNOT gates to simulate the Plinko board interactions
# CNOT gates create entanglement between qubits, representing photon collisions
qc.cx(0, 1)  # First level interaction
qc.cx(1, 0)  # Second level interaction

# Measure both qubits
# This collapses the quantum state, simulating the final position of photons
qc.measure([0, 1], [0, 1])

# Draw the circuit for visualization
qc.draw(output='mpl')
plt.show()

# Create a simulator to run our quantum circuit
simulator = AerSimulator()

# Transpile the circuit for the simulator
# This optimizes the circuit for the specific backend
transpiled_qc = transpile(qc, simulator)

# Run the circuit on the simulator
# We use 1000 shots to get a statistical distribution of outcomes
job = simulator.run(transpiled_qc, shots=1000)

# Get the results of the simulation
result = job.result()
counts = result.get_counts()

# Plot the results as a histogram
# This shows the distribution of photons in different bins
plot_histogram(counts)
plt.title("Quantum Plinko Distribution")
plt.xlabel("Bin (Qubit States)")
plt.ylabel("Photon Count")
plt.show()