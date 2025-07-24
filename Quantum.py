#------------
import numpy as np

import cirq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



tetrahedral_directions = [
    np.array([1, 1, 1]),
    np.array([1, -1, -1]),
    np.array([-1, 1, -1]),
    np.array([-1, -1, 1])
]

def build_hamiltonian(qubits, sequence):
    n = len(sequence)
    terms = []

    for i in range(n):
        for j in range(i + 2, n):
            if sequence[i] == 'H' and sequence[j] == 'H':
                terms.append((cirq.Z(qubits[i]) * cirq.Z(qubits[j]), -1.5))

    return terms

def build_circuit(qubits, params):
    circuit = cirq.Circuit()
    n = len(qubits)

    # 1. Apply Hadamard to all qubits
    for q in qubits:
        circuit.append(cirq.H(q))

    # 2. Add CNOT entanglement in a ring
    for i in range(n - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    # Optionally close the ring
    if n > 2:
        circuit.append(cirq.CNOT(qubits[-1], qubits[0]))

    # 3. Parameterized RX layer
    for i, q in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(q))

    return circuit

#------------------------------------
def expectation(params, qubits, hamiltonian, sim, sequence):  # CHANGED
    circuit = build_circuit(qubits, params)
    circuit.append(cirq.measure(*qubits, key='m'))
    result = sim.run(circuit, repetitions=100)

    bitstrings = []
    for i in range(100):
        bits = ''.join(str(result.measurements['m'][i][j]) for j in range(len(qubits)))
        bitstrings.append(bits)

    unique_bitstrings = list(set(bitstrings))
    best_energy = float('inf')

    for b in unique_bitstrings:
        e = bitstring_energy_3d(b, sequence)
        if e < best_energy:
            best_energy = e

    return best_energy


def bitstring_to_path_3d(bitstring):
    position = np.array([0, 0, 0])
    path = [position.copy()]
    dir_index = 0

    for i in range(0, len(bitstring), 2):
        turn = bitstring[i:i+2]
        if turn == '00':
            pass
        elif turn == '01':
            dir_index = (dir_index + 1) % 4
        elif turn == '10':
            dir_index = (dir_index - 1) % 4
        elif turn == '11':
            dir_index = (dir_index + 2) % 4
        position = position + tetrahedral_directions[dir_index]
        path.append(position.copy())

    return path
#-----------------------

def bitstring_energy_3d(bitstring, sequence):
    path = bitstring_to_path_3d(bitstring)
    energy = 0.0

    for i in range(len(sequence)):
        for j in range(i + 2, len(sequence)):
            if sequence[i] == 'H' and sequence[j] == 'H':
                dist = np.linalg.norm(np.array(path[i]) - np.array(path[j]))
                if np.isclose(dist, np.linalg.norm(tetrahedral_directions[0])):
                    energy += -1.5


    if len(set(map(tuple, path))) < len(path):
        energy += 10.0

    return energy


def visualize_3d_path(bitstring, sequence):
    path = bitstring_to_path_3d(bitstring)
    x, y, z = zip(*path)
    colors = ['red' if s == 'H' else 'blue' for s in sequence]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '-o', color='gray', linewidth=2)
    for i in range(len(path)):
        ax.scatter(x[i], y[i], z[i], color=colors[i], s=100)
    ax.set_title('Best 3D Protein Fold')
    ax.set_xlabel('X axis')  # ADDED
    ax.set_ylabel('Y axis')  # ADDED
    ax.set_zlabel('Z axis')  # ADDED
    plt.show()
#------------------------
def run_quantum_simulation(sequence, num_runs=30):
    n = len(sequence)
    num_qubits = 2 * (n - 1) 
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    sim = cirq.Simulator()
    hamiltonian = build_hamiltonian(qubits, sequence)

    all_solutions = []
    avg_energies = []  # ADDED

    for run in range(num_runs):
        print(f"--- Optimization Run {run + 1} ---")
        init_params = np.random.uniform(0, 2 * np.pi, num_qubits)

        result = minimize(expectation, init_params, args=(qubits, hamiltonian, sim, sequence),  # CHANGED
                          method='COBYLA', options={'maxiter': 100})

        circuit = build_circuit(qubits, result.x)
        circuit.append(cirq.measure(*qubits, key='m'))
        final_result = sim.run(circuit, repetitions=100)

        run_energies = []  # ADDED

        for i in range(100):
            bits = ''.join(str(final_result.measurements['m'][i][j]) for j in range(len(qubits)))
            energy = bitstring_energy_3d(bits, sequence)
            all_solutions.append((bits, energy))
            run_energies.append(energy)  # ADDED

        min_energy = min(run_energies)  # NEW
        avg_energies.append(min_energy)  # NEW

    all_solutions = list(set(all_solutions))
    all_solutions.sort(key=lambda x: x[1])

    best_bitstring, best_energy = all_solutions[0]
    print(f"Bitstring = {best_bitstring}")
    print(f"Energy = {best_energy:.4f}")

    visualize_3d_path(best_bitstring, sequence)

    # ADDED: Plot energy trend
    plt.figure()
    plt.plot(range(1, num_runs + 1), avg_energies, marker='o')
    plt.xlabel('Optimization Run')
    plt.ylabel('Minimum Energy')  # NEW
    plt.title('Energy Trend Across Optimization Runs')
    plt.grid(True)
    plt.ylim(min(-1.5, min(avg_energies)) - 1, max(avg_energies) + 1)  # ADDED
    plt.show()

sequence = "HPHPPHHPHPPH"
run_quantum_simulation(sequence, num_runs=30)
#-----------------------
