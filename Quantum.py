import numpy as np

import cirq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import combinations

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
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
    return circuit

def expectation(params, qubits, hamiltonian, sim, alpha=0.3, shots=200):
    circuit = build_circuit(qubits, params)
    circuit.append(cirq.measure(*qubits, key='m'))
    result = sim.run(circuit, repetitions=shots)

    measurements = result.measurements['m']
    bitstrings = []
    bits = [''.join(str(b) for b in reversed(m)) for m in measurements]
    bitstrings.append(bits)

    unique_bitstrings = list(set(bitstrings))
    best_energy = float('inf')

    cutoff = max(1, int(alpha*shots))

    energies = []

    for b in unique_bitstrings:
        e = bitstring_energy_3d(b, hamiltonian, sequence, qubits)
        energies.append(e)

        if e < best_energy:
            best_energy = e

    sorted_energies = sorted(energies)

    return np.mean(sorted_energies[:cutoff])
    


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


# --- Bitstring Energy (position-aware) ---
def bitstring_energy_3d(bitstring, hamiltonian, residue_sequence, qubits):
    positions = bitstring_to_path_3d(bitstring)

    energy = 0
    # Evaluate from Hamiltonian
    for coeff, pauli_term in hamiltonian:
        val = 1
        for q in pauli_term.qubits:
            bit = int(bitstring[qubits.index(q)])
            val *= 1 if bit == 0 else -1
        energy += coeff * val

    # Extra penalty for actual overlaps (based on position)
    if len(set(map(tuple, positions))) < len(positions):
        energy += 10.0  # strong overlap penalty

    # Bonus H-H contact detection at physical distance
    for i, j in combinations(range(len(positions)), 2):
        if abs(i - j) > 1 and residue_sequence[i] == residue_sequence[j] == 'H':
            dist = np.linalg.norm(positions[i] - positions[j])
            if np.isclose(dist, np.linalg.norm(tetrahedral_directions[0])):
                energy += -1.5

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
    plt.show()

def run_quantum_simulation(sequence, num_runs=30):
    n = len(sequence)
    num_qubits = 2 * (n - 1) 
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    sim = cirq.Simulator()
    hamiltonian = build_hamiltonian(qubits, sequence)

    all_solutions = []

    for run in range(num_runs):
        print(f"--- Optimization Run {run + 1} ---")
        init_params = np.random.uniform(0, 2 * np.pi, num_qubits)

        result = minimize(expectation, init_params, args=(qubits, hamiltonian, sim),
                          method='COBYLA', options={'maxiter': 100})

        circuit = build_circuit(qubits, result.x)
        circuit.append(cirq.measure(*qubits, key='m'))
        final_result = sim.run(circuit, repetitions=100)

        for i in range(100):
            bits = ''.join(str(final_result.measurements['m'][i][j]) for j in range(len(qubits)))
            energy = bitstring_energy_3d(bits, sequence)
            all_solutions.append((bits, energy))

    all_solutions = list(set(all_solutions))
    all_solutions.sort(key=lambda x: x[1])

    best_bitstring, best_energy = all_solutions[0]
    print(f"Bitstring = {best_bitstring}")
    print(f"Energy = {best_energy:.4f}")

    visualize_3d_path(best_bitstring, sequence)

sequence = "HHPH"
run_quantum_simulation(sequence, num_runs=10)

