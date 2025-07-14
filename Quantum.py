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
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
    return circuit

def expectation(params, qubits, hamiltonian, sim):
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

    ax.set_title('3D Protein Fold')
    plt.show()

def run_quantum_simulation(sequence):
    n = len(sequence)
    num_qubits = 2 * (n - 1) 
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    sim = cirq.Simulator()






   
    hamiltonian = build_hamiltonian(qubits, sequence)

    init_params = np.random.uniform(0, 2 * np.pi, num_qubits)
    result = minimize(expectation, init_params, args=(qubits, hamiltonian, sim),
                      method='COBYLA', options={'maxiter': 100})

    print("Optimization finished.")
    print("Best energy found:", result.fun)

    circuit = build_circuit(qubits, result.x)
    circuit.append(cirq.measure(*qubits, key='m'))
    final_result = sim.run(circuit, repetitions=100)

    bitstrings = []
    for i in range(100):
        bits = ''.join(str(final_result.measurements['m'][i][j]) for j in range(len(qubits)))
        bitstrings.append(bits)

    unique_bitstrings = list(set(bitstrings))
    scored = [(b, bitstring_energy_3d(b, sequence)) for b in unique_bitstrings]
    scored.sort(key=lambda x: x[1])

    print("\nTop 5 folding solutions:")
    for i, (b, e) in enumerate(scored[:5]):
        print(f"{i+1}: Bitstring = {b}, Energy = {e}")

    visualize_3d_path(scored[0][0], sequence)

sequence = "HPHPPHHPHPPHPHHPPHPH"
run_quantum_simulation(sequence)
