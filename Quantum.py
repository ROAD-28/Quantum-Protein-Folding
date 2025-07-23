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
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
    return circuit
#------------------------------------
def expectation(params, qubits, hamiltonian, sim):
    circuit = build_circuit(qubits, params)
    circuit.append(cirq.measure(*qubits, key='m'))
    result = sim.run(circuit, repetitions=500)

    bitstrings = []
    for i in range(500):
        bits = ''.join(str(result.measurements['m'][i][j]) for j in range(len(qubits)))
        bitstrings.append(bits)

    unique_bitstrings = list(set(bitstrings))
    energies = []
    for b in unique_bitstrings:
        e = bitstring_energy_3d(b, sequence)
        energies.append(e)
    energies.sort()
    top_10_percent = max(1, len(energies)//10)
    avg_energy = np.mean(energies[:top_10_percent])

    return avg_energy


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
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.show()
    plt.close()

def plot_energy_trend(energies):
    plt.figure()
    plt.plot(range(1, len(energies)+1), energies, marker='o', linestyle='-', color='blue')
    plt.title('Energy Minimization Trend Over Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Lowest Energy Found')
    plt.grid(True)
    plt.show()
    plt.close()
#------------------------
def run_quantum_simulation(sequence, num_runs=30):
    n = len(sequence)
    num_qubits = 2 * (n - 1) 
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    sim = cirq.Simulator()
    hamiltonian = build_hamiltonian(qubits, sequence)

    all_solutions = []
    lowest_energies_per_run = []

    for run in range(num_runs):
        print(f"--- Optimization Run {run + 1} ---")
        init_params = np.random.uniform(0, 2 * np.pi, num_qubits)

        result = minimize(expectation, init_params, args=(qubits, hamiltonian, sim),
                          method='COBYLA', options={'maxiter': 100})

        circuit = build_circuit(qubits, result.x)
        circuit.append(cirq.measure(*qubits, key='m'))
        final_result = sim.run(circuit, repetitions=500)

        run_solutions = []
        for i in range(500):
            bits = ''.join(str(final_result.measurements['m'][i][j]) for j in range(len(qubits)))
            energy = bitstring_energy_3d(bits, sequence)
            run_solutions.append((bits, energy))

        run_solutions = list(set(run_solutions))
        run_solutions.sort(key=lambda x: x[1])
        best_bits, best_energy = run_solutions[0]
        lowest_energies_per_run.append(best_energy)

        all_solutions.extend(run_solutions)

    all_solutions = list(set(all_solutions))
    all_solutions.sort(key=lambda x: x[1])

    best_bitstring, best_energy = all_solutions[0]
    print(f"Bitstring = {best_bitstring}")
    print(f"Energy = {best_energy:.4f}")

    plot_energy_trend(lowest_energies_per_run)
    visualize_3d_path(best_bitstring, sequence)

sequence = "HPHPPHHPHPPHPHHPPHPH"
run_quantum_simulation(sequence, num_runs=100)
