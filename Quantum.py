import matplotlib.pyplot as plt
import numpy as np
import sympy
import itertools
from math import radians, degrees
from scipy.optimize import minimize
import cirq
from mpl_toolkits.mplot3d import Axes3D

tetrahedral_directions = [
    np.array([1, 1, 1]),
    np.array([1, -1, -1]),
    np.array([-1, 1, -1]),
    np.array([-1, -1, 1])
]

def generate_fold_path_3d(direction_indices):
    pos = [np.array([0, 0, 0])]
    for i in range(len(direction_indices)):
        step = tetrahedral_directions[direction_indices[i]]
        pos.append(pos[-1] + step)
    return pos

def bitstring_to_directions(bitstring):
    return [int("".join(bitstring[i:i+2]), 2) for i in range(0, len(bitstring), 2)]

from itertools import combinations

def build_hamiltonian(residue_sequence, qubits, overlap_penalty=5.0, contact_reward=-1.5):
    hamiltonian = []
    n = len(residue_sequence)






    for i, j in combinations(range(n), 2):
        if residue_sequence[i] == 'H' and residue_sequence[j] == 'H' and abs(i - j) > 1:
            if i > 0 and j > 0:
                qi = 2 * (i - 1)
                qj = 2 * (j - 1)
                if qi < len(qubits) and qj < len(qubits):
                    hamiltonian.append((contact_reward, cirq.Z(qubits[qi]) * cirq.Z(qubits[qj])))

    for i in range(len(qubits) - 2):
        hamiltonian.append((overlap_penalty, cirq.Z(qubits[i]) * cirq.Z(qubits[i + 2])))

    return hamiltonian


def bitstring_energy_3d(bitstring, hamiltonian, residue_sequence):
    direction_indices = bitstring_to_directions(bitstring)
    positions = generate_fold_path_3d(direction_indices)

    energy = 0
    for coeff, pauli_term in hamiltonian:
        val = 1
        for q in pauli_term.qubits:
            bit_index = int(str(q)[1:])
            bit = int(bitstring[bit_index])
            val *= 1 if bit == 0 else -1
        energy += coeff * val
    if len(set(map(tuple, positions))) < len(positions):
        energy += 10.0



    for i, j in combinations(range(len(positions)), 2):
        if abs(i - j) > 1 and residue_sequence[i] == residue_sequence[j] == 'H':
            dist = np.linalg.norm(positions[i] - positions[j])
            if np.isclose(dist, np.linalg.norm(tetrahedral_directions[0])):
                energy += -1.5
    return energy





def ansatz(params, qubits):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))

    return circuit


def expectation(params):
    sequence = "HPPHPH"
    num_residues = len(sequence)
    num_folds = num_residues - 1
    num_qubits = 2 * num_folds
    qubits = cirq.NamedQubit.range(num_qubits, prefix="q")

    circuit = ansatz(params, qubits)
    for q in qubits:
        circuit.append(cirq.measure(q, key=str(q)))

    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=1)
    bitstring = ''.join(str(result.measurements[str(q)][0][0]) for q in qubits)

    hamiltonian = build_hamiltonian(sequence, qubits)
    energy = bitstring_energy_3d(bitstring, hamiltonian, sequence)
    return energy



sequence = "HPPHPH"
n_qubits = 2 * (len(sequence) - 1)
init_params = np.random.uniform(0, 2 * np.pi, size=n_qubits)

result = minimize(expectation, init_params, method='COBYLA', options={'disp': True})

def plot_fold_path_3d(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]

    ax.plot(xs, ys, zs, marker='o', color='b')





    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, str(i), size=8, zorder=1, color='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Protein Fold Path')

    plt.show()

print("Optimized parameters:", result.x)
print("Final energy:", result.fun)


class AminoAcidProperties:
    def __init__(self):
        self.hydrophobic_residues = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
        self.hydrophilic_residues = {'R', 'N', 'D', 'Q', 'E', 'K', 'S', 'T', 'H', 'C'}

    def is_hydrophobic(self, residue):
        return residue in self.hydrophobic_residues

    def is_hydrophilic(self, residue):
        return residue in self.hydrophilic_residues

        # polarization effects:


        