import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cirq
from scipy.optimize import minimize
from itertools import combinations

# -----------------------------
# 1. Tetrahedral Directions
# -----------------------------
tetrahedral_directions = [
    np.array([1, 1, 1]),
    np.array([1, -1, -1]),
    np.array([-1, 1, -1]),
    np.array([-1, -1, 1])
]

# -----------------------------
# 2. Protein Sequence (Expanded)
# -----------------------------
sequence = "HPPHPPHPHPHPPHPH"  # 16 residues
num_residues = len(sequence)
num_folds = num_residues - 1
num_qubits = 2 * num_folds
qubits = cirq.NamedQubit.range(num_qubits, prefix="q")

# -----------------------------
# 3. Bitstring to Directions
# -----------------------------
def bitstring_to_directions(bitstring):
    return [int("".join(bitstring[i:i+2]), 2) for i in range(0, len(bitstring), 2)]

# -----------------------------
# 4. Generate 3D Fold Path
# -----------------------------
def generate_fold_path_3d(direction_indices):
    pos = [np.array([0, 0, 0])]
    for idx in direction_indices:
        step = tetrahedral_directions[idx]
        pos.append(pos[-1] + step)
    return pos

# -----------------------------
# 5. Build Hamiltonian
# -----------------------------
def build_hamiltonian(residue_sequence, qubits, overlap_penalty=5.0, contact_reward=-1.5):
    hamiltonian = []
    n = len(residue_sequence)

    for i, j in combinations(range(n), 2):
        if residue_sequence[i] == 'H' and residue_sequence[j] == 'H' and abs(i - j) > 1:
            qi = 2 * (i - 1)
            qj = 2 * (j - 1)
            if qi < len(qubits) and qj < len(qubits):
                hamiltonian.append((contact_reward, cirq.Z(qubits[qi]) * cirq.Z(qubits[qj])))

    for i in range(len(qubits) - 2):
        hamiltonian.append((overlap_penalty, cirq.Z(qubits[i]) * cirq.Z(qubits[i + 2])))

    return hamiltonian

# -----------------------------
# 6. Calculate Energy
# -----------------------------
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
        energy += 10.0  # overlap penalty

    for i, j in combinations(range(len(positions)), 2):
        if abs(i - j) > 1 and residue_sequence[i] == residue_sequence[j] == 'H':
            dist = np.linalg.norm(positions[i] - positions[j])
            if np.isclose(dist, np.linalg.norm(tetrahedral_directions[0])):
                energy += -1.5

    return energy

# -----------------------------
# 7. Quantum Circuit (Ansatz)
# -----------------------------
def ansatz(params, qubits):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
    return circuit

# -----------------------------
# 8. Energy Expectation
# -----------------------------
def expectation(params):
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

# -----------------------------
# 9. Run Optimization
# -----------------------------
init_params = np.random.uniform(0, 2 * np.pi, size=num_qubits)
result = minimize(expectation, init_params, method='COBYLA', options={'disp': True})

# -----------------------------
# 10. Plot 3D Fold Model
# -----------------------------
def plot_3d_fold_model(positions, residue_sequence=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    ax.plot(xs, ys, zs, color='blue', marker='o', linewidth=2)

    if residue_sequence:
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            color = 'red' if residue_sequence[i] == 'H' else 'green'
            ax.scatter(x, y, z, color=color, s=100)
            ax.text(x, y, z, f'{i} ({residue_sequence[i]})', fontsize=9)
    else:
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, str(i), fontsize=9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Fold Model of Protein Chain')
    plt.show()

# -----------------------------
# 11. Decode Result & Plot
# -----------------------------
bitstring = ''.join(['0' if np.cos(p / 2) ** 2 > 0.5 else '1' for p in result.x])
direction_indices = bitstring_to_directions(bitstring)
positions = generate_fold_path_3d(direction_indices)

plot_3d_fold_model(positions, sequence)

print("Final energy:", result.fun)
print("Optimized parameters:", result.x)
print("Bitstring:", bitstring)
