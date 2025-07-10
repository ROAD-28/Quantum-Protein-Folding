import cirq
import matplotlib.pyplot as plt
import numpy as np
import sympy
import itertools
from math import radians, degrees
from scipy.optimize import minimize





def create_quantum_circuit():
    # Create a quantum circuit with 2 qubits
    qubit1 = cirq.GridQubit(0, 0)
    qubit2 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit()

    # Add gates to the circuit
    circuit.append([cirq.H(qubit1), cirq.CNOT(qubit1, qubit2)])

    # Measure the qubits
    circuit.append(cirq.measure(qubit1, key='q1'), cirq.measure(qubit2, key='q2'))

    return circuit
def simulate_quantum_circuit(circuit):
    # Create a simulator
    simulator = cirq.Simulator()

    # Simulate the circuit
    result = simulator.run(circuit, repetitions=10)

    return result
