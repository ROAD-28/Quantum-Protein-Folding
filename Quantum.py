import matplotlib.pyplot as plt
import numpy as np
import sympy
import itertools
from math import radians, degrees
from scipy.optimize import minimize

import cirq
import numpy as np
from scipy.optimize import minimize

n_qubit = 2
qubits = cirq.NamedQubit.range(n_qubits)
#(for those who don't know, ansatz is a trail wave function, search it up)
def ansatz(params):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
 
    return circuit




def protein_folding_hamiltonian():
    # placeholder cause idk how to do hamiltonian yet
    pass



init_params = np.random.rand(n_qubits)
result = minimize(expectation, init_params, method='COBYLA')
print("Minimum energy:", result.fun)
