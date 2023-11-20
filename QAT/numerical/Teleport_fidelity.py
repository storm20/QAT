import netsquid as ns
import numpy as np
import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import *
from netsquid.qubits.qformalism import QFormalism
import random
import pandas as pd
import matplotlib.pyplot as plt


bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
ns.set_qstate_formalism(QFormalism.SPARSEDM)

def Dagger(rho):
    return np.conjugate(rho).T

num_samples = 1000
# for i in range (num_samples):


def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def is_density_matrix(matrix):
    return np.isclose(np.trace(matrix), 1) and is_hermitian(matrix)


# for i in range (num_samples):
#     temp = b[i] * np.exp(1j*c[i])
#     ket_state = np.array([[a[i]],[temp]])
#     dens_matrix = ket_state*Dagger(ket_state)
#     print(is_density_matrix(dens_matrix))

# dens_matrix = 
# print(ket_state*Dagger(ket_state))

# assign_qstate(qubits[0], )
# temp = b[0] * np.exp(1j*c[0])
# ket_state = np.array([[a[0]],[temp]])
# dens_matrix = ket_state*Dagger(ket_state)
# print(dens_matrix)
# rho = dens_matrix

a = np.random.rand(num_samples)
c = 2 * np.pi * np.random.rand(num_samples)
b = np.sqrt(1 - a**2)
probs = np.linspace(0,1,num=9)

fid_error = [] 
for j in range(len(probs)):
    print("Probs: ",probs[j])
    fid_list = []
    for i in range (num_samples):
        temp = b[i] * np.exp(1j*c[i])
        ket_state = np.array([[a[i]],[temp]])
        rho = ket_state*Dagger(ket_state)
        
        
        # print(is_density_matrix(dens_matrix))
        qubits = ns.qubits.create_qubits(3)
        operate(qubits[1],H)
        operate([qubits[1],qubits[2]], CNOT)
        
        # rho = reduced_dm(qubits[0])
        dephase(qubits[1], prob=probs[j])
        dephase(qubits[2], prob=probs[j]) 
        
        assign_qstate(qubits[0],rho)
        meas, prob = ns.qubits.gmeasure([qubits[0],qubits[1]], meas_operators=bell_operators)
        if meas == 1:
            operate(qubits[2],X)
        elif meas == 2:
            operate(qubits[2],Z)
        elif meas == 3:
            operate(qubits[2],X)
            operate(qubits[2],Z)
        output_state = reduced_dm(qubits[2])
        fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
        
        # print(fidelity)
        fid_list.append(fidelity)
    avrg_fid = sum(fid_list)/len(fid_list)
    fid_error.append(avrg_fid)
    print(avrg_fid)

# plt.figure(figsize=(8, 6))  
# plt.plot(probs, fid_error, label=f'Avrg Fidelity')
