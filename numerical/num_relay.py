# import netsquid as ns
# import numpy as np

# b00 = ns.qubits.ketstates.b00
# b00_dm = ns.qubits.ketutil.ket2dm(b00)
# # print(b00_dm)

# proj_b00 = ns.qubits.operators.Operator('proj_00',b00_dm)
# print(proj_b00)

# zero = ns.qubits.ketstates.s0
# one = ns.qubits.ketstates.s1
# basis1 = np.array([zero,one])
# basis_list1 = []
# basis_list1.append(zero)
# basis_list1.append(one)
# # q1 = ns.qubits.qubitapi.create_qubits(1)
# # print((q1[0].qstate.qrepr))
# basis_list2 = []
# # print(basis_list[1])
# # basis2 = np.eye(2)
# # print(basis1[:,1])
# # print(basis2[:,0])
# # basis2 = np.array([basis2[:,0],basis2[:]])
# for i in range(2):
#     for j in range(2):
#         basis_list2.append(np.kron(basis_list1[i],basis_list1[j]))
# # print(basis_list2[0])
# dens_list = []
# for i in range(4):
#     for j in range(4):
        
#         dens_list.append(ns.qubits.ketutil.outerprod(basis_list2[i],basis_list2[j]))

# # print(dens_list[0])

# for i in range(16):
#     print(f"index {i}")
#     out_state1 = (ns.qubits.opmath.operate_dm(dens_list[i],proj_b00,[0,1]))
#     # norm = np.trace(np.matmul(dens_list[i], proj_b00.arr))
#     print(out_state1)
#     # print(out_state1/norm)


import netsquid as ns
import numpy as np
import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import *
from netsquid.qubits.qformalism import QFormalism
import random
import pandas as pd
import matplotlib.pyplot as plt


psi = ns.qubits.ketstates.b00
rho = psi * psi.conj().transpose()

bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)



num_party = 5
ns.set_qstate_formalism(QFormalism.SPARSEDM)
qubit_number = 2*num_party+2
# output_list = [] 
probs = np.linspace(0,1,num=100)
# probs = [0.5]
output_array = np.ndarray(shape=(len(probs),2))
for k in range (len(probs)):
    # print(f"Current index: {k}")
    # Init all qubits to 0 state
    qubits = ns.qubits.create_qubits(qubit_number)
    # Init Bell state
    i = 0
    j = 0
    while i< (num_party):
        operate(qubits[j],H)
        operate([qubits[j],qubits[j+1]], CNOT)
        i = i+1
        j = j+2
    # Apply depolarize to 2nd qubit that is send to next party  
#     i = 0
#     j = 1
#     while i < (num_party-1):
#         depolarize(qubits[j], prob=probs[k])
#         i = i+1
#         j = j+2
    
    # BSM and Correction and CNOT
    i = 0
    j = 0
    while i < (num_party-1):
#         print(f"Current step: {i}")
#         depolarize(qubits[j+1], prob=probs[k])
        dephase(qubits[j+1], prob=probs[k])
#         apply_pauli_noise(qubits[j+1], (1-probs[k], probs[k], 0, 0))
        meas, prob = ns.qubits.gmeasure([qubits[j+1],qubits[j+2]], meas_operators=bell_operators)
        if meas == 1:
            operate(qubits[j+3],X)
        elif meas == 2:
            operate(qubits[j+3],Z)
        elif meas == 3:
            operate(qubits[j+3],X)
            operate(qubits[j+3],Z)
        if i == 0:
            operate([qubits[3],qubits[2*num_party]],CNOT)
        elif i == 1:
            operate([qubits[5],qubits[2*num_party+1]],CNOT)
        i = i+1
        j = j+2
    
    meas1,prob1 =  ns.qubits.measure(qubits[0],X)
    meas2,prob2 = ns.qubits.measure(qubits[2*num_party-1],X)
    if meas1 != meas2:
        operate(qubits[2*num_party],Z)
    
    output_state = reduced_dm([qubits[2*num_party],qubits[2*num_party+1]])
    print(output_state)
    fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
#     print(fidelity)
    output_array[k][0] = probs[k]
    output_array[k][1] = fidelity
fid_data = pd.DataFrame(data = output_array,columns = ['Noise Param','Fidelity'])
print(fid_data)
# fid_data.to_csv('fidelity_4_depol_numerical.csv')


fig,ax1 = plt.subplots(1,1, figsize=(7, 5))
# ax1.plot(x,y,'-', color='green')
ax1.plot(output_array[:,0],output_array[:,1],'-', color='red',label='simulation')

out_funct = (1+pow((1-2*probs),num_party-1))/2

# ax1.plot(output_array[:,0],out_funct,'-*', color='blue',label='function')
ax1.legend()