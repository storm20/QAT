import netsquid as ns
import numpy as np
import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import *
from netsquid.qubits.qformalism import QFormalism
import random
import pandas as pd

psi = ns.qubits.ketstates.b00
rho = psi * psi.conj().transpose()

rho1 = ns.qubits.ketstates.s0 * ns.qubits.ketstates.s0.conj().transpose()



num_party = 4
ns.set_qstate_formalism(QFormalism.DM)

# output_list = [] 
probs = np.linspace(0,1,num=20)
# probs = [0.1]
output_array = np.ndarray(shape=(len(probs),2))
s = 0
r = num_party-1


bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)


for j in range (len(probs)):
    # print(f"Current Index: {j}")
    qubits = ns.qubits.create_qubits(num_party)
    qubit_send = ns.qubits.create_qubits(1)
    operate(qubits[0],H)
    for i in range(num_party-1):
        operate([qubits[0],qubits[i+1]], CNOT)
    for i in range(num_party):
        # pass
        depolarize(qubits[i], prob=probs[j])
        # dephase(qubits[i], prob=probs[j])
        # apply_pauli_noise(qubits[i], (1-probs[j], probs[j], 0, 0))
    sum_1 = 0
    for i in range (num_party):
        if i !=s and i !=r:
            operate(qubits[i],H)
            meas,prob = ns.qubits.measure(qubits[i])
            if meas == 1:
                sum_1 = sum_1 + 1
    # Sender flip coin apply Z gate
    b = random.randint(0,1)
    if b == 1:
        sum_1 = sum_1 + 1
        operate(qubits[s],Z)
    # Receiver perform correction if 1's is odd
    if sum_1 % 2 == 1:
            operate(qubits[r],Z)
            
    output_state1 = reduced_dm([qubits[s],qubits[r]])
    fidelity1 = ns.qubits.dmutil.dm_fidelity(output_state1,rho,squared = True)
    
    meas, prob = ns.qubits.gmeasure([qubit_send[0],qubits[s]], meas_operators=bell_operators)
    if meas == 1:
        operate(qubits[r],X)
    elif meas == 2:
        operate(qubits[r],Z)
    elif meas == 3:
        operate(qubits[r],X)
        operate(qubits[r],Z)      
    
    output_state = reduced_dm([qubits[r]])
    
    # print(output_state)
    
    # print(rho)
    # print(output_state)
    fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho1,squared = True)
    # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
    print(f"Noise:{probs[j]} State Fidelity: {fidelity} QAE Fidelity:{fidelity1} ")
    # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
    output_array[j][0] = probs[j]
    output_array[j][1] = fidelity
# print(output_array)

fid_data = pd.DataFrame(data = output_array,columns = ['Noise Param','Fidelity'])
# print(fid_data)
# fid_data.plot(x = 'Noise Param', y='Fidelity')

fid_data.to_csv(f'Depol_Teleport_K={num_party}_0state.csv')
