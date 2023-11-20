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

rho1 = ns.qubits.ketstates.h0 * ns.qubits.ketstates.h0.conj().transpose()



num_party = 5
ns.set_qstate_formalism(QFormalism.DM)

# output_list = [] 
probs = np.linspace(0,1,num=15)
# probs = [0.1]
output_array = np.ndarray(shape=(len(probs),2))

# Note: Fidelity is affected by:
# Num of party
# Distance between party 
# Position of the sender i.e. given same distance between sender and receiver d, sender position affect the fidelity


s = 4
r = 0


bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)


for j in range (len(probs)):
    # print(f"Current Index: {j}")
    qubits = ns.qubits.create_qubits(num_party)
    
    # Teleported qubit if calculating teleported state fidelity
    # qubit_send = ns.qubits.create_qubits(1)
    # operate(qubit_send[0],H) 
    
    
    operate(qubits[0],H)
    for i in range(num_party-1):
        operate([qubits[0],qubits[i+1]], CNOT)
        # Noise from state preparation
        depolarize(qubits[0], prob=probs[j])
        depolarize(qubits[i+1], prob=probs[j])
        
        # apply_pauli_noise(qubits[0], (1-probs[j], probs[j], 0, 0))
        # apply_pauli_noise(qubits[i+1], (1-probs[j], probs[j], 0, 0))
        
    # Noise from channel during distribution of state    
    # for i in range(num_party):
    #     # pass
    #     # depolarize(qubits[i], prob=probs[j])
    #     # dephase(qubits[i], prob=probs[j])
    #     apply_pauli_noise(qubits[i], (1-probs[j], probs[j], 0, 0))
    
    # Measure in X basis by non sender-receiver node
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
    
    # Calculate anonymous entangled state fidelity            
    output_state1 = reduced_dm([qubits[s],qubits[r]])
    fidelity1 = ns.qubits.dmutil.dm_fidelity(output_state1,rho,squared = True)
    
    # Teleportation by BSM on sender and teleported qubit
    # meas, prob = ns.qubits.gmeasure([qubit_send[0],qubits[s]], meas_operators=bell_operators)
    # # Correction on receiver qubit
    # if meas == 1:
    #     operate(qubits[r],X)
    # elif meas == 2:
    #     operate(qubits[r],Z)
    # elif meas == 3:
    #     operate(qubits[r],X)
    #     operate(qubits[r],Z)      
    
    # output_state = reduced_dm([qubits[r]])
    
    # # print(output_state)
    # # print(rho)
    # # print(output_state)
    # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho1,squared = True)
    # # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
    # print(f"Noise:{probs[j]} State Fidelity: {fidelity} QAE Fidelity:{fidelity1} ")
    # # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
    output_array[j][0] = probs[j]
    output_array[j][1] = fidelity1
# print(output_array)



fid_data = pd.DataFrame(data = output_array,columns = ['Noise Param','Fidelity'])
print(fid_data)
# print(fid_data)
# fid_data.plot(x = 'Noise Param', y='Fidelity')

fid_data.to_csv(f'QAE_Fidelity_K={num_party}_d={4}_s={s}.csv')
