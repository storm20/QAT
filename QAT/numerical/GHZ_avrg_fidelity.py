import netsquid as ns
import numpy as np
# import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import *
from netsquid.qubits.qformalism import QFormalism
import random
import pandas as pd
from datetime import datetime
psi = ns.qubits.ketstates.b00
rho = psi * psi.conj().transpose()

rho1 = ns.qubits.ketstates.h0 * ns.qubits.ketstates.h0.conj().transpose()

np.random.seed(0)

num_party = 10
ns.set_qstate_formalism(QFormalism.DM)
# ns.set_qstate_formalism(QFormalism.SPARSEDM)

# output_list = [] 
# probs = np.linspace(0,1,num=15)
# probs = np.logspace(-4, 0, num=10, endpoint=True)
# probs1 = np.logspace(-4, 0, num=10, endpoint=True)

probs = np.linspace(0, 1, num=25, endpoint=True)
probs1 = np.linspace(0, 1, num=25, endpoint=True)


x_2d, y_2d = np.meshgrid(probs,probs1)
# probs = [0.1]
output_array = np.ndarray(shape=(len(probs),2))
output_array = np.zeros_like(x_2d)
# print(x_2d)

# Note: Fidelity is affected by:
# Num of party
# Distance between party 
# Position of the sender i.e. given same distance between sender and receiver d, sender position affect the fidelity


s = 0
r = 4


bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)


def Dagger(rho):
    return np.conjugate(rho).T

for z in range (len(x_2d)):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print(f"Current Probs: {probs[z]}")
    for j in range (len(y_2d)):
        
        qubits = ns.qubits.create_qubits(num_party)
        
        # Teleported qubit if calculating teleported state fidelity
        
        # operate(qubit_send[0],H) 
        
        
        operate(qubits[0],H)
        for i in range(num_party-1):
            operate([qubits[0],qubits[i+1]], CNOT)
            # Noise from state preparation
            # depolarize(qubits[0], prob=probs1[k])
            # depolarize(qubits[i+1], prob=probs1[k])
            
            depolarize(qubits[0], prob=x_2d[z,j])
            depolarize(qubits[i+1], prob=x_2d[z,j])
            
            # apply_pauli_noise(qubits[0], (1-probs[j], probs[j], 0, 0))
            # apply_pauli_noise(qubits[i+1], (1-probs[j], probs[j], 0, 0))
            
        # Noise from channel during distribution of state    
        for i in range(num_party):
            # pass
            # depolarize(qubits[i], prob=probs[j])
            # dephase(qubits[i], prob=probs[j]/2)
            depolarize(qubits[i], prob=y_2d[z,j]/2)
            # apply_pauli_noise(qubits[i], (1-probs[j]/2, probs[j]/2, 0, 0))
        
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
        # output_state1 = reduced_dm([qubits[s],qubits[r]])
        # fidelity1 = ns.qubits.dmutil.dm_fidelity(output_state1,rho,squared = True)
        
        # Teleportation by BSM on sender and teleported qubit
        num_samples = 1000
        a = np.random.rand(num_samples)
        c = 2 * np.pi * np.random.rand(num_samples)
        b = np.sqrt(1 - a**2)
        fid_list = []
        
        for k in range (num_samples):
            temp = b[k] * np.exp(1j*c[k])
            ket_state = np.array([[a[k]],[temp]])
            rho = ket_state*Dagger(ket_state)
            noisy_bell = reduced_dm([qubits[s],qubits[r]])
            
            qubit_teleport = ns.qubits.create_qubits(3)
            
            assign_qstate(qubit_teleport[0],rho)
            assign_qstate([qubit_teleport[1],qubit_teleport[2]],noisy_bell)
            
            meas, prob = ns.qubits.gmeasure([qubit_teleport[0],qubit_teleport[1]], meas_operators=bell_operators)
            # Correction on receiver qubit
            if meas == 1:
                operate(qubit_teleport[2],X)
            elif meas == 2:
                operate(qubit_teleport[2],Z)
            elif meas == 3:
                operate(qubit_teleport[2],X)
                operate(qubit_teleport[2],Z)      
            
            output_state = reduced_dm([qubit_teleport[2]])
            fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
            fid_list.append(fidelity)
        avrg_fid = sum(fid_list)/len(fid_list)
        # # print(output_state)
        # # print(rho)
        # # print(output_state)
        # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho1,squared = True)
        # # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
        # print(f"Noise:{probs[j]} State Fidelity: {fidelity} QAE Fidelity:{fidelity1} ")
        # # fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
        # output_array[j][0] = probs[j]
        # output_array[j][1] = avrg_fid
        output_array[z,j] = avrg_fid
        
    # print(output_array)



# fid_data = pd.DataFrame(data = output_array,columns = ['Noise Param','Fidelity'])
print(output_array)
np.savetxt('z_values_linear.csv', output_array, delimiter=',')

# print(fid_data)
# fid_data.plot(x = 'Noise Param', y='Fidelity')

# fid_data.to_csv(f'QAE_Avrg_Fidelity_K={num_party}_bitflip_channel.csv')

# fid_data.to_csv(f'QAE_Avrg_Fidelity_log_K={num_party}_dephase_channel.csv')
