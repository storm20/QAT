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

num_party = 6
ns.set_qstate_formalism(QFormalism.DM)

# output_list = [] 
probs = np.linspace(0,1,num=20)
# probs = [0]
output_array = np.ndarray(shape=(len(probs),2))
s = 0
r = 3

for j in range (len(probs)):
    # print(f"Current Index: {j}")
    qubits = ns.qubits.create_qubits(num_party)
    operate(qubits[0],H)
    for i in range(num_party-1):
        operate([qubits[0],qubits[i+1]], CNOT)
    for i in range(num_party):
        # depolarize(qubits[i], prob=probs[j])
        # dephase(qubits[i], prob=probs[j])
        apply_pauli_noise(qubits[i], (1-probs[j], probs[j], 0, 0))
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
            
    output_state = reduced_dm([qubits[s],qubits[r]])
    # print(rho)
    # print(output_state)
    fidelity = ns.qubits.dmutil.dm_fidelity(output_state,rho,squared = True)
    output_array[j][0] = probs[j]
    output_array[j][1] = fidelity
# print(output_array)

fid_data = pd.DataFrame(data = output_array,columns = ['Noise Param','Fidelity'])
print(fid_data)
fid_data.plot(x = 'Noise Param', y='Fidelity')
fid_data.to_csv(f'GHZ_fidelity_bitflip_K{num_party}.csv')

