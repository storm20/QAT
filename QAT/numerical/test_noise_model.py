import netsquid as ns
import numpy as np
import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits import operators as ops
from netsquid.qubits.qformalism import QFormalism
import random
import pandas as pd


data = np.load('fake_lagos_GHZ_5_LSTSQ.npy')
print(data.shape)
ns.set_qstate_formalism(QFormalism.DM)
num_qubits = 5
qubits = ns.qubits.create_qubits(num_qubits)
# # print(len(qubits))

# max_mix_state = (np.eye(2**num_qubits)/2**num_qubits)
# # print(reduced_dm(qubits_temp))
# prob = 0.5
# # p = 0.25 * prob
# print(data[0])


operate(qubits[0],ns.H)

for i in range(num_qubits-1):
    operate([qubits[0],qubits[i+1]], ns.CNOT)
    
# output_state = prob*(max_mix_state) + (1-prob)*reduced_dm(qubits)
# print(output_state)
# assign_qstate(qubits,output_state)
dm = reduced_dm(qubits)
sum_fid = 0
for i in range(data.shape[0]):
    fidelity = ns.qubits.dmutil.dm_fidelity(data[i],dm,squared = True)
    # print(fidelity)
    sum_fid += fidelity
print(sum_fid/data.shape[0])

