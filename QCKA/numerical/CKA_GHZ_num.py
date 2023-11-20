import netsquid as ns
import numpy as np
import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import *
from netsquid.qubits.qformalism import QFormalism
import random
import pandas as pd
from datetime import datetime

psi = ns.qubits.ketstates.b00
rho = psi * psi.conj().transpose()

rho1 = ns.qubits.ketstates.h0 * ns.qubits.ketstates.h0.conj().transpose()



num_party = 15 #  Number of participants in CKA
num_nodes = 15 #  Number of nodes in network

ns.set_qstate_formalism(QFormalism.SPARSEDM)

# output_list = [] 
# probs = np.linspace(0,1,num=10)
# probs = np.logspace(-4, 0, num=5, endpoint=True)
probs = [0.0001,0.001]
# probs = [0.1]
print(probs)


# probs = [0.1]

# list_party = np.random.choice(num_nodes,num_party,replace=False)
# print(list_party)

# list_party = [0,1,2]
# list_party = [1,2,3]
# list_party = [0,1,num_nodes-1]
list_party = [0,1]
# list_party = [1,2,3,4]

# list_meas = []
round_size = 1e6
num_error = 200
round_type = 0 # 0 for key generation
# round_type = 1 # 0 for test round

if round_type == 0:
    # output_array = np.ndarray(shape=(len(probs),num_party))
    output_array = np.ndarray(shape=(len(probs),len(list_party)))
else:
    output_array = np.ndarray(shape=(len(probs),2))
temp_probs = []
for j in range (len(probs)):
    error = 0
    num_exp = 0
    print(j,probs[j])
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Prob: ",probs[j])
    print("Current Time =", current_time)
    print("Current experiment:",0)
    error_avrg = 0
    
    
    # list_sum = [0]*(num_party-1)
    list_sum = [0]*(len(list_party)-1)
    # print(list_sum)
    k = 0 
    temp_probs.append(probs[j])
    while k < round_size and error < num_error:
        num_exp += 1
    # for k in range(round_size):
        # print(f"Current Index: {j}")
        if k % 1000 == 0 and k != 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("Current experiment:",k)
            print("Current error:",error)
        meas_ = []
        qubits = ns.qubits.create_qubits(num_nodes)
        operate(qubits[0],H)
        for i in range(num_nodes-1):
            operate([qubits[0],qubits[i+1]], CNOT)
            # Noise from state preparation
            depolarize(qubits[0], prob=probs[j])
            depolarize(qubits[i+1], prob=probs[j])

            
        # Noise from channel during distribution of state    
        # for i in range(num_party):
        #     # pass
        #     depolarize(qubits[i], prob=probs[j])
        #     dephase(qubits[i], prob=probs[j])
        #     apply_pauli_noise(qubits[i], (1-probs[j], probs[j], 0, 0))
        # Z basis Error
        if round_type == 0:
            for i in range (num_nodes):
                if i in list_party:
                    meas,prob = ns.qubits.measure(qubits[i])
                    meas_.append(meas)
                    # if meas == 1:
                    #     sum_1 = sum_1 + 1
                else:
                    operate(qubits[i],H)
                    meas,prob = ns.qubits.measure(qubits[i])
                    meas_.append(meas)
            for z in range (len(list_party)-1):
                if meas_[list_party[0]] != meas_[list_party[z+1]]:
                    list_sum[z] += 1
                    error += 1
            # print(list_sum)
        # X basis error
        else:
            sum_1 = 0
            for i in range (num_nodes):
                # print(i)
                operate(qubits[i],H)
                meas,prob = ns.qubits.measure(qubits[i])
                sum_1 += meas
                meas_.append(meas)
                # if meas == 1:
                #     sum_1 = sum_1 + 1
            sum_1 = sum_1 % 2
            error += 1
            error_avrg += sum_1
        k += 1
        # list_meas.append(meas_)
    if round_type == 0:
        output_array[j][0] = probs[j]
        for m in range (len(list_sum)):
            # print(m)
            output_array[j][m+1] = list_sum[m]/num_exp 
        # print(output_array)
    else:
        error_avrg = error_avrg/num_exp
        output_array[j][0] = probs[j]
        output_array[j][1] = error_avrg
        

columns = ['Noise Param']
# for i in range (num_party - 1):
#     columns.append(f'Error Qz{i+1}')
    
for i in range (len(list_party) - 1):
    columns.append(f'Error Qz{i+1}')
# print(columns)
# print(output_array.shape)

data = pd.DataFrame(data = output_array,columns = columns)
# data = pd.DataFrame(data = output_array,columns = ['Noise Param','Error Qx'])


# print(fid_data)
# print(fid_data)
# data.plot(x = 'Noise Param', y='Error Qx')

data.to_csv(f'error_prob_QZ_N={num_party}_test.csv')
# data.to_csv(f'error_prob_log_QX_K={num_nodes}.csv')

# data.to_csv(f'data_K=5_lowest/error_prob_log_QX_K={num_nodes}.csv')


# big_round = 100
# sub_round = 10000
# temp_list = []
# counter = 0
# for j in range (big_round):
#     now = datetime.now()
#     current_time = now.strftime("%H:%M:%S")
#     print("Current Time =", current_time)
#     print("Current round:",j)
#     error_avrg = 0
#     if j % 5 == 0 and j != 0:
#         print(f"Current counter: {counter}")
#         temp_list = []
#     for k in range(sub_round):
#         meas_ = []
#         qubits = ns.qubits.create_qubits(num_nodes)
#         operate(qubits[0],H)
#         for i in range(num_nodes-1):
#             operate([qubits[0],qubits[i+1]], CNOT)
#             # Noise from state preparation
#             depolarize(qubits[0], prob=probs[0])
#             depolarize(qubits[i+1], prob=probs[0])
#         sum_1 = 0
#         for i in range (num_nodes):
#             # print(i)
#             operate(qubits[i],H)
#             meas,prob = ns.qubits.measure(qubits[i])
#             sum_1 += meas
#             meas_.append(meas)
#             # if meas == 1:
#             #     sum_1 = sum_1 + 1
#         sum_1 = sum_1 % 2
#         error_avrg += sum_1
#     error_avrg = error_avrg/sub_round
#     temp_list.append(error_avrg)
#     if j % 5 == 0 and j != 0:
#         counter += 1
#         data = pd.DataFrame(data = temp_list,columns = ['Error Qx'])
#         data.to_csv(f'data_K={num_nodes}_0/error_prob_QX_K={num_nodes}_{counter}.csv')
        

        