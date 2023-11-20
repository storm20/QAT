import netsquid as ns
import numpy as np
import cmath
from netsquid.qubits.qubitapi import *
from netsquid.qubits.operators import *
from netsquid.qubits.qformalism import QFormalism
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from scipy.special import comb


b00 = ns.qubits.ketstates.b00
b00_dm = ns.qubits.ketutil.ket2dm(b00)
# print(b00_dm)

# b01 = ns.qubits.ketstates.b01
# b01_dm = ns.qubits.ketutil.ket2dm(b01)
# print(b01_dm)

# b10 = ns.qubits.ketstates.b10
# b10_dm = ns.qubits.ketutil.ket2dm(b10)
# print(b10_dm)

# b11 = ns.qubits.ketstates.b11
# b11_dm = ns.qubits.ketutil.ket2dm(b11)
# print(b11_dm)


# print(probs)




ns.set_qstate_formalism(QFormalism.SPARSEDM)
qubits0 = ns.qubits.create_qubits(2)
qubits1 = ns.qubits.create_qubits(1)
qubits2 = ns.qubits.create_qubits(1)
# qubits3 = ns.qubits.create_qubits(2)
# assign_qstate(qubits0,b00_dm)
# assign_qstate(qubits3,b00_dm)

# apply_pauli_noise(qubits0[1],[0.5,0.5,0,0])

# print(reduced_dm([qubits0[0],qubits0[1]]))
# operate(qubits0[1],X)
# operate([qubits0[1], qubits1[0]], CNOT)
# operate(qubits0[1],X)
# operate([qubits0[1], qubits2[0]], CNOT)
# operate(qubits0[1],X)
# print(reduced_dm([qubits0[0],qubits1[0],qubits0[1]]))
# operate(qubits0[1],X)
# operate(qubits0[1],X)
# df = pd.DataFrame(reduced_dm([qubits0[0],qubits1[0],qubits2[0],qubits0[1]]))
# print(df)
# assign_qstate(qubits1,b00_dm)
# operate(qubits[1],X)
# operate(qubits[1],Z)
# operate([qubits0[1], qubits1[0]], CNOT)
# operate([qubits1[0], qubits2[0]], CNOT)
# operate(qubits0[1], H)
# meas1, prob1 = measure(qubits0[1])
# meas2, prob2 = measure(qubits1[0])
# print(meas1,meas2)
# output = reduced_dm([qubits1[0],qubits2[0],qubits0[1],qubits0[0]])


# if meas1 == 0 and meas2 == 1:
#     operate(qubits1[1],X)
# elif meas1 == 1 and meas2 == 0:
#     operate(qubits1[1],Z)
# elif meas1 == 1 and meas2 == 1:
#     operate(qubits1[1],Z)
#     operate(qubits1[1],X)
# print(reduced_dm([qubits0[0],qubits1[1]]))




num_party = 6
s = 0


K = 50
probs = np.linspace(0,1,num=K)
# probs = [0.5]
# a = num_party
ns.set_qstate_formalism(QFormalism.SPARSEDM)
fig,ax1 = plt.subplots(1,1, figsize=(7, 5))
# x = 4
x = 1
while x <= num_party-1:
    fid_data = []
    # num_party = x
    r = x
    for k in range(len(probs)):
        qubit_list = []
        for i in range(num_party):
            qubits = ns.qubits.create_qubits(2)
            assign_qstate(qubits,b00_dm)
            qubit_list.append(qubits)
        qubits_s = ns.qubits.create_qubits(1)
        qubits_r = ns.qubits.create_qubits(1)
        
        # receiver = False
        sender = False
        receiver = False
        for i in range(num_party-1):
            # print(f"Party: {i}")
            # print(f"sender: {s}")
            if i == s:# If first party is sender
                # print(f"P{i} sender case, apply CNOT")
                operate([qubit_list[i][1], qubits_s[0]], CNOT)
                sender = True
            elif i == r:# If first party is receiver
                # print(f"P{i} receiver case, apply CNOT")
                operate([qubit_list[i][1], qubits_r[0]], CNOT)
                receiver = True
            # Apply Noise
            # if sender and receiver:
            #     df = pd.DataFrame(reduced_dm([qubit_list[0][0],qubits_s[0],qubits_r[0],qubit_list[i][1]]))
            #     print(df)
            # elif sender:
            #     print(reduced_dm([qubit_list[0][0],qubits_s[0],qubit_list[i][1]]))
            # else:
            #     print(reduced_dm([qubit_list[0][0],qubit_list[i][1]]))
            # print("apply noise")
            # dephase(qubit_list[i][1],prob=probs[k])
            depolarize(qubit_list[i][1],prob=probs[k])
            # apply_pauli_noise(qubit_list[i][1],[(1-probs[k]),probs[k],0,0])
            # print(reduced_dm([qubit_list[0][0],qubits_s[0],qubit_list[i][1]]))
            # if sender and receiver:
            #     df = pd.DataFrame(reduced_dm([qubit_list[0][0],qubits_s[0],qubits_r[0],qubit_list[i][1]]))
            #     print(df)
            # elif sender:
            #     print(reduced_dm([qubit_list[0][0],qubits_s[0],qubit_list[i][1]]))
            # else:
            #     print(reduced_dm([qubit_list[0][0],qubit_list[i][1]]))
            
            # print("After noise")
            # print(reduced_dm([qubit_list[i][0], qubit_list[i][1]]))
            # #BSM
            # print("BSM")
            # print("Before BSM, after noise")
            # if sender:
            #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
            # else:
            #     print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            operate([qubit_list[i][1], qubit_list[i+1][0]], CNOT)
            operate(qubit_list[i][1], H)
            meas1, prob1 = measure(qubit_list[i][1])
            meas2, prob2 = measure(qubit_list[i+1][0])
            
            # print("After BSM")
            #BSM Correction
            # print("Correction")
            # if meas1 == 0 and meas2 == 0:
            #     print(0)
                # if sender:
                #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
                # else:
                #     print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            if meas1 == 0 and meas2 == 1:
                # print(1)
                # print(reduced_dm([qubit_list[i][0], qubit_list[i+1][1]]))
                operate(qubit_list[i+1][1],X)
                # if sender:
                #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
                # else:
                #     print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            elif meas1 == 1 and meas2 == 0:
                # print(2)
                # print(reduced_dm([qubit_list[i][0], qubit_list[i+1][1]]))
                operate(qubit_list[i+1][1],Z)
                # if sender:
                #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
                # else:
                #     print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            elif meas1 == 1 and meas2 == 1:
                # print(3)
                # print(reduced_dm([qubit_list[i][0], qubit_list[i+1][1]]))
                operate(qubit_list[i+1][1],Z)
                operate(qubit_list[i+1][1],X)
                # if sender:
                #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
                # else:
                #     print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            
            # if sender and receiver:
            #     df = pd.DataFrame(reduced_dm([qubit_list[0][0],qubits_s[0],qubits_r[0],qubit_list[i+1][1]]))
            #     print(df)
            # elif sender:
            #     print(reduced_dm([qubit_list[0][0],qubits_s[0],qubit_list[i+1][1]]))
            # else:
            #     print(reduced_dm([qubit_list[0][0],qubit_list[i+1][1]]))
            
            # if sender:
            #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
            # else:
            #     print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            # print(reduced_dm([qubit_list[0][0], qubit_list[i+1][1]]))
            
            
            # if i == s: # if sender case but not p1, apply CNOT
            #     print(f"party{i} sender case, apply CNOT")
            #     operate([qubit_list[i+1][1], qubits_s[0]], CNOT)
            #     print(reduced_dm([qubits_s[0],qubit_list[i+1][1],qubit_list[0][0]]))
            #     print(reduced_dm([qubit_list[0][0],qubits_s[0],qubit_list[i+1][1]]))
            #     sender = True
            # if i == r and r != (num_party-1): # if receiver case, not last part apply CNOT
            #     print(f"party{i} receiver case")
            #     operate([qubit_list[i+1][1], qubits_r[0]], CNOT)
            #     global_state1 = reduced_dm([qubits_s[0],qubits_r[0],qubit_list[0][0],qubit_list[num_party-1][1]])
            #     # print(pd.DataFrame(global_state1))
            #     # receiver = True
            #     # pd.DataFrame(global_state1).to_csv("qstate1.csv")
                

        # print(f"index {i}")
        # global_state2 = reduced_dm([qubits_s[0],qubits_r[0],qubit_list[0][0],qubit_list[num_party-1][1]])
        # pd.DataFrame(global_state2).to_csv("qstate2.csv")
        # last party receiver or sender case:
        if r == (num_party-1):
            # print("last party receiver case, apply CNOT")
            operate([qubit_list[num_party-1][1], qubits_r[0]], CNOT)
        elif s == (num_party-1):
            # print("last party receiver case, apply CNOT")
            operate([qubit_list[num_party-1][1], qubits_s[0]], CNOT)
        # elif s == (num_party-1):
        #     operate([qubit_list[num_party-1][1], qubits_s[0]], CNOT)
        # global_state1 = reduced_dm([qubit_list[0][0],qubits_s[0],qubits_r[0],qubit_list[num_party-1][1]])
        # print(pd.DataFrame(global_state1))
        # print("Hadamard measure")
        operate(qubit_list[0][0], H)
        meas1, prob1 = measure(qubit_list[0][0])
        operate(qubit_list[num_party-1][1], H)
        meas2, prob2 = measure(qubit_list[num_party-1][1])
        #Hadamard Correction 
        # print(meas1,meas2)
        if meas1 != meas2:
            # print("Correction")
            operate(qubits_r[0],Z)
        # print("After Correction")
        output_state = reduced_dm([qubits_s[0],qubits_r[0]]) 
        # print(output_state)
        
        # print(output_state) 
        fidelity = ns.qubits.dmutil.dm_fidelity(output_state,b00_dm,squared = True)
        # print(fidelity)
        fid_data.append(fidelity)
        # print(fid_data)
        
    def bitflip_funct(n,probs):
        k = 0
        sum = 0
        while k<= n:
            const = comb(n, k, exact=True)
            sum = sum + const * pow(probs,n-k)*pow(1-probs,k)
            k = k+2
        return sum

    # out_funct = []
    # n = abs(s-r)

    # for i in range (len(probs)):
    #     val = bitflip_funct(n,1-probs[i])
    #     out_funct.append(val)
    
    # ax1.plot(x,y,'-', color='green')
    # ax1.plot(probs,fid_data,'-', color='red',label=f' K ={x}')
    ax1.plot(probs,fid_data,'-', color='red',label=f' d ={x}')
    d = {'Noise Param': probs, 'Fidelity':fid_data}
    fid_data = pd.DataFrame(data = d,columns = ['Noise Param','Fidelity'])
    fid_data.to_csv(f'relay_depol_d{r}_K{num_party}.csv')

    
    # Function for dephasing
    # out_funct = (1+pow((1-2*probs),num_party-1))/2
    # ax1.plot(probs,out_funct,'-', color='blue',label='function')
    x += 1
    

# K1 = 100
# probs1 = np.linspace(0,1,num=K1)
# num_party = 5
# out_funct = (1+pow((1-2*probs1),num_party-1))/2  
# ax1.plot(probs1,out_funct,'-', color='blue',label=f'Relay K={num_party}')
# num_party = 15
# out_funct = (1+pow((1-2*probs1),num_party-1))/2  
# ax1.plot(probs1,out_funct,'-', color='red',label=f'Relay K={num_party}')
# num_party = 45
# out_funct = (1+pow((1-2*probs1),num_party-1))/2  
# ax1.plot(probs1,out_funct,'-', color='green',label=f'Relay K={num_party}')
# num_party = 75
# out_funct = (1+pow((1-2*probs1),num_party-1))/2  
# ax1.plot(probs1,out_funct,'-', color='brown',label=f'Relay K={num_party}')

ax1.legend()


