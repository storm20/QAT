import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# df = pd.read_csv('error_prob_QZ_K=5_N=5_shift1.csv')
# df = pd.read_csv('error_prob_QZ_N=5.csv')
# df1 = pd.read_csv('error_prob_QZ_N=10.csv')
# df2 = pd.read_csv('error_prob_QZ_N=15.csv')



# df = pd.read_csv(f'error_prob_log_QX_K={n}.csv')
# df1 = pd.read_csv(f'error_prob_log_QX_K={7}.csv')
# df2 = pd.read_csv(f'error_prob_log_QX_K={10}.csv')
# df3 = pd.read_csv(f'error_prob_log_QX_K={15}.csv')

# n = 5
# df = pd.read_csv(f'error_prob_log_QX_K={n}.csv')
# path = "./data_K=5_0"
# dir_list = os.listdir(path)
 
# print("Files and directories in '", path, "' :")
 
# prints all files
# print(dir_list[0])




plt.figure(figsize=(8, 6))  
# plt.plot(df['Noise Param'], df['Error Qx'], label=f'n={n}')
# plt.plot(df['Noise Param'], df['Error Qz1'],marker = 'o',linestyle="None", label=f'Qz1')
# plt.plot(df['Noise Param'], df1['Error Qz1'],linestyle="dashed", label=f'Qz1')
# plt.plot(df['Noise Param'], df2['Error Qz1'],linestyle="dashed", label=f'Qz1')


# plt.plot(df['Noise Param'], df['Error Qz2'],marker = '^',linestyle="None", label=f'Qz2')


# plt.plot(df['Noise Param'], df['Error Qz3'],marker = 'v',linestyle="None", label=f'Qz3')
# plt.plot(df['Noise Param'], df['Error Qz2'],marker = 'o',linestyle="None", label=f'Qz4')

# error_data = (df['Error Qx'].values.tolist())
# clm_data = []
# for i in range(len(error_data)):
#     std = np.sqrt(error_data[i]*(1-error_data[i])/10000)
#     print(std)
#     clm_data.append(1.96*std)
# plt.errorbar(df['Noise Param'], df['Error Qx'], yerr=clm_data, label='both limits (default)')


# plt.plot(df['Noise Param'], df['Error Qx'],marker = 'o',linestyle="none", label=f'Qx K=5')
# plt.plot(df1['Noise Param'], df1['Error Qx'],marker = 'x',linestyle="none", label=f'Qx K=7')
# plt.plot(df2['Noise Param'], df2['Error Qx'],marker = '^',linestyle="none", label=f'Qx K=10')
# plt.plot(df3['Noise Param'], df3['Error Qx'],marker = '*',linestyle="none", label=f'Qx K=10')


# df1 = (df.loc[:, ['Error Qz1', 'Error Qz2', 'Error Qz3','Error Qz4']] )


# df1 = (df.loc[:, ['Error Qz1', 'Error Qz2', 'Error Qz3']] )
# df1['mean'] = df1.mean(axis=1)
# print(df1)
# plt.plot(df['Noise Param'], df1['mean'],marker = 's',linestyle="None", label=f'Qz Average Worst')

# plt.yscale('log')

def error_prob_Qx(n,p): # X Error Rate
    # return pow(2,n-1) * (pow((p/4 + (1-p/2)/2),n) - pow(((1-p)/2),n))
    return 0.5 * ( 1 - pow((1 - p),(2*n - 2)) )

def error_prob_Qz_worst(n,p): # Worst Error Average
    sum = 0
    for i in range(n-2):
        sum += pow((1-p),i)
    temp = pow((1-p),3)/(n-2) * sum
    temp = 0.5 * (1 - temp)
    # return pow(2,n-1) * (pow((p/4 + (1-p/2)/2),n) - pow(((1-p)/2),n))
    return temp


def error_prob_Qz1(n,p): # First Qubit error average
    sum = 0
    for i in range(n-1):
        sum += pow((1-p),i)
    temp = pow((1-p),2)/(n-1) * sum
    temp = 0.5 * (1 - temp)
    # return pow(2,n-1) * (pow((p/4 + (1-p/2)/2),n) - pow(((1-p)/2),n))
    return temp

def error_prob_Qz(n,p,t): # First Qubit Alice and t qubit Bob
    return 0.5 * (1 - (pow((1-p), 2+n-t)))

def error_prob_Qz_q(p,q,t): # q Qubit Alice and t qubit Bob
    return 0.5 * (1 - (pow((1-p), 2+t-q)))

# print(error_prob_Qz(5,0))
# print(func1(0,0))
# prob = error_prob(10,1)  
k = 100
# prob_data = np.linspace(0,1,k)  
prob_data = np.logspace(-4, 0, num=k, endpoint=True)

qz_data1 = []
qz_data2 = []
qz_data3 = []
qz_data4 = []
qz_data1_10 = []
qz_data1_15 = []



mean_data1 = []
mean_worst_data = []
worst_data = []


mean_data1_10 = []
mean_worst_data_10 = []
worst_data_10 = []

n = 5
n1 = 7
n2 = 10
n3 = 15
qx_data = []
# qx_data1 = []
# qx_data2 = []
# qx_data3 = []

for i in range(k):
    qz_data1.append(error_prob_Qz(n,prob_data[i],2))
    # qz_data2.append(error_prob_Qz(n,prob_data[i],3))
    # qz_data3.append(error_prob_Qz(n,prob_data[i],4))
    qz_data4.append(error_prob_Qz(n,prob_data[i],5))
    qx_data.append(error_prob_Qx(n,prob_data[i]))
    # qz_data1_10.append(error_prob_Qz(n2,prob_data[i],2))
    # qz_data1_15.append(error_prob_Qz(n3,prob_data[i],2))
    
    # mean_data1.append(error_prob_Qz1(n,prob_data[i]))
    # mean_worst_data.append(error_prob_Qz_worst(n,prob_data[i]))
    # worst_data.append(error_prob_Qz_q(n,prob_data[i],2,n))
    # mean_data1_10.append(error_prob_Qz1(n2,prob_data[i]))
    # mean_worst_data_10.append(error_prob_Qz_worst(n2,prob_data[i]))
    # worst_data_10.append(error_prob_Qz_q(n2,prob_data[i],2,n2))
    
    
    # qx_data.append(error_prob_Qx(n,prob_data[i]))
    # qx_data1.append(error_prob_Qx(n1,prob_data[i]))
    # qx_data2.append(error_prob_Qx(n2,prob_data[i]))
    # qx_data3.append(error_prob_Qx(n3,prob_data[i]))
    # print(error_prob_Qx(n,prob_data[i]))
    # data.append(error_prob_Qz(n,prob_data[i],3))
    
    
# print(data)
# print(prob)
plt.plot(prob_data, qz_data1, label=f'Qz1 func')
# plt.plot(prob_data, qz_data2, label=f'Qz2 func')
# plt.plot(prob_data, qz_data3, label=f'Qz3 func')
plt.plot(prob_data, qz_data4, label=f'Qz4 func')
print(qz_data1)
# plt.plot(prob_data, qz_data1_10, label=f'Qz1_10 func')
# plt.plot(prob_data, qz_data1_15, label=f'Qz1_15 func')


# plt.plot(prob_data, qz_data2_10, label=f'Qz2_10 func')
# plt.plot(prob_data, qz_data3_10, label=f'Qz3_10 func')
# plt.plot(prob_data, qz_data4_10, label=f'Qz4_10 func')
# plt.plot(prob_data, qz_data5_10, label=f'Qz5_10 func')
# plt.plot(prob_data, qz_data6_10, label=f'Qz6_10 func')
# plt.plot(prob_data, qz_data7_10, label=f'Qz7_10 func')
# plt.plot(prob_data, qz_data8_10, label=f'Qz8_10 func')
# plt.plot(prob_data, qz_data9_10, label=f'Qz9_10 func')


# plt.plot(prob_data, qz_data1, label=f'Qz1 func')
# plt.plot(prob_data, qz_data2, label=f'Qz2 func')
# plt.plot(prob_data, qz_data3, label=f'Qz3 func')
# plt.plot(prob_data, qz_data4, label=f'Qz4 func')
# plt.plot(prob_data, mean_data1, label=f'Qz Function Average First Qubit')
# plt.plot(prob_data, mean_worst_data, label=f'Qz Function Worst Error Average')
# plt.plot(prob_data, worst_data, label=f'Qz Function Worst Error',linestyle='dashed')



# plt.plot(prob_data, mean_data1_10, label=f'Qz Function Average First Qubit K=10')
# plt.plot(prob_data, mean_worst_data_10, label=f'Qz Function Worst Error Average K=10')
# plt.plot(prob_data, worst_data_10, label=f'Qz Function Worst Error K=10',linestyle='dashed')

plt.xscale("log")
plt.yscale("log")
# plt.plot(prob_data, qx_data, label=f'Qx Function Error n=5')
# plt.plot(prob_data, qx_data1, label=f'Qx Function Error n=7')
# plt.plot(prob_data, qx_data2, label=f'Qx Function Error n=10')
# plt.plot(prob_data, qx_data3, label=f'Qx Function Error n=15')

ymax, ymin = plt.ylim()
# print(ymin)
# print(ymax)
# plt.xlabel('Noise parameter') 
plt.ylabel('Error Probability') 
plt.xlabel('Noise parameter p', color = 'black') 
plt.legend()

dict = {'Noise Param': prob_data, 'Error Qz': qx_data}  
df = pd.DataFrame(dict) 
# # saving the dataframe 
df.to_csv(f'error_prob_log_QX_K={n}_theory_v1.csv') 

# Display the plot
plt.show()



# ax1.plot(probs, data_error,'*-', color='blue',label='Qudit Protocol 8 nodes')