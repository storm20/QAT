import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# df = pd.read_csv('error_prob_QZ_K=5_N=5_shift1.csv')



df = pd.read_csv('QAE_Fidelity_K=4_d=1_s=1.csv')
df1 = pd.read_csv('QAE_Fidelity_K=4_d=1_s=2.csv')
# df2 = pd.read_csv('QAE_Fidelity_K=5_d=1_s=1.csv')
# df3 = pd.read_csv('QAE_Fidelity_K=5_d=4_s=0.csv')
# df4 = pd.read_csv('QAE_Fidelity_K=5_d=2_s=1.csv')





plt.figure(figsize=(8, 6))  
# plt.plot(df['Noise Param'], df['Error Qx'], label=f'n={n}')
plt.plot(df['Noise Param'], df['Fidelity'],linestyle="solid", label=f's=0')
plt.plot(df['Noise Param'], df1['Fidelity'],linestyle="solid", label=f's=1')
# plt.plot(df['Noise Param'], df2['Fidelity'],linestyle="solid", label=f's=2')
# plt.plot(df['Noise Param'], df3['Fidelity'],linestyle="solid", label=f'd=4')
# plt.plot(df['Noise Param'], df4['Fidelity'],linestyle="solid", label=f's=1 d=2')




# print(error_prob_Qz(5,0))
# print(func1(0,0))
# prob = error_prob(10,1)  
# k = 100
# prob_data = np.linspace(0,1,k)  
# prob_data = np.logspace(-4, 0, num=k, endpoint=True)

# plt.xscale("log")
# plt.yscale("log")


ymax, ymin = plt.ylim()
# print(ymin)
# print(ymax)
# plt.xlabel('Noise parameter') 
plt.ylabel('Fidelity') 
plt.xlabel('Noise parameter p', color = 'black') 
plt.legend()

# dict = {'Noise Param': prob_data, 'Error Qz': qz_data4}  
# df = pd.DataFrame(dict) 
# # # saving the dataframe 
# df.to_csv(f'error_prob_log_QZ_theory.csv') 

# Display the plot
plt.show()



# ax1.plot(probs, data_error,'*-', color='blue',label='Qudit Protocol 8 nodes')