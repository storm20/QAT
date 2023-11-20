
import numpy as np
import matplotlib.pyplot as plt


# probs = np.linspace(0, 1, num=25, endpoint=True)
# probs1 = np.linspace(0, 1, num=25, endpoint=True)

probs = np.logspace(-4, 0, num=10, endpoint=True)
probs1 = np.logspace(-4, 0, num=10, endpoint=True)

x_2d, y_2d = np.meshgrid(probs,probs1)

filename = './z_values.csv'  # Replace with the actual path to your CSV file
z_values = np.genfromtxt(filename, delimiter=',')
print(z_values)

cp1 = plt.contourf(x_2d, y_2d, z_values,levels=10)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('State Preparation Noise Parameter')
plt.ylabel('Channel Noise parameter')
plt.title('QATE Fidelity (Depolarizing) N=10')
plt.colorbar(label='Average Fidelity')

plt.savefig('contour_plot.png')
plt.show()
