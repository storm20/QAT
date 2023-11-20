import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch
import qutip 
# def plot_bloch_vectors(qubit_states):
#     # Create a Bloch sphere
#     bloch_sphere = Bloch()
#     # bloch_sphere = Bloch(axes=[0,0,1])

#     # Add points for each qubit state to the Bloch sphere
#     # bloch_sphere.add_points(qubit_states)
#     for state_vector in qubit_states:
#         # print(state_vector)
#         bloch_sphere.add_points(state_vector)

#     # Show the Bloch sphere
#     bloch_sphere.show()
    
    
    
def transform_states(qubit_states):
    # transformed_states = []
    temp_x = []
    temp_y = []
    temp_z = []
    for state in qubit_states:
        # print(state)
        theta = 2*np.arccos(state[0])
        x = np.sin(state[2])*np.cos(theta)
        temp_x.append(x)
        y = np.sin(state[2])*np.sin(theta)
        temp_y.append(y)
        z = np.cos(state[2])
        temp_z.append(z)
        # print(temp)
    transformed_states = [temp_x,temp_y,temp_z]
    # print(transform_states)
    return np.array(transformed_states)

# Number of qubit states to sample
num_samples = 1000
np.random.seed(0)
# Generate random values for |a|, |b|, and c
a_values = np.sqrt(np.random.rand(num_samples))
# a_values = (np.random.rand(num_samples))
# print(a_values)
# a_values = np.array([0])
c_values = 2 * np.pi * np.random.rand(num_samples)
# c_values = 0
# Calculate |b| based on the constraint |a|^2 + |b|^2 = 1
b_values = np.sqrt(1 - a_values**2)


# Construct the qubit states
qubit_states = np.array([a_values, b_values ,  c_values]).T
# print(qubit_states)

# qubit_states = [[0,1,np.pi]]
# print(np.array([a_values,b_values*np.exp( 1j*  c_values)]).T)
test = transform_states(qubit_states)
# print(test)


bloch_sphere = Bloch()
bloch_sphere.point_size = [5, 62, 65, 75]
bloch_sphere.add_points(test)
bloch_sphere.zlpos = [1.2, -1.32]
bloch_sphere.xlpos = [1.5, -1.1]

# bloch_sphere.render()
bloch_sphere.show()
plt.savefig('bloch_sphere.png')

# Plot multiple qubit states on the Bloch sphere
# plot_bloch_vectors(test)
# th = np.linspace(0, 2*np.pi, 20)
# xp = np.cos(th)
# yp = np.sin(th)
# zp = np.zeros(20)

# pnts = [xp, yp, zp]
# print(pnts)

# Show the plot
plt.show()
