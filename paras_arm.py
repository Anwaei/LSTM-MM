import numpy as np

data_path = 'data/data_arm.npz'

batch_size = 10

T = 10
dt = 0.1

g = 9.81
l0 = 0.6
B = 5
m = [1, 2.5, 5]
J = [1, 2.5, 5]

Q = np.diag([0.1, 0.1])
R = np.diag([0.1])

b = 0.523  # Boundary of s1/s2
q23 = 0.9
r23 = 1.2
q32 = 0.95
r32 = 1.5

x1_c = 0.9
x2_c = 3.5

x0 = np.array([0, 2.4])
s0 = 1
