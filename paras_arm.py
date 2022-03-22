import numpy as np

T = 10
dt = 0.01

g = 9.81
l0 = 0.6
B = 5
m = [1, 2.5, 5]
J = [1, 2.5, 5]

Q = np.diag[0.1, 0.1]
R = np.diag[0.1]

b = 0.523  # Boundary of s1/s2
q23 = 0.8
r23 = 1.2
q32 = 0.9
r32 = 1.5
