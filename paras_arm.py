import numpy as np

data_path = 'data/data_arm.npz'

batch_size = 10

T = 10
dt = 0.01

"""""""""""""""""""""
Model
"""""""""""""""""""""

g = 9.81
l0 = 0.6
B = 2
m = [1, 2.5, 5]
J = [1, 2.5, 5]

Q = np.diag([0.001, 0.001])
R = np.diag([0.1])

"""""""""""""""""""""
Transition
"""""""""""""""""""""

# Boundary of s1/s2
b = 0.2
# b = 0.53

# Parameters of s2<->s3
# q23 = 0.9
# r23 = 1.2
# q32 = 0.95
# r32 = 1.5

q23 = 0.8
r23 = 0.5
q32 = 0.7
r32 = 0.4

# Non switch situation
# q23 = 1
# r23 = 0.5
# q32 = 0
# r32 = 0.3

"""""""""""""""""""""
Constraint
"""""""""""""""""""""
x1_c = 0.9
x2_c = 3.5

# Non constraint situation
# x1_c = 1000
# x2_c = 1000

"""""""""""""""""""""
Initial
"""""""""""""""""""""

x0 = np.array([0, 2.4])
s0 = 1

"""""""""""""""""""""
Network
"""""""""""""""""""""
units_mlp_x = [10, 10]
units_mlp_s = [10, 10]
units_lstm = [10, 10]
units_mlp_c = [15, 15]  # Except last layer

T_max_parallel = [50, 50]  # Mode 2 and 3, no mode 1
T_max_integral = 100

train_prop = 0.7  # Proportion of training data
