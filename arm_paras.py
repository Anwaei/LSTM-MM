import numpy as np

data_path = 'data/data_arm.npz'
net_path_pi_int = 'nets/net_arm_pi_int'
net_path_pi_para2 = 'nets/net_arm_pi_para2'
net_path_pi_para3 = 'nets/net_arm_pi_para3'
net_path_npi_int = 'nets/net_arm_npi_int'
net_path_npi_para2 = 'nets/net_arm_npi_para2'
net_path_npi_para3 = 'nets/net_arm_npi_para3'

batch_size = 20000

T = 5
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
b = -0.2
# b = 0.53

# Parameters of s2<->s3
# q23 = 0.9
# r23 = 1.2
# q32 = 0.95
# r32 = 1.5

q23 = 0.8
r23 = 0.5
q32 = 0.7
r32 = 0.2

# Non switch situation
# q23 = 1
# r23 = 0.5
# q32 = 0
# r32 = 0.3

"""""""""""""""""""""
Constraint
"""""""""""""""""""""
x1_c = 0.5
x2_c = 2.5

# Non constraint situation
# x1_c = 1000
# x2_c = 1000

"""""""""""""""""""""
Initial
"""""""""""""""""""""

x0 = np.array([0, 2.0])
s0 = 1

"""""""""""""""""""""
Network
"""""""""""""""""""""

T_max_parallel = [None, 100, 120]  # Mode 2 and 3, no mode 1
T_max_integrated = 100

units_mlp_x = [10, 10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_pi_para2 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [10, 10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_pi_para3 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [10, 10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_npi_para2 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [10, 10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_npi_para3 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [10, 10, 10]
units_mlp_s = [10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_pi_int = {'mlp_x': units_mlp_x,
                'mlp_s': units_mlp_s,
                'lstm': units_lstm,
                'mlp_c': units_mlp_c}

units_mlp_x = [10, 10, 10]
units_mlp_s = [10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_npi_int = {'mlp_x': units_mlp_x,
                 'mlp_s': units_mlp_s,
                 'lstm': units_lstm,
                 'mlp_c': units_mlp_c}


bs = 64

train_prop = 0.7  # Proportion of training data