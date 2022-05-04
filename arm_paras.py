import numpy as np
from scipy import stats

data_path = 'data/arm_data.npz'
net_path_pi_int = 'nets/arm_net_pi_int'
net_path_pi_para1 = 'nets/arm_net_pi_para1'
net_path_pi_para2 = 'nets/arm_net_pi_para2'
net_path_pi_para3 = 'nets/arm_net_pi_para3'
net_path_npi_int = 'nets/arm_net_npi_int'
net_path_npi_para1 = 'nets/arm_net_npi_para1'
net_path_npi_para2 = 'nets/arm_net_npi_para2'
net_path_npi_para3 = 'nets/arm_net_npi_para3'

filter_data_path = 'data/arm_results'

batch_size = 50000

T = 10
dt = 0.1

"""""""""""""""""""""
Model
"""""""""""""""""""""

g = 9.81
l0 = 0.6
B = 2
# m = [0.1, 1, 10]
# J = [0.1, 1, 10]
m = [0.5, 5, 50]
J = [0.5, 5, 50]

# Q_gene = np.diag([0.001, 0.001])
Q_gene = 0
Q = np.diag([0.001, 0.001])
# R = np.diag([0.05, 0.05])
R = np.diag([0.0005])
R_filter = np.diag([0.05])

nx = 2
nz = 1

"""""""""""""""""""""
Transition
"""""""""""""""""""""

M = 3
t_last = 15
# Boundary of s1/s2
b = -0.25
# b = 0.53

# Parameters of s2<->s3
# q23 = 0.9
# r23 = 1.2
# q32 = 0.95
# r32 = 1.5

q23 = 0.8
r23 = 0.6
q32 = 0.6
r32 = 0.4

# Non switch situation
# q23 = 1
# r23 = 0.5
# q32 = 0
# r32 = 0.3

"""""""""""""""""""""
Constraint
"""""""""""""""""""""
x1_c = 0.3
x2_c = 1.2

# Non constraint situation
# x1_c = 1000
# x2_c = 1000

lambda1 = 100
lambda2 = 50

"""""""""""""""""""""
Initial
"""""""""""""""""""""

x0 = np.array([0, 1.2])
s0 = 3
Q0 = np.diag([0.001, 0.001])

"""""""""""""""""""""
Network
"""""""""""""""""""""

T_max_parallel = [3, 30, 30]  # \del{Mode 2 and 3, no mode 1} Including mode 1
T_max_integrated = 30

units_mlp_x = [10, 10, 10]
units_lstm = [10, 10, 10]
units_mlp_c = [15, 25, 50]  # Except last layer
units_pi_para1 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

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
units_npi_para1 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [16, 16, 16]
units_mlp_c = [15, 25, 50]  # Except last layer
units_npi_para2 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [16, 16, 16]
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

units_mlp_x = [32, 32, 32]
units_mlp_s = [32, 32, 32]
units_lstm = [32, 32, 32, 32]
units_mlp_c = [32, 32, 32]  # Except last layer
units_npi_int = {'mlp_x': units_mlp_x,
                 'mlp_s': units_mlp_s,
                 'lstm': units_lstm,
                 'mlp_c': units_mlp_c}

bs = 64

train_prop = 0.9  # Proportion of training data

"""""""""""""""""""""
Filtering
"""""""""""""""""""""

Np = 500
run_batch = 10

Pi_IMM = np.array([[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]])