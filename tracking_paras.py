import numpy as np

data_path = 'data/tracking_data.npz'
net_path_pi_int = 'nets/tracking_net_pi_int'
net_path_pi_para1 = 'nets/tracking_net_pi_para1'
net_path_pi_para2 = 'nets/tracking_net_pi_para2'
net_path_pi_para3 = 'nets/tracking_net_pi_para3'
net_path_npi_int = 'nets/tracking_net_npi_int'
net_path_npi_para1 = 'nets/tracking_net_npi_para1'
net_path_npi_para2 = 'nets/tracking_net_npi_para2'
net_path_npi_para3 = 'nets/tracking_net_npi_para3'

filter_data_path = 'data/tracking_results'

batch_size = 20000

T = 5
dt = 0.02

"""""""""""""""""""""
Model
"""""""""""""""""""""


Q = np.diag([0.001, 0.001])
R = np.diag([0.01, 0.01])
G = np.array([[1/2*(dt**2), 0],
                  [1/2*(dt**2), 0],
                  [0, dt],
                  [0, dt]])

nx = 4
nz = 2

omegal = 0.5
omegar = -0.5
swl = np.sin(omegal*dt)
cwl = np.cos(omegal*dt)
swr = np.sin(omegar*dt)
cwr = np.cos(omegar*dt)
apos = 10
aneg = -10

psx = 0
psy = 0
psz = 100

"""""""""""""""""""""
Transition
"""""""""""""""""""""

M = 3
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

lambda1 = 15
lambda2 = 10

"""""""""""""""""""""
Initial
"""""""""""""""""""""

x0 = np.array([0, 2.0])
s0 = 1
Q0 = np.diag([0.001, 0.001])

"""""""""""""""""""""
Network
"""""""""""""""""""""

T_max_parallel = [50, 100, 120]  # \del{Mode 2 and 3, no mode 1} Including mode 1
T_max_integrated = 100

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
units_mlp_s = [32, 32]
units_lstm = [32, 32, 32]
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

Np = 5000
run_batch = 10

Pi_IMM = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])