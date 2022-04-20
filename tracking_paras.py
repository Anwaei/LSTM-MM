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

batch_size = 20

T = 100
dt = 0.5

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

M = 5

tlast = 20

alpha12 = 0.1
nu12 = 0.6
px_tcp1 = -50
py_tcp1 = -50
Sigma12 = [5, 5]


alpha21 = 0.1
nu21 = 0.6
psi21 = -0.5
sigma21 = 0.2

alpha13 = 0.1
nu13 = 0.6
px_tcp2 = 100
py_tcp2 = 50
Sigma13 = [5, 5]

alpha31 = 0.1
nu31 = 0.6
psi31 = 0.7
sigma31 = 0.2

alpha14 = 0.1
nu14 = 0.6
psi14 = 1

alpha41 = 0.1
nu41 = 0.6
psi41 = 0.5

alpha15 = 0.1
nu15 = 0.6
psi15 = 0.5

alpha51 = 0.1
nu51 = 0.6
psi51 = 0.5


"""""""""""""""""""""
Constraint
"""""""""""""""""""""
ca3 = 5*10**(-5)
ca2 = -0.004
ca1 = -0.2
ca0 = 75
cb3 = 4.899*10**(-5)
cb2 = -0.0047
cb1 = -0.1521
cb0 = 65.4109

vmax = 12.5
ve = 10

lambda1 = 4
lambda2 = 4
lambda3 = 1

"""""""""""""""""""""
Initial
"""""""""""""""""""""

x0 = np.array([-80, 40, 5, 5])
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