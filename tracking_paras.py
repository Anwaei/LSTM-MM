import numpy as np

data_path = 'data/tracking_data.npz'
net_path_pi_int = 'nets/tracking_net_pi_int'
net_path_pi_para1 = 'nets/tracking_net_pi_para1'
net_path_pi_para2 = 'nets/tracking_net_pi_para2'
net_path_pi_para3 = 'nets/tracking_net_pi_para3'
net_path_pi_para4 = 'nets/tracking_net_pi_para4'
net_path_pi_para5 = 'nets/tracking_net_pi_para5'
net_path_npi_int = 'nets/tracking_net_npi_int'
net_path_npi_para1 = 'nets/tracking_net_npi_para1'
net_path_npi_para2 = 'nets/tracking_net_npi_para2'
net_path_npi_para3 = 'nets/tracking_net_npi_para3'
net_path_npi_para4 = 'nets/tracking_net_npi_para4'
net_path_npi_para5 = 'nets/tracking_net_npi_para5'

filter_data_path = 'data/tracking_results'

batch_size = 50000

T = 30
dt = 0.2

"""""""""""""""""""""
Model
"""""""""""""""""""""


nq = 2
Q = np.diag([0.001, 0.001])
R = np.diag([0.01, 0.01])
G = np.array([[1/2*(dt**2), 0],
                  [1/2*(dt**2), 0],
                  [0, dt],
                  [0, dt]])

nx = 4
nz = 2

omegal = 0.12
omegar = -0.12
swl = np.sin(omegal*dt)
cwl = np.cos(omegal*dt)
swr = np.sin(omegar*dt)
cwr = np.cos(omegar*dt)
apos = 0.8
aneg = -0.4

psx = 0
psy = 0
psz = 100

"""""""""""""""""""""
Transition
"""""""""""""""""""""

M = 5

tlast = 10

alpha12 = 0.001
nu12 = 25
px_tcp1 = -57.4879
py_tcp1 = 56.6634
Sigma12 = [10, 10]


alpha21 = 0.001
nu21 = 0.12
psi21 = -0.2295
sigma21 = 0.005

alpha13 = 0.001
nu13 = 50
px_tcp2 = 68.4919
py_tcp2 = 53.6949
Sigma13 = [15, 15]

alpha31 = 0.001
nu31 = 0.12
psi31 = 0.8516
sigma31 = 0.005

alpha14 = 0.001
nu14 = 0.8
psi14 = 0.5

alpha41 = 0.001
nu41 = 0.4
psi41 = 2

alpha15 = 0.001
nu15 = 0.5
psi15 = 2

alpha51 = 0.001
nu51 = 0.8
psi51 = 1.8


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

lambda1 = 10
lambda2 = 10
lambda3 = 1

"""""""""""""""""""""
Initial
"""""""""""""""""""""
v0 = 6
psi0 = 0.8186
x0 = np.array([-75.9313, 36.9728, v0*np.cos(psi0), v0*np.sin(psi0)])
s0 = 1
Q0 = np.diag([0.001, 0.001, 0.001, 0.001])

"""""""""""""""""""""
Network
"""""""""""""""""""""

T_max_parallel = [50, 50, 50, 50, 50]  #
T_max_integrated = 64

units_mlp_x = [24, 24, 24]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_pi_para1 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [24, 24, 24]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_pi_para2 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [24, 24, 24]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_pi_para3 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [24, 24, 24]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_pi_para4 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [24, 24, 24]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_pi_para5 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_npi_para1 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_npi_para2 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_npi_para3 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_npi_para4 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [16, 16, 16]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
units_npi_para5 = {'mlp_x': units_mlp_x,
                  'lstm': units_lstm,
                  'mlp_c': units_mlp_c}

units_mlp_x = [32, 32, 32]
units_mlp_s = [32, 32]
units_lstm = [32, 32, 32]
units_mlp_c = [32, 32, 64]  # Except last layer
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

Np = 250
run_batch = 10

Pi_IMM = np.array([[0.9, 0.025, 0.025, 0.025, 0.025],
                   [0.025, 0.9, 0.025, 0.025, 0.025],
                   [0.025, 0.025, 0.9, 0.025, 0.025],
                   [0.025, 0.025, 0.025, 0.9, 0.025],
                   [0.025, 0.025, 0.025, 0.025, 0.9]])
