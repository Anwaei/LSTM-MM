import numpy as np
from matplotlib import pyplot as plt
import tracking_paras as tkp


def read_data(data):
    x_all = data['x_all']
    z_all = data['z_all']
    s_all = data['s_all']
    t_all = data['t_all']
    tpm_all = data['tpm_all']
    ifreach_all = data['ifreach_all']
    time_steps_all = data['time_steps_all']
    return x_all, z_all, s_all, t_all, tpm_all, ifreach_all, time_steps_all


def plot_model():
    px = np.arange(start=-80, stop=140, step=1)
    py_above = tkp.ca3*px**3 + tkp.ca2*px**2 + tkp.ca1*px + tkp.ca0
    py_below = tkp.cb3*px**3 + tkp.cb2*px**2 + tkp.cb1*px + tkp.cb0
    plt.figure(1)
    plt.plot(px, py_above, px, py_below)
    plt.axis([-100, 160, 0, 130])
    plt.show()


def plot_single_trajectory(data):
    x_all, z_all, s_all, t_all, tpm_all, ifreach_all, time_steps_all = read_data(data)
    index = np.random.randint(0, time_steps_all.shape[0])
    x_s = x_all[index, :, :]
    z_s = z_all[index, :, :]
    s_s = s_all[index, 0, :]
    t_s = t_all[index, 0, :]
    tpm_s = tpm_all[index, :, :]
    ifreach_s = ifreach_all[index, 0, :]
    time_steps_s = time_steps_all[index, 0, :]

    plt.figure(1)
    plt.plot(time_steps_s, x_s[0, :], time_steps_s, x_s[1, :])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['Pos x', 'Pos y'])

    plt.figure(2)
    plt.plot(x_s[0, :], x_s[1, :])
    plt.scatter(tkp.x0[0], tkp.x0[1])
    plt.axis([-100, 160, 0, 130])

    px = np.arange(start=-80, stop=140, step=1)
    py_above = tkp.ca3 * px ** 3 + tkp.ca2 * px ** 2 + tkp.ca1 * px + tkp.ca0
    py_below = tkp.cb3 * px ** 3 + tkp.cb2 * px ** 2 + tkp.cb1 * px + tkp.cb0
    plt.plot(px, py_above, px, py_below, alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Trajectory', 'Road upper bound', 'Road lower bound'])

    plt.figure(3)
    plt.plot(time_steps_s, x_s[2, :], time_steps_s, x_s[3, :])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['Vel x', 'Vel y'])

    plt.figure(4)
    plt.plot(time_steps_s, ifreach_s)
    plt.xlabel('Time')
    plt.ylabel('If reach the constraint')

    plt.figure(5)
    plt.plot(time_steps_s, s_s)
    plt.xlabel('Time')
    plt.ylabel('Mode')

    plt.figure(6)
    plt.plot(time_steps_s, t_s)
    plt.xlabel('Time')
    plt.ylabel('Sojourn time')

    plt.show()


def plot_result_single(data):
    # ['xtrue_all', 'strue_all', 'xest_all', 'xp_all', 'w_all', 'mu_all', 'z_all', 'cmtp_all',
    # 'what_all', 'what_sum_all', 'zcliprd_all', 'v_all', 'xi_all', 'zeta_all', 'time_steps']
    # for name in data.files:
    #     exec(name+'=data['+name+']')
    time_steps = data['time_steps']
    xtrue_all = data['xtrue_all'][:, 1:, :]
    xest_all = data['xest_all'][:, 1:, :]
    strue_all = data['strue_all'][:, 1:]
    mu_all = data['mu_all'][:, 1:, :]
    index = 0

    plt.figure(1)
    plt.plot(time_steps, xtrue_all[index, :, 0], time_steps, xest_all[index, :, 0])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['True state 1', 'Estimated state 1'])
    plt.title('Trajectory of state 1')

    plt.figure(2)
    plt.plot(time_steps, xtrue_all[index, :, 1], time_steps, xest_all[index, :, 1])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['True state 2', 'Estimated state 2'])

    plt.figure(3)
    plt.plot(time_steps, strue_all[index, :], time_steps, mu_all[index, :])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['True mode', 'Mode1 probability', 'Mode2 probability', 'Mode3 probability'])

    plt.show()


def plot_rmse(data):
    time_steps = data['time_steps']
    xtrue_all = data['xtrue_all'][:, 1:, :]
    xest_all = data['xest_all'][:, 1:, :]
    strue_all = data['strue_all'][:, 1:]
    mu_all = data['mu_all'][:, 1:, :]

    rmse = np.sqrt(np.mean((xtrue_all-xest_all)**2, axis=0))

    plt.figure(1)
    plt.plot(time_steps, rmse)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['RMSE of state 1', 'RMSE of state 2'])
    plt.title('RMSE of states')
    plt.show()


def plot_compare(datas):
    time_steps = datas[0]['time_steps']
    xtrue_all = []
    xest_all = []
    strue_all = []
    mu_all = []
    for k in range(len(datas)):
        data = datas[k]
        xtrue_all.append(data['xtrue_all'][:, 1:, :])
        xest_all.append(data['xest_all'][:, 1:, :])
        strue_all.append(data['strue_all'][:, 1:])
        mu_all.append(data['mu_all'][:, 1:, :])

    index = 2
    plt.figure(1)
    # plt.hold(True)
    plt.plot(time_steps, xtrue_all[0][index, :, 0])
    for k in range(len(datas)):
        plt.plot(time_steps, xest_all[k][index, :, 0])
    plt.xlabel('Time')
    plt.ylabel('Value')
    legends=['True value', 'LSTM-MM Estimation', 'IMM Estimation', 'IMMPF Estimation']
    plt.legend(legends[0: len(datas)+1])
    plt.title('Trajectory of state 1')

    plt.figure(2)
    # plt.hold(True)
    plt.plot(time_steps, xtrue_all[0][index, :, 1])
    for k in range(len(datas)):
        plt.plot(time_steps, xest_all[k][index, :, 1])
    plt.xlabel('Time')
    plt.ylabel('Value')
    legends=['True value', 'LSTM-MM Estimation', 'IMM Estimation', 'IMMPF Estimation']
    plt.legend(legends[0: len(datas)+1])
    plt.title('Trajectory of state 2')

    plt.figure(3)
    # plt.hold(True)
    plt.plot(time_steps, strue_all[0][index, :])
    for k in range(len(datas)):
        plt.plot(time_steps, mu_all[k][index, :])
    plt.xlabel('Time')
    plt.ylabel('Value')
    legends = []
    legends.append('True Mode')
    for j in range(tkp.M):
        legends.append('LSTM-MM Mode' + str(j+1))
    for j in range(tkp.M):
        legends.append('IMM Mode' + str(j+1))
    for j in range(tkp.M):
        legends.append('IMMPF Mode' + str(j+1))
    plt.legend(legends[0: tkp.M * len(datas) + 1])
    plt.title('Mode probabilities')

    plt.figure(4)
    # plt.hold(True)
    legends=[]
    for k in range(len(datas)):
        rmse = np.sqrt(np.mean((xtrue_all[k]-xest_all[k])**2, axis=0))
        plt.plot(time_steps, rmse)
    legends.append('LSTM-MM rmse for state 1')
    legends.append('LSTM-MM rmse for state 2')
    legends.append('IMM rmse for state 1')
    legends.append('IMM rmse for state 2')
    legends.append('IMMPF rmse for state 1')
    legends.append('IMMPF rmse for state 2')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(legends[0: tkp.M * len(datas)])
    plt.title('RMSE')

    plt.show()


if __name__ == '__main__':
    data_path = tkp.data_path
    data = np.load(data_path)
    plot_single_trajectory(data)
    # plot_model()

    # which_net = 'pi_int'
    # data_path = tkp.filter_data_path + '_' + which_net + '.npz'
    # data_npi_int = np.load(data_path)
    # plot_result_single(data_npi_int)
    # plot_rmse(data_npi_int)

    # data_path = tkp.filter_data_path + '_' + 'IMM' + '.npz'
    # data_imm = np.load(data_path)
    # plot_result_single(data_imm)
    # plot_rmse(data_imm)

    # data_path = tkp.filter_data_path + '_' + 'IMMPF' + '.npz'
    # data_immpf = np.load(data_path)
    #
    # plot_compare([data_npi_int, data_imm, data_immpf])
