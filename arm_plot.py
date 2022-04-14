import numpy as np
from matplotlib import pyplot as plt
import arm_paras as ap


def read_data(data):
    x_all = data['x_all']
    z_all = data['z_all']
    s_all = data['s_all']
    t_all = data['t_all']
    tpm_all = data['tpm_all']
    ifreach_all = data['ifreach_all']
    time_steps_all = data['time_steps_all']
    return x_all, z_all, s_all, t_all, tpm_all, ifreach_all, time_steps_all


def plot_single_trajectory(data):
    x_all, z_all, s_all, t_all, tpm_all, ifreach_all, time_steps_all = read_data(data)
    index = np.random.randint(0, time_steps_all.shape[0])
    x_s = x_all[index, :, :]
    z_s = z_all[index, 0, :]
    s_s = s_all[index, 0, :]
    t_s = t_all[index, 0, :]
    tpm_s = tpm_all[index, :, :]
    ifreach_s = ifreach_all[index, 0, :]
    time_steps_s = time_steps_all[index, 0, :]

    plt.figure(1)
    plt.plot(time_steps_s, x_s[0, :], time_steps_s, x_s[1, :])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['State 1', 'State 2'])

    plt.figure(2)
    plt.plot(time_steps_s, ifreach_s)
    plt.xlabel('Time')
    plt.ylabel('If reach the constraint')

    plt.figure(3)
    plt.plot(time_steps_s, s_s)
    plt.xlabel('Time')
    plt.ylabel('Mode')

    plt.figure(4)
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
    xtrue_all = data['xtrue_all'][1:]
    xest_all = data['xest_all'][1:]
    strue_all = data['strue_all'][1:]
    mu_all = data['mu_all'][1:]

    plt.figure(1)
    plt.plot(time_steps, xtrue_all[:, 0], time_steps, xest_all[:, 0])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['True state 1', 'Estimated state 1'])
    plt.title('Trajectory of state 1')

    plt.figure(2)
    plt.plot(time_steps, xtrue_all[:, 1], time_steps, xest_all[:, 1])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['True state 2', 'Estimated state 2'])

    plt.figure(3)
    plt.plot(time_steps, strue_all, time_steps, mu_all)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['True mode', 'Mode1 probability', 'Mode2 probability', 'Mode3 probability'])

    plt.show()


if __name__ == '__main__':
    # data_path = ap.data_path
    # data = np.load(data_path)
    # plot_single_trajectory(data)
    data_path = ap.filter_data_path
    data = np.load(data_path)
    plot_result_single(data)
