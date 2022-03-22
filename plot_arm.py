import numpy as np
from matplotlib import pyplot as plt
import paras_arm as pa


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
    index = 1
    x_s = x_all[index, :, :]
    z_s = z_all[index, :, :]
    s_s = s_all[index, :, :]
    t_s = t_all[index, :, :]
    tpm_s = tpm_all[index, :, :]
    ifreach_s = ifreach_all[index, :, :]
    time_steps_s = time_steps_all[index, :, :]

    plt.figure(1)
    plt.plot(t_s, x_s[:, 0, :], t_s, x_s[:, 1, :], t_s, z_s)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['State 1', 'State 2', 'Measurement'])

    plt.show()


if __name__ == '__main__':
    data_path = pa.data_path
    data = np.load(data_path)
    plot_single_trajectory(data)
