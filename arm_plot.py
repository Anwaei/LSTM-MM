import numpy as np
from matplotlib import pyplot as plt
import arm_paras as ap
from tensorflow import keras


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

    if ap.nz == 1:
        plt.figure(5)
        plt.plot(time_steps_s, x_s[0, :])
        plt.scatter(time_steps_s, z_s, s=12, c='tab:orange')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(['State 1', 'Measurement'])
    else:
        plt.figure(5)
        plt.plot(time_steps_s, z_s[0, :], time_steps_s, z_s[1, :])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(['Measurement 1', 'Measurement 2'])

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


def plot_compare(datas, labels):
    index = 1

    time_steps = datas[0]['time_steps']
    xtrue_all = []
    xest_all = []
    strue_all = []
    mu_all = []
    ztrue = datas[-1]['z_all'][index][1:, 0]
    for k in range(len(datas)):
        data = datas[k]
        xtrue_all.append(data['xtrue_all'][:, 1:, :])
        xest_all.append(data['xest_all'][:, 1:, :])
        strue_all.append(data['strue_all'][:, 1:])
        mu_all.append(data['mu_all'][:, 1:, :])

    plt.figure(1, figsize=(7.2, 5.4))
    plt.plot(time_steps, xtrue_all[0][index, :, 0], c='tab:brown', ls='--', lw=2, label='True value')
    legends = ['True value']
    for k in range(len(datas)):
        plt.plot(time_steps, xest_all[k][index, :, 0], label=labels[k]+' Estimation')
    if ap.nz == 1:
        plt.scatter(time_steps, ztrue, s=12, c='tab:orange', label='Measurement')
    con_a = np.ones(shape=time_steps.shape) * ap.x1_c
    con_b = -np.ones(shape=time_steps.shape) * ap.x1_c
    plt.plot(time_steps, con_a, time_steps, con_b, c='tab:olive', ls='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Trajectory of state 1')

    plt.figure(2, figsize=(7.2, 5.4))
    plt.plot(time_steps, xtrue_all[0][index, :, 1], c='tab:brown', ls='--', lw=2)
    legends = ['True value']
    for k in range(len(datas)):
        plt.plot(time_steps, xest_all[k][index, :, 1])
        legends.append(labels[k] + ' Estimation')
    con_a = np.ones(shape=time_steps.shape) * ap.x2_c
    con_b = -np.ones(shape=time_steps.shape) * ap.x2_c
    plt.plot(time_steps, con_a, time_steps, con_b, c='tab:olive', ls='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(legends)
    plt.title('Trajectory of state 2')

    plt.figure(3)
    plt.subplot(ap.M+1, 1, 1)
    plt.plot(time_steps, strue_all[0][index, :], c='tab:brown')
    plt.xlabel('Time')
    plt.ylabel('Value')
    legends = list()
    legends.append('True Mode')
    plt.legend(legends, loc='upper right')
    ce = np.zeros(len(datas))
    strue_oh = keras.utils.to_categorical(strue_all[0][index, :]-1)
    for j in range(ap.M):
        legends = list()
        plt.subplot(ap.M+1, 1, j+2)
        for k in range(len(datas)):
            w = 1.5 if 'IMM' in labels[k] else 2
            s = '--' if 'IMM' in labels[k] else '-'
            plt.plot(time_steps, mu_all[k][index, :, j], ls=s, lw=w)
            legends.append(labels[k] + ' Mode' + str(j + 1))
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(legends, loc='upper right')
    # plt.suptitle('Mode Probabilities')
    for k in range(len(datas)):
        ce[k] = np.mean(keras.losses.categorical_crossentropy(strue_oh, mu_all[k][index, :, :]))
        print(labels[k] + ' CCE: ' + str(ce[k]))

    for n in range(ap.nx):
        plt.figure(4+n, figsize=(6.4, 4.8))
        legends = list()
        for k in range(len(datas)):
            rmse = np.sqrt(np.mean((xtrue_all[k] - xest_all[k]) ** 2, axis=0))
            plt.plot(time_steps, rmse[:, n])
            legends.append(labels[k] + ' rmse of state ' + str(n + 1))
            print(labels[k] + str(n + 1) + ':' + str(rmse[:, n].mean()))
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(legends, loc='upper right')
        plt.title('RMSE of state'+str(n+1))

    plt.show()


if __name__ == '__main__':
    # data_path = ap.data_path
    # data = np.load(data_path)
    # plot_single_trajectory(data)

    which_net = 'pi_int'
    data_path = ap.filter_data_path+'_'+which_net+'.npz'
    data_pi_int = np.load(data_path)
    # plot_result_single(data_npi_int)
    # plot_rmse(data_npi_int)

    which_net = 'npi_int'
    data_path = ap.filter_data_path+'_'+which_net+'.npz'
    data_npi_int = np.load(data_path)
    # plot_result_single(data_npi_int)
    # plot_rmse(data_npi_int)

    # which_net = 'pi_para'
    # data_path = ap.filter_data_path+'_'+which_net+'.npz'
    # data_pi_para = np.load(data_path)

    # which_net = 'npi_para'
    # data_path = ap.filter_data_path+'_'+which_net+'.npz'
    # data_npi_para = np.load(data_path)

    data_path = ap.filter_data_path + '_' + 'IMM' + '.npz'
    data_imm = np.load(data_path)
    # plot_result_single(data_imm)
    # plot_rmse(data_imm)

    data_path = ap.filter_data_path + '_' + 'IMMPF-5000' + '.npz'
    data_immpf_5000 = np.load(data_path)

    data_path = ap.filter_data_path + '_' + 'IMMPF-500' + '.npz'
    data_immpf_500 = np.load(data_path)

    data_path = ap.filter_data_path + '_' + 'optPF' + '.npz'
    data_optpf = np.load(data_path)

    plot_compare(datas=[data_imm, data_immpf_500, data_immpf_5000, data_npi_int, data_pi_int],
                 labels=['IMM-EKF', 'IMM-PF-500', 'IMM-PF-5000', 'DLMM-NoPi', 'DLMM-Pi'])

    # plot_compare(datas=[data_npi_int, data_pi_int, data_imm, data_immpf_500],
    #              labels=['LSTM-MM-NoPi', 'LSTM-MM-Pi', 'IMM-EKF', 'IMM-PF-500'])

    # plot_compare(datas=[data_imm, data_immpf_500],
    #              labels=['IMM-EKF', 'IMM-PF-500'])
