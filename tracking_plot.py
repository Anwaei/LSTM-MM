import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import tracking_paras as tkp
import tracking_model as tkm


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
    py = np.arange(start=-20, stop=120, step=1)
    pxm, pym = np.meshgrid(px, py)
    py_above = tkp.ca3*px**3 + tkp.ca2*px**2 + tkp.ca1*px + tkp.ca0
    py_below = tkp.cb3*px**3 + tkp.cb2*px**2 + tkp.cb1*px + tkp.cb0
    plt.figure(1)
    plt.plot(px, py_above, px, py_below)
    plt.axis([-100, 160, 0, 130])
    # plt.show()

    angle = np.arange(start=-np.pi/2, stop=np.pi/2, step=0.01)
    vel = np.arange(start=0, stop=15, step=0.1)

    tpm12 = np.zeros(shape=pxm.shape)
    for ix in range(px.size):
        for iy in range(py.size):
            tpm12[iy, ix] = tkp.alpha12 + tkp.nu12 * tkm.pdf_Gaussian(x=[px[ix], py[iy]],
                                                                      mean=[tkp.px_tcp1, tkp.py_tcp1],
                                                                      cov=tkp.Sigma12)
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X=pxm, Y=pym, Z=tpm12)
    ax.plot(px, py_above)
    ax.plot(px, py_below)
    plt.title('tp12')

    tpm21 = np.zeros(shape=angle.size)
    for ia in range(angle.size):
        tpm21[ia] = tkp.alpha21 + tkp.nu21 * tkm.pdf_Gaussian(x=angle[ia], mean=tkp.psi21, cov=tkp.sigma21)
    plt.figure(3)
    plt.plot(angle, tpm21)
    plt.xlabel('Angle')
    plt.title('tp21')

    tpm13 = np.zeros(shape=pxm.shape)
    for ix in range(px.size):
        for iy in range(py.size):
            tpm13[iy, ix] = tkp.alpha13 + tkp.nu13 * tkm.pdf_Gaussian(x=[px[ix], py[iy]],
                                                                      mean=[tkp.px_tcp2, tkp.py_tcp2],
                                                                      cov=tkp.Sigma13)
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X=pxm, Y=pym, Z=tpm13)
    ax.plot(px, py_above)
    ax.plot(px, py_below)
    plt.title('tp13')

    tpm31 = np.zeros(shape=angle.size)
    for ia in range(angle.size):
        tpm31[ia] = tkp.alpha31 + tkp.nu31 * tkm.pdf_Gaussian(x=angle[ia], mean=tkp.psi31, cov=tkp.sigma31)
    plt.figure(5)
    plt.plot(angle, tpm31)
    plt.xlabel('Angle')
    plt.title('tp31')

    tpm14 = np.zeros(shape=vel.size)
    for iv in range(vel.size):
        tpm14[iv] =tkp.alpha14 + tkp.nu14 * tkp.psi14 * np.exp(-tkp.psi14 * vel[iv])
    plt.figure(6)
    plt.plot(vel, tpm14)
    plt.xlabel('Velocity')
    plt.title('tp14')

    tpm41 = np.zeros(shape=vel.size)
    for iv in range(vel.size):
        tpm41[iv] = tkp.alpha41 + tkp.nu41 / (1 + np.exp(-tkp.psi41 * (vel[iv] - tkp.ve)))
    plt.figure(7)
    plt.plot(vel, tpm41)
    plt.xlabel('Velocity')
    plt.title('tp41')

    tpm15 = np.zeros(shape=vel.size)
    for iv in range(vel.size):
        tpm15[iv] = tkp.alpha15 + tkp.nu15 / (1 + np.exp(-tkp.psi15 * (vel[iv] - tkp.vmax)))
    plt.figure(8)
    plt.plot(vel, tpm15)
    plt.xlabel('Velocity')
    plt.title('tp15')

    tpm51 = np.zeros(shape=vel.size)
    for iv in range(vel.size):
        tpm51[iv] = tkp.alpha51 + tkp.nu51 / (1 + np.exp(-tkp.psi51 * (-vel[iv] + tkp.ve)))
    plt.figure(9)
    plt.plot(vel, tpm51)
    plt.xlabel('Velocity')
    plt.title('tp51')

    plt.show()

    # tpm[0, 1] = tkp.alpha12 + tkp.nu12 * pdf_Gaussian(x=[px, py], mean=[tkp.px_tcp1, tkp.py_tcp1], cov=tkp.Sigma12)
    # tpm[1, 0] = tkp.alpha21 + tkp.nu21 * pdf_Gaussian(x=angle, mean=tkp.psi21, cov=tkp.sigma21)
    # tpm[0, 2] = tkp.alpha13 + tkp.nu13 * pdf_Gaussian(x=[px, py], mean=[tkp.px_tcp2, tkp.py_tcp2], cov=tkp.Sigma13)
    # tpm[2, 0] = tkp.alpha31 + tkp.nu31 * pdf_Gaussian(x=angle, mean=tkp.psi31, cov=tkp.sigma31)
    # tpm[0, 3] = tkp.alpha14 + tkp.nu14 * tkp.psi14 * np.exp(-tkp.psi14 * velocity)
    # tpm[3, 0] = tkp.alpha41 + tkp.nu41 / (1 + np.exp(-tkp.psi41 * (velocity - tkp.ve)))
    # tpm[0, 4] = tkp.alpha15 + tkp.nu15 / (1 + np.exp(-tkp.psi15 * (velocity - tkp.vmax)))
    # tpm[4, 0] = tkp.alpha51 + tkp.nu51 / (1 + np.exp(-tkp.psi51 * (velocity - tkp.ve)))


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
    index = 3

    plt.figure(1)
    # plt.hold(True)
    plt.plot(time_steps, xtrue_all[0][index, :, 0])
    for k in range(len(datas)):
        plt.plot(time_steps, xest_all[k][index, :, 0])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.axis([None, None, -100, 200])
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
    plt.axis([None, None, -50, 250])
    legends=['True value', 'LSTM-MM Estimation', 'IMM Estimation', 'IMMPF Estimation']
    plt.legend(legends[0: len(datas)+1])
    plt.title('Trajectory of state 2')

    plt.figure(3)
    # plt.hold(True)
    plt.plot(time_steps, strue_all[0][index, :])
    for k in range(len(datas)):
        plt.plot(time_steps, mu_all[k][index, :], linewidth=(2.0 if k==0 else 1.0))
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
    plt.legend(legends[0: tkp.M * len(datas) + 1], loc='upper right')
    plt.title('Mode probabilities')

    plt.figure(4)
    legends = []
    for k in range(len(datas)):
        rmse = np.sqrt(np.mean((xtrue_all[k] - xest_all[k]) ** 2, axis=0))
        rmse_pos = np.mean(rmse[:, 0:2], axis=1)
        plt.plot(time_steps, rmse_pos)
    legends.append('LSTM-MM rmse of position')
    legends.append('IMM rmse of position')
    legends.append('IMMPF rmse of position')
    plt.axis([None, None, 0, 150])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(legends)
    plt.title('RMSE of position')

    plt.figure(5)
    legends = []
    for k in range(len(datas)):
        rmse = np.sqrt(np.mean((xtrue_all[k] - xest_all[k]) ** 2, axis=0))
        rmse_pos = np.mean(rmse[:, 2:], axis=1)
        plt.plot(time_steps, rmse_pos)
    legends.append('LSTM-MM rmse of velocity')
    legends.append('IMM rmse of velocity')
    legends.append('IMMPF rmse of velocity')
    plt.axis([None, None, 0, 30])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(legends)
    plt.title('RMSE of velocity')

    plt.figure(6)
    plt.plot(xtrue_all[0][index, :, 0], xtrue_all[0][index, :, 1])
    for k in range(len(datas)):
        plt.plot(xest_all[k][index, :, 0], xest_all[k][index, :, 1])
    plt.scatter(tkp.x0[0], tkp.x0[1])
    plt.axis([-100, 160, 0, 130])
    px = np.arange(start=-80, stop=140, step=1)
    py_above = tkp.ca3 * px ** 3 + tkp.ca2 * px ** 2 + tkp.ca1 * px + tkp.ca0
    py_below = tkp.cb3 * px ** 3 + tkp.cb2 * px ** 2 + tkp.cb1 * px + tkp.cb0
    plt.plot(px, py_above, px, py_below, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    legends = []
    legends.append('True trajectory')
    legends.append('LSTM-MM trajectory')
    legends.append('IMM trajectory')
    legends.append('IMMPF trajectory')
    legends.append('Road upper bound')
    legends.append('Road lower bound')
    plt.legend(legends)

    # for n in range(tkp.nx):
    #     plt.figure(4+n)
    #     # plt.hold(True)
    #     legends = []
    #     for k in range(len(datas)):
    #         rmse = np.sqrt(np.mean((xtrue_all[k] - xest_all[k]) ** 2, axis=0))
    #         plt.plot(time_steps, rmse[:, n])
    #     legends.append('LSTM-MM rmse of state '+str(n+1))
    #     legends.append('IMM rmse of state '+str(n+1))
    #     legends.append('IMMPF rmse of state '+str(n+1))
    #     if n<2:
    #         plt.axis([None, None, 0, 150])
    #     else:
    #         plt.axis([None, None, 0, 30])
    #     plt.xlabel('Time')
    #     plt.ylabel('Value')
    #     plt.legend(legends)
    #     plt.title('RMSE of state'+str(n+1))

    plt.show()


if __name__ == '__main__':
    # data_path = tkp.data_path
    # data = np.load(data_path)
    # plot_single_trajectory(data)
    # plot_model()

    which_net = 'npi_int'
    data_path = tkp.filter_data_path + '_' + which_net + '.npz'
    data_npi_int = np.load(data_path)
    # plot_result_single(data_npi_int)
    # plot_rmse(data_npi_int)

    data_path = tkp.filter_data_path + '_' + 'IMM' + '.npz'
    data_imm = np.load(data_path)
    # plot_result_single(data_imm)
    # plot_rmse(data_imm)

    data_path = tkp.filter_data_path + '_' + 'IMMPF' + '.npz'
    data_immpf = np.load(data_path)
    #
    plot_compare([data_npi_int, data_imm, data_immpf])
