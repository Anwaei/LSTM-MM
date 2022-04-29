import numpy as np
import arm_model as am
import arm_paras as ap
import arm_plot as aplot
from tqdm import tqdm


def one_step_EKF(mp, Pp, z, s):
    s = s+1
    if s not in [1, 2, 3]:
        raise ValueError("Invalid mode.")
    Jaf = am.dynamic_Jacobian_arm(x=mp, s=s)
    mt = am.dynamic_arm(x_p=mp, sc=s, q=np.zeros(shape=ap.nx))
    Pt = Jaf @ Pp @ Jaf.transpose() + ap.Q

    Jah = am.measurement_Jacobian_arm(x=mt, s=s)
    v = z - am.measurement_arm(x=mt, r=np.zeros(shape=ap.nz), s=s)
    S = Jah @ Pt @ Jah.transpose() + ap.R
    K = Pt @ Jah.transpose() @ np.linalg.inv(S)

    m = mt + K @ v
    P = Pt - K @ S @ K.transpose()

    lam = am.pdf_Gaussian(x=v, mean=np.zeros(ap.nz), cov=S)

    return m, P, lam


def IMM(ztrue):

    T = ap.T
    dt = ap.dt
    K = int(T/dt)
    M = ap.M
    nx = ap.nx
    nz = ap.nz
    x0 = ap.x0
    s0 = ap.s0
    Pi = ap.Pi_IMM

    m_all = np.zeros(shape=(K+1, nx))
    P_all = np.zeros(shape=(K+1, nx, nx))
    mri_all = np.zeros(shape=(K+1, M, nx))
    Pri_all = np.zeros(shape=(K+1, M, nx, nx))
    ml_all = np.zeros(shape=(K+1, M, nx))
    Pl_all = np.zeros(shape=(K+1, M, nx, nx))
    mu_all = np.zeros(shape=(K+1, M))
    mu_mix_all = np.zeros(shape=(K+1, M, M))

    m_all[0, :] = x0
    P_all[0, :, :] = ap.Q0
    mu_all[0, s0-1] = 1
    for j in range(M):
        ml_all[0, j, :] = x0
        Pl_all[0, j, :, :] = ap.Q0

    for k in range(1, K+1):
        c = np.zeros(shape=M)
        lam = np.zeros(shape=M)
        for j in range(M):
            for i in range(M):
                mu_mix_all[k, i, j] = Pi[i, j] * mu_all[k-1, i]
            c[j] = np.sum(mu_mix_all[k, :, j])
            mu_mix_all[k, :, j] = mu_mix_all[k, :, j]/c[j]

        for j in range(M):
            for i in range(M):
                mri_all[k, j, :] = mri_all[k, j, :] + ml_all[k-1, i, :] * mu_mix_all[k, i, j]
        for j in range(M):
            for i in range(M):
                Ptemp = (ml_all[k-1, i, :] - mri_all[k, j, :]) @ np.transpose(ml_all[k-1, i, :] - mri_all[k, j, :])
                Pri_all[k, j, :, :] = Pri_all[k, j, :, :] + mu_mix_all[k, i, j] * (Pl_all[k-1, j, :, :] + Ptemp)

        for j in range(M):
            ml_all[k, j, :], Pl_all[k, j, :], lam[j] = one_step_EKF(mp=mri_all[k, j, :], Pp=Pri_all[k, j, :, :],
                                                                    s=j, z=ztrue[k, :])

        for j in range(M):
            mu_all[k, j] = lam[j]*c[j]
        mu_all[k, :] = mu_all[k, :]/np.sum(mu_all[k, :])
        # mu_all[k, :] = np.array([0, 0, 3])

        for j in range(M):
            m_all[k, :] = m_all[k, :] + mu_all[k, j] * ml_all[k, j, :]

        # m_all[k, :] = ml_all[k, 2, :]

    return m_all, mu_all


def tpm_immpf_arm(x):
    b = ap.b
    tpm = np.zeros([3, 3])
    ep = 0.95 if x[0] >= b else 0.05
    tpm[0][0] = -1*ep + 1
    tpm[0][1] = 1*ep
    tpm[0][2] = 0
    tpm[1][0] = -1*ep + 1
    tpm[1][1] = 0.96 * ep
    tpm[1][2] = 0.04 * ep
    tpm[2][0] = -1 * ep + 1
    tpm[2][1] = 0.05 * ep
    tpm[2][2] = 0.95 * ep

    return tpm


def resample(v, Np):
    M = ap.M
    if v.shape != (M, Np):
        raise ValueError('Incorrect probability dimension for v')
    v_flatten = np.reshape(v, (M * Np))
    index = np.random.choice(a=M * Np, size=Np, p=v_flatten)
    xi = index // Np
    xi.astype(int)
    zeta = index - xi * Np
    zeta.astype(int)
    return xi, zeta


def IMMPF(ztrue, scale=1):
    T = ap.T
    dt = ap.dt
    K = int(T/dt)
    M = ap.M
    Np = ap.Np*scale
    x0 = ap.x0
    s0 = ap.s0

    # xtrue_all = np.zeros(shape=(K + 1, ap.nx))
    # strue_all = np.zeros(shape=(K + 1))
    xest_all = np.zeros(shape=(K + 1, ap.nx))
    xp_all = np.zeros(shape=(K + 1, M, Np, ap.nx))
    w_all = np.zeros(shape=(K + 1, M, Np))
    what_all = np.zeros(shape=(K + 1, M, M, Np))
    mu_all = np.zeros(shape=(K + 1, M))
    gamma_all = np.zeros(shape=(K + 1, M))
    # z_all = np.zeros(shape=(K + 1, ap.nz))
    xi_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    zeta_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    q_proposal_all = np.random.multivariate_normal(mean=np.zeros(ap.nx), cov=ap.Q, size=(K + 1, M, Np))

    z_all = ztrue

    for j in range(M):
        for l in range(Np):
            w_all[0, j, l] = 1 / (Np*M)
            xp_all[0, j, l, :] = np.random.multivariate_normal(x0, ap.Q0)
        mu_all[0, j] = 0.98 if j == s0-1 else 0.01

    for k in range(1, K + 1):
        z = z_all[k, :]
        for i in range(M):
            for l in range(Np):
                tpm = tpm_immpf_arm(xp_all[k - 1, i, l, :])
                for j in range(M):
                    what_all[k, i, j, l] = tpm[i, j] * mu_all[k-1, i] * w_all[k-1, i, l]
        for j in range(M):
            gamma_all[k, j] = np.sum(what_all[k, :, j, :])
            what_all[k, :, j, :] = what_all[k, :, j, :] / gamma_all[k, j]
        for j in range(M):
            xi_all[n, k - 1, j, :], zeta_all[n, k - 1, j, :] = resample(what_all[k, :, j, :], Np=Np)
        for j in range(M):
            for l in range(Np):
                xi = xi_all[n, k - 1, j, l]
                zeta = zeta_all[n, k - 1, j, l]
                xp_all[k, j, l, :] = am.dynamic_arm(sc=j+1, x_p=xp_all[k - 1, xi, zeta, :]
                                                    , q=q_proposal_all[k - 1, j, l, :])
                zli = am.compute_meas_likelihood(x=xp_all[k, j, l, :], z=z, s=j+1)
                w_all[k, j, l] = gamma_all[k, j]*zli
        for j in range(M):
            mu_all[k, j] = np.sum(w_all[k, j, :])/Np
            w_all[k, j, :] = w_all[k, j, :] / np.sum(w_all[k, j, :])
        mu_all[k, :] = mu_all[k, :]/np.sum(mu_all[k, :])
        xest = np.zeros(ap.nx)
        for j in range(M):
            xestj = np.zeros(ap.nx)
            for l in range(Np):
                xestj = xestj + w_all[k, j, l] * xp_all[k, j, l, :]
            xest = xest + mu_all[k, j] * xestj
        xest_all[k, :] = xest
        # xest_all[k, :] = xestj

    return xest_all, mu_all


def optimalPF(ztrue, strue, scale=1):
    T = ap.T
    dt = ap.dt
    K = int(T/dt)
    M = ap.M
    Np = ap.Np*scale
    x0 = ap.x0
    s0 = ap.s0

    xest_all = np.zeros(shape=(K + 1, ap.nx))
    xp_all = np.zeros(shape=(K + 1, M, Np, ap.nx))
    w_all = np.zeros(shape=(K + 1, M, Np))
    what_all = np.zeros(shape=(K + 1, M, M, Np))
    mu_all = np.zeros(shape=(K + 1, M))
    gamma_all = np.zeros(shape=(K + 1, M))
    xi_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    zeta_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    q_proposal_all = np.random.multivariate_normal(mean=np.zeros(ap.nx), cov=ap.Q, size=(K + 1, M, Np))

    z_all = ztrue
    s_all = strue

    for j in range(M):
        for l in range(Np):
            w_all[0, j, l] = 1 / (Np*M)
            xp_all[0, j, l, :] = np.random.multivariate_normal(x0, ap.Q0)
        mu_all[0, j] = 1 if j == s0-1 else 0
    for k in range(1, K + 1):
        for j in range(M):
            mu_all[k, j] = 0.98 if j == s_all[k]-1 else 0.01

    for k in range(1, K + 1):
        z = z_all[k, :]
        for i in range(M):
            for l in range(Np):
                tpm = tpm_immpf_arm(xp_all[k - 1, i, l, :])
                for j in range(M):
                    what_all[k, i, j, l] = tpm[i, j] * mu_all[k-1, i] * w_all[k-1, i, l]
        for j in range(M):
            gamma_all[k, j] = np.sum(what_all[k, :, j, :])
            what_all[k, :, j, :] = what_all[k, :, j, :]/gamma_all[k, j]
        for j in range(M):
            xi_all[n, k - 1, j, :], zeta_all[n, k - 1, j, :] = resample(what_all[k, :, j, :], Np=Np)
        for j in range(M):
            for l in range(Np):
                xi = xi_all[n, k - 1, j, l]
                zeta = zeta_all[n, k - 1, j, l]
                xp_all[k, j, l, :] = am.dynamic_arm(sc=j+1, x_p=xp_all[k - 1, xi, zeta, :]
                                                    , q=q_proposal_all[k - 1, j, l, :])
                zli = am.compute_meas_likelihood(x=xp_all[k, j, l, :], z=z, s=j+1)
                w_all[k, j, l] = gamma_all[k, j]/Np*zli
            w_all[k, j, :] = w_all[k, j, :] / np.sum(w_all[k, j, :])
        xest = np.zeros(ap.nx)
        for j in range(M):
            xestj = np.zeros(ap.nx)
            for l in range(Np):
                xestj = xestj + w_all[k, j, l] * xp_all[k, j, l, :]
            xest = xest + mu_all[k, j] * xestj
        xest_all[k, :] = xest
    return xest_all, mu_all


if __name__ == '__main__':
    data = np.load(ap.data_path)
    x_data, z_data, s_data, t_data, tpm_data, ifreach_data, time_steps_data = aplot.read_data(data)
    size_run = int(x_data.shape[0] * ap.train_prop)
    xtrue_batch = np.swapaxes(x_data[size_run:, :, :], 1, 2)
    ztrue_batch = np.swapaxes(z_data[size_run:, :, :], 1, 2)
    strue_batch = s_data[size_run:, 0, :]
    ttrue_batch = t_data[size_run:, 0, :]
    time_steps_batch = time_steps_data[size_run:, 0, :]
    time_steps = time_steps_batch[0, :]

    T = ap.T
    dt = ap.dt
    K = int(T/dt)
    run_batch = ap.run_batch
    xtrue_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    strue_all = np.zeros(shape=(run_batch, K + 1))
    xest_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    mu_all = np.zeros(shape=(run_batch, K + 1, ap.M))
    z_all = np.zeros(shape=(run_batch, K + 1, ap.nz))

    for n in tqdm(range(run_batch)):
        xtrue_all[n, 0, :] = ap.x0
        xtrue_all[n, 1:, :] = xtrue_batch[n, :, :]
        # z_all[n, 0, :] = ap.z0
        z_all[n, 1:, :] = ztrue_batch[n, :, :]
        strue_all[n, 0] = ap.s0
        strue_all[n, 1:] = strue_batch[n, :]
        xest_all[n, :, :], mu_all[n, :, :] = IMM(ztrue=z_all[n, :, :])

    np.savez(file=ap.filter_data_path+'_'+'IMM'+'.npz',
             xtrue_all=xtrue_all,
             strue_all=strue_all,
             xest_all=xest_all,
             mu_all=mu_all,
             z_all=z_all,
             time_steps=time_steps)

    # T = ap.T
    # dt = ap.dt
    # K = int(T/dt)
    # run_batch = ap.run_batch
    # xtrue_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    # strue_all = np.zeros(shape=(run_batch, K + 1))
    # xest_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    # mu_all = np.zeros(shape=(run_batch, K + 1, ap.M))
    # z_all = np.zeros(shape=(run_batch, K + 1, ap.nz))
    #
    # for n in tqdm(range(run_batch)):
    #     xtrue_all[n, 0, :] = ap.x0
    #     xtrue_all[n, 1:, :] = xtrue_batch[n, :, :]
    #     # z_all[n, 0, :] = ap.z0
    #     z_all[n, 1:, :] = ztrue_batch[n, :, :]
    #     strue_all[n, 0] = ap.s0
    #     strue_all[n, 1:] = strue_batch[n, :]
    #     xest_all[n, :, :], mu_all[n, :, :] = IMMPF(ztrue=z_all[n, :, :], scale=10)
    #
    # np.savez(file=ap.filter_data_path+'_'+'IMMPF-5000'+'.npz',
    #          xtrue_all=xtrue_all,
    #          strue_all=strue_all,
    #          xest_all=xest_all,
    #          mu_all=mu_all,
    #          z_all=z_all,
    #          time_steps=time_steps)
    #
    T = ap.T
    dt = ap.dt
    K = int(T/dt)
    run_batch = ap.run_batch
    xtrue_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    strue_all = np.zeros(shape=(run_batch, K + 1))
    xest_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    mu_all = np.zeros(shape=(run_batch, K + 1, ap.M))
    z_all = np.zeros(shape=(run_batch, K + 1, ap.nz))

    for n in tqdm(range(run_batch)):
        xtrue_all[n, 0, :] = ap.x0
        xtrue_all[n, 1:, :] = xtrue_batch[n, :, :]
        # z_all[n, 0, :] = ap.z0
        z_all[n, 1:, :] = ztrue_batch[n, :, :]
        strue_all[n, 0] = ap.s0
        strue_all[n, 1:] = strue_batch[n, :]
        xest_all[n, :, :], mu_all[n, :, :] = IMMPF(ztrue=z_all[n, :, :])

    np.savez(file=ap.filter_data_path+'_'+'IMMPF-500'+'.npz',
             xtrue_all=xtrue_all,
             strue_all=strue_all,
             xest_all=xest_all,
             mu_all=mu_all,
             z_all=z_all,
             time_steps=time_steps)

    # T = ap.T
    # dt = ap.dt
    # K = int(T/dt)
    # run_batch = ap.run_batch
    # xtrue_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    # strue_all = np.zeros(shape=(run_batch, K + 1))
    # xest_all = np.zeros(shape=(run_batch, K + 1, ap.nx))
    # mu_all = np.zeros(shape=(run_batch, K + 1, ap.M))
    # z_all = np.zeros(shape=(run_batch, K + 1, ap.nz))
    #
    # for n in tqdm(range(run_batch)):
    #     xtrue_all[n, 0, :] = ap.x0
    #     xtrue_all[n, 1:, :] = xtrue_batch[n, :, :]
    #     # z_all[n, 0, :] = ap.z0
    #     z_all[n, 1:, :] = ztrue_batch[n, :, :]
    #     strue_all[n, 0] = ap.s0
    #     strue_all[n, 1:] = strue_batch[n, :]
    #     xest_all[n, :, :], mu_all[n, :, :] = optimalPF(ztrue=z_all[n, :, :], strue=strue_all[n, :])
    #
    # np.savez(file=ap.filter_data_path+'_'+'optPF'+'.npz',
    #          xtrue_all=xtrue_all,
    #          strue_all=strue_all,
    #          xest_all=xest_all,
    #          mu_all=mu_all,
    #          z_all=z_all,
    #          time_steps=time_steps)


