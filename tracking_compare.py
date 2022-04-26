import numpy as np
import tracking_model as tkm
import tracking_paras as tkp
import tracking_plot as tkplot
from tqdm import tqdm


def one_step_EKF(mp, Pp, z, s):
    s = s+1
    if s not in [1, 2, 3, 4, 5]:
        raise ValueError("Invalid mode.")
    Jaf = tkm.dynamic_Jacobian_tracking(x=mp, s=s)
    mt = tkm.dynamic_tracking(x_p=mp, sc=s, q=np.zeros(shape=tkp.nq))
    Pt = Jaf @ Pp @ Jaf.transpose() + tkp.G @ tkp.Q @ tkp.G.transpose()

    Jah = tkm.measurement_Jacobian_tracking(x=mt)
    v = z - tkm.measurement_tracking(x=mt, r=np.zeros(shape=tkp.nz))
    S = Jah @ Pt @ Jah.transpose() + tkp.R
    K = Pt @ Jah.transpose() @ np.linalg.inv(S)

    m = mt + K @ v
    P = Pt - K @ S @ K.transpose()

    lam = tkm.pdf_Gaussian(x=v, mean=np.zeros(tkp.nz), cov=S)

    return m, P, lam


def IMM(ztrue):

    T = tkp.T
    dt = tkp.dt
    K = int(T/dt)
    M = tkp.M
    nx = tkp.nx
    nz = tkp.nz
    x0 = tkp.x0
    s0 = tkp.s0
    Pi = tkp.Pi_IMM

    m_all = np.zeros(shape=(K+1, nx))
    P_all = np.zeros(shape=(K+1, nx, nx))
    mri_all = np.zeros(shape=(K+1, M, nx))
    Pri_all = np.zeros(shape=(K+1, M, nx, nx))
    ml_all = np.zeros(shape=(K+1, M, nx))
    Pl_all = np.zeros(shape=(K+1, M, nx, nx))
    mu_all = np.zeros(shape=(K+1, M))
    mu_mix_all = np.zeros(shape=(K+1, M, M))

    m_all[0, :] = x0
    P_all[0, :, :] = tkp.Q0
    mu_all[0, s0-1] = 1
    for j in range(M):
        ml_all[0, j, :] = x0
        Pl_all[0, j, :, :] = tkp.Q0

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

        for j in range(M):
            m_all[k, :] = m_all[k, :] + mu_all[k, j] * ml_all[k, j, :]

    return m_all, mu_all


def tpm_immpf_tracking(x):
    tpm = tkm.tpm_tracking(x, tkp.tlast+1)
    return tpm


def resample(v):
    M = tkp.M
    Np = tkp.Np
    if v.shape != (M, Np):
        raise ValueError('Incorrect probability dimension for v')
    v_flatten = np.reshape(v, (M * Np))
    index = np.random.choice(a=M * Np, size=Np, p=v_flatten)
    xi = index // Np
    xi.astype(int)
    zeta = index - xi * Np
    zeta.astype(int)
    return xi, zeta


def IMMPF(ztrue):
    T = tkp.T
    dt = tkp.dt
    K = int(T/dt)
    M = tkp.M
    Np = tkp.Np
    nx = tkp.nx
    nz = tkp.nz
    x0 = tkp.x0
    s0 = tkp.s0

    xtrue_all = np.zeros(shape=(K + 1, tkp.nx))
    strue_all = np.zeros(shape=(K + 1))
    xest_all = np.zeros(shape=(K + 1, tkp.nx))
    xp_all = np.zeros(shape=(K + 1, M, Np, tkp.nx))
    w_all = np.zeros(shape=(K + 1, M, Np))
    what_all = np.zeros(shape=(K + 1, M, M, Np))
    mu_all = np.zeros(shape=(K + 1, M))
    gamma_all = np.zeros(shape=(K + 1, M))
    z_all = np.zeros(shape=(K + 1, tkp.nz))
    xi_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    zeta_all = np.zeros(shape=(run_batch, K + 1, M, Np), dtype='int')
    q_proposal_all = np.random.multivariate_normal(mean=np.zeros(tkp.nq), cov=tkp.Q, size=(K + 1, M, Np))

    z_all = ztrue

    for j in range(M):
        for l in range(Np):
            w_all[0, j, l] = 1 / Np
            xp_all[0, j, l, :] = np.random.multivariate_normal(x0, tkp.Q0)
        mu_all[0, j] = 1 if j == s0 else 0

    for k in range(1, K + 1):
        z = z_all[k, :]
        for i in range(M):
            for l in range(Np):
                tpm = tpm_immpf_tracking(xp_all[k - 1, i, l, :])
                for j in range(M):
                    what_all[k, i, j, l] = tpm[i, j] * mu_all[k - 1, i] * w_all[k - 1, i, l]
        for j in range(M):
            gamma_all[k, j] = np.sum(what_all[k, :, j, :])
            what_all[k, :, j, :] = what_all[k, :, j, :] / gamma_all[k, j]
        for j in range(M):
            xi_all[n, k - 1, j, :], zeta_all[n, k - 1, j, :] = resample(what_all[k, :, j, :])
        for j in range(M):
            for l in range(Np):
                xi = xi_all[n, k - 1, j, l]
                zeta = zeta_all[n, k - 1, j, l]
                xp_all[k, j, l, :] = tkm.dynamic_tracking(sc=j + 1, x_p=xp_all[k - 1, xi, zeta, :],
                                                     q=q_proposal_all[k - 1, j, l, :])
                zli = tkm.compute_meas_likelihood(x=xp_all[k, j, l, :], z=z)
                w_all[k, j, l] = gamma_all[k, j] / Np * zli
        for j in range(M):
            mu_all[k, j] = np.sum(w_all[k, j, :])
        mu_all[k, :] = mu_all[k, :] / np.sum(mu_all[k, :])
        w_all[k, :, :] = w_all[k, :, :] / np.sum(w_all[k, :, :])
        xest = np.zeros(tkp.nx)
        for j in range(M):
            xestj = np.zeros(tkp.nx)
            for l in range(Np):
                xestj = xestj + w_all[k, j, l] / w_all[k, j, :].sum() * xp_all[k, j, l, :]
            xest = xest + mu_all[k, j] * xestj
        xest_all[k, :] = xest

    return xest_all, mu_all


if __name__ == '__main__':
    data = np.load(tkp.data_path)
    x_data, z_data, s_data, t_data, tpm_data, ifreach_data, time_steps_data = tkplot.read_data(data)
    size_run = int(x_data.shape[0] * tkp.train_prop)
    xtrue_batch = np.swapaxes(x_data[size_run:, :, :], 1, 2)
    ztrue_batch = np.swapaxes(z_data[size_run:, :, :], 1, 2)
    strue_batch = s_data[size_run:, 0, :]
    ttrue_batch = t_data[size_run:, 0, :]
    time_steps_batch = time_steps_data[size_run:, 0, :]
    time_steps = time_steps_batch[0, :]

    T = tkp.T
    dt = tkp.dt
    K = int(T/dt)
    run_batch = tkp.run_batch
    xtrue_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    strue_all = np.zeros(shape=(run_batch, K + 1))
    xest_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    mu_all = np.zeros(shape=(run_batch, K + 1, tkp.M))
    z_all = np.zeros(shape=(run_batch, K + 1, tkp.nz))

    for n in tqdm(range(run_batch)):
        xtrue_all[n, 0, :] = tkp.x0
        xtrue_all[n, 1:, :] = xtrue_batch[n, 0:K, :]
        # z_all[n, 0, :] = tkp.z0
        z_all[n, 1:, :] = ztrue_batch[n, 0:K, :]
        strue_all[n, 0] = tkp.s0
        strue_all[n, 1:] = strue_batch[n, 0:K]
        xest_all[n, :, :], mu_all[n, :, :] = IMM(ztrue=z_all[n, :, :])

    np.savez(file=tkp.filter_data_path+'_'+'IMM'+'.npz',
             xtrue_all=xtrue_all,
             strue_all=strue_all,
             xest_all=xest_all,
             mu_all=mu_all,
             z_all=z_all,
             time_steps=time_steps)

    T = tkp.T
    dt = tkp.dt
    K = int(T/dt)
    run_batch = tkp.run_batch
    xtrue_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    strue_all = np.zeros(shape=(run_batch, K + 1))
    xest_all = np.zeros(shape=(run_batch, K + 1, tkp.nx))
    mu_all = np.zeros(shape=(run_batch, K + 1, tkp.M))
    z_all = np.zeros(shape=(run_batch, K + 1, tkp.nz))

    for n in tqdm(range(run_batch)):
        xtrue_all[n, 0, :] = tkp.x0
        xtrue_all[n, 1:, :] = xtrue_batch[n, 0:K, :]
        # z_all[n, 0, :] = ap.z0
        z_all[n, 1:, :] = ztrue_batch[n, 0:K, :]
        strue_all[n, 0] = tkp.s0
        strue_all[n, 1:] = strue_batch[n, 0:K]
        xest_all[n, :, :], mu_all[n, :, :] = IMMPF(ztrue=z_all[n, :, :])

    np.savez(file=tkp.filter_data_path + '_' + 'IMMPF' + '.npz',
             xtrue_all=xtrue_all,
             strue_all=strue_all,
             xest_all=xest_all,
             mu_all=mu_all,
             z_all=z_all,
             time_steps=time_steps)

