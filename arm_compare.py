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

    Jah = am.measurement_Jacobian_arm(x=mt)
    v = z - am.measurement_arm(x=mt, r=np.zeros(shape=ap.nz))
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

        for j in range(M):
            m_all[k, :] = m_all[k, :] + mu_all[k, j] * ml_all[k, j, :]

    return m_all, mu_all


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

