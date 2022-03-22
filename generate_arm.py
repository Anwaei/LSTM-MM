import numpy as np
import paras_arm as pa
import model_arm as ma


if __name__ == '__main__':

    T = pa.T
    dt = pa.dt
    x0 = pa.x0
    s0 = pa.s0
    batch_size = pa.batch_size

    K = T//dt
    x_all = np.zeros([batch_size, 2, K])
    z_all = np.zeros([batch_size, 1, K])
    s_all = np.zeros([batch_size, 1, K])
    t_all = np.zeros([batch_size, 1, K])
    tpm_all = np.zeros([batch_size, 3, 3, K])
    ifreach_all = np.zeros([batch_size, 1, K])

    tk = 1
    for k in range(K):
        qk, rk = ma.noise_arm()
        if k == 1:
            tpm0 = ma.tpm_arm(x=x0, t=tk)
            sk = ma.switch_arm(sp=s0, xp=x0, tp=tk)
            if sk == s0:
                tk = tk + 1
            else:
                tk = 1
            xk = ma.dynamic_arm(sk, x0, qk)
            xk, ifreachk = ma.constraint_arm(xk)
            zk = ma.measurement_arm(xk, rk)
            tpmk = ma.tpm_arm(x=xk, t=tk)
        else:
            sp = s_all[batch_size, 1, k - 1]
            xp = x_all[batch_size, :, k - 1]
            tp = t_all[batch_size, 1, k - 1]
            sk = ma.switch_arm(sp, xp, tp)
            if sk == sp:
                tk = tk + 1
            else:
                tk = 1
            xk = ma.dynamic_arm(sk, xp, qk)
            xk, ifreachk = ma.constraint_arm(xk)
            zk = ma.measurement_arm(xk, rk)
            tpmk = ma.tpm_arm(x=xk, t=tk)

        x_all[batch_size, :, k] = xk
        z_all[batch_size, 1, k] = zk
        s_all[batch_size, 1, k] = sk
        t_all[batch_size, 1, k] = tk
        tpm_all[batch_size, :, :, k] = tpmk
        ifreach_all[batch_size, 1, k] = ifreachk

    np.savez('data/data_arm', x_all=x_all, z_all=z_all, s_all=s_all, tpm_all=tpm_all, ifreach_all=ifreach_all)

