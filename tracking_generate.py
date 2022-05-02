import numpy as np
from tqdm import tqdm
import tracking_paras as tkp
import tracking_model as tkm

if __name__ == '__main__':

    T = tkp.T
    dt = tkp.dt
    x0 = tkp.x0
    s0 = tkp.s0
    batch_size = tkp.batch_size

    K = int(T / dt)
    x_all = np.zeros([batch_size, tkp.nx, K])
    z_all = np.zeros([batch_size, tkp.nz, K])
    s_all = np.zeros([batch_size, 1, K], dtype='int')
    t_all = np.zeros([batch_size, 1, K], dtype='int')
    tpm_all = np.zeros([batch_size, tkp.M, tkp.M, K])
    ifreach_all = np.zeros([batch_size, 1, K], dtype='int')
    time_steps_all = np.zeros([batch_size, 1, K])

    for n in tqdm(range(batch_size)):
        tk = 1
        time_current = 0
        for k in range(K):
            time_current = time_current + dt
            qk, rk = tkm.noise_tracking()
            if k == 0:
                tpm0 = tkm.tpm_tracking(x=x0, t=tk)
                sk = tkm.switch_tracking(sp=s0, xp=x0, tp=tk)
                if sk == s0:
                    tk = tk + 1
                else:
                    tk = 1
                xk = tkm.dynamic_tracking(sk, x0, qk)
                xkn, ifreachk = tkm.constraint_tracking(xk)
                zk = tkm.measurement_tracking(xk, rk)
                tpmk = tkm.tpm_tracking(x=xk, t=tk)
            else:
                sp = s_all[n, 0, k - 1]
                xp = x_all[n, :, k - 1]
                tp = t_all[n, 0, k - 1]
                sk = tkm.switch_tracking(sp, xp, tp)
                if sk == sp:
                    tk = tk + 1
                else:
                    tk = 1
                xk = tkm.dynamic_tracking(sk, xp, qk)
                xkn, ifreachk = tkm.constraint_tracking(xk)
                zk = tkm.measurement_tracking(xk, rk)
                tpmk = tkm.tpm_tracking(x=xk, t=tk)

            x_all[n, :, k] = xk
            z_all[n, :, k] = zk
            s_all[n, 0, k] = sk
            t_all[n, 0, k] = tk
            tpm_all[n, :, :, k] = tpmk
            ifreach_all[n, 0, k] = ifreachk
            time_steps_all[n, 0, k] = time_current

    data_path = tkp.data_path
    np.savez(data_path, x_all=x_all, z_all=z_all, s_all=s_all,
             t_all=t_all, tpm_all=tpm_all, ifreach_all=ifreach_all,
             time_steps_all=time_steps_all)

