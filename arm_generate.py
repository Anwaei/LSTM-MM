import numpy as np
from tqdm import tqdm
import arm_paras as ap
import arm_model as am

if __name__ == '__main__':

    T = ap.T
    dt = ap.dt
    x0 = ap.x0
    s0 = ap.s0
    batch_size = ap.batch_size

    K = int(T / dt)
    x_all = np.zeros([batch_size, 2, K])
    z_all = np.zeros([batch_size, 1, K])
    s_all = np.zeros([batch_size, 1, K], dtype='int')
    t_all = np.zeros([batch_size, 1, K], dtype='int')
    tpm_all = np.zeros([batch_size, 3, 3, K])
    ifreach_all = np.zeros([batch_size, 1, K], dtype='int')
    time_steps_all = np.zeros([batch_size, 1, K])

    tempT = K
    tpm_temp = np.zeros(shape=(2, tempT))
    for ti in range(tempT):
        t = ti+1
        tpm_temp[0, ti] = ap.q23**(t**ap.r23 - (t-1)**ap.r23)
        tpm_temp[1, ti] = ap.q32**(t**ap.r32 - (t-1)**ap.r32)

    for n in tqdm(range(batch_size)):
        tk = 1
        time_current = 0
        for k in range(K):
            time_current = time_current + dt
            qk, rk = am.noise_arm()
            if k == 0:
                tpm0 = am.tpm_arm(x=x0, t=tk, temp=[tpm_temp[0, tk-1], tpm_temp[1, tk-1]])
                sk = am.switch_arm(sp=s0, xp=x0, tp=tk)
                if sk == s0:
                    tk = tk + 1
                else:
                    tk = 1
                xk = am.dynamic_arm(sk, x0, qk)
                xk, ifreachk = am.constraint_arm(xk)
                zk = am.measurement_arm(xk, rk)
                tpmk = am.tpm_arm(x=xk, t=tk, temp=[tpm_temp[0, tk-1], tpm_temp[1, tk-1]])
            else:
                sp = s_all[n, 0, k - 1]
                xp = x_all[n, :, k - 1]
                tp = t_all[n, 0, k - 1]
                sk = am.switch_arm(sp, xp, tp)
                if sk == sp:
                    tk = tk + 1
                else:
                    tk = 1
                xk = am.dynamic_arm(sk, xp, qk)
                xk, ifreachk = am.constraint_arm(xk)
                zk = am.measurement_arm(xk, rk)
                tpmk = am.tpm_arm(x=xk, t=tk, temp=[tpm_temp[0, tk-1], tpm_temp[1, tk-1]])

            x_all[n, :, k] = xk
            z_all[n, 0, k] = zk
            s_all[n, 0, k] = sk
            t_all[n, 0, k] = tk
            tpm_all[n, :, :, k] = tpmk
            ifreach_all[n, 0, k] = ifreachk
            time_steps_all[n, 0, k] = time_current

    data_path = ap.data_path
    np.savez(data_path, x_all=x_all, z_all=z_all, s_all=s_all,
             t_all=t_all, tpm_all=tpm_all, ifreach_all=ifreach_all,
             time_steps_all=time_steps_all)

