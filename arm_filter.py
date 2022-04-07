import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow import keras
import arm_paras as ap


if __name__ == '__main__':

    T = ap.T
    dt = ap.dt
    K = int(T // dt)
    M = ap.M
    Np = ap.Np

    net_pi_int = keras.models.load_model(ap.net_path_pi_int)
    net_pi_para2 = keras.models.load_model(ap.net_path_pi_para2)
    net_pi_para3 = keras.models.load_model(ap.net_path_pi_para3)
    net_npi_int = keras.models.load_model(ap.net_path_npi_int)
    net_npi_para2 = keras.models.load_model(ap.net_path_npi_para2)
    net_npi_para3 = keras.models.load_model(ap.net_path_npi_para3)
    which_net = 'pi_int'

    w = np.zeros(shape=(K, M, Np))
    xp = np.zeros(shape=(K, M, Np, ap.nx))
    mu = np.zeros(shape=(K, M))

    for k in range(K):
        for j in range(M):
            if k == 1:
                for l in range(Np):
                    w[k, j, l] = 1/Np
                    xp[k, j, l, :] = np.random.multivariate_normal(ap.x0, ap.Q0)
                mu[k, j] = 1 if j == ap.s0 else 0
            else:
                pass
