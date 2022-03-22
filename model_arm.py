import numpy as np
import paras_arm as pa


def dynamic_arm(x_p, q, sc):
    """
    :param x_p: np.array (2,)
    :param q: np.array (2,)
    :param sc: current mode, in {1,2,3}
    :return:
    """

    dt = pa.dt
    g = pa.g
    l0 = pa.l0
    B = pa.B
    m = pa.m[sc - 1]
    J = pa.m[sc - 1]

    x1 = x_p[0] + x_p[1]*dt + q[0]
    x2 = x_p[1] - g*l0*m/J*np.sin(x_p[0])*dt - B/J*dt + q[1]

    x = np.array([x1, x2])

    return x


def measurement_arm(x, r):
    return x[0] + r


def noise_arm():
    Q = pa.Q
    R = pa.R
    q = np.random.multivariate_normal([0, 0], Q)
    r = np.random.multivariate_normal(0, R)
    return q, r


def tpm_arm(x, t):
    b = pa.b
    q23 = pa.q23
    r23 = pa.r23
    q32 = pa.q32
    r32 = pa.r32

    tpm = np.zeros([3, 3])
    ep = 1 if x >= b else 0
    tpm[0][0] = -1*ep + 1
    tpm[0][1] = 1*ep
    tpm[0][2] = 0
    tpm[1][0] = -1*ep + 1
    tpm[1][1] = q23**(t**r23-(t-1)**r23) * ep
    tpm[1][2] = (1-q23**(t**r23-(t-1)**r23)) * ep
    tpm[2][0] = -1 * ep + 1
    tpm[2][1] = (1-q32**(t**r32-(t-1)**r32)) * ep
    tpm[2][2] = q32**(t**r32-(t-1)**r32) * ep

    return tpm


def switch_arm(sp, x, t):
    mode_list = [1, 2, 3]
    if sp not in mode_list:
        raise ValueError("Invalid mode.")
    tpm = tpm_arm(x, t)
    tp = [tpm[sp-1, 0], tpm[sp-1, 1], tpm[sp-1, 2]]
    s = np.random.choice(mode_list, 1, p=tp)
    return s

