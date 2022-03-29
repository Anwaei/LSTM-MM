import numpy as np
import paras_arm as pa


def dynamic_arm(sc, x_p, q):
    """
    :param sc: current mode, in {1,2,3}
    :param x_p: np.array (2,)
    :param q: np.array (2,)
    :return:
    """

    if sc not in [1, 2, 3]:
        raise ValueError("Invalid mode.")

    dt = pa.dt
    g = pa.g
    l0 = pa.l0
    B = pa.B
    m = pa.m[sc - 1]
    J = pa.J[sc - 1]

    x1 = x_p[0] + x_p[1]*dt + q[0]
    x2 = x_p[1] - g*l0*m/J*np.sin(x_p[0])*dt - B/J*x_p[1]*dt + q[1]

    x = np.array([x1, x2])

    return x


def measurement_arm(x, r):
    return x[0] + r


def noise_arm():
    Q = pa.Q
    R = pa.R
    # q = np.zeros(2)
    q = np.random.multivariate_normal([0, 0], Q)
    r = np.random.multivariate_normal([0], R)
    return q, r


def tpm_arm(x, t):

    if (type(t) is not int) and (type(t) is not np.int32):
        raise ValueError("Invalid sojourn time.")

    b = pa.b
    q23 = pa.q23
    r23 = pa.r23
    q32 = pa.q32
    r32 = pa.r32

    tpm = np.zeros([3, 3])
    ep = 1 if x[0] >= b else 0
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


def switch_arm(sp, xp, tp):
    mode_list = [1, 2, 3]
    if sp not in mode_list:
        raise ValueError("Invalid mode.")
    if (type(tp) is not int) and (type(tp) is not np.int32):
        raise ValueError("Invalid sojourn time.")
    tpm = tpm_arm(xp, tp)
    probability = [tpm[sp-1, 0], tpm[sp-1, 1], tpm[sp-1, 2]]
    sc = np.random.choice(mode_list, 1, p=probability)[0]
    return sc


def constraint_arm(x):
    x1_c = pa.x1_c
    x2_c = pa.x2_c

    if abs(x[0]) <= x1_c and abs(x[1]) <= x2_c:
        if_reach_constraint = 0
    else:
        if x[0] > x1_c:
            x[0] = x1_c
        if x[0] < -x1_c:
            x[0] = -x1_c
        if x[1] > x2_c:
            x[1] = x2_c
        if x[1] < -x2_c:
            x[1] = -x2_c
        if_reach_constraint = 1

    return x, if_reach_constraint
