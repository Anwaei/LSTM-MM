import numpy as np


def dynamic_arm(x_p, q, dt, mode, g=9.81, l0=0.6, B=5):
    """
    :param x_p: np.array (2,)
    :param q: np.array (2,)
    :param mode: 1,2,3
    :return:
    """

    if mode == 1:
        m = 1
        J = 1
    elif mode == 2:
        m = 2.5
        J = 2.5
    elif mode == 3:
        m = 5
        J = 5
    else:
        raise ValueError('Invalid mode.')

    x1 = x_p[0] + x_p[1]*dt + q[0]
    x2 = x_p[1] - g*l0*m/J*np.sin(x_p[0])*dt - B/J*dt + q[1]

    x = np.array([x1, x2])

    return x


def measurement_arm(x, r):
    return x[0] + r


def noise_arm(Q, R):
    q = np.random.multivariate_normal([0, 0], Q)
    r = np.random.multivariate_normal(0, R)
    return q, r


def switch_arm(sp, x, t):
    b = 0.523
    q23 = 0.8
    r23 = 1.2
    q32 = 0.9
    r32 = 1.5

    ep = 1 if x>=b else 0
    if sp == 1:
        pi1 = -1*ep + 1
        pi2 = 1*ep
        pi3 = 0*ep
    elif sp == 2:
        pi1 = -1*ep + 1
        pi2 = q23**(t**r23-(t-1)**r23) * ep
        pi3 = (1-q23**(t**r23-(t-1)**r23)) * ep
    elif sp == 3:
        pi1 = -1 * ep + 1
        pi2 = (1-q32**(t**r32-(t-1)**r32)) * ep
        pi3 = q32**(t**r32-(t-1)**r32) * ep
    else:
        raise ValueError('Invalid mode.')

    mode_list = [1, 2, 3]
    s = np.random.choice(mode_list, 1, p=[pi1, pi2, pi3])

    return s
