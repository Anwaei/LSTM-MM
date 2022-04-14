import numpy as np
import arm_paras as ap
from scipy import stats
import time

def dynamic_arm(sc, x_p, q):
    """
    :param sc: current mode, in {1,2,3}
    :param x_p: np.array (2,)
    :param q: np.array (2,)
    :return:
    """

    if sc not in [1, 2, 3]:
        raise ValueError("Invalid mode.")

    dt = ap.dt
    g = ap.g
    l0 = ap.l0
    B = ap.B
    m = ap.m[sc - 1]
    J = ap.J[sc - 1]

    x1 = x_p[0] + x_p[1]*dt + q[0]
    x2 = x_p[1] - g*l0*m/J*np.sin(x_p[0])*dt - B/J*x_p[1]*dt + q[1]

    x = np.array([x1, x2])

    return x


def measurement_arm(x, r):
    return x[0] + r


def compute_meas_likelihood(x, z, s):
    # Gaussian noise
    mean = measurement_arm(x, r=0)
    li = pdf_Gaussian(x=z-mean, mean=np.zeros(ap.nz), cov=ap.R)
    return li


def compute_meas_loglikelihood(x, z, s):
    # Gaussian noise
    mean = measurement_arm(x, r=0)
    if mean.size == 1:
        logli = -(z - mean) ** 2 / (2 * ap.R)
    else:
        logli = -1/2 * np.matmul(np.matmul(z-mean, np.linalg.inv(ap.R)), z-mean)
    return logli[0, 0]


def noise_arm():
    Q = ap.Q
    R = ap.R
    # q = np.zeros(2)
    q = np.random.multivariate_normal([0, 0], Q)
    r = np.random.multivariate_normal([0], R)
    return q, r


def tpm_arm(x, t):

    if (type(t) is not int) and (type(t) is not np.int32):
        raise ValueError("Invalid sojourn time.")

    b = ap.b
    q23 = ap.q23
    r23 = ap.r23
    q32 = ap.q32
    r32 = ap.r32

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
    x1_c = ap.x1_c
    x2_c = ap.x2_c

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


def compute_constraint_likelihood(x):
    h1 = abs(x[0]) - ap.x1_c
    h2 = abs(x[1]) - ap.x2_c
    # t1 = time.clock()
    # p1x = 1 - stats.expon.cdf(x=h1, scale=1/ap.lambda1)
    # p2x = 1 - stats.expon.cdf(x=h2, scale=1/ap.lambda2)
    # t2 = time.clock()
    p1 = np.exp(-ap.lambda1*h1) if h1 > 0 else 1
    p2 = np.exp(-ap.lambda2*h2) if h2 > 0 else 1
    # t3 = time.clock()
    return p1*p2


def compute_constraint_loglikelihood(x):
    h1 = abs(x[0]) - ap.x1_c
    h2 = abs(x[1]) - ap.x2_c
    logli1 = -ap.lambda1 * h1 if h1 > 0 else 0
    logli2 = -ap.lambda2 * h2 if h2 > 0 else 0
    return logli1*logli2


def pdf_Gaussian(x, mean, cov):
    if mean.size == 1:
        pdf = 1/np.sqrt(2*np.pi*cov)*np.exp(-(x-mean)**2/(2*cov))
    else:
        pdf = np.power(2*np.pi, -mean.size/2) * np.power(np.linalg.det(cov), -1/2) \
              * np.exp(-1/2 * np.matmul(np.matmul(x-mean, np.linalg.inv(cov)), x-mean))
    return pdf[0, 0]


def cdf_expon(x, lam):
    c = 1 - np.exp(-lam*x) if x > 0 else 0
    return c




