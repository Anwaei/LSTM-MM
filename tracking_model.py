import numpy as np
import tracking_paras as tkp
import time


def dynamic_tracking(sc, x_p, q):
    """
    :param sc: current mode, in {1,2,3}
    :param x_p: np.array (2,)
    :param q: np.array (2,)
    :return:
    """

    if sc not in [1, 2, 3, 4, 5]:
        raise ValueError("Invalid mode.")

    dt = tkp.dt
    px = x_p[0]
    py = x_p[1]
    vx = x_p[2]
    vy = x_p[3]
    qc = tkp.G @ q

    if sc == 1:
        x1 = px + vx*dt
        x2 = py + vy*dt
        x3 = vx
        x4 = vy
    elif sc == 2:
        w = tkp.omegar
        sw = tkp.swr
        cw = tkp.cwr
        x1 = px + sw/w*vx - (1-cw)/w*vy
        x2 = py + (1-cw)/w*vx + sw/w*vy
        x3 = vx + cw*vx - sw*vy
        x4 = vy + sw*vx + cw*vy
    elif sc == 3:
        w = tkp.omegal
        sw = tkp.swl
        cw = tkp.cwl
        x1 = px + sw/w*vx - (1-cw)/w*vy
        x2 = py + (1-cw)/w*vx + sw/w*vy
        x3 = vx + cw*vx - sw*vy
        x4 = vy + sw*vx + cw*vy
    elif sc == 4:
        a = tkp.apos
        v = np.sqrt(vx**2 + vy**2)
        x1 = px + vx*dt + vx/(2*v)*a*dt**2
        x2 = py + vy*dt + vy/(2*v)*a*dt**2
        x3 = vx + vx/v*a*dt
        x4 = vy + vy/v*a*dt
    elif sc == 5:
        a = tkp.aneg
        v = np.sqrt(vx**2 + vy**2)
        x1 = px + vx*dt + vx/(2*v)*a*dt**2
        x2 = py + vy*dt + vy/(2*v)*a*dt**2
        x3 = vx + vx/v*a*dt
        x4 = vy + vy/v*a*dt
    else:
        raise ValueError("Invalid mode.")

    x = np.array([x1, x2, x3, x4]) + qc

    return x


def dynamic_Jacobian_tracking(x, s):
    dt = tkp.dt
    px = x[0]
    py = x[1]
    vx = x[2]
    vy = x[3]

    Ja = np.zeros(shape=(tkp.nx, tkp.nx))

    if s==1:
        Ja[0, 0] = 1
        Ja[0, 1] = 0
        Ja[0, 2] = dt
        Ja[0, 3] = 0
        Ja[1, 0] = 0
        Ja[1, 1] = 1
        Ja[1, 2] = 0
        Ja[1, 3] = dt
        Ja[2, 0] = 0
        Ja[2, 1] = 0
        Ja[2, 2] = 1
        Ja[2, 3] = 0
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = 0
        Ja[3, 3] = 1
    elif s == 2:
        w = tkp.omegar
        sw = tkp.swr
        cw = tkp.cwr
        Ja[0, 0] = 1
        Ja[0, 1] = 0
        Ja[0, 2] = sw/w
        Ja[0, 3] = -(1-cw)/w
        Ja[1, 0] = 0
        Ja[1, 1] = 1
        Ja[1, 2] = (1-cw)/w
        Ja[1, 3] = sw/w
        Ja[2, 0] = 0
        Ja[2, 1] = 0
        Ja[2, 2] = 1+cw
        Ja[2, 3] = -sw
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = sw
        Ja[3, 3] = 1+cw
    elif s == 3:
        w = tkp.omegar
        sw = tkp.swl
        cw = tkp.cwl
        Ja[0, 0] = 1
        Ja[0, 1] = 0
        Ja[0, 2] = sw/w
        Ja[0, 3] = -(1-cw)/w
        Ja[1, 0] = 0
        Ja[1, 1] = 1
        Ja[1, 2] = (1-cw)/w
        Ja[1, 3] = sw/w
        Ja[2, 0] = 0
        Ja[2, 1] = 0
        Ja[2, 2] = 1+cw
        Ja[2, 3] = -sw
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = sw
        Ja[3, 3] = 1+cw
    elif s == 4:
        a = tkp.apos
        v = np.sqrt(vx ** 2 + vy ** 2)
        tempx = vx**2/v**3
        tempy = vy**2/v**3
        tempxy = vx*vy/v**3
        Ja[0, 0] = 1
        Ja[0, 1] = 0
        Ja[0, 2] = dt + a*dt**2/2 * tempy
        Ja[0, 3] = -a*dt**2/2 * tempxy
        Ja[1, 0] = 0
        Ja[1, 1] = 1
        Ja[1, 2] = -a*dt**2/2 * tempxy
        Ja[1, 3] = dt + a*dt**2/2 * tempx
        Ja[2, 0] = 0
        Ja[2, 1] = 0
        Ja[2, 2] = a*dt * tempy
        Ja[2, 3] = a*dt * tempxy
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = a*dt * tempxy
        Ja[3, 3] = a*dt * tempx
    elif s == 5:
        a = tkp.aneg
        v = np.sqrt(vx ** 2 + vy ** 2)
        tempx = vx ** 2 / v ** 3
        tempy = vy ** 2 / v ** 3
        tempxy = vx * vy / v ** 3
        Ja[0, 0] = 1
        Ja[0, 1] = 0
        Ja[0, 2] = dt + a * dt ** 2 / 2 * tempy
        Ja[0, 3] = -a * dt ** 2 / 2 * tempxy
        Ja[1, 0] = 0
        Ja[1, 1] = 1
        Ja[1, 2] = -a * dt ** 2 / 2 * tempxy
        Ja[1, 3] = dt + a * dt ** 2 / 2 * tempx
        Ja[2, 0] = 0
        Ja[2, 1] = 0
        Ja[2, 2] = a * dt * tempy
        Ja[2, 3] = a * dt * tempxy
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = a * dt * tempxy
        Ja[3, 3] = a * dt * tempx
    else:
        raise ValueError("Invalid mode.")

    return Ja


def measurement_tracking(x, r):
    z = np.zeros(tkp.nz)
    px = x[0]
    py = x[1]
    z[0] = np.arctan((py-tkp.psy)/(px-tkp.psx))
    z[1] = np.arctan(tkp.psz/np.sqrt((px-tkp.psx)**2 + (py-tkp.psy)**2))
    return z + r


def measurement_Jacobian_tracking(x):
    Ja = np.zeros(shape=(tkp.nz, tkp.nx))
    px = x[0]
    py = x[1]
    temp = (px-tkp.psx)**2 + (py-tkp.psy)**2
    stemp = np.sqrt(temp)
    Ja[0, 0] = -(py-tkp.psy)/temp
    Ja[0, 1] = (px-tkp.psx)/temp
    Ja[0, 2] = 0
    Ja[0, 3] = 0
    Ja[1, 0] = -(px-tkp.psx)/stemp**3
    Ja[1, 1] = -(py-tkp.psy)/stemp**3
    Ja[1, 2] = 0
    Ja[1, 3] = 0
    return Ja


def compute_meas_likelihood(x, z, s=1):
    # Gaussian noise
    mean = measurement_tracking(x, r=0)
    li = pdf_Gaussian(x=z-mean, mean=np.zeros(tkp.nz), cov=tkp.R)
    return li


def compute_meas_loglikelihood(x, z, s):
    # Gaussian noise
    mean = measurement_tracking(x, r=0)
    if mean.size == 1:
        logli = -(z - mean) ** 2 / (2 * tkp.R)
    else:
        logli = -1/2 * np.matmul(np.matmul(z-mean, np.linalg.inv(tkp.R)), z-mean)
    return logli[0, 0]


def noise_tracking():
    Q = tkp.Q
    R = tkp.R
    # q = np.zeros(2)
    q = np.random.multivariate_normal([0, 0], Q)
    r = np.random.multivariate_normal([0, 0], R)
    return q, r


def tpm_tracking(x, t):

    if (type(t) is not int) and (type(t) is not np.int32):
        raise ValueError("Invalid sojourn time.")

    b = tkp.b
    q23 = tkp.q23
    r23 = tkp.r23
    q32 = tkp.q32
    r32 = tkp.r32

    tpm = np.zeros([3, 3])
    ep = 0.99 if x[0] >= b else 0.01
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


def switch_tracking(sp, xp, tp):
    mode_list = [1, 2, 3]
    if sp not in mode_list:
        raise ValueError("Invalid mode.")
    if (type(tp) is not int) and (type(tp) is not np.int32):
        raise ValueError("Invalid sojourn time.")
    tpm = tpm_tracking(xp, tp)
    probability = [tpm[sp-1, 0], tpm[sp-1, 1], tpm[sp-1, 2]]
    sc = np.random.choice(mode_list, 1, p=probability)[0]
    return sc


def constraint_tracking(x):
    x1_c = tkp.x1_c
    x2_c = tkp.x2_c

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
    h1 = abs(x[0]) - tkp.x1_c
    h2 = abs(x[1]) - tkp.x2_c
    # t1 = time.clock()
    # p1x = 1 - stats.expon.cdf(x=h1, scale=1/tkp.lambda1)
    # p2x = 1 - stats.expon.cdf(x=h2, scale=1/tkp.lambda2)
    # t2 = time.clock()
    p1 = np.exp(-tkp.lambda1*h1) if h1 > 0 else 1
    p2 = np.exp(-tkp.lambda2*h2) if h2 > 0 else 1
    # t3 = time.clock()
    return p1*p2


def compute_constraint_loglikelihood(x):
    h1 = abs(x[0]) - tkp.x1_c
    h2 = abs(x[1]) - tkp.x2_c
    logli1 = -tkp.lambda1 * h1 if h1 > 0 else 0
    logli2 = -tkp.lambda2 * h2 if h2 > 0 else 0
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




