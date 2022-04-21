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
        x3 = cw*vx - sw*vy
        x4 = sw*vx + cw*vy
    elif sc == 3:
        w = tkp.omegal
        sw = tkp.swl
        cw = tkp.cwl
        x1 = px + sw/w*vx - (1-cw)/w*vy
        x2 = py + (1-cw)/w*vx + sw/w*vy
        x3 = cw*vx - sw*vy
        x4 = sw*vx + cw*vy
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
    px, py, vx, vy = x

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
        Ja[2, 2] = cw
        Ja[2, 3] = -sw
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = sw
        Ja[3, 3] = cw
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
        Ja[2, 2] = cw
        Ja[2, 3] = -sw
        Ja[3, 0] = 0
        Ja[3, 1] = 0
        Ja[3, 2] = sw
        Ja[3, 3] = cw
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

    tpm = np.zeros([tkp.M, tkp.M])
    if t <= tkp.tlast:
        for j in range(tkp.M):
            tpm[j, j] = 1
    else:
        px, py, vx, vy = x
        angle = np.arctan(vy/vx)
        velocity = np.sqrt(vx**2 + vy**2)

        tpm[0, 1] = tkp.alpha12 + tkp.nu12*pdf_Gaussian(x=[px, py], mean=[tkp.px_tcp1, tkp.py_tcp1], cov=tkp.Sigma12)
        tpm[1, 0] = tkp.alpha21 + tkp.nu21*pdf_Gaussian(x=angle, mean=tkp.psi21, cov=tkp.sigma21)
        tpm[0, 2] = tkp.alpha13 + tkp.nu13*pdf_Gaussian(x=[px, py], mean=[tkp.px_tcp2, tkp.py_tcp2], cov=tkp.Sigma13)
        tpm[2, 0] = tkp.alpha31 + tkp.nu31*pdf_Gaussian(x=angle, mean=tkp.psi31, cov=tkp.sigma31)
        tpm[0, 3] = tkp.alpha14 + tkp.nu14*tkp.psi14*np.exp(-tkp.psi14*velocity)
        tpm[3, 0] = tkp.alpha41 + tkp.nu41/(1+np.exp(-tkp.psi41*(velocity-tkp.ve)))
        tpm[0, 4] = tkp.alpha15 + tkp.nu15/(1+np.exp(-tkp.psi15*(velocity-tkp.vmax)))
        tpm[4, 0] = tkp.alpha51 + tkp.nu51/(1+np.exp(-tkp.psi51*(-velocity+tkp.ve)))

        tpm[0, 0] = 1 - tpm[0, 1] - tpm[0, 2] - tpm[0, 3] - tpm[0, 4]
        tpm[1, 1] = 1 - tpm[1, 0]
        tpm[2, 2] = 1 - tpm[2, 0]
        tpm[3, 3] = 1 - tpm[3, 0]
        tpm[4, 4] = 1 - tpm[4, 0]
        if tpm[0, 0] < 0:
            raise ValueError('Transition probability sum of mode 1 exceed 1')

    return tpm


def switch_tracking(sp, xp, tp):
    mode_list = [1, 2, 3, 4, 5]
    if sp not in mode_list:
        raise ValueError("Invalid mode.")
    if (type(tp) is not int) and (type(tp) is not np.int32):
        raise ValueError("Invalid sojourn time.")
    tpm = tpm_tracking(xp, tp)
    probability = tpm[sp-1, :]
    sc = np.random.choice(mode_list, 1, p=probability)[0]
    return sc


def constraint_tracking(x):
    px, py, vx, vy = x
    if py > tkp.ca3*px**3 + tkp.ca2*px**2 + tkp.ca1*px + tkp.ca0:
        py = tkp.ca3*px**3 + tkp.ca2*px**2 + tkp.ca1*px + tkp.ca0
        x[1] = py
        v = np.sqrt(vx**2 + vy**2)
        phi = np.arctan(3*tkp.ca3*px**2 + 2*tkp.ca2*px + tkp.ca1)
        x[2] = v * np.cos(phi)
        x[3] = v * np.sin(phi)
        if_reach_constraint = True
    elif py < tkp.cb3*px**3 + tkp.cb2*px**2 + tkp.cb1*px + tkp.cb0:
        py = tkp.cb3*px**3 + tkp.cb2*px**2 + tkp.cb1*px + tkp.cb0
        x[1] = py
        v = np.sqrt(vx**2 + vy**2)
        phi = np.arctan(3*tkp.cb3*px**2 + 2*tkp.cb2*px + tkp.cb1)
        x[2] = v * np.cos(phi)
        x[3] = v * np.sin(phi)
        if_reach_constraint = True
    else:
        if_reach_constraint = False

    return x, if_reach_constraint


def compute_constraint_likelihood(x):
    px, py, vx, vy = x
    h1 = py - (tkp.ca3*px**3 + tkp.ca2*px**2 + tkp.ca1*px + tkp.ca0)
    h2 = -py + tkp.cb3*px**3 + tkp.cb2*px**2 + tkp.cb1*px + tkp.cb0
    h3 = np.sqrt(vx**2+vy**2) - tkp.vmax
    # t1 = time.clock()
    # p1x = 1 - stats.expon.cdf(x=h1, scale=1/tkp.lambda1)
    # p2x = 1 - stats.expon.cdf(x=h2, scale=1/tkp.lambda2)
    # t2 = time.clock()
    p1 = np.exp(-tkp.lambda1*h1) if h1 > 0 else 1
    p2 = np.exp(-tkp.lambda2*h2) if h2 > 0 else 1
    p3 = np.exp(-tkp.lambda3*h3) if h3 > 0 else 1
    # t3 = time.clock()
    return p1*p2*p3


def compute_constraint_loglikelihood(x):
    px, py, vx, vy = x
    h1 = py - (tkp.ca3*px**3 + tkp.ca2*px**2 + tkp.ca1*px + tkp.ca0)
    h2 = -py + tkp.cb3*px**3 + tkp.cb2*px**2 + tkp.cb1*px + tkp.cb0
    h3 = np.sqrt(vx**2+vy**2) - tkp.vmax
    logli1 = -tkp.lambda1 * h1 if h1 > 0 else 0
    logli2 = -tkp.lambda2 * h2 if h2 > 0 else 0
    logli3 = -tkp.lambda3 * h3 if h3 > 0 else 0
    return logli1*logli2*logli3


def pdf_Gaussian(x, mean, cov):
    if isinstance(x, list) or isinstance(x, float) or isinstance(x, int):
        x = np.array(x)
    mean = np.array(mean)
    if mean.size == 1:
        pdf = 1/np.sqrt(2*np.pi*cov)*np.exp(-(x-mean)**2/(2*cov))
    else:
        if isinstance(cov, list):
            # Fast implementation for diag cov
            cov_diag = np.array(cov)
            cov_det = cov_diag.prod()
            cov_inv = 1 / cov_diag
            pdf = 1 / ((2*np.pi) ** (mean.size/2) * np.sqrt(cov_det)) * np.exp(-1/2 * np.sum((x - mean)**2 * cov_inv))
        elif cov.ndim == 1:
            cov_diag = cov
            cov_det = cov_diag.prod()
            cov_inv = 1 / cov_diag
            pdf = 1 / ((2*np.pi) ** (mean.size/2) * np.sqrt(cov_det)) * np.exp(-1/2 * ((x - mean)**2 * cov_inv))
        else:
            pdf = np.power(2*np.pi, -mean.size/2) * np.power(np.linalg.det(cov), -1/2) \
                  * np.exp(-1/2 * np.matmul(np.matmul(x - mean, np.linalg.inv(cov)), x - mean))
    if pdf.size != 1:
        raise ValueError('Error pdf size')
    return pdf.mean()


def cdf_expon(x, lam):
    c = 1 - np.exp(-lam*x) if x > 0 else 0
    return c




