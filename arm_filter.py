import numpy as np
import arm_paras as ap


if __name__ == '__main__':

    T = ap.T
    dt = ap.dt
    K = int(T // dt)

    weights = np.zeros()

