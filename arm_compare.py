import numpy as np
import arm_model as am
import arm_paras as ap


def one_step_EKF(xp, Pp, z, s):
    Ja = am.dynamic_Jacobian_arm(x=xp, s=s)
    
