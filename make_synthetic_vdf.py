import sys, os
import numpy as np
import matplotlib.pyplot as plt


def Maxwellian(x,n, u, w):
    """
    Generate a Maxwellian velocity distribution with given:

    n --- density.
    u --- bulk velocity.
    w --- thermal speed.

    Note: the thermal speed w = sqrt(2 kb T_s/ m_s) where the s
    corresponds to the species.
    """
    return n*np.exp(-((x - u)**2)/w**2)/(np.sqrt(np.pi)*w)


