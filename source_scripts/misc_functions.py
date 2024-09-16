import numpy as np
NAX = np.newaxis

def grid_pol2cart(lnE, tt, pp):
    # making the Cartesian grid
    Vmag = 13.8 * np.sqrt(np.power(10, lnE))
    THETA, PHI = tt, pp
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]
    
    return VX, VY, VZ