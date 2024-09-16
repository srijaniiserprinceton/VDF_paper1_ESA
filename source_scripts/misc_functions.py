import numpy as np
NAX = np.newaxis

def grid_pol2cart(lnE, tt, pp, savegrids=False):
    # making the Cartesian grid
    Vmag = 13.8 * np.sqrt(np.power(10, lnE))
    THETA, PHI = tt, pp
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]

    if(savegrids==True):
        np.save('./output_data_files/VX.npy', VX)
        np.save('./output_data_files/VY.npy', VY)
        np.save('./output_data_files/VZ.npy', VZ)
    
    return VX, VY, VZ