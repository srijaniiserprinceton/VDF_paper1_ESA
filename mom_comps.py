import numpy as np
import scipy as sp
import xarray as xr
import matplotlib.pyplot as plt
import cdflib 

from calculations.calc_moments import calc_moments, spher_moments

def plot_vdferr_per_vdf(energy, theta, phi, vdf_err, vdf, TROLL=0, PROLL=0, LOG=False):
    fig, ax = plt.subplots(nrows=4, ncols=8, figsize=(16, 16), layout='constrained')

    for i in range(4):
        for j in range(8):
            idx = 8*i + j
            if LOG is True:
                ax[i, j].pcolormesh(theta, phi, np.log10(np.roll(np.roll(vdf_err[idx]/vdf[idx], PROLL, axis=0), TROLL, axis=1)))
            else:
                ax[i, j].pcolormesh(theta, phi, np.roll(np.roll(vdf_err[idx]/vdf[idx], PROLL, axis=0), TROLL, axis=1))

    plt.show()


def generate_reconstructed_vdf():
    # Load in the reconstructed VDF data.
    log_E      = np.load('/home/michael/Research/VDF_paper1_ESA/calculations/lnE_mesh.npy')

    phi_mesh   = np.load('/home/michael/Research/VDF_paper1_ESA/calculations/phi_mesh.npy')
    theta_mesh = np.load('/home/michael/Research/VDF_paper1_ESA/calculations/theta_mesh.npy')
    vdf        = np.load('/home/michael/Research/VDF_paper1_ESA/calculations/VDF_3D_rec.npy')
    vdf        = 10**(vdf)
    
    vdf_min    = 2.81734931544879e-27
    vdf_result = vdf * vdf_min * 1e12
    
    energy = 10.0**(log_E)
    theta  = np.degrees(theta_mesh[:, 0])
    phi    = np.degrees(phi_mesh[0, :])

    velocity = 13.85 * np.sqrt(energy) * 1000

    return(vdf_result, velocity, theta, phi)



if __name__ == "__main__":
    # Load in the file that we are interested in looking at. 
    file = "/home/michael/Research/VDF_paper1_ESA/input_data_files/MMS_2016-01-11_VDF_and_ERRs.cdf"

    data_xr = cdflib.cdf_to_xarray(file, to_datetime=True)

    energy  = data_xr.energy.data[533, :, 0, 0]
    theta   = data_xr.theta.data[533, 0, 0, :]
    phi     = data_xr.phi.data[533, 0, :, 0]
    vdf     = data_xr.vdf.data[533, :, :, :] * 1e12    
    vdf_err = data_xr.vdf_err.data[533, :, :, :] * 1e12

    err_mask = vdf_err/vdf < 0.7
    vdf_noise = vdf.copy()
    vdf_noise[err_mask] = 0

    n_noise, u_noise, p_noise = calc_moments(np.transpose(vdf_noise, [0, 2, 1]), 13.8*np.sqrt(energy) * 1000, np.radians(theta), np.radians(phi))
    n_sig, u_sig, p_sig       = calc_moments(np.transpose(vdf, [0, 2, 1]), 13.8*np.sqrt(energy) * 1000, np.radians(theta), np.radians(phi))

    print(n_noise/100**3, n_sig/100**3, n_sig/100**3 - n_noise/100**3)

    n_rec, u_rec, p_rec = calc_moments(*generate_reconstructed_vdf())

    

