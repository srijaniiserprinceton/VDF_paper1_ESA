import os
import numpy as np
import spherepy as sp
NAX = np.newaxis

def grid_pol2cart(lnE, tt, pp, savegrids=False):
    # making the Cartesian grid
    Vmag = 13.85 * np.sqrt(np.power(10, lnE))
    THETA, PHI = tt, pp
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]

    if(savegrids==True):
        np.save('./output_data_files/VX.npy', VX)
        np.save('./output_data_files/VY.npy', VY)
        np.save('./output_data_files/VZ.npy', VZ)
    
    return VX, VY, VZ

def gen_SH(L, NPHI, NTHETA):
    sh_basis = np.zeros(((L+1)**2, NTHETA, NPHI))
    sh_coefs = sp.zeros_coefs(nmax=L, mmax=L)

    basis_count = 0
    for ell in range(L+1):
        for m in range(-ell, ell+1):
            sh_coefs[ell, m] += 1.0
            sh_basis[int(basis_count)] = sp.ispht(sh_coefs, nrows=NTHETA, ncols=NPHI).array.real
            sh_coefs[ell, m] *= 0.0
            basis_count += 1

    return sh_basis

def gen_SLEP(rec_dict, N2D_restrict=False):
    import matlab.engine as matlab
    # generating the low and high resolution Slepians-on-polar-cap
    eng = matlab.start_matlab()
    s = eng.genpath(rec_dict.slep_dir)
    eng.addpath(s, nargout=0)
    [G, V, lon, lat] = eng.glmalphapto(f'VDF_polarcap_{rec_dict.instrument}', rec_dict.Lmax, 'HIGHRES', nargout=4)
    rec_dict.G = np.asarray(G)
    rec_dict.V = np.asarray(V).squeeze()
    rec_dict.SLEP_PHI = np.asarray(lon)
    rec_dict.SLEP_THETA = np.asarray(lat)
    
    # keeping only until the Shannon number
    if(N2D_restrict):
        N2D = np.argmin(np.abs(rec_dict.V - 0.5))
        rec_dict.G = rec_dict.G[:N2D]
        rec_dict.V = rec_dict.V[:N2D]

    eng.quit()
        

def read_config():
    package_dir = os.getcwd()  # os.path.dirname(current_dir)
    with open(f"{package_dir}/.config", "r") as f:
        dirnames = f.read().splitlines()

    return dirnames