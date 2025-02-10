import os
import numpy as np
import spherepy as sp
from scipy.special import sph_harm
from scipy.interpolate import griddata, interp1d
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

def gen_SH(rec_dict): # L, NPHI, NTHETA):

    # Swap the order of the basis function.
    sh_basis = np.zeros((rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA, (rec_dict.Lmax+1)**2), dtype='complex128')
    sh_coefs = sp.zeros_coefs(nmax=rec_dict.Lmax, mmax=rec_dict.Lmax)

    basis_count = 0
    for ell in range(rec_dict.Lmax+1):
        for m in range(-ell, ell+1):
            sh_coefs[ell, m] += 1.0
            sh_basis[:,:,int(basis_count)] = sp.ispht(sh_coefs, nrows=rec_dict.NTHETA_ESA, ncols=rec_dict.NPHI_ESA).array.T
            sh_coefs[ell, m] *= 0.0j
            basis_count += 1

    rec_dict.G = sh_basis

    
def gen_SH_scipy(rec_dict, time_idx):
    sh_basis = np.zeros((rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA, (rec_dict.Lmax+1)**2), dtype='complex128')

    basis_count = 0
    for ell in range(rec_dict.Lmax+1):
        for m in range(-ell, ell+1):
            norm = np.sqrt( ((2 * ell + 1)/(4*np.pi)) * (np.math.factorial(ell - m)/np.math.factorial(ell + m))  )
            sh_basis[:,:,int(basis_count)] = sph_harm(m, ell, np.radians(rec_dict.ESA_PHI[time_idx,0]), np.radians(rec_dict.ESA_THETA[time_idx,0]))
            basis_count += 1

    rec_dict.G = sh_basis
    

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

def reject_outliers(data, m = 3):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def interpolate_mask(mask, NEfine, NPfine, NTfine):
    NE, NT, NP = mask.shape
    P, T = np.linspace(-1,1,NP), np.linspace(-1,1,NT)
    PP, TT = np.meshgrid(P, T, indexing='ij')
    Pfine, Tfine = np.linspace(-1,1,NPfine), np.linspace(-1,1,NTfine)
    PPfine, TTfine = np.meshgrid(Pfine, Tfine, indexing='ij')

    # first interpolating in theta and phi
    mask_interp1 = []
    for Eidx in range(NE):
        mask_interp1.append(griddata((PP.flatten(), TT.flatten()), mask[Eidx].flatten(),
                           (PPfine, TTfine), method='nearest'))

    mask_interp1 = np.asarray(mask_interp1)

    # next interpolating in along the energy direction
    f_interp = interp1d(np.linspace(0,1,NE), mask_interp1, axis=0, kind='nearest')
    mask_interp = f_interp(np.linspace(0,1,NEfine))
    
    return mask_interp