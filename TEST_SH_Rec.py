# import statements
import os
import cdflib, sys, pickle
import numpy as np
import spherepy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm
from scipy.io import savemat
from scipy.stats import norm
from datetime import datetime, timedelta
from scipy.special import sph_harm
func = np.vectorize(datetime.utcfromtimestamp)

def read_config():
    package_dir = os.getcwd()  # os.path.dirname(current_dir)
    with open(f"{package_dir}/.config", "r") as f:
        dirnames = f.read().splitlines()

    return dirnames

class MMS:
    def __init__(self, data_ESA, TH, Lmax=None, Nmesh=(100, 201, 101), Espline_order=3, PHI_CEN_IDX=16, THETA_CEN_IDX=0, makeplot=True):
        self.instrument = 'MMS'

        NTIME, NENERGY, NPHI, NTHETA = data_ESA.vdf.data.shape
        VDF = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        VDF_ERR = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        ENERGY = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        THETA = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        PHI = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))

        VDF[:,:,:NPHI,:] = data_ESA.vdf.data
        VDF_ERR[:,:,:NPHI,:] = data_ESA.vdf_err.data
        ENERGY[:,:,:NPHI,:] = data_ESA.energy.data
        THETA[:,:,:NPHI,:] = data_ESA.theta.data
        PHI[:,:,:NPHI,:] = data_ESA.phi.data

        # we want to scale VDF such that the lowest non-zero entry is 1.0
        VDF[VDF == 0] = np.nan
        self.VDF_minval_true = np.nanmin(VDF)
        VDF = VDF / self.VDF_minval_true

        # populating the ghost cell in phi
        VDF[:,:,NPHI,:] = (VDF[:,:,0,:] + VDF[:,:,NPHI-1,:])/2.
        VDF_ERR[:,:,NPHI,:] = (VDF_ERR[:,:,0,:] + VDF_ERR[:,:,NPHI-1,:])/2.
        ENERGY[:,:,NPHI,:] = ENERGY[:,:,0,:]
        THETA[:,:,NPHI,:] = THETA[:,:,0,:]
        PHI[:,:,NPHI,:] = PHI[:,:,NPHI-1,:] + 11.25   # IN DEGREES

        self.VDF = VDF * 1.0
        self.VDF_ERR = VDF_ERR * 1.0
        self.ENERGY = ENERGY * 1.0

        # the true MMS FPI grid
        self.ESA_THETA = THETA * 1.0
        self.ESA_PHI = PHI * 1.0

        self.NENERGY = NENERGY
        self.NPHI_ESA = NPHI + 1
        self.NTHETA_ESA = NTHETA

        # the grid to be used for Slepian reconstruction
        self.SLEP_THETA = np.linspace(0, 180, self.NTHETA_ESA) - 90
        self.SLEP_PHI = np.linspace(0, 360, self.NPHI_ESA)

        self.NPHI_SLEP = NPHI + 1
        self.NTHETA_SLEP = NTHETA

        self.SLEP_PP, self.SLEP_TT = np.meshgrid(self.SLEP_PHI, self.SLEP_THETA, indexing='ij')

        # Setup slepian directory
        self.slep_dir = read_config()[0]

        # Check Lmax
        self.Lmax_Nyq = int(self.NTHETA_ESA / 2)
        if Lmax is None: 
            self.Lmax = self.Lmax_Nyq
        else:
            if Lmax > self.Lmax_Nyq: print("Lmax exceeds Nyquist. Resetting to Nyquist.")
            self.Lmax = min(Lmax, self.Lmax_Nyq)
        
        # to be initialized later in the workflow
        self.G = None
        self.V = None
        self.TH = TH

        # the location of the phi and theta center
        self.PHI_CEN_IDX = PHI_CEN_IDX
        self.THETA_CEN_IDX = THETA_CEN_IDX

        # Init parameters for Energy Shell interpolation. 
        self.NEmesh, self.NPmesh, self.NTmesh = Nmesh
        self.Espline_order = Espline_order


        self.makeplot = makeplot

        # finding the vmin and vmax according to the time 
        self.vmax_t = np.nanmax(np.log10(self.VDF), axis=(1,2,3)).astype('int')
        # self.vmax_t = np.nan_to_num(nan = 1.1, posinf=1.0, neginf=1.0)
        self.vmin_t = np.ones_like(self.vmax_t)

def gen_SH_scipy(rec_dict, time_idx): # L, NPHI, NTHETA, PHI, THETA):
    sh_basis = np.zeros(((rec_dict.Lmax+1)**2, rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA), dtype='complex128')

    basis_count = 0
    for ell in range(rec_dict.Lmax+1):
        for m in range(-ell, ell+1):
            sh_basis[int(basis_count)] = sph_harm(m, ell, np.radians(rec_dict.ESA_PHI[time_idx,0]), np.radians(rec_dict.ESA_THETA[time_idx,0]))
            basis_count += 1

    rec_dict.SH_hr = sh_basis

if __name__ == "__main__":
    NEmesh, NPmesh, NTmesh = 200, 201, 101
    Espline_order = 3

    filename = './input_data_files/MMS_2016-01-11_VDF_and_ERRs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # calculating time in units of milliseconds
    times_datetime = func(data.unix_time.values)
    times = (times_datetime - times_datetime[0]) / timedelta(seconds=1)

    rec_dict = MMS(data, 45, Lmax=None, Nmesh=(NEmesh, NPmesh, NTmesh), Espline_order=3, makeplot=False)

    # Now let us generate the SH basis
    time_idx = 479

    gen_SH_scipy(rec_dict, time_idx)

    rec = np.zeros((rec_dict.NENERGY, rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA))
    for E_idx in range(16,17): # range(rec_dict.NENERGY):
        E  = rec_dict.ENERGY[time_idx, E_idx, 0, 0]
        vv = rec_dict.VDF[time_idx, E_idx, :, :]

        data_vv = np.log10(vv)
        data_vv = np.nan_to_num(data_vv, posinf=np.nan, neginf=np.nan)

        img_hr = data_vv * 1.0

        coeffs = np.zeros((rec_dict.NENERGY, (rec_dict.Lmax + 1)**2))

        # Fitting with the Spherical Harmonics
        nan_mask = np.isnan(img_hr)
        SH_nonan = rec_dict.SH_hr[:, ~nan_mask]
        M = SH_nonan @ SH_nonan.T
        _, rec_dict.S_hr, _ = np.linalg.svd(M)
        I = np.identity(M.shape[0])
        coeffs[E_idx] = np.linalg.inv(M + rec_dict.S_hr.max() * 0 * I) @ (SH_nonan) @ img_hr[~nan_mask]

        rec_coefs = np.dot(np.moveaxis(rec_dict.SH_hr, 0, -1), coeffs[E_idx])
        rec[E_idx] += rec_coefs.real

        print(img_hr.shape, nan_mask.shape)


