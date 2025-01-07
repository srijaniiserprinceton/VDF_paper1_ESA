# import statements
import os
import cdflib, sys, pickle
import cdflib.xarray
import numpy as np
import spherepy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm
from scipy.io import savemat
from scipy.stats import norm
from datetime import datetime, timedelta
from scipy.special import sph_harm
from scipy.integrate import simps
func = np.vectorize(datetime.utcfromtimestamp)

from calculations.calc_moments import calc_moments
from scipy.stats import linregress

from matplotlib.colors import LogNorm

def read_config():
    package_dir = os.getcwd()  # os.path.dirname(current_dir)
    with open(f"{package_dir}/.config", "r") as f:
        dirnames = f.read().splitlines()

    return dirnames

class MMS:
    def __init__(self, data_ESA, TH, Lmax=None, Nmesh=(100, 201, 101), Espline_order=3, PHI_CEN_IDX=16, THETA_CEN_IDX=0, makeplot=True):
        self.instrument = 'MMS'

        NTIME, NENERGY, NPHI, NTHETA = data_ESA.vdf.data.shape
        VDF = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        VDF_ERR = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        ENERGY = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        THETA = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        PHI = np.zeros((NTIME, NENERGY, NPHI, NTHETA))

        VDF[:,:,:,:] = data_ESA.vdf.data
        VDF_ERR[:,:,:,:] = data_ESA.vdf_err.data
        ENERGY[:,:,:,:] = data_ESA.energy.data
        THETA[:,:,:,:] = data_ESA.theta.data
        PHI[:,:,:,:] = data_ESA.phi.data

        # NTIME, NENERGY, NPHI, NTHETA = data_ESA.vdf.data.shape
        # VDF = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        # VDF_ERR = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        # ENERGY = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        # THETA = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        # PHI = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))

        # VDF[:,:,:NPHI,:] = data_ESA.vdf.data
        # VDF_ERR[:,:,:NPHI,:] = data_ESA.vdf_err.data
        # ENERGY[:,:,:NPHI,:] = data_ESA.energy.data
        # THETA[:,:,:NPHI,:] = data_ESA.theta.data
        # PHI[:,:,:NPHI,:] = data_ESA.phi.data

        # we want to scale VDF such that the lowest non-zero entry is 1.0
        VDF[VDF == 0] = np.nan
        self.VDF_minval_true = np.nanmin(VDF)
        VDF = VDF / self.VDF_minval_true

        # # populating the ghost cell in phi
        # VDF[:,:,NPHI,:] = (VDF[:,:,0,:] + VDF[:,:,NPHI-1,:])/2.
        # VDF_ERR[:,:,NPHI,:] = (VDF_ERR[:,:,0,:] + VDF_ERR[:,:,NPHI-1,:])/2.
        # ENERGY[:,:,NPHI,:] = ENERGY[:,:,0,:]
        # THETA[:,:,NPHI,:] = THETA[:,:,0,:]
        # PHI[:,:,NPHI,:] = PHI[:,:,NPHI-1,:] + 11.25   # IN DEGREES

        self.VDF = VDF * 1.0
        self.VDF_ERR = VDF_ERR * 1.0
        self.ENERGY = ENERGY * 1.0

        # the true MMS FPI grid
        self.ESA_THETA = THETA * 1.0
        self.ESA_PHI = PHI * 1.0

        self.NENERGY = NENERGY
        # self.NPHI_ESA = NPHI + 1
        self.NPHI_ESA = NPHI
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
        self.Lmax_Nyq = min(int(self.NTHETA_ESA - 2), int((self.NPHI_ESA - 2)/ 2))
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

def gen_SH(rec_dict): # L, NPHI, NTHETA):

    Lmax = rec_dict.Lmax
    sh_basis = np.zeros((rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA, (Lmax+1)**2), dtype='complex128')
    sh_coefs = sp.zeros_coefs(nmax=Lmax, mmax=Lmax)

    basis_count = 0
    for ell in range(Lmax+1):
        for m in range(-ell, ell+1):
            sh_coefs[ell, m] = 1.0
            sh_basis[:,:,int(basis_count)] = sp.ispht(sh_coefs, nrows=rec_dict.NTHETA_ESA, ncols=rec_dict.NPHI_ESA).array.T
            sh_coefs[ell, m] *= 0.0j
            basis_count += 1

    rec_dict.SH_hr = sh_basis

def gen_SH_scipy(rec_dict, time_idx):
    Lmax = rec_dict.Lmax
    sh_basis = np.zeros((rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA, (Lmax+1)**2), dtype='complex128')

    basis_count = 0
    for ell in range(rec_dict.Lmax+1):
        for m in range(-ell, ell+1):
            norm = np.sqrt( ((2 * ell + 1)/(4*np.pi)) * (np.math.factorial(ell - m)/np.math.factorial(ell + m))  )
            print(norm)
            sh_basis[:,:,int(basis_count)] = sph_harm(m, ell, np.radians(rec_dict.ESA_PHI[time_idx,0]), np.radians(rec_dict.ESA_THETA[time_idx,0])) * norm
            basis_count += 1

    rec_dict.SH_hr = sh_basis

def make_Y_arr(Lmax, Ncols, Nrows):
    #Ncols and Nrows are being swapped in sph_rec function
    zeroes_coefs = sp.zeros_coefs(nmax = Lmax, mmax = Lmax)

    Y_arr = []
    for i in range(Lmax + 1):  # i ranges from 0 to 20 inclusive
        for j in range(max(-Lmax, -i), min(Lmax, i) + 1):  # j ranges from -i to i inclusive
            zeroes_coefs[i,j] = 1
            img_rec = sp.ispht(zeroes_coefs, nrows=Nrows, ncols=Ncols).array
            Y_arr.append(img_rec)
            zeroes_coefs = sp.zeros_coefs(nmax = Lmax, mmax = Lmax)

    Y_arr = np.asarray(Y_arr)
    #Y_arr.shape = (Nsph, Nrows, Ncols)
    return Y_arr

def svd_sph_rec(image_svd, Lmax_svd, u, return_coeffs = False):
    #image being passed in is NOT transposed
    #get information from input data
    Nrows, Ncols = image_svd.shape
    Yarr = make_Y_arr(Lmax_svd, Nrows, Ncols)
    image_svd = image_svd.T
    Nsph = (Lmax_svd + 1 )**2

    #put input data in desired format for spherical harmonic recomposition using nan masks to remove the nan values
    mask = np.isnan(image_svd)
    img_masked = image_svd[~mask]
    N_non_nan = np.sum(~mask)
    Y_arr_masked_L = np.zeros((N_non_nan, len(Yarr[:,:Nsph])), dtype = 'complex128')

    for i in range(Nsph):
        datai = Yarr[i][~mask]
        Y_arr_masked_L[:,i] = datai
    #c = (M^T M)^-1 M^T d
    #finding M^T M and performing svd
    inverse_matrix = np.conjugate(Y_arr_masked_L).T @ Y_arr_masked_L  # Matrix multiplication
    U, A, V = np.linalg.svd(inverse_matrix)
    inverse = np.linalg.inv(inverse_matrix + (u * np.identity(A.size)))
    coefficients = inverse @ np.conjugate(Y_arr_masked_L).T @ img_masked
    image_rec = np.sum(Yarr[:Nsph] * coefficients[:, np.newaxis, np.newaxis], axis = 0)
    if return_coeffs:
        return image_rec, coefficients
    return Yarr, Y_arr_masked_L, inverse_matrix, inverse, img_masked, image_rec, coefficients

def calc_moments_MMS_SH(time_idx, mask_noisy=False):
    vv = rec_dict.VDF[time_idx, :, :, :] * 1.0
    data_vv = vv
    data_vv = np.nan_to_num(data_vv, nan=0, posinf=0, neginf=0)

    DATA_VDF = np.transpose(data_vv, [0, 2, 1]) * 1e12 * rec_dict.VDF_minval_true
    REC_VDF = np.transpose(rec_final[time_idx], [0, 2, 1]) * 1e12 * rec_dict.VDF_minval_true

    # removing the parts of the data and reconstructed VDF which have larger than NSR = 0.7
    if(mask_noisy):
        DATA_VDF_ERR = np.transpose(rec_dict.VDF_ERR[time_idx], [0, 2, 1]) * 1e12
        err_mask = DATA_VDF_ERR/DATA_VDF > 0.5
        DATA_VDF[err_mask] = 0.0

    velocity = 13.8 * np.sqrt(rec_dict.ENERGY[time_idx, :, 0, 0]) * 1000

    DATA_theta, DATA_phi = rec_dict.ESA_THETA[time_idx,0,0,:], rec_dict.ESA_PHI[time_idx,0,:,0]
    REC_theta, REC_phi = rec_dict.SLEP_THETA+90, rec_dict.SLEP_PHI

    # adjsuting the data phi to go from 0->360
    DATA_phi = DATA_phi - DATA_phi[0]

    # in SI units
    DATA_VDF[np.isnan(DATA_VDF)] = 0.0

    DATA_moments = calc_moments(DATA_VDF, velocity, np.radians(DATA_theta), np.radians(DATA_phi), METHOD='simps')
    REC_moments = calc_moments(REC_VDF, velocity, np.radians(DATA_theta), np.radians(DATA_phi), METHOD='simps')

    return DATA_moments, REC_moments

def log_calc_moments_MMS_SH(time_idx, mask_noisy=False):
    vv = np.log10(rec_dict.VDF[time_idx, :, :, :]) * 1.0
    data_vv = vv
    data_vv = np.nan_to_num(data_vv, nan=0, posinf=0, neginf=0)

    DATA_VDF = np.transpose(np.power(10,data_vv), [0, 2, 1]) * 1e12 * rec_dict.VDF_minval_true
    REC_VDF = np.transpose(np.power(10, rec_final[time_idx]), [0, 2, 1]) * 1e12 * rec_dict.VDF_minval_true

    # removing the parts of the data and reconstructed VDF which have larger than NSR = 0.7
    if(mask_noisy):
        DATA_VDF_ERR = np.transpose(rec_dict.VDF_ERR[time_idx], [0, 2, 1]) * 1e12
        err_mask = DATA_VDF_ERR/DATA_VDF > 0.5
        DATA_VDF[err_mask] = 0.0

    velocity = 13.8 * np.sqrt(rec_dict.ENERGY[time_idx, :, 0, 0]) * 1000

    DATA_theta, DATA_phi = rec_dict.ESA_THETA[time_idx,0,0,:], rec_dict.ESA_PHI[time_idx,0,:,0]
    REC_theta, REC_phi = rec_dict.SLEP_THETA+90, rec_dict.SLEP_PHI

    # adjsuting the data phi to go from 0->360
    DATA_phi = DATA_phi - DATA_phi[0]

    # in SI units
    DATA_VDF[np.isnan(DATA_VDF)] = 0.0

    DATA_moments = calc_moments(DATA_VDF, velocity, np.radians(DATA_theta), np.radians(DATA_phi), METHOD='simps')
    REC_moments = calc_moments(REC_VDF, velocity, np.radians(DATA_theta), np.radians(DATA_phi), METHOD='simps')

    return DATA_moments, REC_moments

def get_mu(chi2data, chi2rho):
    # normalize
    chi2data = (chi2data - chi2data.min())/(chi2data.max() - chi2data.min())
    chi2rho = (chi2rho - chi2rho.min())/(chi2rho.max() - chi2rho.min())

    
    line1 = linregress(chi2rho[0:10], chi2data[0:10])
    line2 = linregress(chi2rho[-10:], chi2data[-10:])
    xpoint = -(line1[1] - line2[1])/(line1[0] - line2[0])
    ypoint = line1[0] * xpoint + line1[1]
    return(np.argmin((chi2data - ypoint)**2 + (chi2rho - xpoint)**2))

if __name__ == "__main__":
    LOG = True
    NEmesh, NPmesh, NTmesh = 200, 201, 101
    Espline_order = 3

    filename = './input_data_files/MMS_2016-01-11_VDF_and_ERRs.cdf'
    data = cdflib.xarray.cdf_to_xarray(filename, to_datetime=True)

    # calculating time in units of milliseconds
    times_datetime = func(data.unix_time.values)
    times = (times_datetime - times_datetime[0]) / timedelta(seconds=1)

    rec_dict = MMS(data, 45, Lmax=14, Nmesh=(NEmesh, NPmesh, NTmesh), Espline_order=3, makeplot=False)

    gen_SH_scipy(rec_dict, 563)
    #gen_SH(rec_dict)

    n_mu = 32
    rec_final = np.zeros((1399, rec_dict.NENERGY, rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA))

    data_mom, rec_mom = {}, {}
    for time_idx in tqdm(range(563, 564)):
        # rec = np.zeros((n_mu, rec_dict.NENERGY, rec_dict.NPHI_ESA, rec_dict.NTHETA_ESA))
        coeffs = np.zeros((rec_dict.NENERGY, (rec_dict.Lmax + 1)**2), dtype='complex128')

        chi2data = np.zeros(n_mu)
        chi2rho  = np.zeros(n_mu)

        mu_vals = np.linspace(-5,5, n_mu)

        rho_look = 0
        rec_rho_look = 0

        # for mu_idx, mu_power in enumerate(mu_vals[0:1]):
        #     mu = 0 # np.power(10, mu_power)
    
        #     find_peak = np.unravel_index(np.nanargmax(rec_dict.VDF[time_idx, :, :, :]), rec_dict.VDF[time_idx, :, :, :].shape)

        #     for E_idx in (range(find_peak[0],find_peak[0]+1)): 
        #         E  = rec_dict.ENERGY[time_idx, E_idx, 0, 0] * 1.0
        #         vv = rec_dict.VDF[time_idx, E_idx, :, :] * 1.0

        #         data_vv = np.log10(vv)
        #         data_vv = np.nan_to_num(data_vv, nan=0, posinf=0, neginf=0)

        #         img_hr = data_vv * 1.0

        #         # Fitting with the Spherical Harmonics
        #         nan_mask = np.isnan(img_hr)
        #         peak_mask = (np.nanmax(img_hr) - img_hr) <= 2 

        #         G = rec_dict.SH_hr * 1.0
        #         Gmask = G[~nan_mask]

        #         esa_phi_rad = np.radians(rec_dict.ESA_PHI[time_idx, 0, :, 0])
        #         esa_theta_rad = np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :])
        #         rho = simps(simps(img_hr, x = esa_phi_rad, axis=0) * np.sin(esa_theta_rad), x = esa_theta_rad)

        #         img_mask = img_hr.copy()
        #         img_mask[~peak_mask] = 0

        #         rho_mask = simps(simps(img_mask, x = esa_phi_rad, axis=0) * np.sin(esa_theta_rad), x = esa_theta_rad)

        #         rho_power = simps(simps(np.power(10,img_hr), x = esa_phi_rad, axis=0) * np.sin(esa_theta_rad), x = esa_theta_rad)

        #         S = simps(simps(G, x = esa_phi_rad, axis=0) * np.sin(esa_theta_rad)[:, None], x = esa_theta_rad, axis=0)
                
        #         G_peak = G.copy()
        #         G_peak[~peak_mask] = 0

        #         S_mask = simps(simps(G_peak, x = esa_phi_rad, axis=0) * np.sin(esa_theta_rad)[:, None], x = esa_theta_rad, axis=0)

        #         SS = np.outer(np.conjugate(S_mask).T, S_mask)

        #         rhoS = rho_mask * np.conjugate(S_mask)

        #         M = (np.conjugate(Gmask).T @ Gmask)
        #         _, rec_dict.S_hr, _ = np.linalg.svd(M)
        #         I = np.identity(M.shape[0])
        #         # inverted_M = np.linalg.inv(M + rec_dict.S_hr.max() * 1e-4 * I)
        #         inverted_M = np.linalg.inv(M + mu * SS)
                
        #         coeffs[E_idx] = inverted_M @ (np.conjugate(Gmask).T @ img_hr[~nan_mask] + mu * rhoS)

        #         rec_coefs = np.dot(G, coeffs[E_idx])

        #         rec[mu_idx, E_idx] = rec_coefs

        #         rec_coefs_masks = rec_coefs.copy()
        #         rec_coefs_masks[~peak_mask] = 0

        #         rec_rho = simps(simps(np.power(10,rec_coefs_masks), x = esa_phi_rad, axis=0) * np.sin(esa_theta_rad), x = esa_theta_rad, axis=0)

        #         print(rho_power, rec_rho,  rho_power / rec_rho.real, mu, mu_idx)

        #         rho_look += rho_power
        #         rec_rho_look += rec_rho

        #         chi2data[mu_idx] += np.nansum(np.abs(img_hr - rec_coefs)**2)
        #         chi2rho[mu_idx] += np.abs(np.power(10,rho) - np.power(10,rec_rho))**2

        #     print("HERE:", rho_look, rec_rho_look, mu_idx)

        # sys.exit()
        rho_look = 0
        rec_rho_look = 0
        mu_knee_idx = 17 #get_mu(chi2data, chi2rho)
        mu = 0 #np.power(10,mu_vals[mu_knee_idx ])
        # print('Mu value:', mu, mu_knee_idx)
        for E_idx in (range(0,32)): 
            E  = rec_dict.ENERGY[time_idx, E_idx, 0, 0] * 1.0
            vv = rec_dict.VDF[time_idx, E_idx, :, :] * 1.0

            data_vv = np.log10(vv)
            data_vv = np.nan_to_num(data_vv, nan=0, posinf=0, neginf=0)

            img_hr = np.ones_like(data_vv) * 1.0

            # Fitting with the Spherical Harmonics
            nan_mask = np.isnan(img_hr)

            G = rec_dict.SH_hr * 1.0
            Gmask = G[~nan_mask]

            rho = simps(simps(np.abs(img_hr), x = np.radians(rec_dict.ESA_PHI[time_idx, 0, :, 0]), axis=0) \
                        * np.sin(np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :])), x = np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :]))

            S = simps(simps(G, x = np.radians(rec_dict.ESA_PHI[time_idx, 0, :, 0]), axis=0) \
                        * np.sin(np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :]))[:, None], x = np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :]), axis=0)
            

            SS = np.outer(np.conjugate(S).T, S)

            rhoS = rho * np.conjugate(S)

            M = (np.conjugate(Gmask).T @ Gmask)
            _, rec_dict.S_hr, _ = np.linalg.svd(M)
            I = np.identity(M.shape[0])
            # inverted_M = np.linalg.inv(M + rec_dict.S_hr.max() * 1e-4 * I)
            inverted_M = np.linalg.inv(M + mu * SS)
            
            coeffs[E_idx] = inverted_M @ (np.conjugate(Gmask).T @ img_hr[~nan_mask] + mu * rhoS)

            rec_coefs = np.dot(G, coeffs[E_idx])

            rec_final[time_idx, E_idx] = rec_coefs
            
            rec_rho = simps(simps(rec_coefs, x = np.radians(rec_dict.ESA_PHI[time_idx, 0, :, 0]), axis=0) \
                            * np.sin(np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :])), x = np.radians(rec_dict.ESA_THETA[time_idx, 0, 0, :]))

            rho_look += rho
            rec_rho_look += rec_rho

        data_mom[time_idx], rec_mom[time_idx] = calc_moments_MMS_SH(time_idx)
        # print("HERE 2:", rho_look, rec_rho_look)
        # print(data_mom[0], rec_mom[0])
        # sys.exit()
        # Nrows, Ncols = 4, 8
        # fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

        # for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        #     # finding the row and the column of the subplots
        #     row, col = i//Ncols, i%Ncols

        #     E = rec_dict.ENERGY[time_idx, E_idx, 0, 0]
            
        #     # note that for FPI, theta array stays the same at different times but phi array changes
        #     tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
        #                         rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
        #                         rec_dict.VDF[time_idx, E_idx, :, :]

        #     if LOG:
        #         ax[row, col].pcolormesh(pp_orig, tt_orig, np.roll(np.log10(vv), 16, axis=0), vmin=1, vmax=8, cmap='inferno')
        #     else:
        #         ax[row, col].pcolormesh(pp_orig, tt_orig, np.roll((vv), 16, axis=0), cmap='inferno', norm=LogNorm(vmin=1e1, vmax=1e7))
        #     ax[row, col].set_aspect('equal')

        # fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True, layout='constrained')

        # for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        #     # finding the row and the column of the subplots
        #     row, col = i//Ncols, i%Ncols

        #     E = rec_dict.ENERGY[time_idx, E_idx, 0, 0]
            
        #     # note that for FPI, theta array stays the same at different times but phi array changes
        #     tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
        #                         rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
        #                         rec_dict.VDF[time_idx, E_idx, :, :]

        #     if LOG:
        #         ax[row, col].pcolormesh(pp_orig, tt_orig, np.roll((rec_final[time_idx, i]), 16, axis=0), vmin=1, vmax=8, cmap='inferno')
        #     else:
        #         ax[row, col].pcolormesh(pp_orig, tt_orig, np.roll((rec_final[time_idx, i]), 16, axis=0), norm=LogNorm(vmin=1e1, vmax=1e7), cmap='inferno')
        #     ax[row, col].set_title(f'{E}')
        #     ax[row, col].set_aspect('equal')

    # plt.suptitle('With Scipy', fontsize=20)