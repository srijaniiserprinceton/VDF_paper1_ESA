'''
---------------------------------------------------------------------------
This is the driver script that should be run to get the reconstructed VDFs.
---------------------------------------------------------------------------
'''

# import statements
import cdflib, sys, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm
from scipy.io import savemat
from datetime import datetime, timedelta
func = np.vectorize(datetime.utcfromtimestamp)

from source_scripts import import_script, setup_rec_grid
from source_scripts import misc_functions as misc_funcs
import plot_3D_VDF
from calculations.calc_moments import calc_moments, spher_moments

def reconstruct_from_PSP(time_idx):
    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta, phi_theta_cen = locate_axis.find_gyroaxis(rec_dict, time_idx, bslopes[time_idx], bvars[time_idx],
                                                                TH=TH, Nrows=4, Ncols=8, makeplot=True)
    # return None
    
    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    StepI_bundle = sph2slep.get_StepI_dict(mu_phi, mu_theta, TH, rec_dict.ESA_PHI[time_idx,0,:,0],\
                   rec_dict.ESA_THETA[time_idx,0,0], instrument=instrument)

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps(rec_dict, StepI_bundle, time_idx, iterative_fit=iterative_fit,
                                                        Lmin=Lmin, Lmax=Lmax, rcond=rcond_polcap, makeplot=True, instrument=instrument)
    # return StepII_bundle
    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    VDF_2D_rec = VDF_rec_final.VDF_rec_cartesian(StepII_bundle, time_idx, N=Ncart, Vmin_shell=Vmin_shell,
                                                        rcond=rcond_cart, makeplot=makeplot)
    return VDF_2D_rec
    # saving the final reconstructed VDF for post-processing calculations
    VDF_rec_dict = {}
    VDF_rec_dict['VDF_2D_rec'] = VDF_2D_rec.VDF_Sleprec
    VDF_rec_dict['X'] = VDF_2D_rec.XX[0,:]
    VDF_rec_dict['Y'] = VDF_2D_rec.YY[:,0]
    VDF_rec_dict['phi0'] = VDF_2D_rec.phi0
    VDF_rec_dict['theta0'] = VDF_2D_rec.theta0
    write_pickle(VDF_rec_dict, f'./output_data_files/VDF_rec_pklfiles/VDF_2D_rec_{time_idx}')

def reconstruct_from_MMS_Slepians(time_idx):
    #=============STEP I: Plots the VDF on each Energy Shell (Does not really find gyrocenter)=======================#
    locate_axis.find_gyroaxis(rec_dict, time_idx, Nrows=4, Ncols=8)

    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    # StepI_bundle = sph2slep.get_StepI_Slepdict(rec_dict, TH)
    # return None

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps_Slepians(rec_dict, time_idx, rcond=rcond_polcap,
                                                                 makeplot=True)

    # return StepII_bundle
    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec = VDF_rec_final.get_3D_VDF(StepII_bundle)
    # sys.exit()
    return lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec, StepII_bundle


def reconstruct_from_SolO_Slepians(time_idx):
    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta = locate_axis.find_gyroaxis(rec_dict, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=True)

    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    # StepI_bundle = sph2slep.get_StepI_Slepdict(rec_dict, TH)
    # return None

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps_Slepians(mu_theta, mu_phi, rec_dict, time_idx, Lmax=Lmax, rcond=rcond_polcap,
                                                                 makeplot=True)

    return StepII_bundle

def reconstruct_from_MMS_SphericalHarmonics(time_idx):
    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta, phi_theta_cen = locate_axis.find_gyroaxis(DATA, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=True)

    #--------------------------------------------saving the theta and phi grid -------------------------------------#
    StepI_bundle = sph2slep.get_StepI_SHdict(DATA.PHI[0,:,0], DATA.THETA[0,0], instrument=instrument)

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps_SphericalHarmonics(DATA, StepI_bundle, time_idx, SH_basis, 
                                                                           Lmax=Lmax, rcond=rcond_polcap, makeplot=True,
                                                                           instrument=instrument)

    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec = VDF_rec_final.get_3D_VDF(StepII_bundle, NEmesh=100, spline_order=3)
    return lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec, StepII_bundle

    # saving the final reconstructed VDF for post-processing calculations
    VDF_rec_dict = {}
    VDF_rec_dict['VDF_2D_rec'] = VDF_2D_rec.VDF_Sleprec
    VDF_rec_dict['X'] = VDF_2D_rec.XX[0,:]
    VDF_rec_dict['Y'] = VDF_2D_rec.YY[:,0]
    VDF_rec_dict['phi0'] = VDF_2D_rec.phi0
    VDF_rec_dict['theta0'] = VDF_2D_rec.theta0
    write_pickle(VDF_rec_dict, f'./output_data_files/VDF_rec_pklfiles/VDF_2D_rec_{time_idx}')


def calc_moments_MMS(time_idx, mask_noisy=False):
    DATA_VDF = np.transpose(StepII_bundle.VDF[time_idx], [0, 2, 1]) * 1e12 * rec_dict.VDF_minval_true
    REC_VDF = np.power(10, StepII_bundle.fine_from_fine) * 1e12 * rec_dict.VDF_minval_true

    # removing the parts of the data and reconstructed VDF which have larger than NSR = 0.7
    if(mask_noisy):
        DATA_VDF_ERR = np.transpose(StepII_bundle.VDF_ERR[time_idx], [0, 2, 1]) * 1e12
        err_mask = DATA_VDF_ERR/DATA_VDF > 0.5
        DATA_VDF[err_mask] = 0.0

    velocity = 13.8 * np.sqrt(StepII_bundle.ENERGY[time_idx, :, 0, 0]) * 1000

    DATA_theta, DATA_phi = StepII_bundle.ESA_THETA[time_idx,0,0,:], StepII_bundle.ESA_PHI[time_idx,0,:,0]
    REC_theta, REC_phi = StepII_bundle.SLEP_THETA[:,0]+90, StepII_bundle.SLEP_PHI[0,:]

    # adjsuting the data phi to go from 0->360
    DATA_phi = DATA_phi - DATA_phi[0]

    # in SI units
    DATA_moments = calc_moments(DATA_VDF, velocity, np.radians(DATA_theta), np.radians(DATA_phi))
    REC_moments = calc_moments(REC_VDF, velocity, np.radians(REC_theta), np.radians(REC_phi))

    return DATA_moments, REC_moments

def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    instrument = 'MMS'             # currently we have 'PSP-SPAN', 'MMS' and 'SolO' (under construction)
    angular_basis = 'Slepians'     # 'SphericalHarmonics'
    makeplot = True                # whether we want to save the diagnostic plots
    TH = 85                        # the angular radius of the polar cap [in degrees]
    iterative_fit = False          # if we want the polar cap to be iteratively fitted from Lmin -> Lmax
    Lmin = 8                       # minimum angular degree for polar Slepian generation
    Lmax = 8                       # maximum angular degree for polar Slepian generation
    Ncart = 50                     # effective Shannon number of 2D Cartesian Slepian functions
    Vmin_shell = 250               # Minimum reliable energy shell [in km/s]
    rcond_polcap = 0.0             # Condition number for the inversion in polar caps
    rcond_cart = 1e-4              # Condition number for the inversion on a 2D plane
    ignore_last_anode = False      # if we want to set the last anode counts to nan
    N2D_restrict = False

    NEmesh, NPmesh, NTmesh = 100, 201, 101    # High resolution grid for final interpolation.
    Espline_order = 3                         # Spline order for final interpolation in energy.

    #----------------------READING THE SOURCE FILE----------------------------------#
    # filename = './input_data_files/2020-01-26_VDFs.cdf'
    filename = './input_data_files/MMS_2016-01-11_VDF_and_ERRs.cdf'
    # filename='input_data_files/SO_Test.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # calculating time in units of milliseconds
    times = func(data.unix_time.values)
    times = (times - times[0]) / timedelta(seconds=1)

    if(instrument=='PSP-SPAN'): 
        rec_dict = setup_rec_grid.PSP(data, TH)
        reconstruct_func = reconstruct_from_PSP
    elif(instrument=='PSP-SPAN-MT'): 
        reconstruct_func = reconstruct_from_PSP

    # full FOV instrument
    elif(instrument=='MMS'): 
        rec_dict = setup_rec_grid.MMS(data, TH, Lmax=Lmax, Nmesh=(NEmesh, NPmesh, NTmesh), Espline_order=Espline_order, makeplot=makeplot)
        if(angular_basis == 'Slepians'):
            reconstruct_func = reconstruct_from_MMS_Slepians
        elif(angular_basis == 'SphericalHarmonics'):
            reconstruct_func = reconstruct_from_MMS_SphericalHarmonics
    
    # full FOV instrument
    elif(instrument=='SolO'): 
        rec_dict = setup_rec_grid.SolO(data, TH, Lmax=Lmax, Nmesh=(NEmesh, NPmesh, NTmesh), Espline_order=Espline_order, makeplot=makeplot)
        if(angular_basis == 'Slepians'):
            reconstruct_func = reconstruct_from_SolO_Slepians
        elif(angular_basis == 'SphericalHarmonics'):
            reconstruct_func = reconstruct_from_SolO_SphericalHarmonics

    
    if(angular_basis == 'Slepians'):
        # saving these files as MATLAB readable arrays for generating Slepian functions
        mdict = {'phi0': 180, 'theta0': 90, 'cap_extent': rec_dict.TH, 'phi_grid': rec_dict.SLEP_PP.flatten(),
                'theta_grid': rec_dict.SLEP_TT.flatten(), 'Nphi': rec_dict.NPHI_SLEP, 'Ntheta': rec_dict.NTHETA_SLEP}
        savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{instrument}_HIGHRES.mat', mdict)

        # generating the Slepian basis functions
        misc_funcs.gen_SLEP(rec_dict, N2D_restrict=N2D_restrict)
    

    if(angular_basis == 'SphericalHarmonics'):
        SH_basis = misc_funcs.gen_SH(Lmax, NPHI, NTHETA)        

    # loading the B-slopes
    bslopes = np.load('./input_data_files/slopes.npy')
    bvars = np.load('./input_data_files/vars.npy')

    # importing scripts based on which instrument we are using
    sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_final, plot_VDF =\
                                        import_script.import_instrument_scripts(instrument)

    data_moments = {}
    rec_moments = {}

    # for time_idx in tqdm(range(len(times))):
    for time_idx in tqdm(range(533, 534)):
        #------------------USER SPECIFIED PARAMETERS------------------------------------#
        # time_idx = 0         # time index of VDF to be reconstructed
        time_HMS = func(data.unix_time.values)[time_idx].strftime('%Y-%m-%d %H:%M:%S')

        lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec, StepII_bundle = reconstruct_func(time_idx)
        # StepII_bundle = reconstruct_func(time_idx)

        # # calculating the moments for comparison
        # data_moments[time_idx], rec_moments[time_idx] = calc_moments_MMS(time_idx)

        # # plotting the uninterpolated VDF
        # plot_VDF.plot_VDF(StepII_bundle, time_idx)
        # continue

        # converting the grid to unstructured Cartesian
        VX, VY, VZ = misc_funcs.grid_pol2cart(lnE_mesh, theta_mesh, phi_mesh, savegrids=False)

        # plotting the 2D slice
        plt.style.use('dark_background')
        plt.figure()
        plt.pcolormesh(VX[:,NTmesh//2], VY[:,NTmesh//2], VDF_3D_rec[:,NTmesh//2], vmin=0, vmax=7, cmap='inferno', rasterized=True)
        plt.gca().set_aspect('equal')
        plt.title(f'Time = {time_HMS}')
        plt.colorbar()
        plt.savefig(f'./VDF_paper1_plots/2D_MMS/2D_MMS_{time_idx}.png')
        plt.close()

        plot_3D_VDF.plot_VDF(VX, VY, VZ, VDF_3D_rec, time_idx, time_HMS)
        sys.exit()