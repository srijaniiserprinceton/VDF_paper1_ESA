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

from source_scripts import import_script

def reconstruct_from_PSP():
    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta, phi_theta_cen = locate_axis.find_gyroaxis(DATA, time_idx, bslopes[time_idx], vars[time_idx],
                                                                TH=TH, Nrows=4, Ncols=8, makeplot=True)
    return None
    
    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    StepI_bundle = sph2slep.get_StepI_dict(mu_phi, mu_theta, TH, DATA.PHI[0,:,0], DATA.THETA[0,0], instrument=instrument)

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps(DATA, StepI_bundle, time_idx, iterative_fit=iterative_fit,
                                                        Lmin=Lmin, Lmax=Lmax, rcond=rcond_polcap, makeplot=True, instrument=instrument)
    sys.exit()
    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    VDF_2D_rec = VDF_rec_cartesian.VDF_rec_cartesian(StepII_bundle, time_idx, N=Ncart, Vmin_shell=Vmin_shell,
                                                        rcond=rcond_cart, makeplot=makeplot)
    # saving the final reconstructed VDF for post-processing calculations
    VDF_rec_dict = {}
    VDF_rec_dict['VDF_2D_rec'] = VDF_2D_rec.VDF_Sleprec
    VDF_rec_dict['X'] = VDF_2D_rec.XX[0,:]
    VDF_rec_dict['Y'] = VDF_2D_rec.YY[:,0]
    VDF_rec_dict['phi0'] = VDF_2D_rec.phi0
    VDF_rec_dict['theta0'] = VDF_2D_rec.theta0
    write_pickle(VDF_rec_dict, f'./output_data_files/VDF_rec_pklfiles/VDF_2D_rec_{time_idx}')

def reconstruct_from_MMS():
    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta, phi_theta_cen = locate_axis.find_gyroaxis(DATA, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=True)
    # sys.exit()
    
    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    StepI_bundle = sph2slep.get_StepI_dict(mu_phi, mu_theta, TH, DATA.PHI[0,:,0], DATA.THETA[0,0], instrument=instrument)

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps(DATA, StepI_bundle, time_idx, iterative_fit=iterative_fit,
                                                        Lmin=Lmin, Lmax=Lmax, rcond=rcond_polcap, makeplot=True, instrument=instrument)
    sys.exit()
    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    VDF_2D_rec = VDF_rec_cartesian.VDF_rec_cartesian(StepII_bundle, time_idx, N=Ncart, Vmin_shell=Vmin_shell,
                                                        rcond=rcond_cart, makeplot=makeplot)
    # saving the final reconstructed VDF for post-processing calculations
    VDF_rec_dict = {}
    VDF_rec_dict['VDF_2D_rec'] = VDF_2D_rec.VDF_Sleprec
    VDF_rec_dict['X'] = VDF_2D_rec.XX[0,:]
    VDF_rec_dict['Y'] = VDF_2D_rec.YY[:,0]
    VDF_rec_dict['phi0'] = VDF_2D_rec.phi0
    VDF_rec_dict['theta0'] = VDF_2D_rec.theta0
    write_pickle(VDF_rec_dict, f'./output_data_files/VDF_rec_pklfiles/VDF_2D_rec_{time_idx}')


def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    instrument = 'MMS'        # currently we have 'PSP-SPAN' and 'MMS' (under construction)
    makeplot = True                # whether we want to save the diagnostic plots
    TH = 85                        # the angular radius of the polar cap [in degrees]
    iterative_fit = False          # if we want the polar cap to be iteratively fitted from Lmin -> Lmax
    Lmin = 8                       # minimum angular degree for polar Slepian generation
    Lmax = 12                       # maximum angular degree for polar Slepian generation
    Ncart = 50                     # effective Shannon number of 2D Cartesian Slepian functions
    Vmin_shell = 250               # Minimum reliable energy shell [in km/s]
    rcond_polcap = 0.0             # Condition number for the inversion in polar caps
    rcond_cart = 1e-4              # Condition number for the inversion on a 2D plane
    ignore_last_anode = False      # if we want to set the last anode counts to nan

    #----------------------READING THE SOURCE FILE----------------------------------#
    if(instrument=='PSP-SPAN'): 
        filename = './input_data_files/2020-01-26_VDFs.cdf'
        reconstruct_func = reconstruct_from_PSP
    elif(instrument=='PSP-SPAN-MT'): 
        filename = './input_data_files/2020-01-26_VDFs.cdf'
        reconstruct_func = reconstruct_from_PSP
    elif(instrument=='MMS'): 
        filename = './input_data_files/MMS_2016-01-11_VDFs.cdf'
        reconstruct_func = reconstruct_from_MMS
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # loading the B-slopes
    bslopes = np.load('./input_data_files/slopes.npy')
    vars = np.load('./input_data_files/vars.npy')

    # importing scripts based on which instrument we are using
    sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian = import_script.import_instrument_scripts(instrument)

    Ntimes = data.energy.data.shape[0]
    for time_idx in tqdm(range(0,25)):
        #------------------USER SPECIFIED PARAMETERS------------------------------------#
        # time_idx = 0         # time index of VDF to be reconstructed

        # extracting the required timestamp
        DATA = extract_data.extract_VDF_data(data, time_idx, instrument=instrument)
        # we want to scale VDF such that the lowest non-zero entry is 1.0
        DATA.VDF[DATA.VDF == 0] = np.nan
        DATA.minval_true = np.nanmin(DATA.VDF)
        DATA.VDF = DATA.VDF / DATA.minval_true

        # putting nan in the last anode if requested
        if(ignore_last_anode):
            DATA.VDF[:,-1,:] = np.nan

        # try:
        reconstruct_func()
        sys.exit()
        # except: continue