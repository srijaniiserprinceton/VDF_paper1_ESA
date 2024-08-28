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

def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_gyroframe_diag(mu_phi, phi_theta_cen):
    # making marker size dependent on the number of counts
    s = 20/(phi_theta_cen[:,1] / phi_theta_cen[:,1].max())**2
    plt.figure()
    plt.plot(phi_theta_cen[:,0], phi_theta_cen[:,2], '.-k')
    plt.scatter(phi_theta_cen[:,0], phi_theta_cen[:,2], color='k', s=s)
    plt.axhline(180, color='k', label='last anode')
    plt.axhline(mu_phi, color='r', label=r'$\phi_0$')
    plt.axhline(mu_phi - 11.25, color='r', ls='dashed', label=r'$\phi_0 - 11.25$')
    plt.axhline(mu_phi + 11.25, color='r', ls='dashed', label=r'$\phi_0 + 11.25$')
    plt.ylim([90,250])
    plt.xlim([0, 3500])
    plt.legend()
    plt.xlabel(r'Energy shell [eV]')
    plt.ylabel(r'$\phi_{shell}$')
    plt.tight_layout()
    plt.savefig(f'./VDF_paper1_plots/plot_gyroframe_diag/check_center_{time_idx}.png')


if __name__=='__main__':
    #----------------------READING THE SOURCE FILE----------------------------------#
    # filename = './input_data_files/2020-01-26_VDFs.cdf'
    filename = './input_data_files/MMS_2016-01-11_VDFs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    Ntimes = data.energy.data.shape[0]

    instrument = 'MMS'        # currently we have 'PSP' and 'MMS' (under construction)
    makeplot = True           # whether we want to save the diagnostic plots
    TH = 45                   # the angular radius of the polar cap [in degrees]
    iterative_fit = False      # if we want the polar cap to be iteratively fitted from Lmin -> Lmax
    Lmin = 8                  # minimum angular degree for polar Slepian generation
    Lmax = 5                 # maximum angular degree for polar Slepian generation
    Ncart = 50                # effective Shannon number of 2D Cartesian Slepian functions
    Vmin_shell = 250          # Minimum reliable energy shell [in km/s]
    rcond_polcap = 0.0        # Condition number for the inversion in polar caps
    rcond_cart = 1e-4         # Condition number for the inversion on a 2D plane
    ignore_last_anode = False # if we want to set the last anode counts to nan

    # importing scripts based on which instrument we are using
    sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian = import_script.import_instrument_scripts(instrument)
    sys.exit()

    for time_idx in tqdm(range(Ntimes)):
        #------------------USER SPECIFIED PARAMETERS------------------------------------#
        time_idx = 1000         # time index of VDF to be reconstructed

        # extracting the required timestamp
        DATA = extract_data.extract_VDF_data(data, time_idx, instrument=instrument)
        # we want to scale VDF such that the lowest non-zero entry is 1.0
        DATA.VDF[DATA.VDF == 0] = np.nan
        DATA.VDF = DATA.VDF / np.nanmin(DATA.VDF)

        # putting nan in the last anode if requested
        if(ignore_last_anode):
            DATA.VDF[:,-1,:] = np.nan

        # try:
        #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
        mu_phi, mu_theta, phi_theta_cen = locate_axis.find_gyroaxis(DATA, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=True)
        if(instrument=='SPAN'): plot_gyroframe_diag(mu_phi, phi_theta_cen)
        
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
        # except: continue