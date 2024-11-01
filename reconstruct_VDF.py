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
from scipy.stats import norm
from datetime import datetime, timedelta
func = np.vectorize(datetime.utcfromtimestamp)

from source_scripts import import_script, setup_rec_grid
from source_scripts import misc_functions as misc_funcs
import plot_3D_VDF
from calculations.calc_moments import calc_moments, spher_moments
from source_scripts import fit_2D_gaussian as fit_gauss

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

def reconstruct_from_MMS_Slepians(time_idx, angular_basis):
    #=============STEP I: Plots the VDF on each Energy Shell (Does not really find gyrocenter)=======================#
    locate_axis.find_gyroaxis(rec_dict, time_idx, Nrows=4, Ncols=8)

    if(angular_basis == 'Slepians'):
        # saving these files as MATLAB readable arrays for generating Slepian functions
        mdict = {'phi0': 180, 'theta0': 90, 'cap_extent': rec_dict.TH, 'phi_grid': rec_dict.SLEP_PP.flatten(),
                'theta_grid': rec_dict.SLEP_TT.flatten(), 'Nphi': rec_dict.NPHI_SLEP, 'Ntheta': rec_dict.NTHETA_SLEP}
        savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{instrument}_HIGHRES.mat', mdict)

        # generating the Slepian basis functions
        misc_funcs.gen_SLEP(rec_dict, N2D_restrict=N2D_restrict)

    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    # StepI_bundle = sph2slep.get_StepI_Slepdict(rec_dict, TH)
    # return None

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps_Slepians(rec_dict, time_idx, rcond=rcond_polcap)

    # return StepII_bundle
    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec = VDF_rec_final.get_3D_VDF(StepII_bundle)
    # sys.exit()
    return lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec, StepII_bundle


def reconstruct_from_SolO_Slepians(time_idx):
    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    locate_axis.find_gyroaxis(rec_dict, time_idx, Nrows=4, Ncols=8)

    sys.exit()

    if(angular_basis == 'Slepians'):
        # saving these files as MATLAB readable arrays for generating Slepian functions
        mdict = {'phi0': rec_dict.mu_phi[time_idx], 'theta0': rec_dict.mu_theta[time_idx], 'cap_extent': rec_dict.TH, 'phi_grid': rec_dict.SLEP_PP.flatten(),
                'theta_grid': rec_dict.SLEP_TT.flatten(), 'Nphi': rec_dict.NPHI_SLEP, 'Ntheta': rec_dict.NTHETA_SLEP}
        savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{instrument}_HIGHRES.mat', mdict)

        # generating the Slepian basis functions
        misc_funcs.gen_SLEP(rec_dict, N2D_restrict=N2D_restrict)
    
    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps_Slepians(rec_dict, time_idx, rcond=rcond_polcap,
                                                                 Nrows=4, Ncols=8)

    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec = VDF_rec_final.get_3D_VDF(StepII_bundle)

    return lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec, StepII_bundle

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
    instrument = 'SolO'             # currently we have 'PSP-SPAN', 'MMS' and 'SolO' (under construction)
    angular_basis = 'Slepians'      # 'SphericalHarmonics'
    makeplot = False                # whether we want to save the diagnostic plots
    TH = 45                         # the angular radius of the polar cap [in degrees]
    iterative_fit = False           # if we want the polar cap to be iteratively fitted from Lmin -> Lmax
    Lmin = 8                        # minimum angular degree for polar Slepian generation
    Lmax = None                     # maximum angular degree for polar Slepian generation
    Ncart = 50                      # effective Shannon number of 2D Cartesian Slepian functions
    Vmin_shell = 250                # Minimum reliable energy shell [in km/s]
    rcond_polcap = 0.0              # Condition number for the inversion in polar caps
    rcond_cart = 1e-4               # Condition number for the inversion on a 2D plane
    ignore_last_anode = False       # if we want to set the last anode counts to nan
    N2D_restrict = True            # if we want to truncate the basis functions to Shannon number
    datascan_mode = True          # if we want to scan over the time interval to find the centroid

    NEmesh, NPmesh, NTmesh = 200, 201, 101    # High resolution grid for final interpolation.
    Espline_order = 3                         # Spline order for final interpolation in energy.

    #----------------------READING THE SOURCE FILE----------------------------------#
    # filename = './input_data_files/2020-01-26_VDFs.cdf'
    # filename = './input_data_files/MMS_2016-01-11_VDF_and_ERRs.cdf'
    filename='input_data_files/SO_2020-07-16_VDF.cdf'       # Change the naming convention so that is it SolO...
    # filename='input_data_files/SO_Test.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # calculating time in units of milliseconds
    times_datetime = func(data.unix_time.values)
    times = (times_datetime - times_datetime[0]) / timedelta(seconds=1)

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

    if(datascan_mode):
        for time_idx in tqdm(range(len(times))):
            locate_axis.find_gyroaxis(rec_dict, time_idx)

        plt.style.use('default')
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(7,4))
        # plotting a dummy background to show the ESA grid
        dummy_ESA = np.ones_like(rec_dict.VDF[101,56,:,:])
        plt.pcolormesh(rec_dict.ESA_PHI[101,0], 90 - rec_dict.ESA_THETA[101,0], dummy_ESA, cmap='binary', vmin=0, vmax=1, alpha=0.5)

        # overplotting the raw data
        plt.pcolormesh(rec_dict.ESA_PHI[101,0], 90 - rec_dict.ESA_THETA[101,0], np.log10(rec_dict.VDF[101,56,:,:]), cmap='inferno')
        plt.colorbar(shrink=0.95, aspect=30, fraction=0.05, pad=0.01)
        plt.gca().set_aspect('equal')
        plt.xlim([95, 265])
        plt.ylim([-50,50])
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\theta$')

        # fitting a 2D Gaussian
        theta_clockangle = np.linspace(0, 2 * np.pi, 100)
        xcirc, ycirc = rec_dict.mu_sigma[time_idx] * np.cos(theta_clockangle) + rec_dict.mu_phi[time_idx],\
                       rec_dict.mu_sigma[time_idx] * np.sin(theta_clockangle) + rec_dict.mu_theta[time_idx]

        # plotting the contours of the 2D Gaussian
        plt.plot(xcirc, ycirc, '--k', label=r'$4\sigma$ polar cap')
        plt.plot(rec_dict.mu_phi[time_idx], rec_dict.mu_theta[time_idx], 'xk',
                 label=r'$(\phi_0,\theta_0)=(%.2f,%.2f)$'%(rec_dict.mu_phi[time_idx], rec_dict.mu_theta[time_idx]))
        plt.legend(loc=2)
        plt.subplots_adjust(top=0.98, bottom=0.13, left=0.13, right=0.95)
        plt.savefig('VDF_paper1_plots/final_plots/capfit1.pdf')

        # plotting the histograms demo for fitting the mu_phi and mu_theta
        locate_axis.find_gyroaxis(rec_dict, 101)
        # plt.style.use('default')
        fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)

        (mu, sigma) = norm.fit(rec_dict.Eshell_info[:,2])
        n, bins, patches = ax[0].hist(rec_dict.Eshell_info[:,2], 10, density=True, alpha=0.75, color='skyblue', edgecolor='black')
        y = norm.pdf(bins, mu, sigma)
        ax[0].plot(bins, y, 'r--', linewidth=2)
        ax[0].axvline(mu, ls='dotted', color='black', label=r'$\phi_{\rm{cen}}$ = %.2f'%mu)
        ax[0].legend(loc=2)

        (mu, sigma) = norm.fit(rec_dict.Eshell_info[:,3])
        n, bins, patches = ax[1].hist(rec_dict.Eshell_info[:,3], 10, density=True, alpha=0.75, color='skyblue', edgecolor='black')
        y = norm.pdf(bins, mu, sigma)
        ax[1].plot(bins, y, 'r--', linewidth=2)
        ax[1].axvline(mu, ls='dotted', color='black', label=r'$\theta_{\rm{cen}}$ = %.2f'%mu)
        ax[1].legend(loc=2)

        ax[0].set_ylabel('Normalized histogram')
        ax[0].set_xlabel(r'$\phi_0$')
        ax[1].set_xlabel(r'$\theta_0$')

        plt.subplots_adjust(wspace=0.1, left=0.09, right=0.98, top=0.98, bottom=0.15)
        plt.savefig('VDF_paper1_plots/final_plots/capfit2.pdf')

        # plotting the centroid location as a function of time
        # plt.style.use('default')
        fig, ax = plt.subplots(2, 1, figsize=(10,4), sharex=True)

        ax[0].fill_between(times_datetime, rec_dict.mu_phi-rec_dict.mu_sigma, rec_dict.mu_phi+rec_dict.mu_sigma, alpha=0.25, color='k')
        ax[1].fill_between(times_datetime, rec_dict.mu_theta-rec_dict.mu_sigma, rec_dict.mu_theta+rec_dict.mu_sigma, alpha=0.25, color='k')
        ax[0].plot(times_datetime, rec_dict.mu_phi, 'k')
        ax[1].plot(times_datetime, rec_dict.mu_theta, 'k')

        # setting the limits in the yaxis
        phimin_ESA, phimax_ESA = rec_dict.ESA_PHI.max(), rec_dict.ESA_PHI.min()
        thetamin_ESA, thetamax_ESA = rec_dict.ESA_THETA.max(), rec_dict.ESA_THETA.min()
        phimin, phimax = 90, 270
        thetamin, thetamax = 0, 180

        ax[0].set_ylim([phimin, phimax])
        ax[1].set_ylim([thetamin-90, thetamax-90])
        ax[0].set_xlim([times_datetime[0], times_datetime[-1]])
        ax[1].set_xlim([times_datetime[0], times_datetime[-1]])

        ax[0].axhline(phimin_ESA, color='r')
        ax[0].axhline(phimax_ESA, color='r')
        ax[1].axhline(thetamin_ESA-90, color='r')
        ax[1].axhline(thetamax_ESA-90, color='r')
        ax[0].axhline(rec_dict.mu_phi.mean()-45, color='k', ls='--')
        ax[0].axhline(rec_dict.mu_phi.mean()+45, color='k', ls='--')
        ax[1].axhline(rec_dict.mu_theta.mean()-45, color='k', ls='--')
        ax[1].axhline(rec_dict.mu_theta.mean()+45, color='k', ls='--')

        ax[0].set_ylabel(r'$\phi_{\rm{cen}}$', rotation=0, labelpad=20)
        ax[1].set_ylabel(r'$\theta_{\rm{cen}}$', rotation=0, labelpad=20)
        ax[1].set_xlabel('Time (UTC)')

        plt.subplots_adjust(bottom=0.15, right=0.98, top=0.98)
        plt.savefig('VDF_paper1_plots/final_plots/capfit3.pdf')

    else:
        # for time_idx in tqdm(range(len(times))):
        for time_idx in tqdm(range(479, 480)):
            time_HMS = func(data.unix_time.values)[time_idx].strftime('%Y-%m-%d %H:%M:%S')

            lnE_mesh, theta_mesh, phi_mesh, VDF_3D_rec, StepII_bundle = reconstruct_func(time_idx)
            continue
            # StepII_bundle = reconstruct_func(time_idx)

            # # calculating the moments for comparison
            # data_moments[time_idx], rec_moments[time_idx] = calc_moments_MMS(time_idx)

            # # plotting the uninterpolated VDF
            # plot_VDF.plot_VDF(StepII_bundle, time_idx)
            # continue

            # converting the grid to unstructured Cartesian
            VX, VY, VZ = misc_funcs.grid_pol2cart(lnE_mesh, theta_mesh, phi_mesh, savegrids=False)

            VDF_3D_bundle = {}
            VDF_3D_bundle['VDF_3D_rec'] = VDF_3D_rec
            VDF_3D_bundle['VX'] = VX
            VDF_3D_bundle['VY'] = VY
            VDF_3D_bundle['VZ'] = VZ

            StepII_bundle_plotdict = {}
            StepII_bundle_plotdict['ENERGY'] = StepII_bundle.ENERGY
            StepII_bundle_plotdict['ESA_THETA'] = StepII_bundle.ESA_THETA
            StepII_bundle_plotdict['ESA_PHI'] = StepII_bundle.ESA_PHI
            StepII_bundle_plotdict['VDF'] = StepII_bundle.VDF
            StepII_bundle_plotdict['SLEP_THETA'] = StepII_bundle.SLEP_THETA
            StepII_bundle_plotdict['SLEP_PHI'] = StepII_bundle.SLEP_PHI
            StepII_bundle_plotdict['fine_from_fine'] = StepII_bundle.fine_from_fine
            StepII_bundle_plotdict['G'] = StepII_bundle.G
            StepII_bundle_plotdict['V'] = StepII_bundle.V
            StepII_bundle_plotdict['Slep_coeffs'] = StepII_bundle.SLEP_coeffs
            write_pickle(VDF_3D_bundle, 'VDF3Dbundle_533_MMSplot')
            write_pickle(StepII_bundle_plotdict, 'StepIIbundle_533_MMSplot')

            # plotting the 2D slice
            plt.style.use('dark_background')
            plt.figure()
            plt.pcolormesh(VX[:,NTmesh//2], VY[:,NTmesh//2], VDF_3D_rec[:,NTmesh//2], vmin=0, vmax=7, cmap='inferno', rasterized=True)
            plt.gca().set_aspect('equal')
            plt.title(f'Time = {time_HMS}')
            plt.colorbar()
            plt.savefig(f'./VDF_paper1_plots/2D_{instrument}/2D_{instrument}_{time_idx}.png')
            plt.close()

            plot_3D_VDF.plot_VDF(VX, VY, VZ, VDF_3D_rec, time_idx, time_HMS, instrument)
            # sys.exit()


# cmaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

# for cmap in cmaps:
#     plt.figure()
#     plt.pcolormesh(VX[:,50], VY[:,50], VDF_3D_rec[:,50], vmin=0, vmax=7, cmap=cmap, rasterized=True)
#     plt.gca().set_aspect('equal')
#     plt.colorbar()
#     plt.savefig(f'test_cmaps/{cmap}.png')
#     plt.close()