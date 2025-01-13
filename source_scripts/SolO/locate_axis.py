import numpy as np
from scipy.ndimage import center_of_mass as com
from scipy.stats import norm   
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')

from source_scripts import fit_2D_gaussian as fit_gauss
from source_scripts import misc_functions as misc_fn

def find_gyroaxis(rec_dict, time_idx, Nrows=4, Ncols=8):
    if(rec_dict.makeplot): Eshell_info, fig, ax = with_plot(rec_dict, time_idx, Nrows, Ncols)
    else: Eshell_info = without_plot(rec_dict, time_idx, Nrows, Ncols)

    Eshell_info = np.asarray(Eshell_info)

    Eshell_info[:,2] = Eshell_info[:,2] / (rec_dict.NPHI_ESA) * (rec_dict.ESA_PHI[0,0,-1,0]-rec_dict.ESA_PHI[0,0,0,0])\
                       + rec_dict.ESA_PHI[0,0,0,0]
    Eshell_info[:,3] = 90 - (Eshell_info[:,3] / (rec_dict.NTHETA_ESA) * (rec_dict.ESA_THETA[0,0,0,-1]-rec_dict.ESA_THETA[0,0,0,0])\
                             + rec_dict.ESA_THETA[0,0,0,0])

    # finding the effective centroid across shells
    (mu_phi, sig_phi) = norm.fit(misc_fn.reject_outliers(Eshell_info[:,2]))
    (mu_theta, sig_theta) = norm.fit(misc_fn.reject_outliers(Eshell_info[:,3]))
    mu_sigma = np.max(misc_fn.reject_outliers(Eshell_info[:,4]))

    # finding the centroids in phi and theta
    mu_phi_calibrated = mu_phi
    mu_theta_calibrated = mu_theta
    mu_sigma_calibrated = mu_sigma / (rec_dict.NPHI_ESA) * (rec_dict.ESA_PHI[0,0,-1,0]-rec_dict.ESA_PHI[0,0,0,0])
    rec_dict.mu_phi[time_idx] = mu_phi_calibrated
    rec_dict.mu_theta[time_idx] = mu_theta_calibrated
    rec_dict.mu_sigma[time_idx] = 4 * mu_sigma_calibrated

    rec_dict.Eshell_info = Eshell_info

    if(rec_dict.makeplot):
        # making the polar cap extent in degrees
        clock_angle = np.linspace(0, 2 * np.pi, 100)
        x_cap, y_cap = rec_dict.mu_phi[time_idx] + rec_dict.TH * np.cos(clock_angle),\
                       rec_dict.mu_theta[time_idx] + rec_dict.TH * np.sin(clock_angle)

        x_cap_sig, y_cap_sig = rec_dict.mu_phi[time_idx] + rec_dict.mu_sigma[time_idx] * np.cos(clock_angle),\
                               rec_dict.mu_theta[time_idx] + rec_dict.mu_sigma[time_idx] * np.sin(clock_angle)

        # plotting the effective centroid in all shells
        for axs in ax.flatten():
            axs.plot(x_cap, y_cap, '--w')
            axs.plot(x_cap_sig, y_cap_sig, '--r')
            axs.set_xlim([100, 300])
            axs.set_ylim([-50, 50])
            # axs.set_xlim([0, 360])
            # axs.set_ylim([-90, 90])
            axs.set_aspect('equal')

        plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
        # to put common x and y labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
        plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)
        plt.suptitle(f'{time_idx}')
        plt.savefig(f'VDF_paper1_plots/VDF_{rec_dict.instrument}_polar_plot/{time_idx}.png')
        plt.close()

def with_plot(rec_dict, time_idx, Nrows, Ncols):
    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # making list to store the gyroaxis locations for each shell
    Eshell_info = []

    # finding the start and stop indices in energy according to the highest intensity 32 shells
    logvv = np.log10(rec_dict.VDF[time_idx, :, :, :])
    logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)
    E_intensities = np.nansum(logvv, axis=(1,2))
    E_maxintensity_idx = np.argmax(E_intensities)
    E_minidx, E_maxidx = E_maxintensity_idx - 16, E_maxintensity_idx + 16

    if(E_minidx < 0):
        E_minidx, E_maxidx = 0, 32
    if(E_maxidx > rec_dict.NENERGY - 1): 
        E_minidx, E_maxidx = rec_dict.NENERGY - 33, rec_dict.NENERGY - 1

    rec_dict.E_minidx, rec_dict.E_maxidx = E_minidx, E_maxidx

    # for i, E_idx in enumerate(np.arange(rec_dict.E_minidx, rec_dict.E_maxidx)):
    for i, E_idx in enumerate(np.arange(40, 40+32)):
        # finding the row and the column of the subplots
        row, col = i//Ncols, i%Ncols

        E = rec_dict.ENERGY[time_idx, E_idx, 0, 0]

        tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
                               rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
                               rec_dict.VDF[time_idx, E_idx, :, :]

        # skipping Gaussian fitting if there is less than or equal to 3 counts in the E-shell
        if((np.sum(np.isfinite(rec_dict.VDF[time_idx, E_idx]))) <= 3): continue

        # the number of bins which are non-zero in an energy shell
        Intensity = np.sum(vv[~np.isnan(vv)])

        # Calculating the log and setting +\- inf to nan
        logvv = np.log10(vv)
        logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)

        # finding the centroid (in linear scale)
        # xcen, ycen = com(np.nan_to_num(vv))
        H, xcen, ycen, sigxy = fit_gauss.fitgaussian(logvv)

        # bad 2D gaussian fits are discarded 
        if(xcen > rec_dict.NPHI_ESA or ycen > rec_dict.NTHETA_ESA): continue

        #------------------MAKING DIAGNOSTIC PLOTS IF REQUIRED-----------------------#
        vmin, vmax = rec_dict.vmin_t[time_idx], rec_dict.vmax_t[time_idx]
        vmin, vmax = 0, 3
        plot_diagnostic_panels(ax[row,col], E, pp_orig, tt_orig, logvv, vmin, vmax)

        # appending the located centers
        Eshell_info.append([E, Intensity, xcen, ycen, sigxy])

    return Eshell_info, fig, ax
    
def without_plot(rec_dict, time_idx, Nrows, Ncols):
    # making list to store the gyroaxis locations for each shell
    Eshell_info = []

    # finding the start and stop indices in energy according to the highest intensity 32 shells
    logvv = np.log10(rec_dict.VDF[time_idx, :, :, :])
    logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)
    E_intensities = np.nansum(logvv, axis=(1,2))
    E_maxintensity_idx = np.argmax(E_intensities)
    E_minidx, E_maxidx = E_maxintensity_idx - 16, E_maxintensity_idx + 16

    if(E_minidx < 0):
        E_minidx, E_maxidx = 0, 32
    if(E_maxidx > rec_dict.NENERGY - 1): 
        E_minidx, E_maxidx = rec_dict.NENERGY - 33, rec_dict.NENERGY - 1

    rec_dict.E_minidx, rec_dict.E_maxidx = E_minidx, E_maxidx

    for i, E_idx in enumerate(np.arange(rec_dict.E_minidx, rec_dict.E_maxidx)):
        E = rec_dict.ENERGY[time_idx, E_idx, 0, 0] 
        tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
                               rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
                               rec_dict.VDF[time_idx, E_idx, :, :]

        # skipping Gaussian fitting if there is less than or equal to 3 counts in the E-shell
        if((np.sum(np.isfinite(rec_dict.VDF[time_idx, E_idx]))) <= 3): continue

        # the number of bins which are non-zero in an energy shell
        Intensity = np.sum(vv[~np.isnan(vv)])

        # Calculating the log and setting +\- inf to nan
        logvv = np.log10(vv)
        logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)

        # finding the centroid (in linear scale)
        # xcen, ycen = com(np.nan_to_num(vv))

        # if there are some non-zero values in the shell
        if(np.nansum(np.log10(rec_dict.VDF[time_idx, E_idx])) > 0):
            H, xcen, ycen, sigxy = fit_gauss.fitgaussian(logvv)
        else:
            continue

        # bad 2D gaussian fits are discarded 
        if(xcen > rec_dict.NPHI_ESA or ycen > rec_dict.NTHETA_ESA): continue

        # appending the located centers
        Eshell_info.append([E, Intensity, xcen, ycen, sigxy])

    return Eshell_info

def plot_diagnostic_panels(ax, E, pp_orig, tt_orig, logvv, vmin, vmax):
    im = ax.pcolormesh(pp_orig, 90 - tt_orig, logvv, cmap='inferno', rasterized=True, vmin=vmin, vmax=vmax)
    ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
            va='bottom', ha='left', color='white', fontweight='bold')