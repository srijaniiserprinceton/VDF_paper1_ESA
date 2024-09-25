import numpy as np
from scipy.ndimage import center_of_mass as com
from scipy.stats import norm   
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')

# imports from our custom package
from . import fit_2D_gaussian as fit_gauss

def find_gyroaxis(rec_dict, time_idx, TH=45, Nrows=4, Ncols=8, makeplot=True, all_shell_info=True):
    if(makeplot): Eshell_info, fig, ax = with_plot(rec_dict, time_idx, Nrows, Ncols)
    else: Eshell_info = without_plot(rec_dict, Nrows, Ncols)

    Eshell_info = np.asarray(Eshell_info)

    # finding the effective centroid across shells
    (mu_phi, sig_phi) = norm.fit(Eshell_info[:,2])
    (mu_theta, sig_theta) = norm.fit(Eshell_info[:,3])

    if(makeplot):
        # making the polar cap extent in degrees
        clock_angle = np.linspace(0, 2 * np.pi, 100)
        x_cap, y_cap = mu_phi + TH * np.cos(clock_angle), mu_theta + TH * np.sin(clock_angle)

        # plotting the effective centroid in all shells
        for axs in ax.flatten():
            axs.plot(x_cap, y_cap, '--r')
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

    return mu_phi, mu_theta

def with_plot(rec_dict, time_idx, Nrows, Ncols):
    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # making list to store the gyroaxis locations for each shell
    Eshell_info = []

    # for i, E_idx in enumerate(np.arange(2, rec_dict.NENERGY, rec_dict.NENERGY // (Nrows * Ncols))):
    for i, E_idx in enumerate(np.arange(28, 28 + (Nrows * Ncols))):
        # finding the row and the column of the subplots
        row, col = i//Ncols, i%Ncols

        E = rec_dict.ENERGY[time_idx, E_idx, :, :][0, 0]

        tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
                               rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
                               rec_dict.VDF[time_idx, E_idx, :, :]

        # skipping plotting if there is zero counts in the E-shell
        if(np.sum(~np.isnan(vv)) < 1): continue

        # finding the centroid (in linear scale)
        xcen, ycen = com(np.nan_to_num(vv))

        # the number of bins which are non-zero in an energy shell
        Intensity = np.sum(vv[~np.isnan(vv)])

        # For plotting: Calculating the log and setting +\- inf to nan
        logvv = np.log10(vv)
        logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)

        #------------------MAKING DIAGNOSTIC PLOTS IF REQUIRED-----------------------#
        plot_diagnostic_panels(ax[row,col], E, pp_orig, tt_orig, logvv)

        # appending the located centers
        Eshell_info.append([E, Intensity, xcen, ycen])

    return Eshell_info, fig, ax
    
def without_plot(rec_dict, Nrows, Ncols):
    # making list to store the gyroaxis locations for each shell
    Eshell_info = []

    for i, E_idx in enumerate(np.arange(2, rec_dict.NENERGY, rec_dict.NENERGY // (Nrows * Ncols))):
        E = rec_dict.ENERGY[E_idx, :, :][0, 0] 
        tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
                               rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
                               rec_dict.VDF[time_idx, E_idx, :, :]

        # skipping plotting if there is zero counts in the E-shell
        if(np.sum(~np.isnan(vv)) < 1): continue

        # finding the centroid (in linear scale)
        xcen, ycen = com(vv)

        # the number of bins which are non-zero in an energy shell
        Intensity = np.sum(vv[~np.isnan(vv)])

        # appending the located centers
        Eshell_info.append([E, Intensity, xcen, ycen])

    return Eshell_info

def plot_diagnostic_panels(ax, E, pp_orig, tt_orig, logvv):
    vmin, vmax = 1, 7
    im = ax.pcolormesh(pp_orig, 90 - tt_orig, logvv, cmap='inferno', rasterized=True, vmin=vmin, vmax=vmax)
    ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
            va='bottom', ha='left', color='white', fontweight='bold')