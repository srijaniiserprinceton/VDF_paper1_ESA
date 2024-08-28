import numpy as np
from scipy import interpolate
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.ion()    

# imports from our custom package
from . import fit_2D_gaussian as fit_gauss

def find_gyroaxis(DATA, time_idx, TH=45, Nrows=4, Ncols=8, makeplot=True, all_shell_info=True):
    if(makeplot): phi_theta_cen, fig, ax = with_plot(DATA, time_idx, Nrows, Ncols)
    else: phi_theta_cen = without_plot(DATA, Nrows, Ncols)

    phi_theta_cen = np.asarray(phi_theta_cen)

    # removing the shells with less than 0.5 counts of the max
    weight_mask = phi_theta_cen[:,1] / np.max(phi_theta_cen[:,1]) >= 0.7

    phi_theta_cen_purged = []
    for i in range(len(phi_theta_cen)):
        if(weight_mask[i] == False): continue
        phi_theta_cen_purged.append(phi_theta_cen[i])
    phi_theta_cen_purged = np.asarray(phi_theta_cen_purged)

    # finding the effective centroid across shells
    (mu_phi, sig_phi) = norm.fit(phi_theta_cen_purged[:,2])
    (mu_theta, sig_theta) = norm.fit(phi_theta_cen_purged[:,3])

    if(makeplot):
        # making the polar cap extent in degrees
        clock_angle = np.linspace(0, 2 * np.pi, 100)
        x_cap, y_cap = mu_phi + TH * np.cos(clock_angle), mu_theta + TH * np.sin(clock_angle)

        # plotting the effective centroid in all shells
        for axs in ax.flatten():
            axs.scatter(mu_phi, mu_theta, marker='o', color='white')
            axs.plot(x_cap, y_cap, '--r')
            axs.set_xlim([DATA.PHI.min(),DATA.PHI.max()])
            axs.set_ylim([DATA.THETA.min(),DATA.THETA.max()])
            axs.set_aspect('equal')

        plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
        # to put common x and y labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
        plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)
        plt.suptitle(f'{time_idx}')
        plt.savefig(f'VDF_paper1_plots/VDF_{DATA.instrument}_polar_plot/{time_idx}.png')
        plt.close()

        # making a plot of the histograms in theta and phi with weights built from the count of each shell
        hist_weights = (phi_theta_cen_purged[:,1] / np.max(phi_theta_cen_purged[:,1]))**2

        # plotting the histograms
        fig, ax = plt.subplots(1,3,figsize=(15,8))

        ax[0].semilogx(phi_theta_cen[:,0], phi_theta_cen[:,1], 'ok')
        ax[0].set_ylabel('Count fraction')
        ax[0].set_xlabel('Energy [eV]')

        __, bins_phi, __ = ax[1].hist(phi_theta_cen_purged[:,2], bins=7, weights=hist_weights, density=True)
        __, bins_theta, __ = ax[2].hist(phi_theta_cen_purged[:,3], bins=7, weights=hist_weights, density=True)

        y_phi = norm.pdf(bins_phi, mu_phi, sig_phi)
        y_theta = norm.pdf(bins_theta, mu_theta, sig_theta)
        ax[1].plot(bins_phi, y_phi, '--', linewidth=2)
        ax[2].plot(bins_theta, y_theta, '--', linewidth=2)
        ax[1].set_title(r'Normalized histogram of $\phi$ gyrocenter', fontsize=14)
        ax[1].set_xlabel(r'$\phi$ in degrees')
        ax[2].set_title(r'Normalized histogram of $\theta$ gyrocenter', fontsize=14)
        ax[2].set_xlabel(r'$\theta$ in degrees')
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.96, wspace=0.3, hspace=0.3)
        plt.close()

    if(all_shell_info):
        return mu_phi, mu_theta, phi_theta_cen
    else: mu_phi, mu_theta

def with_plot(DATA, time_idx, Nrows, Ncols):
    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # levels chosen just to plot the 2D Gaussian over the VDF
    levels = np.linspace(0, 7, 10)

    # making list to store the gyroaxis locations for each shell
    phi_theta_cen = []

    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # finding the row and the column of the subplots
        row, col = i//Ncols, i%Ncols
        # using try/except so that we can loop over the bad data shells with low counts
        try:
            E = DATA.ENERGY[E_idx, :, :][0, 0]
            tt_orig, pp_orig, vv = DATA.THETA[E_idx, :, :], DATA.PHI[E_idx, :, :], DATA.VDF[E_idx, :, :] 

            # get log of vv in the theta-phi grid after interpolating
            pp, tt, logvv = interpolate_vdf(pp_orig, tt_orig, vv)

            # fitting the 2D Gaussian to the finer interpolated grid
            fit_params = fit_gauss.fitgaussian(logvv)
            fit_params = fit_gauss.scale_fitparams(fit_params, pp, tt)

            # make Gaussian from fit
            gauss = fit_gauss.gaussian(fit_params, pp, tt)

            # the number of bins which are non-zero in an energy shell
            Ncount = np.sum(~np.isnan(vv)) / len(vv.flatten())

            # appending the located centers
            phi_theta_cen.append([E, Ncount, fit_params[1], fit_params[2]])

        except: continue

        #------------------MAKING DIAGNOSTIC PLOTS IF REQUIRED-----------------------#
        plot_diagnostic_panels(ax[row,col], E, pp_orig, tt_orig, pp, tt, vv, gauss, fit_params)

    return phi_theta_cen, fig, ax
    
def without_plot(DATA, Nrows, Ncols):
    # making list to store the gyroaxis locations for each shell
    phi_theta_cen = []

    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # using try/except so that we can loop over the bad data shells with low counts
        try:
            E = DATA.ENERGY[E_idx, :, :][0, 0]
            tt_orig, pp_orig, vv = DATA.THETA[E_idx, :, :], DATA.PHI[E_idx, :, :], DATA.VDF[E_idx, :, :] 

            # the number of bins which are non-zero in an energy shell
            Ncount = np.sum(~np.isnan(vv)) / len(vv.flatten())

            # get log of vv in the theta-phi grid after interpolating
            pp, tt, logvv = interpolate_vdf(pp_orig, tt_orig, vv)

            # fitting the 2D Gaussian to the finer interpolated grid
            fit_params = fit_gauss.fitgaussian(logvv)
            fit_params = fit_gauss.scale_fitparams(fit_params, pp, tt)

            # make Gaussian from fit
            gauss = fit_gauss.gaussian(fit_params, pp, tt)

            # appending the located centers
            phi_theta_cen.append([E, Ncount, fit_params[1], fit_params[2]])
        
        except: continue

    return phi_theta_cen
    
def interpolate_vdf(pp, tt, vdf, Nphi= 201, Ntheta = 101):
    '''
    Interpolating the energy shell in the theta-phi space to a higher
    number of (Ntheta, Nphi) prior to 2D Gaussian fitting.

    Parameters:
    -----------
    pp: array_like of floats, shape (Nphi_data, Ntheta_data)
        2D meshgrid of phi grid.
    tt: array_like of floats, shape (Nphi_data, Ntheta_data)
        2D meshgrid of theta grid.
    vdf: array_like of floats, shape (Nphi_data, Ntheta_data)
        2D grid of VDF values
    Nphi: scalar, optional
          The number of points in phi to interpolate to.
    Ntheta: scalar, optional
            The number of points in theta to interpolate to.

    Returns:
    --------
    phim: array_like of floats, shape (Nphi, Ntheta)
          2D dense phi meshgrid.
    thetam: array_like of floats, shape (Nphi, Ntheta)
            2D dense phi meshgrid.
    logvdf_: array_like of floats, shape (Nphi, Ntheta)
             2D interpolated VDF on denser meshgrid.
    '''
    phi = np.linspace(0, 2*np.pi, Nphi) * 180 / np.pi
    theta = np.linspace(0, np.pi, Ntheta) * 180 / np.pi

    phim, thetam = np.meshgrid(phi, theta, indexing='ij')

    phi_flat, theta_flat, vdf_flat = pp.flatten(), tt.flatten(), vdf.flatten()

    # we want to interpolate the log base 10 VDF
    logvdf_flat = np.log10(vdf_flat)
    # replacing inf values since we want interpolator to ignore it
    logvdf_flat[np.abs(logvdf_flat) == np.inf] = np.nan
    logvdf_flat[np.isnan(logvdf_flat)] = np.nanmin(logvdf_flat)

    logvdf_interp = interpolate.CloughTocher2DInterpolator(list(zip(phi_flat, theta_flat)),
                                                           logvdf_flat, fill_value=np.nan)
    logvdf_ = logvdf_interp(phim, thetam)

    return phim, thetam, logvdf_

def plot_diagnostic_panels(ax, E, pp_orig, tt_orig, pp, tt, vv, gauss, fit_params):
    vmin, vmax = 0, 6
    im = ax.pcolormesh(pp_orig, tt_orig, np.log10(vv), cmap='BuPu', rasterized=True, vmin=vmin, vmax=vmax)

    # plotting the 2D contours of the fitted Gaussian
    ax.contour(pp, tt, gauss, colors='k', linestyles='dashed', linewidths=1, alpha=0.5, levels=5)
    ax.text(0.99, 0.95, f'({fit_params[1]:.2f}, {fit_params[2]:.2f})', transform=ax.transAxes,
                    va='top', ha='right', color='blue')
    ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
                    va='bottom', ha='left', color='red', fontweight='bold')
    ax.plot(fit_params[1], fit_params[2], 'xk')