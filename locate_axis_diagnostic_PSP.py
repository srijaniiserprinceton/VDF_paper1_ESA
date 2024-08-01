'''
This is the first script that should be run so that we know the range of
energy shells that contain valid data. Use the visual results from the 
diagnostic plot to find E_idx_min and E_idx_max for making histogram plots
to find the effect axis of gyrotropy.
'''

# import statements
import cdflib, sys
import numpy as np
from math import atan
import spherepy as sp
from scipy import interpolate
from scipy.stats import norm
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib import rc
plt.ion()
font = {'size'   : 12}
rc('font', **font)

from source_scripts import fit_2D_gaussian as fit_gauss

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

if __name__=='__main__':
    '''
    Replace Nrows and Ncols such that they multiply to give total number of shells
    '''
    Nrows, Ncols = 4, 8

    # the source file containing the VDF data
    # filename = './input_data_files/2022-02-27_Avg350s_VDFs.cdf'
    filename = './input_data_files/2020-01-26_VDFs.cdf'
    time_stamp = '2020-01-26'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # Each array should be in the shape of [Ntime, dim1, dim2, dim3]
    # where Ntime is the number of time stamps
    # dim1 is the phi dimension, dim2 is the energy, and dim3 is theta index
    # flipping the theta, Energy and phi dimensions to have them monotonically increasing
    Energy = data.energy.data[:,:,::-1,:]
    Theta = data.theta.data[:,:,::-1,:] + 90
    Phi = data.phi.data[:,:,::-1,:]
    VDF = data.vdf.data[:,:,::-1,:]

    # time index with a nice VDF realization
    time_idx = 12359//10 # 51

    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # levels chosen just to plot the 2D Gaussian over the VDF
    levels = np.linspace(0, 7, 10)

    # making list to store the gyroaxis locations for each shell
    phi_theta_cen = []

    VDF[VDF == 0] = np.nan
    VDF = VDF / np.nanmin(VDF)

    # making the diagnostic subplots
    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # using try/except so that we can loop over the bad data shells
        try:
            E = Energy[time_idx, E_idx, :, :][0, 0]
            tt, pp, vv = Theta[time_idx, E_idx, :, :], Phi[time_idx, E_idx, :, :], VDF[time_idx, E_idx, :, :] 

            # the number of bins which are non-zero in an energy shell
            Ncount = np.sum(~np.isnan(vv)) / len(vv.flatten())

            # finding the row and the column of the subplots
            row, col = i//Ncols, i%Ncols

            im = ax[row,col].pcolormesh(pp, tt, np.log10(vv), cmap='rainbow', rasterized=True)# , levels=levels)
            ax[row,col].set_xlim([90,180])
            ax[row,col].set_ylim([30,150])
            ax[row,col].set_aspect('equal')
            # ax[row,col].set_title(f'E = {E:.2f} [eV]')
            ax[row,col].set_aspect('equal')

            # get vv in the theta-phi grid after interpolating
            pp_orig, tt_orig = pp * 1.0, tt * 1.0
            pp, tt, logvv = interpolate_vdf(pp, tt, vv)

            # fitting the 2D Gaussian to the finer interpolated grid
            fit_params = fit_gauss.fitgaussian(logvv)
            fit_params = fit_gauss.scale_fitparams(fit_params, pp, tt)

            # make Gaussian from fit
            gauss = fit_gauss.gaussian(fit_params, pp, tt)
            # plotting the 2D contours of the fitted Gaussian
            ax[row,col].contour(pp, tt, gauss, colors='k', linestyles='dashed', linewidths=1, alpha=0.5, levels=5)
            ax[row,col].text(0.99, 0.95, f'({fit_params[1]:.2f}, {fit_params[2]:.2f})', transform=ax[row,col].transAxes,
                            va='top', ha='right', color='blue')
            ax[row,col].text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax[row,col].transAxes,
                            va='bottom', ha='left', color='red', fontweight='bold')
            ax[row,col].plot(fit_params[1], fit_params[2], 'xk')

            # appending the located centers
            phi_theta_cen.append([E, Ncount, fit_params[1], fit_params[2]])

        except: continue

    # saving the locations of the centers
    phi_theta_cen = np.asarray(phi_theta_cen)
    np.save(f'output_data_files/phi_theta_cen_PSP_{time_stamp}.npy', phi_theta_cen)

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

    # plotting the effective centroid in all shells
    for axs in ax.flatten():
        axs.scatter(mu_phi, mu_theta, marker='o', color='white')

    # making a plot of the histograms in theta and phi with weights built from the count of each shell
    hist_weights = (phi_theta_cen_purged[:,1] / np.max(phi_theta_cen_purged[:,1]))**2

    plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
    # to put common x and y labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
    # plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)

    plt.savefig(f'VDF_paper1_plots/locate_axis_diagnostic_{time_stamp}.pdf')

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