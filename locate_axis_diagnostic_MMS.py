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

    phim, thetam = np.meshgrid(phi, theta)

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
    filename = 'input_data_files/mms_data_period.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # Each array should be in the shape of [Ntime, dim1, dim2, dim3]
    # where Ntime is the number of time stamps
    # dim1 is the phi dimension, dim3 is the energy, and dim2 is theta index
    Energy, Theta, Phi, VDF = data.energy.data, data.theta.data,\
                              data.phi.data, data.distribution.data

    # time index with a nice VDF realization
    time_idx = 700

    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # levels chosen just to plot the 2D Gaussian over the VDF
    levels = np.linspace(0, 7, 10)

    phi_theta_cen = []

    # making the diagnostic subplots
    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # using try/except so that we can loop over the bad data shells
        try:
            E = Energy[time_idx, E_idx] / 1.602e-19
            t, p, vv = Theta, Phi[time_idx, :], VDF[time_idx, :, :, E_idx]
            pp, tt = np.meshgrid(p, t, indexing='ij')

            vv[vv == 0] = np.nan
            vv = vv / np.nanmin(vv)
            vv = np.log10(vv)
            vv[np.abs(vv) == np.inf] = np.nan
            vv = np.roll(vv, (15, 2), axis=(0,1))
            vv = vv[:]

            # changing the convention of theta from (-90, 90) to (0, 180)
            # tt = -(tt - 90)

            # finding the row and the column of the subplots
            row, col = i//Ncols, i%Ncols

            im = ax[row,col].contourf(pp, tt, vv, cmap='rainbow', rasterized=True)#, levels=levels)

            ax[row,col].set_xlim([0,360])
            ax[row,col].set_ylim([0,180])
            ax[row,col].set_aspect('equal')
            ax[row,col].set_title(f'E = {E:.2f} [eV]')
            ax[row,col].set_aspect('equal')

            # get vv in the theta-phi grid after interpolating
            pp_orig, tt_orig = pp * 1.0, tt * 1.0

            # fitting the 2D Gaussian to the finer interpolated grid
            fit_params = fit_gauss.fitgaussian(vv)
            fit_params = fit_gauss.scale_fitparams_MMS(fit_params, pp, tt)

            # make Gaussian from fit
            gauss = fit_gauss.gaussian(fit_params, pp, tt)
            # plotting the 2D contours of the fitted Gaussian
            ax[row,col].contour(pp, tt, gauss, colors='k', linestyles='dashed', linewidths=1, alpha=0.5, levels=5)
            ax[row,col].text(0.99, 0.95, f'({fit_params[1]:.2f}, {fit_params[2]:.2f})', transform=ax[row,col].transAxes,
                            va='top', ha='right', color='blue')
            ax[row,col].plot(fit_params[1], fit_params[2], 'xk')

            # appending the located centers
            phi_theta_cen.append([E_idx, fit_params[1], fit_params[2]])

        except: continue

    plt.subplots_adjust(top=0.99, bottom=0.01, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
    # to put common x and y labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
    # plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)

    plt.savefig('VDF_paper1_plots/locate_axis_diagnostic_MMS.pdf')

    # saving the locations of the centers
    phi_theta_cen = np.asarray(phi_theta_cen)
    np.save('output_data_files/phi_theta_cen_MMS.npy', phi_theta_cen)
