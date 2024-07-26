import cdflib, sys
import numpy as np
from math import atan
import spherepy as sp
from scipy import interpolate
from scipy.stats import norm
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import rc
plt.ion()
font = {'size'   : 12}
rc('font', **font)

sys.path.append("..")
from proposal_codes import fit_2D_gaussian as fit_gauss

def get_subplot_axes():
    # definitions for the axes
    left, width = 0.1, 0.2
    bottom, height = 0.72, 0.2
    bottom_h, left_h = bottom + height + 0.02, left + width + 0.02

    ax_dict = {}

    ax_dict['rect_scatter'] = [0.13, 0.6, 0.57, 0.2]
    ax_dict['rect_histx'] = [0.14, 0.82, 0.55, 0.15]
    ax_dict['rect_histy'] = [0.72, 0.6, 0.23, 0.2]
    ax_dict['panel_b'] = [0.13, 0.2, 0.85, 0.3]
    ax_dict['panel_c'] = [0.13, 0.05, 0.85, 0.2]
    
    ax = []
    for key in ax_dict.keys():
        ax.append(plt.axes(ax_dict[key]))

    return np.asarray(ax)

def scatter_hist_2d_1d(phi_theta_data):
    x, y = phi_theta_data.T
    nullfmt = NullFormatter()         # no labels

    # # definitions for the axes
    # left, width = 0.1, 0.65
    # bottom, height = 0.1, 0.65
    # bottom_h = left_h = left + width + 0.02

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    # axScatter.set_xlim((-lim, lim))
    # axScatter.set_ylim((-lim, lim))
    axScatter.set_xlim((0, 360))
    axScatter.set_ylim((0, 180))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=10, range=(120, 200))
    axHisty.hist(y, bins=10, orientation='horizontal', range=(70, 100))

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

def interpolate_vdf(pp, tt, vdf, Nphi= 201, Ntheta = 101):
    # interpolating onto a regular grid of ZZ and YY
    phi = np.linspace(0, 2*np.pi, Nphi) * 180 / np.pi
    theta = np.linspace(0, np.pi, Ntheta) * 180 / np.pi

    phim, thetam = np.meshgrid(phi, theta)

    phi_flat, theta_flat, vdf_flat = pp.flatten(), tt.flatten(), vdf.flatten()

    logvdf_flat = np.log10(vdf_flat)
    logvdf_flat[np.abs(logvdf_flat) == np.inf] = np.nan
    logvdf_flat[np.isnan(logvdf_flat)] = np.nanmin(logvdf_flat)

    logvdf_interp = interpolate.CloughTocher2DInterpolator(list(zip(phi_flat, theta_flat)),
                                                           logvdf_flat, fill_value=np.nan)
    logvdf_ = logvdf_interp(phim, thetam)

    return phim, thetam, logvdf_

if __name__=='__main__':
    filename = '../proposal_codes/2022-02-27_Avg350s_VDFs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    # Each array should be in the shape of [Ntime, dim1, dim2, dim3]
    # where Ntime is the number of time stamps
    # dim1 is the phi dimension, dim2 is the energy, and dim3 is theta index
    # flipping the theta, Energy and phi dimensions to have them monotonically increasing
    Energy = data.energy.data[:,::-1,::-1,::-1]
    Theta = data.theta.data[:,::-1,::-1,::-1]
    Phi = data.phi.data[:,::-1,::-1,::-1]
    VDF = data.vdf.data[:,::-1,::-1,::-1]

    # time index with a nice VDF realization
    time_idx = 51

    # range of energy shells within which we have nice VDFs
    # found from running locate_axis_diagnostic.py
    E_idx_min, E_idx_max = 2, 22

    # there are 15 energy shells in this case
    # fig = plt.figure(figsize=(5,8), constrained_layout=True)
    # gs = fig.add_gridspec(3, 2)
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[0, 1])
    # ax2 = fig.add_subplot(gs[1, :])
    # ax3 = fig.add_subplot(gs[2, :])

    levels = np.linspace(0, 7, 10)

    # lists to store the location of the centers of VDFs in each energy shell
    phi_theta_cen = []

    for i, E_idx in enumerate(np.arange(E_idx_min, E_idx_max+1)):
        E = Energy[time_idx, :, E_idx, :][0, 0]
        tt, pp, vv = Theta[time_idx, :, E_idx, :], Phi[time_idx, :, E_idx, :], VDF[time_idx, :, E_idx, :]

        # changing the convention of theta from (-90, 90) to (0, 180)
        tt = -(tt - 90)

        # finding the row and the column of the subplots
        row, col = i//8, i%8

        # get vv in the theta-phi grid after interpolating
        pp_orig, tt_orig = pp * 1.0, tt * 1.0
        pp, tt, logvv = interpolate_vdf(pp, tt, vv)

        # fitting the 2D Gaussian to the finer interpolated grid
        # logvv = np.log10(vv)
        # logvv[np.abs(logvv) == np.inf] = np.nan
        fit_params = fit_gauss.fitgaussian(logvv)
        fit_params = fit_gauss.scale_fitparams(fit_params, pp, tt)

        # make Gaussian from fit
        gauss = fit_gauss.gaussian(fit_params, pp, tt)

        # appending the located centers
        phi_theta_cen.append([E_idx, fit_params[1], fit_params[2]])

    phi_theta_cen = np.asarray(phi_theta_cen)

    # special plot only for one of the energy shells
    E_idx = 17
    E = Energy[time_idx, :, E_idx, :][0, 0]
    tt, pp, vv = Theta[time_idx, :, E_idx, :], Phi[time_idx, :, E_idx, :], VDF[time_idx, :, E_idx, :]
    # changing the convention of theta from (-90, 90) to (0, 180)
    tt = -(tt - 90)
    pp_orig, tt_orig = pp * 1.0, tt * 1.0
    pp, tt, logvv = interpolate_vdf(pp, tt, vv)
    fit_params = fit_gauss.fitgaussian(logvv)
    fit_params = fit_gauss.scale_fitparams(fit_params, pp, tt)
    gauss = fit_gauss.gaussian(fit_params, pp, tt)

    '''
    ax1.contourf(pp_orig, tt_orig, np.log10(vv), cmap='rainbow', rasterized=True, levels=levels)
    ax1.set_xlim([90,200])
    ax1.set_ylim([0,180])
    ax1.set_aspect('equal')

    ax0.contourf(pp_orig, tt_orig, np.log10(vv), cmap='rainbow', rasterized=True, levels=levels)
    ax0.set_xlim([90,200])
    ax0.set_ylim([0,180])
    ax0.set_aspect('equal')
    ax0.contour(pp, tt, gauss, colors='k', linestyles='dashed', linewidths=1, levels=5)
    ax0.text(0.99, 0.95, f'({fit_params[1]:.2f}, {fit_params[2]:.2f})', transform=ax0.transAxes,
             va='top', ha='right', color='blue')
    ax0.plot(fit_params[1], fit_params[2], 'xk')

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.99, wspace=0.05, hspace=0.05)

    ax1.set_xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=14)
    ax0.set_ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=14)
    ax0.set_xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=14)
    '''
    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(5, 6))
    ax = get_subplot_axes()

    # sys.exit()

    ax[0].contourf(pp_orig, tt_orig, np.log10(vv), cmap='rainbow', rasterized=True, levels=levels)
    ax[0].set_xlim([0,360])
    ax[0].set_ylim([10,170])
    ax[0].set_aspect('equal')
    ax[0].contour(pp, tt, gauss, colors='k', linestyles='dashed', linewidths=1, levels=5)
    ax[0].text(0.99, 0.95, f'({fit_params[1]:.2f}, {fit_params[2]:.2f})', transform=ax[0].transAxes,
             va='top', ha='right', color='blue')
    ax[0].plot(fit_params[1], fit_params[2], 'xk')
    ax[0].set_ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=14)
    ax[0].set_xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=14)

    print(phi_theta_cen)

    ax[1].hist(phi_theta_cen[:,1], bins=5, range=(150, 175), facecolor='grey')
    ax[2].hist(phi_theta_cen[:,2], bins=5, range=(75, 90), orientation='horizontal', facecolor='grey')
    ax[1].set_xlim([20,330])
    ax[2].set_ylim([0,180])

    # plotting the histogram of the located 2D gaussian peaks
    '''
    (mu_phi, sigma_phi) = norm.fit(phi_theta_cen[:,1])
    (mu_theta, sigma_theta) = norm.fit(phi_theta_cen[:,2])
    __, bins_phi, __ = ax2.hist(phi_theta_cen[:,1], 10, facecolor='blue', alpha=0.5, density=True)
    __, bins_theta, __ = ax2.hist(phi_theta_cen[:,2], 10, facecolor='grey', alpha=0.5, density=True)
    phi_hist_gaussfit = norm.pdf(bins_phi, mu_phi, sigma_phi)
    theta_hist_gaussfit = norm.pdf(bins_theta, mu_theta, sigma_theta)
    ax2.plot(bins_phi, phi_hist_gaussfit, '--', color='blue', linewidth=2)
    ax2.plot(bins_theta, theta_hist_gaussfit, 'k--', linewidth=2)
    ax2.axvline(mu_phi, color='blue')
    ax2.axvline(mu_theta, color='black')

    # plotting this mean center in the top panel
    ax1.plot(fit_params[1], fit_params[2], '.r')
    ax1.axvline(pp_orig[-1,0], color='red', alpha=0.5)
    '''

    for res_idx in range(2,8):
        print(res_idx)
        ax[3].contourf(pp_orig[:res_idx+1] + res_idx * 120, tt_orig[:res_idx+1], np.log10(vv)[:res_idx+1],
                       cmap='rainbow', rasterized=True, levels=levels)
        if(res_idx < 7):
            ax[3].contourf(pp_orig[res_idx:] + res_idx * 120, tt_orig[res_idx:], np.log10(vv)[res_idx:],
                        cmap='rainbow', rasterized=True, levels=levels, alpha=0.2)
        # ax[3].set_ylim([0,180])
        ax[3].set_aspect('equal')
    ax[3].set_xticks([])
    ax[3].set_ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=14)
    ax[3].set_title(r'$\leftarrow$ Increasing FOV restriction', fontsize=14)

    # plotting the third panel of restricted FOV
    E_idx_min, E_idx_max = 10, 16
    phi_theta_mean = []
    for res_idx in range(3,9):
        phi_theta_cen = []
        for i, E_idx in enumerate(np.arange(E_idx_min, E_idx_max+1)):
            E = Energy[time_idx, :res_idx, E_idx, :][0, 0]
            tt, pp, vv = Theta[time_idx, :res_idx, E_idx, :],\
                         Phi[time_idx, :res_idx, E_idx, :],\
                         VDF[time_idx, :res_idx, E_idx, :]

            # changing the convention of theta from (-90, 90) to (0, 180)
            tt = -(tt - 90)

            # finding the row and the column of the subplots
            row, col = i//8, i%8

            # get vv in the theta-phi grid after interpolating
            pp_orig, tt_orig = pp * 1.0, tt * 1.0
            pp, tt, logvv = interpolate_vdf(pp, tt, vv)

            # fitting the 2D Gaussian to the finer interpolated grid
            # fit_params = fit_gauss.fitgaussian(logvv)
            fit_params = fit_gauss.fitgaussian(logvv)
            fit_params = fit_gauss.scale_fitparams(fit_params, pp, tt)

            # make Gaussian from fit
            gauss = fit_gauss.gaussian(fit_params, pp, tt)

            # appending the located centers
            phi_theta_cen.append([E, fit_params[1], fit_params[2]])

        phi_theta_cen = np.asarray(phi_theta_cen)
        (mu_phi, __) = norm.fit(phi_theta_cen[:,1])
        (mu_theta, __) = norm.fit(phi_theta_cen[:,2])
        phi_theta_mean.append([pp_orig[-1,0], mu_phi, mu_theta])
    
    phi_theta_mean = np.asarray(phi_theta_mean)

    for res_idx in range(3,9):
        ax[3].plot(phi_theta_mean[res_idx-3,1] + 120 * (res_idx-1), phi_theta_mean[res_idx-3, 2], '.r')
        ax[3].plot(phi_theta_mean[5,1] + 120 * (res_idx-1), phi_theta_mean[5, 2], 'xk')

    ax[4].plot(pp_orig[2:8,0], phi_theta_mean[:,1], 'or')
    ax[4].axhline(phi_theta_mean[-1,1], color='black', ls='dashed')
    ax[4].set_ylim([120, 260])

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.07, right=0.98, wspace=0.05, hspace=0.05)
    # to put common x and y labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.savefig('plots/Fig1_trial.pdf')

    # saving the locations of the centers
    # np.save('center_locs/phi_theta_cen.npy', np.array(phi_theta_cen))
