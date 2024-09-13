import warnings
import numpy as np
from scipy import interpolate
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from photutils.centroids import centroid_2dg
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.ion()    

# imports from our custom package
from . import fit_2D_gaussian as fit_gauss

def find_gyroaxis(DATA, time_idx, threshold=-3, TH=45, Nrows=4, Ncols=8, makeplot=True, all_shell_info=True):
    if(makeplot): phi_theta_cen, fig, ax = with_plot(DATA, time_idx, Nrows, Ncols, threshold)
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
    (mu_phi, sig_phi) = norm.fit(phi_theta_cen_purged[:,3])
    (mu_theta, sig_theta) = norm.fit(phi_theta_cen_purged[:,2])

    # finding the largest TH
    TH = np.max(phi_theta_cen_purged[:,-1])

    if(makeplot):
        # making the polar cap extent in degrees
        clock_angle = np.linspace(0, 2 * np.pi, 100)
        x_cap, y_cap = mu_phi + TH * np.cos(clock_angle), mu_theta - 90 + TH * np.sin(clock_angle)

        # plotting the effective centroid in all shells
        for axs in ax.flatten():
            # axs.scatter(mu_phi, mu_theta, marker='o', color='white')
            axs.plot(x_cap, y_cap, '--r')
            # axs.set_xlim([DATA.PHI.min(),DATA.PHI.max()])
            # axs.set_ylim([DATA.THETA.min(),DATA.THETA.max()])
            axs.set_xlim([0, 360])
            axs.set_ylim([-90, 90])
            axs.set_aspect('equal')

        plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
        # to put common x and y labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
        plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)
        plt.suptitle(f'{time_idx}')
        plt.savefig(f'VDF_paper1_plots/VDF_MMS_polar_plot/{time_idx}.png')
        # plt.close()

    if(all_shell_info):
        return mu_phi, mu_theta, phi_theta_cen
    else: mu_phi, mu_theta

def with_plot(DATA, time_idx, Nrows, Ncols, threshold):
    roll_idx_peak = []

    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # using try/except so that we can loop over the bad data shells with low counts
        # try:
        vv = DATA.VDF[E_idx, :, :] 
        if(np.sum(~np.isnan(vv)) < 10): continue
        peak_idx_flat = np.argmax(vv)
        roll_val = np.abs(15 - peak_idx_flat)
        roll_idx_peak.append(roll_val)

        # except: continue

    # finding the mean roll value
    roll_val = int(np.mean(roll_val))

    # rolling the vdf by that amount
    DATA.VDF = np.roll(DATA.VDF, roll_val, axis=1)

    x_idx_cen = []
    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # using try/except so that we can loop over the bad data shells with low counts
        # try:
        vv = DATA.VDF[E_idx, :, :] 
        if(np.sum(~np.isnan(vv)) < 10): continue
        logvv = np.log10(vv)
        logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)
        x_cen, __ = centroid_2dg(logvv)
        x_idx_cen.append(x_cen)

        # except: continue
    
        # finding the mean roll value
    roll_val = int(15 - np.mean(x_idx_cen))

    # rolling the vdf by that amount
    DATA.VDF = np.roll(DATA.VDF, roll_val, axis=1)

    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # levels chosen just to plot the 2D Gaussian over the VDF
    levels = np.linspace(0, 7, 10)

    # making list to store the gyroaxis locations for each shell
    phi_theta_cen = []

    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # finding the row and the column of the subplots
        row, col = i//Ncols, i%Ncols
        # using try/except so that we can loop over the bad data shells with low counts
        # try:
        E = DATA.ENERGY[E_idx, :, :][0, 0]
        tt_orig, pp_orig, vv = DATA.THETA[E_idx, :, :], DATA.PHI[E_idx, :, :], DATA.VDF[E_idx, :, :] 
        xmin, xmax, ymin, ymax = tt_orig.min(), tt_orig.max(), pp_orig.min(), pp_orig.max()
        if(np.sum(~np.isnan(vv)) < 10): continue

        # finding the centroid and the angular extend for this shell upto a given threshold
        logvv = np.log10(vv)
        logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)

        # moving pattern to the center by rolling it
        x_cen, y_cen, gauss = Gauss_2dg(logvv)
        # scaling the centers
        x_cen = x_cen/16 * 180
        y_cen = y_cen/32 * 360

        # the number of bins which are non-zero in an energy shell
        Intensity = np.sum(vv[~np.isnan(vv)])

        #------------------MAKING DIAGNOSTIC PLOTS IF REQUIRED-----------------------#
        TH = plot_diagnostic_panels(ax[row,col], E, pp_orig, tt_orig, logvv, gauss, x_cen, y_cen, threshold)

        # appending the located centers
        phi_theta_cen.append([E, Intensity, x_cen, y_cen, 85])

        # except: continue

    return phi_theta_cen, fig, ax
    
def without_plot(DATA, Nrows, Ncols):
    # making list to store the gyroaxis locations for each shell
    phi_theta_cen = []

    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # using try/except so that we can loop over the bad data shells with low counts
        try:
            E_idx = 15
            # finding the row and the column of the subplots
            row, col = i//Ncols, i%Ncols
            # using try/except so that we can loop over the bad data shells with low counts
            E = DATA.ENERGY[E_idx, :, :][0, 0]
            tt_orig, pp_orig, vv = DATA.THETA[E_idx, :, :], DATA.PHI[E_idx, :, :], DATA.VDF[E_idx, :, :] 

            # finding the centroid and the angular extend for this shell upto a given threshold
            # x_cen, y_cen = centroid_2dg(np.roll(np.log10(vv), 16))
            logvv = np.roll(logvv, -(16 - int(x_cen)), axis=0)
            # x_cen, y_cen, gauss = Gauss_2dg(logvv, 16)
            x_cen, y_cen = 0, 0

            # the number of bins which are non-zero in an energy shell
            Intensity = np.sum(vv[~np.isnan(vv)])

            # appending the located centers
            phi_theta_cen.append([E, Intensity, x_cen, y_cen, TH])
        
        except: continue

    return phi_theta_cen

def Gauss_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be a background-subtracted
        cutout image containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """
    # prevent circular import
    from photutils.morphology import data_properties

    Nx, Ny = data.shape
    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.0e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Gaussian plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.0

    mask = data.mask
    data.fill_value = 0.0
    data = data.filled()

    # Subtract the minimum of the data as a rough background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties. Moments from negative data
    # values can yield undefined Gaussian parameters, e.g., x/y_stddev.
    props = data_properties(data - np.min(data), mask=mask)

    constant_init = 0.0  # subtracted data minimum above
    g_init = (Const2D(constant_init)
              + Gaussian2D(amplitude=np.ptp(data),
                           x_mean=props.xcentroid,
                           y_mean=props.ycentroid,
                           x_stddev=props.semimajor_sigma.value,
                           y_stddev=props.semiminor_sigma.value,
                           theta=props.orientation.value))
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    return gfit.x_mean_1.value, gfit.y_mean_1.value, gfit(x, y)

def plot_diagnostic_panels(ax, E, pp_orig, tt_orig, logvv, gauss, x_cen, y_cen, threshold):
    vmin, vmax = 1, 7 #logvv.min(), logvv.max()

    im = ax.pcolormesh(pp_orig, 90 - tt_orig, logvv, cmap='plasma', rasterized=True, vmin=vmin, vmax=vmax)
    # im = ax.pcolormesh(logvv, cmap='seismic', rasterized=True, vmin=vmin, vmax=vmax)

    # plotting the 2D contours of the fitted Gaussian
    levels = np.linspace(0, 8, 8)
    ax.contour(pp_orig, 90 - tt_orig, gauss, colors='k', linestyles='dashed', linewidths=1, alpha=0.5, levels=levels)
    # ax.contour(gauss, colors='k', linestyles='dashed', linewidths=1, alpha=0.5, levels=levels)
    ax.text(0.99, 0.95, f'({x_cen:.2f}, {y_cen - 90:.2f})', transform=ax.transAxes,
            va='top', ha='right', color='black')
    ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
            va='bottom', ha='left', color='red', fontweight='bold')
    ax.plot(y_cen, x_cen - 90, 'xw')

    plt.figure()
    img = plt.contourf(gauss/gauss.max(), cmap='gnuplot2', levels=[0.1, 1])
    plt.close()
    p = img.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]

    TH = np.max(np.sqrt((x - x_cen)**2 + (y - y_cen)**2))

    return TH