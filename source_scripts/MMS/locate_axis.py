import warnings
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.ion()    

def find_gyroaxis(rec_dict, time_idx, Nrows=4, Ncols=8):
    if(rec_dict.makeplot): 
        fig, ax = with_plot(rec_dict, time_idx, Nrows, Ncols)

        mu_phi, mu_theta = rec_dict.ESA_PHI[time_idx,0,rec_dict.PHI_CEN_IDX,0],\
                           rec_dict.ESA_THETA[time_idx,0,0,rec_dict.THETA_CEN_IDX]

        # making the polar cap extent in degrees
        clock_angle = np.linspace(0, 2 * np.pi, 100)
        x_cap, y_cap = mu_phi + rec_dict.TH * np.cos(clock_angle), mu_theta + rec_dict.TH * np.sin(clock_angle)

        # plotting the effective centroid in all shells
        for axs in ax.flatten():
            axs.plot(x_cap, y_cap, '--r')
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
        plt.close()

def with_plot(rec_dict, time_idx, Nrows, Ncols):
    roll_val = rec_dict.PHI_CEN_IDX
    # rolling the vdf by that amount
    rec_dict.VDF[time_idx] = np.roll(rec_dict.VDF[time_idx], roll_val, axis=1)

    fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

    # making list to store the gyroaxis locations for each shell
    Eshell_info = []

    for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
        # finding the row and the column of the subplots
        row, col = i//Ncols, i%Ncols

        E = rec_dict.ENERGY[time_idx, E_idx, 0, 0]
        
        # note that for FPI, theta array stays the same at different times but phi array changes
        tt_orig, pp_orig, vv = rec_dict.ESA_THETA[time_idx, E_idx, :, :],\
                               rec_dict.ESA_PHI[time_idx, E_idx, :, :],\
                               rec_dict.VDF[time_idx, E_idx, :, :]

        # skipping plotting if there is zero counts in the E-shell
        if(np.sum(~np.isnan(vv)) < 1): continue

        # calculating the log and setting +\- inf to nan
        logvv = np.log10(vv)
        logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)

        plot_diagnostic_panels(ax[row,col], E, pp_orig, tt_orig, logvv)

    return fig, ax
    
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

def plot_diagnostic_panels(ax, E, pp_orig, tt_orig, logvv):
    vmin, vmax = 1, 5
    im = ax.pcolormesh(pp_orig, 90 - tt_orig, logvv, cmap='inferno', rasterized=True, vmin=vmin, vmax=vmax)
    ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
            va='bottom', ha='left', color='white', fontweight='bold')