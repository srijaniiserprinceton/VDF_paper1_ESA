import numpy as np
from scipy import optimize
np.set_printoptions(precision=4)
from sklearn import mixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.ion()


def gaussian(p, x, y):
    # height, center_x, center_y, width_x, width_y = p
    height, center_x, center_y, width_xy = p
    # return height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return height*np.exp(-(((center_x-x)/width_xy)**2+((center_y-y)/width_xy)**2)/2)

def moments(data):
    total = np.nansum(data)
    X, Y = np.indices(data.shape)
    center_x = np.nansum(X*data)/total
    center_y = np.nansum(Y*data)/total
    row = data[int(center_x), :]
    col = data[:, int(center_y)]
    width_x = np.nansum(np.sqrt(abs((np.arange(col.size)-center_y)**2*col))
                        /np.nansum(col))
    width_y = np.nansum(np.sqrt(abs((np.arange(row.size)-center_x)**2*row))
                        /np.nansum(row))
    # using a circular gaussian
    width_xy = 0.5 * (np.abs(width_x) + np.abs(width_y))

    height = np.nanmax(data)

    # return height, center_x, center_y, width_x, width_y
    return height, center_x, center_y, width_xy

def errorfunction(p, x, y, data):
    return gaussian(p, x, y) - data

def fitgaussian(data):
    params = moments(data)
    X, Y = np.indices(data.shape)
    mask = ~np.isnan(data)
    x = X[mask]
    y = Y[mask]
    data = data[mask]
    p, success = optimize.leastsq(errorfunction, params, args=(x, y, data))
    return p

def scale_fitparams(fit_params, pp, tt):
    height, center_x, center_y, width_xy = fit_params

    Nx, Ny = pp.shape

    # x is theta and y is phi
    center_y = (center_y - Ny // 2) / Ny * 180 + 90
    center_x = (center_x - Nx // 2) / Nx * 360 + 180

    # adjusting the widths
    width_xy = width_xy / Nx * 180

    return height, center_x, center_y, width_xy


def scale_fitparams_MMS(fit_params, pp, tt):
    # height, center_x, center_y, width_x, width_y = fit_params
    height, center_x, center_y, width_xy = fit_params

    Nx, Ny = pp.shape

    phi_min = pp[0,0]
    phi_max = pp[-1,0]
    theta_min = tt[0,0]
    theta_max = tt[0,-1]

    # x is phi and y is theta
    center_y = (center_y - Ny // 2) / Ny * 180 + 90
    center_x = (center_x - Nx // 2) / Nx * 360 + 180
    # center_y = (center_y - Ny // 2) / Ny * (theta_max - theta_min) + (theta_max - theta_min)/2
    # center_x = (center_x - Nx // 2) / Nx * (phi_max - phi_min) + (phi_max - phi_min)/2

    # adjusting the widths
    width_xy = width_xy / Nx * (phi_max - phi_min)/2

    # return height, center_y, center_x, width_xy
    return height, center_x, center_y, width_xy

def draw_ellipse(position, covariance, ax=None, **kwargs):
    '''
    Draw an ellipse with a given position and covariance.
    '''
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    nsig_max = 4
    for nsig in range(nsig_max-1, nsig_max):
        ax.add_patch(Ellipse(xy=position, width=nsig * width, height=nsig * height,
                             angle=angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c='black', s=1, zorder=2, alpha=0.1)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='black', s=1, zorder=2, alpha=0.1)
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def GMM_on_VDF(VDF_2D, n_populations=2, Nsamples=int(1e4), plot_GMM=True):
    '''
    Function to draw random sample of points with VDF as the distribution.
    This is done to facilitate performing a GMM on the VDF to obtain the contours for Slepians.
    '''
    # removing the negative entries and setting them to zero (probability cannot be negative)
    VDF_2D[VDF_2D < 0.0] = 0.0

    # normalizing the VDF to emulate properties of a probability distribution function
    VDF_2D = VDF_2D / np.nansum(VDF_2D)
    # Create a flat copy of the distribution function
    flat = VDF_2D.flatten()

    # sampling a set of 2D points of length Nsamples from the distribution function
    sample_index = np.random.choice(a=flat.size, p=flat, size=Nsamples)

    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, VDF_2D.shape)
    adjusted_index = np.array(list(zip(*adjusted_index)))

    # performing Gaussian Mixture Modeling
    gmm = mixture.GaussianMixture(n_components=n_populations, covariance_type='full', random_state=42)

    if(plot_GMM):
        fig, ax = plt.subplots(1,1)
        plot_gmm(gmm, adjusted_index, ax=ax)
        ax.set_xlim([0,32])

    return gmm


