import numpy as np
from scipy import optimize
np.set_printoptions(precision=4)


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

