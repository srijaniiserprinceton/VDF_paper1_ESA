import cdflib
import sys
import numpy as np
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from astropy import coordinates as coor
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import Rbf
# from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

plt.ion()

sys.path.append('..')

import rotate_funcs
# from locate_axis import interpolate_vdf

filename = 'mms_data_period.cdf'
data = cdflib.cdf_to_xarray(filename, to_datetime=True)

# Each array should be in the shape of [Ntime, dim1, dim2, dim3]
# where Ntime is the number of time stamps
# dim1 is the phi dimension, dim3 is the energy, and dim2 is theta index
Energy, Theta, Phi, VDF = data.energy.data, data.theta.data,\
                            data.phi.data, data.distribution.data

# time index with a nice VDF realization
time_idx = 700

# range of energy shells within which we have nice VDFs
E_idx_min, E_idx_max = 10, 18

def create_gridspec():
    # fig, ax = plt.subplots(3, 5, figsize=(15,6), sharex=True, sharey=True)
    fig = plt.figure(figsize=(15,7))
    nrows, ncols = 3, 3
    gs_outer = gridspec.GridSpec(nrows, ncols, figure=fig)
    gs_inner = []
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            gs_inner.append(gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[row_idx,col_idx]))

    ax2d = []
    for i in range(len(gs_inner)):
        ax2d.append(fig.add_subplot(gs_inner[i][0]))

    ax3d = []
    for i in range(len(gs_inner)):
        ax3d.append(fig.add_subplot(gs_inner[i][1], projection='3d'))

    return ax2d, ax3d

levels = np.linspace(0, 7, 10)

# phi, theta centers from locate_axis.py
phi_theta_cen = np.load('center_locs_MMS/phi_theta_cen_MMS.npy')

def interpolate_on_rotated_grid(p, t, v, betas):
    Ntheta, Nphi = 101, 201
    phi = np.linspace(0, 2*np.pi, Nphi)
    theta = np.linspace(0, np.pi, Ntheta)

    # the regular (theta, phi) meshgrid to interpolate onto
    phim, thetam = np.meshgrid(phi, theta)

    # converting t and p 
    t = t * np.pi/180
    p = p * np.pi/180

    X = np.sin(t) * np.cos(p)
    Y = np.sin(t) * np.sin(p)
    Z = np.cos(t)

    # unrotated original grid
    r = np.array([X, Y, Z])

    # finding the rotated grid
    r_ = rotate_funcs.rotate_cartesian(r, 'derot', *betas)

    # using astropy to get the latitude and longitude of the 
    # rotated coordinate
    X_, Y_, Z_ = r_
    # __, lat_, lon_ = coor.cartesian_to_spherical(X_, Y_, Z_)

    # intepolation step
    # logvdf_interp = CloughTocher2DInterpolator(points, v, fill_value=np.nan)
    logvdf_interp = Rbf(X_, Y_, Z_, v, epsilon=0.25, smooth=0)

    # query at new regular grid points
    Xn = np.sin(thetam) * np.cos(phim)
    Yn = np.sin(thetam) * np.sin(phim)
    Zn = np.cos(thetam)

    logvdf_ = logvdf_interp(Xn, Yn, Zn)

    return phim, thetam, logvdf_

# array to store the derotated profile of logvv
logvv_derotated = np.zeros((E_idx_max - E_idx_min + 1, 101, 201))

# making the plots to see what we are dealing with
for j in range(50):
    ax2d, ax3d = create_gridspec()
    for i, E_idx in enumerate(np.arange(E_idx_min, E_idx_max+1)):
        # Each array should be in the shape of [Ntime, dim1, dim2, dim3]
        # where Ntime is the number of time stamps
        # dim1 is the phi dimension, dim3 is the energy, and dim2 is theta index
        Energy, Theta, Phi, VDF = data.energy.data, data.theta.data,\
                                  data.phi.data, data.distribution.data
        E = Energy[time_idx, E_idx] / 1.602e-19
        t, p, vv = Theta, Phi[time_idx, :], VDF[time_idx, :, :, E_idx]
        pp, tt = np.meshgrid(p, t, indexing='ij')

        vv[vv == 0] = np.nan
        vv = vv / np.nanmin(vv)
        vv = np.log10(vv)
        vv[np.abs(vv) == np.inf] = np.nan
        vv = np.roll(vv, (15, 2), axis=(0,1))
        vv = vv[:]

        # # changing the convention of theta from (-90, 90) to (0, 180)
        # tt = -(tt - 90)

        # tt = tt[::-1,::-1]
        # pp = pp[::-1,::-1]
        # vv = vv[::-1,::-1]

        x = np.sin(tt*np.pi/180) * np.cos(pp*np.pi/180)
        y = np.sin(tt*np.pi/180) * np.sin(pp*np.pi/180)
        z = np.cos(tt*np.pi/180)

        # vabsmax=1
        # row, col = i//5, i%5

        # finding the rotated logvdf
        phi_flat, theta_flat, vdf_flat = pp.flatten(), tt.flatten(), vv.flatten()
        logvdf_flat = vdf_flat #np.log10(vdf_flat)

        # logvdf_flat = vv
        logvdf_flat[np.abs(logvdf_flat) == np.inf] = np.nan
        logvdf_flat[np.isnan(logvdf_flat)] = 0.0 #np.nanmin(logvdf_flat)
        
        idx = np.argmin(np.abs(phi_theta_cen[:,0] - E_idx))
        phi_cen = phi_theta_cen[idx,1]
        theta_cen = phi_theta_cen[idx,2]

        # betay_arr = np.linspace(0, 90-7.73, 50)
        # betaz_arr = np.linspace(0, 360-161.96, 50)

        betay_arr = np.linspace(0, theta_cen, 50)
        betaz_arr = np.linspace(0, 360-phi_cen, 50)

        logvv_rot = interpolate_on_rotated_grid(phi_flat, theta_flat, logvdf_flat, [0, -betay_arr[j], betaz_arr[j]])

        # setting negative values to nan
        logvv_rot[2][logvv_rot[2] < 0] = np.nan

        theta, phi = logvv_rot[1], logvv_rot[0]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        
        # plotting
        ax2d[i].contourf(logvv_rot[0], logvv_rot[1], logvv_rot[2], cmap='rainbow', rasterized=True, levels=levels)
        ax2d[i].set_aspect('equal')

        ax3d[i].plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.rainbow(logvv_rot[2]/ np.nanmax(logvv_rot[2])),
                        rasterized=True, vmin=levels[0], vmax=levels[-1])
        ax3d[i].plot_wireframe(x,y,z, rstride=5, cstride=5, color='k', lw=1)
        # Turn off the axis planes
        ax3d[i].set_axis_off()
        ax3d[i].set_aspect('equal')
        

        # storing the derotated profile
        logvv_derotated[i] = logvv_rot[2]

    plt.savefig(f'derotation_gif_MMS/{j}.png')
    plt.close()

# saving the derotated logvv
np.save('derotated_logvv.npy', logvv_derotated)

