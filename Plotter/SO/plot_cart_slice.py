import numpy as np

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from matplotlib import rc

font = {'size'   : 12}
rc('font', **font)

def plot_slices(dist_xr, ts=0, PINDX=None, TINDX=None, CONTOUR=True, SUMMATION=True):
    # Plot on the Peak theta and phi slice
    max_vdf = np.nanargmax(dist_xr.vdf[ts].data)
    eind, pind, tind = np.unravel_index(max_vdf, dist_xr.vdf[ts].data.shape)

    vx_peak = dist_xr.vx[ts, eind, pind, tind].data
        
    if TINDX:
        tind = TINDX
    if PINDX:
        pind = PINDX

    PINDX, TINDX = pind, tind    


    fig = plt.figure(figsize=(12,6), dpi=120)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], hspace=0.3)

    gs.update(left=0.08, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])

    cbar_ax = plt.subplot(gs[1, :])

    if SUMMATION:
        vdf_phi_plane   = np.sum(dist_xr.vdf[ts], axis=1)
        vdf_theta_plane = np.sum(dist_xr.vdf[ts], axis=2)

        vx_phi_plane = dist_xr.vx[ts, :, PINDX, :]
        vz_phi_plane = dist_xr.vz[ts, :, PINDX, :]

        vx_theta_plane = dist_xr.vx[ts, :, :, TINDX]
        vy_theta_plane = dist_xr.vy[ts, :, :, TINDX]
    else:
        vdf_phi_plane   = dist_xr.vdf[ts, :, PINDX, :]
        vdf_theta_plane = dist_xr.vdf[ts, :, :, TINDX]

        vx_phi_plane = dist_xr.vx[ts, :, PINDX, :]
        vz_phi_plane = dist_xr.vz[ts, :, PINDX, :]

        vx_theta_plane = dist_xr.vx[ts, :, :, TINDX]
        vy_theta_plane = dist_xr.vy[ts, :, :, TINDX]

    if CONTOUR:
        im1 = ax1.contourf(vx_phi_plane, vz_phi_plane, vdf_phi_plane, norm=LogNorm())
        im2 = ax2.contourf(vx_theta_plane, vy_theta_plane, vdf_theta_plane, norm=LogNorm())

        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    else:
        im1 = ax1.pcolormesh(vx_phi_plane, vz_phi_plane, vdf_phi_plane, norm=LogNorm())
        im2 = ax2.pcolormesh(vx_theta_plane, vy_theta_plane, vdf_theta_plane, norm=LogNorm())

        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

    frame = dist_xr.vx.attrs['frame']
    ax1.set_xlabel(f'$v_x$ {frame}')
    ax1.set_ylabel(f'$v_z$ {frame}')

    ax1.set_xlim([vx_peak - 400, vx_peak + 400])
    ax1.set_ylim([-400, 400])

    ax2.set_xlim([vx_peak - 400, vx_peak + 400])
    ax2.set_ylim([-400, 400])

    ax2.set_xlabel(f'$v_x$ {frame}')
    ax2.set_ylabel(f'$v_y$ {frame}')

    vdf_units = dist_xr.vdf.attrs['units']
    cbar.set_label(f'f(v) [${vdf_units}$]')

    plt.show()