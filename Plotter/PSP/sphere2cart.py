import numpy as np
import xarray as xr

"""
TODO : Add rotation into SC and RTN frame.
"""


def vdf_sphere2cart(xr_ds, FRAME='INST'):
    """
    Take in the original xarray dataset and convert over to Cartesian Coordinates.

    Kwargs:
    -------
    FRAME : str corresponding to instrument frame 
    """
    energy     = xr_ds.energy.data
    elevation  = np.radians(xr_ds.theta.data)
    azimuth    = np.radians(xr_ds.phi.data)

    # Convert the energy in eV to velocity
    velocity = 13.85 * np.sqrt(energy)

    if FRAME == 'INST':
        vx = -velocity * np.cos(elevation) * np.cos(azimuth)
        vy = -velocity * np.cos(elevation) * np.sin(azimuth)
        vz =  velocity * np.sin(elevation)

    xr_ds['vx'] = xr.DataArray(vx, dims = list(xr_ds.dims), coords=dict(xr_ds.coords), attrs={'frame' : f'{FRAME}', 'description':'Cartesian velocity in x-direction'})
    xr_ds['vy'] = xr.DataArray(vy, dims = list(xr_ds.dims), coords=dict(xr_ds.coords), attrs={'frame' : f'{FRAME}', 'description':'Cartesian velocity in y-direction'})
    xr_ds['vz'] = xr.DataArray(vz, dims = list(xr_ds.dims), coords=dict(xr_ds.coords), attrs={'frame' : f'{FRAME}', 'description':'Cartesian velocity in z-direction'})

    return(xr_ds)