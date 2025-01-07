import sys, os

import numpy as np
import xarray as xr
import scipy 
import cdflib
import pyspedas

import matplotlib.pyplot as plt

def get_psp_span_mom(trange, CREDENTIALS=None):
    '''
    Get and download the latest version of the MMS data. 

    Parameters:
    -----------
    trange : list of str, datetime object
             Timerange to download the data
    probe : int or list of ints
            Which MMS probe to get the data from.
    
    Returns:
    --------

    TODO : Add check if file is already downloaded and use local file.
    TODO : Replace with a cdaweb or wget download procedure.
    '''

    if CREDENTIALS:
        files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L3', notplot=True, time_clip=True, downloadonly=True, last_version=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
    else:
        files = pyspedas.psp.spi(trange, datatype='spi_sf00_l3_mom', level='l3', notplot=True, time_clip=True, downloadonly=True, last_version=True)

    return(files)

def init_psp_moms(filename):
    xr_data = cdflib.cdf_to_xarray(filename)

    xr_time_object = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(xr_data.Epoch.data)
    xr_time_array = xr_time_object.utc.datetime 

    xr_data['Epoch'] = xr_time_array
    
    return(xr_data)

def plot_vdf_with_bfield(psp_dist, bfield_inst, bulk_flow, TINDX=0, PSLICE=4, TSLICE=4):
    import matplotlib as mpl
    nlines = bfield_inst.shape[0]
    cmap = mpl.colormaps['plasma']

    colors = cmap(np.linspace(0, 1, nlines))


    vdf = psp_dist.vdf.data
    
    energy = psp_dist.energy.data
    theta = psp_dist.theta.data
    phi = psp_dist.phi.data

    # convert to Cartesian coordinates
    vx = 13.8 * np.sqrt(energy) * np.cos(np.radians(phi)) * np.cos(np.radians(theta))
    vy = 13.8 * np.sqrt(energy) * np.sin(np.radians(phi)) * np.cos(np.radians(theta))
    vz = 13.8 * np.sqrt(energy) * np.sin(np.radians(theta))

    # Define the phi and theta planes that we are interested in
    phi_plane = PSLICE
    theta_plane = TSLICE

    fig, ax = plt.subplots(1, 2, figsize=(16,8), layout='constrained')
    im = ax[0].contourf(vx[TINDX, :, phi_plane, :], vz[TINDX, :, phi_plane, :], np.log10(np.sum(vdf[TINDX, :, :, :], axis=1)))
    [ax[0].plot([0,bfield_inst[i,0]], [0,bfield_inst[i,2]], color=colors[i]) for i in range(bfield_inst.shape[0])]
    ax[0].set_xlim([-1000,  0])
    ax[0].set_ylim([-600, 600])
    plt.colorbar(im)

    im2 = ax[1].contourf(vx[TINDX, :, :, theta_plane], vy[TINDX, :, :, theta_plane], np.log10(np.sum(vdf[TINDX, :, :, :], axis=2)))
    [ax[1].plot([0,bfield_inst[i,0]], [0,bfield_inst[i,1]], color=colors[i]) for i in range(bfield_inst.shape[0])]
    ax[1].set_xlim([-1000,  0])
    ax[1].set_ylim([-100, 600])
    plt.show()

# Load in the PSP VDF
psp_dist = cdflib.cdf_to_xarray('./input_data_files/2020-01-26_VDFs.cdf', to_datetime=True)

# Get the time from the psp_dist
time_dist = psp_dist.time.data

# Get the magnetic field data
trange = [str(time_dist[0]), str(time_dist[-1])]

fields_datafile = pyspedas.psp.fields(trange, datatype='mag_sc_4_per_cycle', level='l2', downloadonly=True, notplot=True, last_version=True)
# fields_data = xr.concat([cdflib.cdf_to_xarray(file) for file in fields_datafile], dim='Epoch')
fields_data = cdflib.cdf_to_xarray(fields_datafile[0])

# Change the variable names
fields_data = fields_data.rename({'epoch_mag_SC_4_Sa_per_Cyc' : 'epoch', 'epoch_mag_SC_zero' : 'epoch_zero'})

# Convert to time
epoch_2_time_obj = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(fields_data.epoch.data)
epoch_2_time = epoch_2_time_obj.utc.datetime

epoch_zero_2_time_obj = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(fields_data.epoch_zero.data)
epoch_zero_2_time = epoch_zero_2_time_obj.utc.datetime

fields_data['epoch'] = epoch_2_time
fields_data['epoch_zero'] = epoch_zero_2_time

# Now we have to rotate from the SPC frame to the SPAN-i frame. 
bfield_sc = fields_data.psp_fld_l2_mag_SC_4_Sa_per_Cyc.data

# INST TO SC ROTATION MATRIX
MAT = np.array([[0, -np.cos(np.radians(20)), -np.sin(np.radians(20))],
               [0,  np.sin(np.radians(20)), -np.cos(np.radians(20))],
               [1, 0, 0]])

MAT_T = np.linalg.inv(MAT)

bfield_inst = np.einsum('ij, tj -> ti', MAT_T, bfield_sc)

xr_mag_inst = xr.DataArray(bfield_inst, dims=['epoch', 'component_index_INST'], coords = dict(epoch = epoch_2_time, component_index_INST = np.arange(3)))

fields_data['psp_fld_l2_mag_INST_Sa_per_Cyc'] = xr_mag_inst

# Define the timebins that we are interested in. 
time_bins = np.column_stack([time_dist - np.timedelta64(4, 's'), time_dist])

list_vals = []
for bin_edges in time_bins:
    vals = fields_data.sel(epoch=slice(bin_edges[0], bin_edges[1]))  # Select the values in each window.
    list_vals.append(vals)

all_slopes_arr = []
slope_arr = []
var_arr = []
for vals in list_vals:
    b_vecs = vals.psp_fld_l2_mag_INST_Sa_per_Cyc.data
    bmag = np.linalg.norm(b_vecs, axis=1)

    # rect 
    sign_bx = np.sign(b_vecs[:,0])

    bunit = (b_vecs) / bmag[:,None]
    bunit2 = bunit * [-1, -1, 0]

    # slope of y over x
    slopes = bunit2[:,1] / bunit2[:,0]

    all_slopes_arr.append(slopes)
    slope_arr.append(np.mean(slopes))
    var_arr.append(np.var(np.arctan(slopes)*180/np.pi))


