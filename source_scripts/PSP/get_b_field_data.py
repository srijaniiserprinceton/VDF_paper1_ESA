import sys, os

import numpy as np
import pyspedas
import cdflib
import xarray as xr

def load_mag_data(trange, DATATYPE='mag_sc_4_per_cycle'):
    '''
    Get the magnetic field data for the supplied timerange.

    Parameters:
    -----------
    trange - list of strings or datetime objects

    Returns:
    --------
    filename - name of files corresponding to the magnetic field CDF.
    
    TODO: Add file explorer to check if magnetic field data is locally 
    downloaded and load in the given data.
    '''
    filenames = pyspedas.psp.fields(trange, 
                                    datatype=DATATYPE, 
                                    level='l2', 
                                    downloadonly=True, 
                                    notplot=True, 
                                    last_version=True)
    return(filenames)
    
def init_mag_data(trange, CLIP=True, PAD=None, INST_2_SPAN=False): #, DATATYPE='mag_sc_4_per_cycle'):
    """
    
    """
    filenames = load_mag_data(trange, DATATYPE='mag_sc_4_per_cycle')
    rename_dict = {'epoch_mag_SC_4_Sa_per_Cyc' : 'epoch', 'epoch_mag_SC_zero' : 'epoch_zero'}

    if len(filenames) > 1:    
        fields_data = xr.concat([cdflib.cdf_to_xarray(f).rename(rename_dict) for f in filenames], dim='epoch')
    else:
        fields_data = cdflib.cdf_to_xarray(filenames[0]).rename(rename_dict)

    # Convert Epoch to time
    epoch_2_time_obj = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(fields_data.epoch.data)
    epoch_2_time = epoch_2_time_obj.utc.datetime

    epoch_zero_2_time_obj = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(fields_data.epoch_zero.data)
    epoch_zero_2_time = epoch_zero_2_time_obj.utc.datetime

    fields_data['epoch'] = epoch_2_time
    fields_data['epoch_zero'] = epoch_zero_2_time

    # Now we have to rotate SC data into the SPAN-i frame.
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

    if CLIP == True:
        tstart, tend = trange[0], trange[1]

        if PAD:     # integer or float seconds
            if isinstance(PAD, int):
                tstart = np.datetime64(tstart) - np.timedelta64(PAD, 's')
                tend   = np.datetime64(tend) + np.timedelta64(PAD, 's')
            else:
                tstart = np.datetime64(tstart) - np.timedelta64(int(1000*PAD), 'ms')
                tend   = np.datetime64(tend) + np.timedelta64(int(1000*PAD), 'ms')

        fields_data = fields_data.sel(epoch = slice(tstart, tend))
    

    return(fields_data)

def group_mag_data(trange, source_time, window_dt, CLIP=True, SLOPES_ONLY=True, ALL_VECTORS=False):
    """
    Find the mangetic field vector that the given ion VDF experiences. 

    Parameters:
    -----------
    trange - list of str or numpy.datetime objects
        The time range of interest. This will load in a full day of data. Use the CLIP
        keyword to restrict to given time range. 
    source_time - numpy.array 
        The time array corresponding to the VDF.
    window_dt - float or int
        Time in seconds over which to consider the magnetic field vectors.

    Keywords:
    ---------
    CLIP - Bool. Default == True.
        Used to clip timeseries to be only a subset of the full day of magnetic 
        field data.
    SLOPES_ONLY - Bool. Default == True.
        Return only the xy, xz, and yz slopes of the magnetic field. If False, function returns 
        the corresponding slopes, spherical coordinates, and cartesian coordinates in the instrument
        frame.
    ALL_VECTORS - Bool. Default == False.
        If True, return array containing instentaneous magnetic field at 4 sample per cycle resolution.
    
    Returns:
    --------
    slopes_array (Default) - numpy.array
        Average slope in xy, xz, and yz planes. 

    sphere_array - numpy.array
        Average mangetic field data in spherical coordinates.
    cart_array - numpy.array 
        Average mangetic field data in Cartesian coordinates.

    Example:

    [1]: from source_scripts.PSP.get_b_field_data import group_mag_data 
    [2]: trange = [psp_dist.time.data[0], psp_dist.time.data[-1]]
    [3]: slopes = group_mag_data(trange, psp_dist.time.data, window_dt = 4, ALL_VECTORS=True) 
    """
    mag_data = init_mag_data(trange, CLIP=CLIP, PAD=window_dt)

    if isinstance(window_dt, int):
        time_bins = np.column_stack([source_time - np.timedelta64(window_dt, 's'), source_time])
    else:
        time_bins = np.column_stack([source_time - np.timedelta64(int(1000*window_dt), 'ms'), source_time])

    list_vals = []
    for bin_edges in time_bins:
        vals = mag_data.sel(epoch=slice(*bin_edges))
        list_vals.append(vals)

    list_cart = []

    list_r = []
    list_theta = []
    list_phi = []

    list_xy = []
    list_xz = []
    list_yz = []
    for vals in list_vals:
        # Get the magnetic field vectors for each vector in time window.
        b_vecs = vals.psp_fld_l2_mag_INST_Sa_per_Cyc.data

        # Rectify the x and y directions such that phi is defined from the 
        # -x direction to the +x direction.
        b_vecs = [-1, -1, 1] * vals.psp_fld_l2_mag_INST_Sa_per_Cyc.data

        bmag = np.linalg.norm(b_vecs, axis=1)

        bunit = b_vecs/bmag[:, None]

        # Concert the data from Cartesian to Spherical Coordinates.
        theta = np.degrees(np.arccos(b_vecs[:, 2]/bmag)) - 90
        phi = np.degrees(np.sign(b_vecs[:, 1]) * np.arccos(b_vecs[:, 0]/np.sqrt(b_vecs[:, 0]**2 + b_vecs[:, 1]**2)))

        xy_slope = bunit[:, 1] / bunit[:, 0]
        xz_slope = bunit[:, 2] / bunit[:, 0]
        yz_slope = bunit[:, 2] / bunit[:, 1]

        list_xy.append(xy_slope)
        list_xz.append(xz_slope)
        list_yz.append(yz_slope)

        list_r.append(bmag)
        list_theta.append(theta)
        list_phi.append(phi)

        list_cart.append(b_vecs)

    # # Stack the slopes together into one array.
    slopes_array = np.array([np.column_stack([list_xy[i], list_xz[i], list_yz[i]]) for i in range(len(list_xy))])
    
    sphere_array = np.array([np.column_stack([list_r[i], list_phi[i], list_theta[i]]) for i in range(len(list_xy))])
    
    cart_array   = np.array(list_cart)

    if SLOPES_ONLY:
        if ALL_VECTORS:
            return(slopes_array)
        else:
            return(np.nanmean(slopes_array, axis=1))

    else:
        if ALL_VECTORS:
            return(slopes_array, sphere_array, cart_array)
        else:
            return(np.nanmean(slopes_array, axis=1), np.nanmean(sphere_array, axis=1), np.nanmean(cart_array, axis=1))
        