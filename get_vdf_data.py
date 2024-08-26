import sys, os
import numpy as np
import xarray as xr
import pyspedas
import cdflib

from pathlib import Path

def _get_psp_vdf(trange, CREDENTIALS=None):
    '''
    Get and download the latest version of PSP data. 

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
        files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L2', notplot=True, time_clip=True, downloadonly=True, last_version=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
    else:
        files = pyspedas.psp.spi(trange, datatype='spi_sf00_8dx32ex8a', level='l2', notplot=True, time_clip=True, downloadonly=True, last_version=True)

    return(files)

def _get_mms_vdf(trange, probe):
    """
    Get and download the latest verision of the MMS data.

    Parameters:
    -----------
    trange : list of str, datetime object
             Timerange to download the data

    Returns:
    --------
    files : List of str
            Files that will be loaded into the dataset.
    """
    pyspedas.mms.mms_config.CONFIG['download_only'] = True

    files = pyspedas.mms.fpi(trange, probe=probe, data_rate = 'brst', datatype='dis-dist', level='l2', notplot=True)

    return(files)

def init_mms_vdf(filename):
    '''
    Parameters:
    -----------
    filename : list containing the files that are going to be loaded in.

    Returns:
    --------
    vdf_ds : xarray dataset containing the key VDF parameters from the given filename.
    
    NOTE: This will only load in a single day of data.
    '''
    # Constants
    mass_p = 0.010438870
    charge_p = 1

    # List of path names
    filestem = Path(filename).stem      # Get the file stem away from the path.
    parts = filestem.split('_')         # List of the split segments of filestem

    probe, inst, data_rate, level, product, timestamp, version = parts

    # Generate the preamble
    dist_preamble = f'{probe}_'+str(product.split('-')[0])+f'_dist_{data_rate}'
    energy_preamble = f'{probe}_'+str(product.split('-')[0])+f'_energy_{data_rate}'
    theta_preamble = f'{probe}_'+str(product.split('-')[0])+f'_theta_{data_rate}'
    phi_preamble = f'{probe}_'+str(product.split('-')[0])+f'_phi_{data_rate}'
    print(dist_preamble)

    xr_data = cdflib.cdf_to_xarray(filename)

    # Get the instrument time
    xr_time_object = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(xr_data.Epoch.data)
    xr_time_array  = xr_time_object.utc.datetime    # Ensure we are in utc!

    # Get the Unix time
    unix_time = xr_time_object.utc.unix

    # Distribution function
    dist = xr_data[f'{dist_preamble}']

    energy = xr_data[f'{energy_preamble}']
    theta  = xr_data[f'{theta_preamble}']
    phi    = xr_data[f'{phi_preamble}']

    theta_dim = 16
    phi_dim = 32
    energy_dim = 32

    LEN = dist.shape[0]

    # Now we have to expand dimensions such that the data arrays all match shapes.
    energy_unsort = np.repeat(np.repeat(energy.data, 32).reshape(1399, 32, 32), 16).reshape(1399, 32, 32, 16)   # Time, Energy, Phi, Theta
    phi_unsort = np.repeat(np.repeat(phi.data, 32).reshape(1399, 32, 32), 16).reshape(1399, 32, 32, 16)         # Time, Phi, Energy, Theta
    theta_unsort = np.repeat(np.repeat(np.repeat(theta.data, 1399).reshape(16, 1399), 32).reshape(16, 1399, 32), 32).reshape(16, 1399, 32, 32)  # Theta, Time, Energy, Phi

    # Convert data to uniform shape (Time, Energy, Phi, Theta)
    energy_sort = energy_unsort  # No-need to convert order
    phi_sort    = np.transpose(phi_unsort, [0, 2, 1, 3])
    theta_sort  = np.transpose(theta_unsort, [1, 2, 3, 0])
    dist_sort   = np.transpose(dist, [0, 3, 1, 2])

    vdf = dist_sort

    # Generate the xarray dataArrays for each value we are going to pass
    xr_energy = xr.DataArray(energy_sort, dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(32), theta_dim = np.arange(16)), attrs={'units':'eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.01', 'validmax' : '100000.', 'scale' : 'log'})
    xr_phi    = xr.DataArray(phi_sort,    dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(32), theta_dim = np.arange(16)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_theta  = xr.DataArray(theta_sort,  dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(32), theta_dim = np.arange(16)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_vdf    = xr.DataArray(vdf,         dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(32), theta_dim = np.arange(16)), attrs={'units':'s^3/cm^6', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})

    xr_unix   = xr.DataArray(unix_time, dims=['time'], coords=dict(time = xr_time_array), attrs={'units' : 'time', 'description':'Unix time'}) 

    # Generate the xarray.Dataset
    xr_ds = xr.Dataset({
                        'unix_time' : xr_unix,
                        'energy' : xr_energy,
                        'phi' : xr_phi,
                        'theta' : xr_theta,
                        'vdf' : xr_vdf
                       },
                       attrs={'description' : 'SPAN-i data recast into proper format. VDF unit is in s^3/cm^6.'})
    
    return(xr_ds)

def init_psp_vdf(filename):
    '''
    Parameters:
    -----------
    filename : list containing the files that are going to be loaded in.

    Returns:
    --------
    vdf_ds : xarray dataset containing the key VDF parameters from the given filename.
    
    NOTE: This will only load in a single day of data.
    '''
    # Constants
    mass_p = 0.010438870        # eV/(km^2/s^2)
    charge_p = 1

    xr_data = cdflib.cdf_to_xarray(filename)

    # Get the instrument time
    xr_time_object = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(xr_data.Epoch.data)
    xr_time_array  = xr_time_object.utc.datetime    # Ensure we are in utc!

    # Keep the unix time as a check
    unix_time = xr_data.TIME.data

    # Differential energy flux taken from PSP
    energy_flux = xr_data.EFLUX.data

    energy = xr_data.ENERGY.data
    theta  = xr_data.THETA.data
    phi    = xr_data.PHI.data

    theta_dim = 8
    phi_dim = 8
    energy_dim = 32

    LEN = energy_flux.shape[0]

    # Now reshape all of our data: phi_dim, energy_dim, phi_dim
    eflux_sort  = energy_flux.reshape(LEN, phi_dim, energy_dim, theta_dim)
    theta_sort  = theta.reshape(LEN, phi_dim, energy_dim, theta_dim)
    phi_sort    = phi.reshape(LEN, phi_dim, energy_dim, theta_dim)
    energy_sort = energy.reshape(LEN, phi_dim, energy_dim, theta_dim)

    # Convert the data to be in uniform shape (E, phi, theta)
    eflux_sort  = np.transpose(eflux_sort, [0, 2, 1, 3])
    theta_sort  = np.transpose(theta_sort, [0, 2, 1, 3])
    phi_sort    = np.transpose(phi_sort, [0, 2, 1, 3])
    energy_sort = np.transpose(energy_sort, [0, 2, 1, 3])

    # Resort the arrays so the energy is increasing
    eflux_sort  = eflux_sort[:, ::-1, :, :]  
    theta_sort  = theta_sort[:, ::-1, :, :]  
    phi_sort    = phi_sort[:, ::-1, :, :]    
    energy_sort = energy_sort[:, ::-1, :, :]

    # Convert energy flux into differential energy flux
    vdf = eflux_sort * ((mass_p * 1e-10)**2) /(2 * energy_sort**2)      # 1e-10 is used to convert km^2 to cm^2

    # Generate the xarray dataArrays for each value we are going to pass
    xr_eflux  = xr.DataArray(eflux_sort,  dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(8), theta_dim = np.arange(8)), attrs={'units':'eV/cm2-s-ster-eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_energy = xr.DataArray(energy_sort, dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(8), theta_dim = np.arange(8)), attrs={'units':'eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.01', 'validmax' : '100000.', 'scale' : 'log'})
    xr_phi    = xr.DataArray(phi_sort,    dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(8), theta_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_theta  = xr.DataArray(theta_sort,  dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(8), theta_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_vdf    = xr.DataArray(vdf,         dims = ['time', 'energy_dim', 'phi_dim', 'theta_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), phi_dim = np.arange(8), theta_dim = np.arange(8)), attrs={'units':'s^3/cm^6', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    
    xr_unix   = xr.DataArray(unix_time, dims=['time'], coords=dict(time = xr_time_array), attrs={'units' : 'time', 'description':'Unix time'}) 

    # Generate the xarray.Dataset
    xr_ds = xr.Dataset({
                        'unix_time' : xr_unix,
                        'eflux'  : xr_eflux,
                        'energy' : xr_energy,
                        'phi' : xr_phi,
                        'theta' : xr_theta,
                        'vdf' : xr_vdf
                       },
                       attrs={'description' : 'SPAN-i data recast into proper format. VDF unit is in s^3/cm^6.'})
    
    return(xr_ds)

def save_vdf_data(trange, spacecraft, PROBE='1', CREDENTIALS=None):
    '''
    Saving VDF data in streamlined format to be used in the rest of the workflow.

    Parameters:
    -----------
    trange: string
            Time range in [tstart, tend] format where timestamps are 'YYYY-MM-DDThh:mm:ss'.
    spacecraft: string 
                Name of the spacecraft whose ESA measurement we want.
    PROBE : str
            default value is set to be 1. Can range from 1 to 4. 
    CREDENTIALS: Tuple of strings ['<username>', '<password>']
                 User credentials in the above format for unrealeased data. Only used for the PSP data.
    '''
    if spacecraft == 'PSP':
        files   = _get_psp_vdf(trange, CREDENTIALS=CREDENTIALS)
        dataset = init_psp_vdf(files[0])
        cdflib.xarray_to_cdf(dataset, f'./input_data_files/PSP_{trange[0][:10]}_VDFs.cdf')

    if spacecraft == 'MMS':
        files   = _get_mms_vdf(trange, probe=PROBE)
        dataset = init_mms_vdf(files[0])
        cdflib.xarray_to_cdf(dataset, f'./input_data_files/MMS_{trange[0][:10]}_VDFs.cdf')

if __name__ == "__main__":
    # This is where the tests are going to be performed
    # target = '2020-01-26T14:10:42'
    # tstart = '2020-01-26T00:00:00'
    # tend   = '2020-01-26T23:00:00'

    # making the trange tuple
    # trange = [tstart, tend]
    trange = ['2016-01-11/00:57:04', '2016-01-11/01:57:04']

    # saving the .cdf file with the formatted VDF from desired time interval
    save_vdf_data(trange, spacecraft='MMS')

