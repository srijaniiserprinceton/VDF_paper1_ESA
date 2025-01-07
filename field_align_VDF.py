import os
import sys
import numpy as np
import xarray as xr
import pyspedas
import cdflib

import matplotlib.pyplot as plt

import astropy.constants as c
import astropy.units as u

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

def field_aligned_coordinates(B_vec):
    if B_vec.shape[0] > 3:
        Bmag = np.nanmean(np.linalg.norm(B_vec, axis=1))

        # The defined unit vector
        Nx = B_vec[:,0]/Bmag
        Ny = B_vec[:,1]/Bmag
        Nz = B_vec[:,2]/Bmag

        # Some random unit vector
        Rx = np.zeros(len(Nx))
        Ry = np.ones(len(Ny))
        Rz = np.zeros(len(Nz))

        # Get the first perp component
        TEMP_Px = (Ny * Rz) - (Nz * Ry)
        TEMP_Py = (Nz * Rx) - (Nx * Rz)
        TEMP_Pz = (Nx * Ry) - (Ny * Rx)

        Pmag = np.sqrt(TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2)

        Px = TEMP_Px / Pmag
        Py = TEMP_Py / Pmag
        Pz = TEMP_Pz / Pmag

        Qx = (Pz * Ny) - (Py * Nz)
        Qy = (Px * Nz) - (Pz * Nx)
        Qz = (Py * Nx) - (Px * Ny)

        return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    else:
        Bmag = np.linalg.norm(B_vec)

        # The defined unit vector
        Nx = B_vec[0]/Bmag
        Ny = B_vec[1]/Bmag
        Nz = B_vec[2]/Bmag

        # Some random unit vector
        Rx = 0
        Ry = 1
        Rz = 0

        # Get the first perp component
        TEMP_Px = (Ny * Rz) - (Nz * Ry)
        TEMP_Py = (Nz * Rx) - (Nx * Rz)
        TEMP_Pz = (Nx * Ry) - (Ny * Rx)

        Pmag = np.sqrt(TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2)

        Px = TEMP_Px / Pmag
        Py = TEMP_Py / Pmag
        Pz = TEMP_Pz / Pmag

        Qx = (Pz * Ny) - (Py * Nz)
        Qy = (Px * Nz) - (Pz * Nx)
        Qz = (Py * Nx) - (Px * Ny)

        return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    

def rotateVectorIntoFieldAligned(Ax, Ay, Az, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    # For some Vector A in the SAME COORDINATE SYSTEM AS THE ORIGINAL B-FIELD VECTOR:
    if Ax.ndim == 4:
        An = (Ax * Nx[:, None, None, None]) + (Ay * Ny[:, None, None, None]) + (Az * Nz[:, None, None, None])  # A dot N = A_parallel
        Ap = (Ax * Px[:, None, None, None]) + (Ay * Py[:, None, None, None]) + (Az * Pz[:, None, None, None])  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
        Aq = (Ax * Qx[:, None, None, None]) + (Ay * Qy[:, None, None, None]) + (Az * Qz[:, None, None, None])  # 
    
    else:
        An = (Ax * Nx) + (Ay * Ny) + (Az * Nz)  # A dot N = A_parallel
        Ap = (Ax * Px) + (Ay * Py) + (Az * Pz)  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
        Aq = (Ax * Qx) + (Ay * Qy) + (Az * Qz)  # 

    return(An, Ap, Aq)



psp_vdf = cdflib.cdf_to_xarray('./input_data_files/2020-01-26_VDFs.cdf', to_datetime=True)

time = psp_vdf.time.data

energy = psp_vdf.energy.data
theta = psp_vdf.theta.data
phi = psp_vdf.phi.data

vdf = psp_vdf.vdf.data

m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
q_p = 1

velocity = np.sqrt(2 * q_p * energy / m_p)

# Define the Cartesian Coordinates
vx = velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
vy = velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
vz = velocity * np.sin(np.radians(theta))

# Get the magnetic field data for the given time range
tstart = '2020-01-26T00:00:00'
tend   = '2020-01-26T23:00:00'

trange = [tstart, tend]

file = get_psp_span_mom(trange)
data = init_psp_moms(file[0])

b_span = data.MAGF_INST.data
v_span = data.VEL_INST.data

v_field_aligned = np.array(rotateVectorIntoFieldAligned(v_span[:,0], v_span[:,1], v_span[:,2], *field_aligned_coordinates(b_span)))

v_para, vperp1, vperp2 = np.array(rotateVectorIntoFieldAligned(vx, vy, vz, *field_aligned_coordinates(b_span)))

density = data.DENS.data
avg_den = np.convolve(density, np.ones(10)/10, 'same')      # 1-minute average

va_vec = ((b_span * u.nT) / (np.sqrt(c.m_p * c.mu0 * avg_den[:,None] * u.cm**(-3)))).to(u.km/u.s).value
va_mag = np.linalg.norm(va_vec, axis=1)



