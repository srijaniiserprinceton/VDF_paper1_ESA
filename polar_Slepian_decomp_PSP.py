import cdflib
import sys
import numpy as np
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from astropy import coordinates as coor
import mat73
plt.ion()

# this is where we unpack the Slepian dictionary and perform other Slepian backend operations
from source_scripts import sph2slep


#----------the source file containing the VDF data-------------#
# filename = './input_data_files/2022-02-27_Avg350s_VDFs.cdf'
filename = './input_data_files/2020-01-26_VDFs.cdf'
time_stamp = '2020-01-26'
data = cdflib.cdf_to_xarray(filename, to_datetime=True)

# Each array should be in the shape of [Ntime, dim1, dim2, dim3]
# where Ntime is the number of time stamps
# dim1 is the phi dimension, dim2 is the energy, and dim3 is theta index
# flipping the theta, Energy and phi dimensions to have them monotonically increasing
Energy = data.energy.data[:,:,::-1,:]
Theta = data.theta.data[:,:,::-1,:] + 90
Phi = data.phi.data[:,:,::-1,:]
VDF = data.vdf.data[:,:,::-1,:]

# time index with a nice VDF realization
time_idx = 12359//5 # 51

# identifying the zeros and then scaling up the VDF with the non-zero minimum value
zeros_mask = VDF == 0.0
VDF[zeros_mask] = np.nan
VDF = VDF / np.nanmin(VDF[~zeros_mask])

# setting the zeros to a very tiny value so log operation goes through
VDF[zeros_mask] = 1e-16

#-----------reading the Slepian basis on a polar cap-----------------------#
# dictionary of axisymmetric indices for different Slepian Lmax
axisym_idx = {}
axisym_idx['12'] = np.array([0, 5, 14])
axisym_idx['15'] = np.array([0, 5, 14, 29])
axisym_idx['20'] = np.array([0, 5, 14, 29, 46])

Lmax = 20
not_axisym = True #False

Slep_fname = f'glmalphapto-45-{Lmax}-1.738618e+02-9.230719e+01-0_lowres.mat'
# Slep_fname = 'glmalphapto-45-20-8.386185e+01--9.230719e+01-0.mat'
Slepian_dict = mat73.loadmat(f'./input_data_files/Slepian_functions/{Slep_fname}')
lon, lat = Slepian_dict['lon'], Slepian_dict['lat']

# the (theta, phi) coorindate system we want our basis functions on
# pp, tt = np.meshgrid(lon, lat)
pp, tt = lon, 90 - lat
# pp = pp[:,::-1]
# tt = tt[:,::-1]

# converting these to Slepian functions on a spherical polar coordinate system
make_Slep = sph2slep.sph2slep(Slepian_dict)

# number of Slepians to consider
if(not_axisym):
    Nsleps = 50 #int(make_Slep.N)
    slep_idx = np.arange(0, Nsleps)
else:
    slep_idx = axisym_idx[f'{Lmax}']
    Nsleps = len(slep_idx)

# padding the data with nan
Nrows, Ncols = 4, 8
fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)
fig2, ax2 = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

Slep_coeffs_each_shell = {}
for i, E_idx in enumerate(np.arange(Nrows * Ncols)):
    # E_idx=15
    # try:
    # finding the row and the column of the subplots
    row, col = i//Ncols, i%Ncols
    E = Energy[time_idx, E_idx, :, :][0, 0]
    tt_span, pp_span, vv = Theta[time_idx, E_idx, :, :], Phi[time_idx, E_idx, :, :], VDF[time_idx, E_idx, :, :] 
    data_vv = np.log10(vv)
    data = np.zeros((12, 32)) + np.nan
    data[2:10, 8:16] = data_vv.T
    # data[2:10, 8:15] = data_vv.T[:,:-1]

    nan_mask = np.isnan(data)
    Eshell_VDF_nonan = data[~nan_mask]
    # now decomposing this structure
    G_nonan = np.zeros((Nsleps, len(Eshell_VDF_nonan)))

    for idx, j in enumerate(slep_idx):
        G_nonan[idx,:] = make_Slep.G[j,~nan_mask]

    M = G_nonan @ G_nonan.T 
    I = np.identity(M.shape[0])
    __, V, __ = np.linalg.svd(M)
    plt.figure()
    plt.semilogy(V, '.k')
    plt.savefig(f'./VDF_paper1_plots/{E_idx}.pdf')
    plt.close()

    # finding coeffs from incomplete mask
    coeffs_from_inc = np.linalg.inv(G_nonan @ G_nonan.T +  I) @ G_nonan @ Eshell_VDF_nonan
    Slep_coeffs_each_shell[f'{E_idx}'] = {}
    Slep_coeffs_each_shell[f'{E_idx}']['coeffs'] = coeffs_from_inc 
    Eshell_rec_inc = np.dot(np.moveaxis(make_Slep.G[:Nsleps], 0, -1), coeffs_from_inc)

    # print()
    vmin, vmax = 0, np.max([np.nanmax(data), 1e-16])
    print(vmax)
    Slep_coeffs_each_shell[f'{E_idx}']['vmax'] = vmax

    # # setting nans with zeros
    # data[np.isnan(data)] = 0.0

    ax[row,col].pcolormesh(pp, tt, data, cmap='rainbow', rasterized=True, vmin=vmin, vmax=vmax)
    ax[row,col].set_xlim([90,180])
    ax[row,col].set_ylim([30,150])
    ax[row,col].text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax[row,col].transAxes,
                        va='bottom', ha='left', color='black', fontweight='bold')

    ax2[row,col].pcolormesh(pp, tt, Eshell_rec_inc, cmap='rainbow', rasterized=True, vmin=vmin, vmax=vmax)
    ax2[row,col].set_xlim([90,180])
    ax2[row,col].set_ylim([30,150])
    ax2[row,col].text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax2[row,col].transAxes,
                        va='bottom', ha='left', color='black', fontweight='bold')
    # except: continue

fig.suptitle('PSP data from 2020-01-26')
fig2.suptitle('Slepian reconstructed VDFs')
fig.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
fig2.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)

fig.savefig('VDF_paper1_plots/SPAN_data.pdf')
fig2.savefig('VDF_paper1_plots/Slepain_rec_polarcap.pdf')
plt.close(fig)
plt.close(fig2)

# making the highres plot
Slep_fname = f'glmalphapto-45-{Lmax}-1.738618e+02-9.230719e+01-0_highres.mat'
# Slep_fname = 'glmalphapto-45-20-8.386185e+01--9.230719e+01-0.mat'
Slepian_dict = mat73.loadmat(f'./input_data_files/Slepian_functions/{Slep_fname}')
lon, lat = Slepian_dict['lon'], Slepian_dict['lat']

# the (theta, phi) coorindate system we want our basis functions on
# pp, tt = np.meshgrid(lon, lat)
pp, tt = lon, 90 - lat
# pp = pp[:,::-1]
# tt = tt[:,::-1]

# converting these to Slepian functions on a spherical polar coordinate system
make_Slep = sph2slep.sph2slep(Slepian_dict)
print(make_Slep.G.shape)

# mask_arr = np.zeros(121, dtype='bool')
# mask_arr[0] = True
# mask_arr[5] = True
# mask_arr[14] = True
# mask_arr[29] = True
# make_Slep.G[~mask_arr] = 0.0

# make_Slep.G = make_Slep.G[:,:,::-1]
# make_Slep.G = np.roll(make_Slep.G, (-90, -180), axis=(1,2))

Nrows, Ncols = 4, 8
fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)

for i, E_idx in enumerate(np.arange(Nrows * Ncols)):
    try:
        row, col = i//Ncols, i%Ncols
        E = Energy[time_idx, E_idx, :, :][0, 0]
        Eshell_rec_inc = np.dot(np.moveaxis(make_Slep.G[slep_idx], 0, -1), Slep_coeffs_each_shell[f'{E_idx}']['coeffs'])

        vmin, vmax = 0, Slep_coeffs_each_shell[f'{E_idx}']['vmax']
        ax[row,col].pcolormesh(pp, tt, Eshell_rec_inc, cmap='rainbow', rasterized=True, vmin=vmin, vmax=vmax)
        ax[row,col].set_xlim([90,180])
        ax[row,col].set_ylim([30,150])
        ax[row,col].text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax[row,col].transAxes,
                            va='bottom', ha='left', color='black', fontweight='bold')
        ax[row,col].set_aspect('equal')
    except: continue

fig.suptitle('Slepian reconstruction high resolution')
fig.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
fig.savefig('VDF_paper1_plots/Slepain_rec_polarcap_highres_agyro.pdf')