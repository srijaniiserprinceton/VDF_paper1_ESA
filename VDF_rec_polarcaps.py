import numpy as np
import mat73, cdflib
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage
from scipy.io import savemat

from source_scripts import fit_2D_gaussian
import generate_2D_contour as gen_contour

slep_idx = 3
Lmax = 12

#-------------reading the low resolution Slepian basis functions------------------#
Slep_fname = f'glmalphapto-45-{Lmax}-1.738618e+02-9.230719e+01-0_lowres.mat'
# Slep_fname = f'glmalphapto-45-{Lmax}-8.386185e+01--9.230719e+01-0.mat'
Slepian_dict = mat73.loadmat(f'./input_data_files/Slepian_functions/{Slep_fname}')
lon_lr, lat_lr = Slepian_dict['lon'], Slepian_dict['lat']
G_lr = Slepian_dict['G_slep']

# plotting the low resolution basis
fig, ax = plt.subplots(1, 2, figsize=(10,6))
ax[0].pcolormesh(lon_lr, lat_lr, G_lr[slep_idx], cmap='seismic')
ax[0].set_aspect('equal')

#-------------reading the high resolution Slepian basis functions-----------------#
Slep_fname = f'glmalphapto-45-{Lmax}-1.738618e+02-9.230719e+01-0_highres.mat'
# Slep_fname = 'glmalphapto-45-20-8.386185e+01--9.230719e+01-0.mat'
Slepian_dict = mat73.loadmat(f'./input_data_files/Slepian_functions/{Slep_fname}')
lon_hr, lat_hr = Slepian_dict['lon'], Slepian_dict['lat']
G_hr = Slepian_dict['G_slep']

# plotting the high resolution basis
ax[1].pcolormesh(lon_hr, lat_hr, G_hr[slep_idx], cmap='seismic')
ax[1].set_aspect('equal')

#--------------reading the VDF data for a certain time interval---------------------#
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
zeros_mask = VDF == 0.0
VDF = VDF / np.nanmin(VDF[~zeros_mask])
VDF[zeros_mask] = 1e0
E_idx = 15

time_idx = 7301

vv = VDF[time_idx, E_idx, :, :] 
data_vv = np.log10(vv)
data = np.zeros((13, 33)) + np.nan
# tiling the SPAN-Ai data in the correct location
data[2:10, 8:16] = data_vv.T
img_lr = data

# # removing the first anode
img_lr[:, 15] = np.nan

# interpolating low res image
pp, tt = np.meshgrid(Phi[time_idx, E_idx, 0, :], Theta[time_idx, E_idx, :, 0])
# img_lr = griddata((pp.flatten(), tt.flatten()), data.flatten(), (lon_lr, lat_lr))

vmin, vmax = np.nanmin(img_lr), np.nanmax(img_lr)

# plotting
fig, ax = plt.subplots(2, 2, figsize=(10,6))
ax[0,0].pcolormesh(lon_lr, lat_lr, img_lr, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[0,0].set_aspect('equal')
ax[0,0].set_title('Coarse-grid image from true coeffs')

# generating coefficients from coarse grid structure on the coarse grid Slepians
nan_mask_lr = np.isnan(img_lr)
G_nonan_lr = G_lr[:,~nan_mask_lr]
M_lr = G_nonan_lr @ G_nonan_lr.T 
__, S_lr, __ = np.linalg.svd(M_lr)
I_lr = np.identity(M_lr.shape[0])
coeffs_lr_16 = np.linalg.inv(G_nonan_lr @ G_nonan_lr.T +  1e-12 * I_lr) @ G_nonan_lr @ img_lr[~nan_mask_lr]
coeffs_lr_1 = np.linalg.inv(G_nonan_lr @ G_nonan_lr.T +  1e-13 * I_lr) @ G_nonan_lr @ img_lr[~nan_mask_lr]

img_from_coeffs_lr_16 = np.dot(np.moveaxis(G_lr, 0, -1), coeffs_lr_16)
# vmin, vmax = np.nanmin(img_lr), np.nanmax(img_lr)
ax[0,1].pcolormesh(lon_lr, lat_lr, img_from_coeffs_lr_16, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[0,1].set_aspect('equal')

img_from_coeffs_lr_1 = np.dot(np.moveaxis(G_lr, 0, -1), coeffs_lr_1)
# vmin, vmax = np.nanmin(img_lr), np.nanmax(img_lr)
ax[1,0].pcolormesh(lon_lr, lat_lr, img_from_coeffs_lr_1, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[1,0].set_aspect('equal')

coeffs_lr = coeffs_lr_1

# making the fine grid structure from coarse grid coefficients
fine_from_coarsecoefs = np.dot(np.moveaxis(G_hr, 0, -1), coeffs_lr)
# vmin, vmax = 0, np.nanmax(fine_from_coarsecoefs)
ax[1,1].pcolormesh(lon_hr, lat_hr, fine_from_coarsecoefs, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[1,1].set_aspect('equal')
ax[1,1].set_title('Fine-grid image from coarse coeffs')
# ax[1,1].axvline(158, ls='dashed', color='k')

plt.savefig('Full_view_demo.pdf')
# plt.savefig('Partial_view_demo.pdf')

# decomposing in high res Slepians
# retaining only the axisymetric Slepians
gyro_mask = np.zeros(len(G_hr), dtype='bool')
gyro_mask[0] = True
gyro_mask[5] = True
gyro_mask[14] = True
gyro_mask[29] = True
G_hr = G_hr[gyro_mask]

Ntheta_lr, Nphi_lr = data.shape
tt_lr_idx, pp_lr_idx = np.meshgrid(np.linspace(0, 180, Ntheta_lr), np.linspace(0, 360, Nphi_lr), indexing='ij')
tt_hr_idx, pp_hr_idx = np.meshgrid(np.linspace(0, 180, 181), np.linspace(0, 360, 361), indexing='ij')

VDF_2D = np.zeros((32, 181))

fig, ax = plt.subplots(4, 8, figsize=(16,8), sharex=True, sharey=True)

# looping over energy shells -> fitting polar Slepians -> plotting the reconstructed energy shell VDFs
for E_idx in range(0, 32):
    E = Energy[time_idx, E_idx, 0, 0]
    vv = VDF[time_idx, E_idx, :, :] 
    data_vv = np.log10(vv)
    data = np.zeros((13, 33)) + np.nan
    # tiling the SPAN-Ai data in the correct location
    data[2:10, 8:16] = data_vv.T

    # interpolating the data to higher resolution before fitting polar Slepians
    img_hr = griddata((tt_lr_idx.flatten(), pp_lr_idx.flatten()), data.flatten(), (tt_hr_idx, pp_hr_idx), method='linear')

    # fitting the polar Slepians
    nan_mask_hr = np.isnan(img_hr)
    G_nonan_hr = G_hr[:,~nan_mask_hr]
    M_hr = G_nonan_hr @ G_nonan_hr.T 
    # __, S_hr, __ = np.linalg.svd(M_hr)
    I_hr = np.identity(M_hr.shape[0])
    coeffs_hr = np.linalg.inv(G_nonan_hr @ G_nonan_hr.T +  0.0 * I_hr) @ G_nonan_hr @ img_hr[~nan_mask_hr]

    # reconstructing from the polar Slepians and plotting
    fine_from_finecoefs = np.dot(np.moveaxis(G_hr, 0, -1), coeffs_hr)
    ax[E_idx//8, E_idx%8].pcolormesh(lon_hr, lat_hr, fine_from_finecoefs, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
    ax[E_idx//8, E_idx%8].set_aspect('equal')
    ax[E_idx//8, E_idx%8].set_xlim([75, 275])
    ax[E_idx//8, E_idx%8].text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax[E_idx//8, E_idx%8].transAxes,
                               va='bottom', ha='left', color='white', fontweight='bold')


    # saving the 2D VDF
    VDF_2D[E_idx] = fine_from_finecoefs[:,173]


plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# converting grids to velocity space
m_p = 0.010438870      #eV/c^2 where c = 299792 km/s
q_p = 1 
vmag = np.sqrt(2 * q_p * Energy[time_idx, :, 0, 0] / m_p)   # in km/s

theta_hr = lat_hr[:,0]
V1 = vmag[:, np.newaxis] * np.cos(theta_hr[np.newaxis,:] * np.pi/180)
V2 = vmag[:, np.newaxis] * np.sin(theta_hr[np.newaxis:,] * np.pi/180)

fig, ax = plt.subplots(1,1)
ax.contourf(V1, V2, VDF_2D, cmap='gnuplot2', vmin=-1e-3, vmax=6)
ax.set_aspect('equal')

# applying Gaussian filter on the 2D VDF
VDF_2D_GF = ndimage.gaussian_filter(VDF_2D, sigma=5.0, order=0)

# method I of getting the contour 
'''
plt.figure()
img = plt.contourf(V1, V2, VDF_2D, cmap='gnuplot2', vmin=-1e-3, vmax=6, levels=[1.0, 6.0])
plt.close()
p = img.collections[0].get_paths()[0]
v = p.vertices
x = v[:,0]
y = v[:,1]
'''

# method II of getting the contour
'''
plt.figure()
# img = plt.contourf(V1, V2, VDF_2D_GF, cmap='gnuplot2', vmin=-1e-6, vmax=6, levels=[1.0, 6.0])
img = plt.contourf(V1, V2, VDF_2D, cmap='gnuplot2', vmin=-1e-6, vmax=6, levels=[1.0, 6.0])
plt.close()
p = img.collections[0].get_paths()[0]
v = p.vertices
x = v[:,0]
y = v[:,1]
'''

# method III of getting the contour

# setting up the grid and interpolating to compare the fitting with
x, y = np.ravel(V1, 'F'), np.ravel(V2, 'F')
z = np.ravel(VDF_2D, 'F')

X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                   np.linspace(y.min(), y.max(), 100))
Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

#  setting up the model and indicating independent variables
gencontourdemo = gen_contour.gen_contour(z, x, y, 'Gaussian_ycentered')


# fitting the model with the data
result = gencontourdemo.fit_2D_VDF()
fit = gencontourdemo.model.func(X, Y, **result.best_values)

plt.figure()
img = plt.contourf(X, Y, fit, cmap='gnuplot2', vmin=-1e-3, vmax=6, levels=[1.0, 6.0])
plt.close()
p = img.collections[0].get_paths()[0]
v = p.vertices
x = v[:,0]
y = v[:,1]

ax.plot(x, y, color='white')

# storing the contour
curve2storeXY = {'X': v.T[0], 'Y': v.T[1]}
savemat('XY_pts.mat', curve2storeXY)

# storing the evaluation points
evalpts2storeXY = {'XP': np.ravel(V1,'F'), 'YP': np.ravel(V2,'F')}
savemat('XYP.mat', evalpts2storeXY)

# saving the 2D VDF and domain points
np.save('output_data_files/VDF_2D.npy', VDF_2D)
np.save('output_data_files/V1.npy', V1)
np.save('output_data_files/V2.npy', V2)

