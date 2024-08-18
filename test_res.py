import numpy as np
import mat73
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import griddata

slep_idx = 10
Lmax = 20

Slep_fname = f'glmalphapto-45-{Lmax}-1.738618e+02-9.230719e+01-0_lowres.mat'
# Slep_fname = f'glmalphapto-45-{Lmax}-8.386185e+01--9.230719e+01-0.mat'
Slepian_dict = mat73.loadmat(f'./input_data_files/Slepian_functions/{Slep_fname}')
lon_lr, lat_lr = Slepian_dict['lon'], Slepian_dict['lat']

G_lr = Slepian_dict['G_slep']
# G = np.roll(Slepian_dict['G_slep'], (-16), axis=1)

# plotting the high res basis
fig, ax = plt.subplots(1, 2, figsize=(10,6))
ax[0].pcolormesh(lon_lr, lat_lr, G_lr[slep_idx], cmap='seismic')
ax[0].set_aspect('equal')

Slep_fname = f'glmalphapto-45-{Lmax}-1.738618e+02-9.230719e+01-0_highres.mat'
# Slep_fname = 'glmalphapto-45-20-8.386185e+01--9.230719e+01-0.mat'
Slepian_dict = mat73.loadmat(f'./input_data_files/Slepian_functions/{Slep_fname}')
lon_hr, lat_hr = Slepian_dict['lon'], Slepian_dict['lat']

G_hr = Slepian_dict['G_slep']
# G = np.roll(Slepian_dict['G_slep'], (180, 90), axis=(2,1))

ax[1].pcolormesh(lon_hr, lat_hr, G_hr[slep_idx], cmap='seismic')
ax[1].set_aspect('equal')

# generating random coefficients
Slep_coefs = np.random.rand(len(G_lr))

# setting some to zero randomly
rand_idx = np.random.randint(50, size=10)
Slep_coefs[rand_idx] = 0.0

# loweres image 
img_lr = np.dot(np.moveaxis(G_lr, 0, -1), Slep_coefs)
img_lr[:,17:] = np.nan
img_lr[:2,:] = np.nan
img_lr[-2:,:] = np.nan
img_lr[:,:9] = np.nan

# highres image
img_hr = np.dot(np.moveaxis(G_hr, 0, -1), Slep_coefs)

vmin, vmax = img_hr.min(), img_hr.max()

# plotting
fig, ax = plt.subplots(2, 2, figsize=(10,6))
ax[0,0].pcolormesh(lon_lr, lat_lr, img_lr, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[0,0].set_aspect('equal')
ax[0,0].set_title('Coarse-grid image from true coeffs')
ax[0,1].pcolormesh(lon_hr, lat_hr, img_hr, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[0,1].set_aspect('equal')
ax[0,1].set_title('Fine-grid image from true coeffs')
ax[0,1].axvline(158, ls='dashed', color='k')

'''
# making a downsampling
img_lr_interp = griddata((lon_hr.flatten(), lat_hr.flatten()), img_hr.flatten(), (lon_lr, lat_lr))
ax[2].pcolormesh(lon_lr, lat_lr, img_lr_interp, cmap='seismic')
ax[2].set_aspect('equal')
'''

# generating coefficients from coarse grid structure on the coarse grid Slepians
nan_mask_lr = np.isnan(img_lr)
G_nonan_lr = G_lr[:,~nan_mask_lr]
M_lr = G_nonan_lr @ G_nonan_lr.T 
__, S_lr, __ = np.linalg.svd(M_lr)
I_lr = np.identity(M_lr.shape[0])
coeffs_lr = np.linalg.inv(G_nonan_lr @ G_nonan_lr.T +  np.max(S_lr) * 1e-9 * I_lr) @ G_nonan_lr @ img_lr[~nan_mask_lr]

# generating coefficients from fine grid structure on the fine grid Slepians
nan_mask_hr = np.isnan(img_hr)
G_nonan_hr = G_hr[:,~nan_mask_hr]
M_hr = G_nonan_hr @ G_nonan_hr.T 
__, S_hr, __ = np.linalg.svd(M_hr)
I_hr = np.identity(M_hr.shape[0])
coeffs_hr = np.linalg.inv(G_nonan_hr @ G_nonan_hr.T +  np.max(S_hr) * 1e-9 * I_hr) @ G_nonan_hr @ img_hr[~nan_mask_hr]

# comparing the coefficients
ax[1,0].plot(Slep_coefs, 'ok', label='True coefs')
ax[1,0].plot(Slep_coefs, 'k', label='True coefs', alpha=0.5)
ax[1,0].plot(coeffs_lr, 'xr', label='Coarse coefs')
ax[1,0].plot(coeffs_lr, '--r', label='Coarse coefs', alpha=0.5)
ax[1,0].plot(coeffs_hr, 'xb', label='Fine coefs')
ax[1,0].plot(coeffs_hr, '-.b', label='Fine coefs', alpha=0.5)
ax[1,0].set_xlim([0, 50])
ax[1,0].legend()

# making the fine grid structure from coarse grid coefficients
coeffs_lr = np.linalg.inv(G_nonan_lr @ G_nonan_lr.T +  np.max(S_lr) * 1e-4 * I_lr) @ G_nonan_lr @ (img_lr[~nan_mask_lr] +\
                          np.random.rand(np.sum(~nan_mask_lr))*1e-2)
fine_from_coarsecoefs = np.dot(np.moveaxis(G_hr, 0, -1), coeffs_lr)

ax[1,1].pcolormesh(lon_hr, lat_hr, fine_from_coarsecoefs, cmap='seismic', vmin=vmin, vmax=vmax, rasterized=True)
ax[1,1].set_aspect('equal')
ax[1,1].set_title('Fine-grid image from coarse coeffs')
ax[1,1].axvline(158, ls='dashed', color='k')

# plt.savefig('Full_view_demo.pdf')
plt.savefig('Partial_view_demo.pdf')


