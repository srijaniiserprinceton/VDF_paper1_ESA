import numpy as np
import mat73
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import griddata

N=50
Slep_fname = f'localization2D-{N}.mat'
Slepian_dict = mat73.loadmat(f'../../../Codes/Helioseismology/Slepians/Slepian_Git/IFILES/LOCALIZATION2D/{Slep_fname}')

# loading the components of Slepian dictionary
G, H, V, K, XYP, XY = Slepian_dict['G'], Slepian_dict['H'], Slepian_dict['V'],\
                      Slepian_dict['K'], Slepian_dict['XYP'], Slepian_dict['XY']

# G = H * 1.0

# locating the Shannon number
Nshannon = np.argmin(np.abs(V-0.5))

# truncating the G to Shannon number
# G = G[:,:,:Nshannon]

# converting XYP from a flattened array into a meshgrid
NX, NY, Nslep = G.shape
YY, XX = np.reshape(XYP[:,1], (NX, NY), 'F'), np.reshape(XYP[:,0], (NX,NY), 'F')

# plot limits
xmin, xmax = XX.min(), XX.max()
ymin, ymax = YY.min(), YY.max()

plt.figure()
plt.pcolormesh(XX, YY, G[:,:,1])
plt.plot(XY[:,0], XY[:,1], '--k')
plt.gca().set_aspect('equal')

# loading the data
VDF_2D = np.load('output_data_files/VDF_2D.npy')
V1, V2 = np.load('output_data_files/V1.npy'), np.load('output_data_files/V2.npy')

# igoring upto 6th energy shell
# VDF_2D[:9,:] = np.nan
VDF_2D = VDF_2D[9:,:]
V1, V2 = V1[9:,:], V2[9:,:]

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
# ax[0].contourf(V1, V2, VDF_2D, vmin=0, vmax=6, levels=15)
ax[0].pcolormesh(V1, V2, VDF_2D, vmin=0, vmax=6)
ax[0].plot(XY[:,0], XY[:,1], '--k')
ax[0].set_xlim([xmin, xmax])
ax[0].set_ylim([ymin, ymax])
ax[0].set_aspect('equal')
ax[0].set_title('VDF 2D raw')

# interpolating the 2D raw VDF into the Slepian domain
VDF_interp = griddata((np.ravel(V1,'F'), np.ravel(V2,'F')), np.ravel(VDF_2D, 'F'), (XX, YY), method='nearest')

# setting shells under 313 to nan
nan_mask_data = np.sqrt(XX**2 + YY**2) < 314
VDF_interp[nan_mask_data] = np.nan

# plotting the interpolated VDF
# ax[1].contourf(XX, YY, VDF_interp, vmin=0, vmax=6, levels=15)
ax[1].pcolormesh(XX, YY, VDF_interp, vmin=0, vmax=6)
ax[1].plot(XY[:,0], XY[:,1], '--k')
ax[1].set_xlim([xmin, xmax])
ax[1].set_ylim([ymin, ymax])
ax[1].set_aspect('equal')
ax[1].set_title('VDF 2D interpolated')

# reconstructing the interpolated VDF in Cartesian Slepians
nan_mask = np.isnan(VDF_interp)
G_nonan = G[~nan_mask,:]
M = G_nonan.T @ G_nonan
__, S, __ = np.linalg.svd(M)
I = np.identity(M.shape[0])
coeffs = np.linalg.inv(M +  0.0 * I) @ G_nonan.T @ VDF_interp[~nan_mask]
VDF_Sleprec = np.dot(G, coeffs)


# plotting the reconstructed distribution
# ax[2].contourf(XX, YY, VDF_Sleprec, vmin=0, vmax=6, levels=15)
ax[2].pcolormesh(XX, YY, VDF_Sleprec, vmin=0, vmax=6)
ax[2].contour(XX, YY, VDF_Sleprec, levels=10, color='k')
ax[2].plot(XY[:,0], XY[:,1], '--k')
ax[2].set_xlim([xmin, xmax])
ax[2].set_ylim([ymin, ymax])
ax[2].set_aspect('equal')
ax[2].set_title('Cartesian Slepian reconstruction')