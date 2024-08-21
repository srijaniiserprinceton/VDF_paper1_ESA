import numpy as np
import mat73
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import griddata

#---------------operating Matlab from within Python------------------#
import matlab.engine as matlab
eng = matlab.start_matlab()
s = eng.genpath('/Users/srijanbharatidas/Documents/Research/Codes/Helioseismology/Slepians/Slepian_Git')
eng.addpath(s, nargout=0)

# adding custom package files
import generate_2D_contour as gen_contour

#----------------USER SPECIFIED PARAMETES/VARIABLES--------------------#
N = 30                  # effective Shannon number input to localization2D.m
Vmin_shell = 350        # in km/s
rcond = 1e-4            # condition number of the matrix to be inverted (Moore-Penrose Inverse)

#--------------PREVIOUS METHOD OF GOING BACK AND FORTH FROM MATLAB TO PYTHON TO GENERATE SLEPIANS----------------#
'''
N=50
Slep_fname = f'localization2D-{N}.mat'
Slepian_dict = mat73.loadmat(f'../../../Codes/Helioseismology/Slepians/Slepian_Git/IFILES/LOCALIZATION2D/{Slep_fname}')

# loading the components of Slepian dictionary
G, H, V, K, XYP, XY = Slepian_dict['G'], Slepian_dict['H'], Slepian_dict['V'],\
                      Slepian_dict['K'], Slepian_dict['XYP'], Slepian_dict['XY']
'''

#--------------CURRENT METHOD TO GENERATE SLEPIANS BY RUNING MATLAB FROM WITHIN PYTHON----------------#
N = 30
[G, H, V, K, XYP, XY] = eng.localization2D('demo_VDF', N, nargout=6)
G = np.asarray(G)
H = np.asarray(H)
V = np.asarray(V)
K = np.asarray(K)
XYP = np.asarray(XYP)
XY = np.asarray(XY)

# G = H * 1.0

# locating the Shannon number
Nshannon = np.argmin(np.abs(V-0.5))

# converting XYP from a flattened array into a meshgrid
NX, NY, Nslep = G.shape
XX, YY =  np.reshape(XYP[:,0], (NX,NY), 'F'), np.reshape(XYP[:,1], (NX, NY), 'F')

# plotting limits
xmin, xmax = XX.min(), XX.max()
ymin, ymax = YY.min(), YY.max()

# loading the data generated in the gyrotropization step
VDF_2D = np.load('output_data_files/VDF_2D.npy')
V1, V2 = np.load('output_data_files/V1.npy'), np.load('output_data_files/V2.npy')

# plotting the 2D raw VDF from gyrotropization step
fig, ax = plt.subplots(2, 2, figsize=(8, 9), sharex=True, sharey=True)
# ax[0,0].contourf(V1, V2, VDF_2D, vmin=0, vmax=6, levels=15)
ax[0,0].pcolormesh(V1, V2, VDF_2D, vmin=0, vmax=6, cmap='hot', rasterized=True)
ax[0,0].contour(V1, V2, VDF_2D, levels=10, cmap='hot')
ax[0,0].plot(XY[:,0], XY[:,1], '--k')
ax[0,0].axhline(0, color='white', ls='dashed')
ax[0,0].set_xlim([xmin, xmax])
ax[0,0].set_ylim([ymin, ymax])
ax[0,0].set_aspect('equal')
ax[0,0].set_title('VDF 2D raw')

# interpolating the 2D raw VDF into the Slepian domain
VDF_interp = griddata((np.ravel(V1,'F'), np.ravel(V2,'F')), np.ravel(VDF_2D, 'F'), (XX, YY), method='linear')

# # setting shells under 350 km/s to nan
nan_mask_data = np.sqrt(XX**2 + YY**2) < Vmin_shell
VDF_interp[nan_mask_data] = np.nan

# plotting the interpolated VDF
ax[0,1].pcolormesh(XX, YY, VDF_interp, vmin=0, vmax=6, cmap='hot', rasterized=True)
ax[0,1].contour(XX, YY, VDF_interp, levels=10, cmap='hot')
ax[0,1].plot(XY[:,0], XY[:,1], '--k')
ax[0,1].axhline(0, color='white', ls='dashed')
ax[0,1].set_xlim([xmin, xmax])
ax[0,1].set_ylim([ymin, ymax])
ax[0,1].set_aspect('equal')
ax[0,1].set_title('VDF 2D interpolated')

# reconstructing the interpolated VDF in Cartesian Slepians
nan_mask = np.isnan(VDF_interp)
G_nonan = G[~nan_mask,:]
M = G_nonan.T @ G_nonan
__, S, __ = np.linalg.svd(M)
I = np.identity(M.shape[0])
coeffs = np.linalg.inv(M +  S.max() * rcond * I) @ G_nonan.T @ VDF_interp[~nan_mask]
VDF_Sleprec = np.dot(G, coeffs)

# plotting the reconstructed distribution
ax[1,0].pcolormesh(XX, YY, VDF_Sleprec, vmin=0, vmax=6, cmap='hot', rasterized=True)
ax[1,0].contour(XX, YY, VDF_Sleprec, levels=10, cmap='hot')
ax[1,0].plot(XY[:,0], XY[:,1], '--k')
ax[1,0].axhline(0, color='white', ls='dashed')
ax[1,0].set_xlim([xmin, xmax])
ax[1,0].set_ylim([ymin, ymax])
ax[1,0].set_aspect('equal')
ax[1,0].set_title('Cartesian Slepian reconstruction')

# plotting the reconstructed distribution vs. the raw 2D VDF from gyrotropization
ax[1,1].pcolormesh(V1, V2, VDF_2D, vmin=0, vmax=6, cmap='hot', rasterized=True)
ax[1,1].pcolormesh(XX[81:], YY[81:], VDF_Sleprec[81:], vmin=0, vmax=6, cmap='hot', rasterized=True)
ax[1,1].set_xlim([xmin, xmax])
ax[1,1].set_ylim([ymin, ymax])
ax[1,1].set_aspect('equal')
ax[1,1].set_title('2D raw vs. Reconstructed')

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.97)