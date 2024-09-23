import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import cdflib, pickle
from scipy.integrate import simps
simps = np.trapz

from calculations import calc_moments

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

DATA = read_pickle('DATA_533')

# loading the reconstruction
VDF_rec = np.power(10, np.load('VDF_rec.npy')) * DATA.minval_true * 1e12
VDF_ESA, lat_ESA, lon_ESA = DATA.VDF * DATA.minval_true * 1e12, DATA.THETA, DATA.PHI
vmin, vmax = 1e1 * DATA.minval_true, 1e6 * DATA.minval_true

# VDF_rec = np.load('VDF_rec.npy')
# VDF_ESA, lat_ESA, lon_ESA = np.log10(DATA.VDF), DATA.THETA, DATA.PHI
# vmin, vmax = 1, 6


lat_hr = np.load('lat_hr.npy') * np.pi / 180
lon_hr = np.load('lon_hr.npy') * np.pi / 180


VDF_ESA = np.moveaxis(VDF_ESA, 1, 2)
lat_ESA = np.moveaxis(lat_ESA, 1, 2) * np.pi / 180
lon_ESA = np.moveaxis(lon_ESA, 1, 2) * np.pi / 180

ENERGY = np.moveaxis(DATA.ENERGY, 1, 2)
E_idx = 16


plt.figure()
# plt.pcolormesh(lon_ESA[E_idx], lat_ESA[E_idx], VDF_ESA[E_idx], vmin=vmin, vmax=vmax)
plt.pcolormesh(lon_ESA[E_idx], lat_ESA[E_idx], np.log10(VDF_ESA[E_idx]/(DATA.minval_true * 1e12)), vmin=1, vmax=6)
plt.colorbar()
plt.figure()
# plt.pcolormesh(lon_hr, lat_hr+90, VDF_rec[E_idx,::-1], vmin=vmin, vmax=vmax)
plt.pcolormesh(lon_hr, lat_hr+np.pi/2, np.log10(VDF_rec[E_idx,::-1]/(DATA.minval_true * 1e12)), vmin=1, vmax=6)
plt.colorbar()


VDF_ESA_E = simps(simps(VDF_ESA * np.sin(lat_ESA), x=lat_ESA[0,:,0], axis=1), x=lon_ESA[0,0,:], axis=1)
VDF_REC_E = simps(simps(VDF_rec * np.sin(lat_hr+np.pi/2)[np.newaxis,:,:], x=lat_hr[:,0]+np.pi/2, axis=1), x=lon_hr[0,:], axis=1)

plt.figure()
plt.loglog(ENERGY[:,0,0], VDF_ESA_E * ENERGY[:,0,0]*(13.8**2), '.-k', label='MMS Data')
plt.loglog(ENERGY[:,0,0], VDF_REC_E * ENERGY[:,0,0]*(13.8**2), '.-r', label='Reconstruction')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('')
plt.grid()

plt.figure()
plt.semilogx(ENERGY[:,0,0], VDF_ESA_E * ENERGY[:,0,0]*(13.8**2), '.-k', label='MMS Data')
plt.semilogx(ENERGY[:,0,0], VDF_REC_E * ENERGY[:,0,0]*(13.8**2), '.-r', label='Reconstruction')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('')
plt.grid()

start_idx = 14

# integrating in the energy dimension
v = np.sqrt(ENERGY[start_idx:,0,0]) * 13.85 * 1000
rho_ESA = simps(VDF_ESA_E[start_idx:] * v**2, x=v)
rho_REC = simps(VDF_REC_E[start_idx:] * v**2, x=v)
rho_calcmom,__,__ = calc_moments.trap_moments(VDF_ESA[start_idx:], v, lat_ESA[0,:,0], lon_ESA[0,0,:])

print(rho_ESA / 100**3, rho_REC / 100**3, rho_calcmom / 100**3)

# mask high discrepancy
mask1 = np.log10(VDF_ESA / VDF_rec[:,::-1,:])
# mask2 = np.log10(VDF_rec[:,::-1,:]/(DATA.minval_true * 1e12)) < 3
mask = mask1 #* mask2

plt.figure()
plt.pcolormesh(lat_hr.T+np.pi/2, lon_hr.T, mask[16].T, vmin=0, vmax=2)
plt.colorbar()