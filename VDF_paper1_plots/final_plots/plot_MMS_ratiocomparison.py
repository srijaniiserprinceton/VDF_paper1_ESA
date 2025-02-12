import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.patches as patches
plt.rcParams.update({'font.size': 14})
import pickle, cdflib
from datetime import datetime
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import curve_fit

NAX = np.newaxis
func = np.vectorize(datetime.utcfromtimestamp)

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def grid_pol2cart():
    E, THETA, PHI = StepII_bundle['ENERGY'][0,:,0,0], np.radians(StepII_bundle['ESA_THETA'][0,0]),\
                    np.radians(StepII_bundle['ESA_PHI'][0,0] + 180)
    # making the Cartesian grid
    Vmag = 13.85 * np.sqrt(E)
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]
    
    return VX, VY, VZ


# loading the saved dictionaries
tstamp = 533
StepII_bundle = read_pickle(f'StepIIbundle_{tstamp}_MMSplot')
VDF_rec_dict = read_pickle(f'VDF3Dbundle_{tstamp}_MMSplot')
VDF_minval = np.load('VDF_minval_true_MMS.npy') * StepII_bundle['ENERGY'][:,:,0,0]**2 

# finding the global minval
VDF_minval_global = np.min(VDF_minval[tstamp][VDF_minval[tstamp] > 0])

# making the first set of plots (energy shell comparison)
fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8.5,6))

vmin, vmax = -1, 1
cmap = 'coolwarm'

count = 1

for i, E_idx in enumerate(range(16,22)):
    # plotting the data
    row, col = i%3, i//3

    # note that for FPI, theta array stays the same at different times but phi array changes
    tt_orig, pp_orig, vv = StepII_bundle['ESA_THETA'][tstamp, E_idx, :, :],\
                           StepII_bundle['ESA_PHI'][tstamp, E_idx, :, :],\
                           np.roll(StepII_bundle['VDF'][tstamp, E_idx, :, :], 16, axis=1)

    # calculating the log and setting +\- inf to nan
    # logvv = np.log10(vv)
    # logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)
    # logvv[logvv==0.0] = np.nan

    vv = StepII_bundle['VDF'] / StepII_bundle['ENERGY'][:, :, :, :]**2
    vv_norm = vv/np.nanmax(vv[tstamp])
    vv_norm[StepII_bundle['VDF'] == 1.0] = np.nan

    REC_VDF = np.power(10, StepII_bundle['smooth_vdf'])
    REC_VDF = REC_VDF/np.nanmax(vv[tstamp])

    E = StepII_bundle['ENERGY'][tstamp, E_idx, 0, 0]
    ratio = (np.log10(vv_norm)[tstamp,E_idx] - np.log10(REC_VDF)[E_idx].T) / (np.log10(vv_norm)[tstamp,E_idx])
    im = ax[row,col].pcolormesh(pp_orig, 90 - tt_orig, ratio, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
    ax[row,col].set_aspect('equal')
    ax[row,col].set_xlim([0, 360])
    t = ax[row,col].text(0.02, 0.92, f'(A{count}) {E:.2f} [eV]', transform=ax[row,col].transAxes,
                     va='bottom', ha='left', color='black', fontsize=10, fontweight='bold')
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
    count += 1

fig.text(0.5, 0.04, r'Azimuth ($\phi$)', ha='center')
fig.text(0.001, 0.5, r'Elevation ($\theta$)', va='center', rotation='vertical')

plt.subplots_adjust(left=0.07, right=0.91, hspace=0.06, wspace=0.01, top=0.95)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.75])
fig.colorbar(im, cax=cbar_ax)

plt.savefig('Eshell_ratiocompare_MMS.pdf')

# plotting the slices in Cartesian coordinates
# getting the Cartesian coordinates for the original grid
VX, VY, VZ = grid_pol2cart()

# fig, ax = plt.subplots(1, 3, figsize=(12,4.5), sharex=True, sharey=True)
# vv = StepII_bundle['VDF']
# nanmask = vv == 1
# vv = vv * VDF_minval[:, :, np.newaxis, np.newaxis]
# vv = vv / VDF_minval_global
# logvv = np.nan_to_num(np.log10(vv), nan=np.nan, posinf=np.nan, neginf=np.nan)
# logvv[nanmask] = np.nan

vv = StepII_bundle['VDF'] / StepII_bundle['ENERGY'][:, :, :, :]**2
vv_norm = vv/np.nanmax(vv[tstamp])
vv_norm = np.transpose(vv_norm, [0, 1, 3, 2])
# logvv = np.nan_to_num(np.log10(vv_norm), nan=np.nan, posinf=np.nan, neginf=np.nan)
# logvv[StepII_bundle['VDF'] == 1.0] = np.nan

REC_VDF = np.power(10, StepII_bundle['smooth_vdf'])
REC_VDF = REC_VDF/np.nanmax(vv[tstamp])

# performing the same scaling for the reconstructed VDF
tenpow_StepII_bundle = np.power(10, StepII_bundle['fine_from_fine']) \
                                * VDF_minval[tstamp, :, np.newaxis, np.newaxis]
StepII_bundle['fine_from_fine'] = np.log10(tenpow_StepII_bundle/ VDF_minval_global)

# plotting the cut in theta
theta_lr_slice = 8
theta_hr_slice = 53

'''
im = ax[0].pcolormesh(VX[:,:,theta_lr_slice], VY[:,:,theta_lr_slice], logvv[tstamp, :, :, theta_lr_slice],
                 vmin=-7, vmax=0, cmap='inferno', rasterized=True)
ax[0].set_xlim([-1500,1500])
ax[0].set_ylim([-1500,1500])
ax[0].set_aspect('equal')
ax[0].set_title(r'$f_{\mathrm{MMS}}$', fontsize=18, fontweight='bold')

REC_VDF = np.power(10, StepII_bundle['smooth_vdf'])
REC_VDF = REC_VDF/np.nanmax(vv[tstamp])

ax[1].set_facecolor('black')
ax[1].pcolormesh(VX[:,:,theta_lr_slice], VY[:,:,theta_lr_slice], np.log10(REC_VDF)[:,theta_lr_slice,:],
                 vmin=-7, vmax=0, cmap='inferno', rasterized=True)
ax[1].set_xlim([-1500,1500])
ax[1].set_ylim([-1500,1500])
ax[1].set_aspect('equal')
ax[1].set_title(r'$f_{\mathrm{rec}}$', fontsize=18, fontweight='bold')

'''
fig, ax = plt.subplots(1, 1, figsize=(5,4.5), sharex=True, sharey=True)
# ratio = -1.0 * np.abs(logvv[tstamp, :, :, theta_lr_slice] - np.log10(REC_VDF)[:,theta_lr_slice,:])
nanmask = StepII_bundle['VDF'][tstamp] == 1.0
nanmask = np.transpose(nanmask, [0, 2, 1])
ratio = np.log10(REC_VDF[~nanmask] / vv_norm[tstamp,~nanmask])
# ax[2].set_facecolor('black')
ax.hist(ratio, range=(-2,2), bins=85, density=True)
# ax.set_aspect('equal')
ax.set_xlabel(r'$log_{10}(f_{\mathrm{rec}}/f_{\mathrm{MMS}})$', fontsize=18, fontweight='bold')

hist, bin_edges = np.histogram(ratio, bins=85, density=True, range=(-2,2))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * norm.pdf(x, mean, stddev)

# Fit the Gaussian to the histogram
popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(ratio), np.std(ratio)])

# Extract the fitted parameters
amplitude_fit, mean_fit, stddev_fit = popt

x = np.sort(ratio)
ax.plot(x, gaussian(x, amplitude_fit, mean_fit, stddev_fit))
ax.set_title(f'$\mu$ = {mean_fit:.3f}, $\sigma$ = {stddev_fit:.3f}')
ax.set_xlim([-2,2])
ax.set_yticks([])

plt.subplots_adjust(top=0.94, bottom=0.15, left=0.05, right=0.95)
plt.savefig('MMS_ratiocompare.pdf')