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
                    np.radians(StepII_bundle['ESA_PHI'][0,0])
    # making the Cartesian grid
    Vmag = 13.85 * np.sqrt(E)
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]
    
    return VX, VY, VZ

def grid_pol2cart_SLEP():
    E, THETA, PHI = StepII_bundle['ENERGY'][0,:,0,0], np.radians(90+StepII_bundle['SLEP_THETA'].T),\
                    np.radians(StepII_bundle['SLEP_PHI'].T)
    # making the Cartesian grid
    Vmag = 13.85 * np.sqrt(E)
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]
    
    return VX, VY, VZ

# loading the saved dictionaries
tstamp = 79
StepII_bundle = read_pickle(f'StepIIbundle_{tstamp}_SolOplot')
VDF_rec_dict = read_pickle(f'VDF3Dbundle_{tstamp}_SolOplot')
DATA_VDF = read_pickle('DATA_VDF_SolO')
REC_VDF = read_pickle('REC_VDF_SolO')
nanmask = read_pickle('nanmask_SolO')
nanmask = np.flip(nanmask, axis=1)

DATA_VDF = DATA_VDF / np.nanmax(DATA_VDF)
REC_VDF = REC_VDF / np.nanmax(REC_VDF)

# this is from the lowest value we choose in the Cartesian slice plots
# mask = np.log10(DATA_VDF) < -4
# setting the bad data to nan
DATA_VDF[nanmask] = np.nan
REC_VDF[nanmask] = np.nan

# plotting the slices in Cartesian coordinates
# getting the Cartesian coordinates for the original grid
VX, VY, VZ = grid_pol2cart_SLEP()

# plotting the cut in theta
theta_lr_slice = 4
theta_hr_slice = 19
theta_vhr_slice = 50

fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0,1].pcolormesh(VX[:,:,theta_hr_slice], VY[:,:,theta_hr_slice], np.log10(DATA_VDF / REC_VDF)[:,theta_hr_slice,:],
                   vmin=-2, vmax=2, cmap='twilight')
ax[0,1].set_xlim([-750,0])
ax[0,1].set_ylim([-350,350])

DATA_VDF = read_pickle('DATA_VDF_SolO')
REC_VDF = read_pickle('REC_VDF_SolO')

ratio = np.log10(DATA_VDF[~nanmask] / REC_VDF[~nanmask]).flatten()
ax[1,1].hist(ratio, range=(-2,2), bins=50, density=True)
# ax.set_aspect('equal')
ax[1,1].set_xlabel(r'$log_{10}(f_{\mathrm{rec}}/f_{\mathrm{SolO}})$', fontsize=18, fontweight='bold')

hist, bin_edges = np.histogram(ratio, bins=100, density=True, range=(-2,2))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * norm.pdf(x, mean, stddev)

# Fit the Gaussian to the histogram
popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(ratio), np.std(ratio)])

# Extract the fitted parameters
amplitude_fit, mean_fit, stddev_fit = popt

x = np.linspace(-2, 2, 100)
ax[1,1].plot(x, gaussian(x, amplitude_fit, mean_fit, stddev_fit))
ax[1,1].set_title(f'$\mu$ = {mean_fit:.3f}, $\sigma$ = {stddev_fit:.3f}')
ax[1,1].set_xlim([-2,2])
ax[1,1].set_yticks([])

plt.subplots_adjust(top=0.94, bottom=0.15, left=0.05, right=0.95)
plt.savefig('ratiocompare.pdf')



# loading the saved dictionaries
tstamp = 533
StepII_bundle = read_pickle(f'StepIIbundle_{tstamp}_MMSplot')
VDF_rec_dict = read_pickle(f'VDF3Dbundle_{tstamp}_MMSplot')
DATA_VDF = read_pickle('DATA_VDF_MMS')
REC_VDF = read_pickle('REC_VDF_MMS')
nanmask = np.transpose(read_pickle('nanmask_MMS'), [0,2,1])

DATA_VDF = DATA_VDF / np.nanmax(DATA_VDF)
REC_VDF = REC_VDF / np.nanmax(REC_VDF)

# this is from the lowest value we choose in the Cartesian slice plots
# mask = np.log10(DATA_VDF) < -4
# setting the bad data to nan
DATA_VDF[nanmask] = np.nan
REC_VDF[nanmask] = np.nan

# plotting the slices in Cartesian coordinates
# getting the Cartesian coordinates for the original grid
VX, VY, VZ = grid_pol2cart_SLEP()

# plotting the cut in theta
theta_lr_slice = 8

im = ax[0,0].pcolormesh(-VX[:,:,theta_lr_slice], -VY[:,:,theta_lr_slice], np.log10(DATA_VDF / REC_VDF)[:,theta_lr_slice,:],
                        vmin=-2, vmax=2, cmap='twilight')
ax[0,0].set_xlim([-1500,1500])
ax[0,0].set_ylim([-1500,1500])

DATA_VDF = read_pickle('DATA_VDF_MMS')
REC_VDF = read_pickle('REC_VDF_MMS')

DATA_NORM = DATA_VDF / np.nanmax(DATA_VDF)

# this is from the limits of the Cartesian slice plot
mask2 = np.log10(DATA_NORM) < -6

mask_total = nanmask + mask2

ratio = np.log10(DATA_VDF[~mask_total] / REC_VDF[~mask_total])
ax[1,0].hist(ratio, range=(-2,2), bins=50, density=True)
# ax.set_aspect('equal')
ax[1,0].set_xlabel(r'$log_{10}(f_{\mathrm{rec}}/f_{\mathrm{MMS}})$', fontsize=18, fontweight='bold')

hist, bin_edges = np.histogram(ratio, bins=100, density=True, range=(-2,2))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * norm.pdf(x, mean, stddev)

# Fit the Gaussian to the histogram
popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(ratio), np.std(ratio)])

# Extract the fitted parameters
amplitude_fit, mean_fit, stddev_fit = popt

x = np.linspace(-2, 2, 100)
ax[1,0].plot(x, gaussian(x, amplitude_fit, mean_fit, stddev_fit))
ax[1,0].set_title(f'$\mu$ = {mean_fit:.3f}, $\sigma$ = {stddev_fit:.3f}')
ax[1,0].set_xlim([-2,2])
ax[1,0].set_yticks([])

# Draw a horizontal lines at those coordinates
arrow_len = 0.12

p1a = patches.FancyArrowPatch((0.09, 0.96), (0.09+arrow_len*0.8,0.96), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
text1 = plt.text(s='MMS-FPI', x=0.25, y=0.95, transform=fig.transFigure)
p1b = patches.FancyArrowPatch((0.5-arrow_len*0.85,0.96), (0.5,0.96), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
fig.add_artist(p1a)
fig.add_artist(text1)
fig.add_artist(p1b)

p1a = patches.FancyArrowPatch((0.56, 0.96), (0.56+arrow_len*0.85,0.96), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
text1 = plt.text(s='SolO-PAS', x=0.72, y=0.95, transform=fig.transFigure)
p1b = patches.FancyArrowPatch((0.96-arrow_len*0.85,0.96), (0.96,0.96), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
fig.add_artist(p1a)
fig.add_artist(text1)
fig.add_artist(p1b)

fig.text(0.5, 0.51, r'$V_x$ [km/s]', ha='center')
fig.text(0.001, 0.75, r'$V_y$ [km/s]', va='center', rotation='vertical')

fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.952, 0.68, 0.01, 0.15])
# cbar_ax.text(0.5, 1.15, r'$log_{10}\left(\frac{f_{\rm{rec}}}{f_{\rm{obs}}}\right)$', ha='center', va='top', transform=cbar_ax.transAxes)
fig.colorbar(im, cax=cbar_ax)

ax[0,0].text(0.86, 0.92, f'(A1)', transform=ax[0,0].transAxes,
             va='bottom', ha='left', color='black', fontweight='bold')
ax[1,0].text(0.86, 0.92, f'(A2)', transform=ax[1,0].transAxes,
             va='bottom', ha='left', color='black', fontweight='bold')
ax[0,1].text(0.86, 0.92, f'(B1)', transform=ax[0,1].transAxes,
             va='bottom', ha='left', color='black', fontweight='bold')
ax[1,1].text(0.86, 0.92, f'(B2)', transform=ax[1,1].transAxes,
             va='bottom', ha='left', color='black', fontweight='bold')

t = ax[0,0].text(0.03, 0.05, r'$log_{10}\left(\frac{f_{\rm{rec}}}{f_{\rm{MMS}}}\right)$', transform=ax[0,0].transAxes,
                 va='bottom', ha='left', color='black', fontweight='bold')
t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
t = ax[0,1].text(0.03, 0.05, r'$log_{10}\left(\frac{f_{\rm{rec}}}{f_{\rm{SolO}}}\right)$', transform=ax[0,1].transAxes,
                 va='bottom', ha='left', color='black', fontweight='bold')
t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))

plt.subplots_adjust(top=0.94, bottom=0.08, left=0.1, right=0.94, hspace=0.3)
plt.savefig('ratiocompare.pdf')