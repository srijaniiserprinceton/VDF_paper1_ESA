import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.patches as patches
plt.rcParams.update({'font.size': 14})
import pickle, cdflib
from datetime import datetime
from tqdm import tqdm

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

# making the first set of plots (energy shell comparison)
fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(7,6))

vmin, vmax = -1, 1
cmap = 'coolwarm'

count = 1

for i, E_idx in enumerate(range(39,45)):
    # plotting the data
    row, col = i%3, i//3

    # note that for FPI, theta array stays the same at different times but phi array changes
    tt_orig, pp_orig, vv = StepII_bundle['ESA_THETA'][tstamp, E_idx, :, :],\
                           StepII_bundle['ESA_PHI'][tstamp, E_idx, :, :],\
                           StepII_bundle['VDF'][tstamp, E_idx, :, :]

    # calculating the log and setting +\- inf to nan
    logvv = np.log10(vv)
    # logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)
    logvv[logvv==0.0] = np.nan
    # tiling the SolO observations to match the reconstruction in all space
    logvv_tiled = np.zeros_like(StepII_bundle['fine_from_fine'][E_idx]) + np.nan
    logvv_tiled[15:24,26:37] = logvv.T
    logvv_tiled = np.flip(logvv_tiled, axis=0)

    E = StepII_bundle['ENERGY'][tstamp, E_idx, 0, 0]
    ratio = (logvv_tiled / StepII_bundle['fine_from_fine'][E_idx]) -1
    im = ax[row,col].pcolormesh(StepII_bundle['SLEP_PHI'], StepII_bundle['SLEP_THETA'], ratio, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
    ax[row,col].set_aspect('equal')
    ax[row,col].set_xlim([140, 230])
    ax[row,col].set_ylim([-30, 30])
    t = ax[row,col].text(0.02, 0.92, f'(B{count}) {E:.2f} [eV]', transform=ax[row,col].transAxes,
                     va='bottom', ha='left', color='black', fontsize=10, fontweight='bold')
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
    count += 1

fig.text(0.5, 0.04, r'Azimuth ($\phi$)', ha='center')
fig.text(0.001, 0.5, r'Elevation ($\theta$)', va='center', rotation='vertical')

plt.subplots_adjust(left=0.07, right=0.91, hspace=0.02, wspace=0.01, top=0.95)

fig.subplots_adjust(right=0.91)
cbar_ax = fig.add_axes([0.89, 0.15, 0.01, 0.75])
fig.colorbar(im, cax=cbar_ax)

plt.savefig('Eshell_ratiocompare_SolO.pdf')

# plotting the slices in Cartesian coordinates
# getting the Cartesian coordinates for the original grid
VX, VY, VZ = grid_pol2cart()

fig, ax = plt.subplots(1, 3, figsize=(12,4.5), sharex=True, sharey=True)
vv = StepII_bundle['VDF']
logvv = np.nan_to_num(np.log10(vv), nan=np.nan, posinf=np.nan, neginf=np.nan)
logvv[logvv==0] = np.nan

# plotting the cut in theta
theta_lr_slice = 4
theta_hr_slice = 19

im = ax[0].pcolormesh(VX[:,:,theta_lr_slice], VY[:,:,theta_lr_slice], logvv[tstamp, :, :, theta_lr_slice],
                 vmin=0, vmax=3, cmap='inferno', rasterized=True)
ax[0].set_xlim([-750,0])
ax[0].set_ylim([-350,350])
ax[0].set_aspect('equal')
ax[0].set_title(r'$f_{\mathrm{SolO}}$', fontsize=18, fontweight='bold')

VX, VY, VZ = grid_pol2cart_SLEP()

ax[1].set_facecolor('black')
ax[1].pcolormesh(VX[:,:,theta_hr_slice], VY[:,:,theta_hr_slice], StepII_bundle['fine_from_fine'][:,theta_hr_slice,:],
                 vmin=0, vmax=3, cmap='inferno', rasterized=True)
ax[1].set_xlim([-750,0])
ax[1].set_ylim([-350,350])
ax[1].set_aspect('equal')
ax[1].set_title(r'$f_{\mathrm{rec}}$', fontsize=18, fontweight='bold')

logvv_tiled = np.zeros_like(StepII_bundle['fine_from_fine'][:,theta_lr_slice,:]) + np.nan
logvv_tiled[:,26:37] = logvv[tstamp, :, :, theta_lr_slice]

ratio = np.abs(logvv_tiled - StepII_bundle['fine_from_fine'][:,theta_hr_slice,:])
# ax[2].set_facecolor('black')
ax[2].pcolormesh(VX[:,:,theta_hr_slice], VY[:,:,theta_hr_slice], ratio,
               vmin=0, vmax=3, cmap='inferno', rasterized=True)
ax[2].set_xlim([-750,0])
ax[2].set_ylim([-350,350])
ax[2].set_aspect('equal')
ax[2].set_title(r'$|f_{\mathrm{SolO}} - f_{\mathrm{rec}}|$', fontsize=18, fontweight='bold')

fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.955, 0.15, 0.01, 0.75])
fig.colorbar(im, cax=cbar_ax)

plt.subplots_adjust(left=0.07, right=0.95, hspace=0.05, wspace=0.05, top=0.95, bottom=0.1)

fig.text(0.55, 0.04, r'$V_x$ [km/s]', ha='center')
fig.text(0.001, 0.5, r'$V_y$ [km/s]', va='center', rotation='vertical')

plt.savefig('Cart_slice_ratiocompare_SolO.pdf')