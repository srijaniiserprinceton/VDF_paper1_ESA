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


# loading the saved dictionaries
tstamp = 533
StepII_bundle = read_pickle(f'StepIIbundle_{tstamp}_MMSplot')
VDF_rec_dict = read_pickle(f'VDF3Dbundle_{tstamp}_MMSplot')

# making the first set of plots (energy shell comparison)
fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12,6))

vmin, vmax = 1, 7
cmap = 'inferno'

count = 1

for i, E_idx in enumerate(range(16,22)):
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


    E = StepII_bundle['ENERGY'][tstamp, E_idx, 0, 0]
    im = ax[row,col].pcolormesh(pp_orig, 90 - tt_orig, logvv, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
    ax[row,col].set_aspect('equal')
    ax[row,col].set_xlim([0, 360])
    t = ax[row,col].text(0.02, 0.92, f'(A{count}) {E:.2f} [eV]', transform=ax[row,col].transAxes,
                     va='bottom', ha='left', color='black', fontsize=10, fontweight='bold')
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
    # ax[row,col].set_title(f'(A{count}) E = {E:.2f} [eV]', fontweight='bold', fontsize=12)

    # plotting the reconstruction
    col = col + 2

    E = StepII_bundle['ENERGY'][tstamp, E_idx, 0, 0]
    ax[row,col].pcolormesh(StepII_bundle['SLEP_PHI'], StepII_bundle['SLEP_THETA'], StepII_bundle['fine_from_fine'][E_idx],
                           cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax[row,col].set_aspect('equal')
    ax[row,col].set_xlim([0, 360])
    t = ax[row,col].text(0.02, 0.92, f'(B{count}) {E:.2f} [eV]', transform=ax[row,col].transAxes,
                       va='bottom', ha='left', color='black', fontsize=10, fontweight='bold')
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
    # ax[row,col].set_title(f'(B{count}) E = {E:.2f} [eV]', fontweight='bold', fontsize=12)

    count += 1

fig.text(0.5, 0.04, r'Azimuth ($\phi$)', ha='center')
fig.text(0.001, 0.5, r'Elevation ($\theta$)', va='center', rotation='vertical')

plt.subplots_adjust(left=0.06, right=0.97, hspace=0.05, wspace=0.05, top=0.95)

fig.subplots_adjust(right=0.94)
cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.75])
fig.colorbar(im, cax=cbar_ax)

# Draw a horizontal lines at those coordinates
arrow_len = 0.12

p1a = patches.FancyArrowPatch((0.06, 0.96), (0.06+arrow_len*0.8,0.96), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
text1 = plt.text(s='MMS-FPI observation', x=0.2, y=0.95, transform=fig.transFigure)
p1b = patches.FancyArrowPatch((0.5-arrow_len*0.85,0.96), (0.5,0.96), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
fig.add_artist(p1a)
fig.add_artist(text1)
fig.add_artist(p1b)

p1a = patches.FancyArrowPatch((0.5, 0.96), (0.5+arrow_len*0.85,0.96), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
text1 = plt.text(s='Slepian reconstruction', x=0.63, y=0.95, transform=fig.transFigure)
p1b = patches.FancyArrowPatch((0.95-arrow_len*0.85,0.96), (0.95,0.96), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
fig.add_artist(p1a)
fig.add_artist(text1)
fig.add_artist(p1b)

plt.savefig('Eshell_plots_MMS.pdf')

# plotting the slices in Cartesian coordinates
# getting the Cartesian coordinates for the original grid
VX, VY, VZ = grid_pol2cart()

fig, ax = plt.subplots(1, 3, figsize=(12,4.5), sharex=True, sharey=True)
vv = StepII_bundle['VDF']
logvv = np.nan_to_num(np.log10(vv), nan=np.nan, posinf=np.nan, neginf=np.nan)

# plotting the cut in theta
theta_lr_slice = 8
theta_hr_slice = 53

ax[0].pcolormesh(VX[:,:,theta_lr_slice], VY[:,:,theta_lr_slice], logvv[tstamp, :, :, theta_lr_slice],
                 vmin=0, vmax=7, cmap='inferno', rasterized=True)
ax[0].set_xlim([-1500,1500])
ax[0].set_ylim([-1500,1500])
ax[0].set_aspect('equal')
ax[0].set_title('(C1) MMS-FPI data', fontsize=12, fontweight='bold')

ax[1].set_facecolor('black')
ax[1].pcolormesh(VX[:,:,theta_lr_slice], VY[:,:,theta_lr_slice], StepII_bundle['fine_from_fine'][:,theta_lr_slice,:],
                 vmin=0, vmax=7, cmap='inferno', rasterized=True)
ax[1].set_xlim([-1500,1500])
ax[1].set_ylim([-1500,1500])
ax[1].set_aspect('equal')
ax[1].set_title('(C2) Slepian fit at FPI resolution', fontsize=12, fontweight='bold')

ax[2].set_facecolor('black')
ax[2].pcolormesh(VDF_rec_dict['VX'][:,theta_hr_slice], VDF_rec_dict['VY'][:,theta_hr_slice], VDF_rec_dict['VDF_3D_rec'][:,theta_hr_slice],
               vmin=0, vmax=7, cmap='inferno', rasterized=True)
ax[2].set_xlim([-1500,1500])
ax[2].set_ylim([-1500,1500])
ax[2].set_aspect('equal')
ax[2].set_title('(C3) Reconstruction at high resolution', fontsize=12, fontweight='bold')

plt.subplots_adjust(left=0.08, right=0.97, hspace=0.05, wspace=0.05, top=0.95, bottom=0.1)

fig.text(0.55, 0.04, r'$V_x$ [km/s]', ha='center')
fig.text(0.001, 0.5, r'$V_y$ [km/s]', va='center', rotation='vertical')

plt.savefig('Cart_slice_rec.pdf')

# comparing the data moments with the reconstructed moments
Ntimes = StepII_bundle['ENERGY'].shape[0]

data_n_arr = np.zeros((Ntimes, 1))
data_u_arr = np.zeros((Ntimes, 3))
data_p_arr = np.zeros((Ntimes, 6))
rec_n_arr = np.zeros((Ntimes, 1))
rec_u_arr = np.zeros((Ntimes, 3))
rec_p_arr = np.zeros((Ntimes, 6))

data_moments = read_pickle('data_moments')
rec_moments = read_pickle('rec_moments')

for key_idx in tqdm(data_moments.keys()):
    data_n_s, data_u_s, data_p_s = data_moments[key_idx]
    data_n_arr[key_idx] = data_n_s / 100**3   # converting from m^-3 to cm^-3
    data_u_arr[key_idx] = data_u_s / 1e3      # converting from m/s to km/s
    data_p_arr[key_idx] = data_p_s[np.triu_indices(3)]

    rec_n_s, rec_u_s, rec_p_s = rec_moments[key_idx]
    rec_n_arr[key_idx] = rec_n_s / 100**3   # converting from m^-3 to cm^-3
    rec_u_arr[key_idx] = rec_u_s / 1e3      # converting from m/s to km/s
    rec_p_arr[key_idx] = rec_p_s[np.triu_indices(3)]

# comparing the moments
fig, ax = plt.subplots(1, 4, figsize=(12, 3))

ax[0].plot(data_n_arr[:,0], rec_n_arr[:,0], '.k')
xabsmax = np.max(np.abs(rec_n_arr[:,0]))
ax[0].set_xlim([0, 150])
ax[0].set_ylim([0, 150])
ax[0].set_aspect('equal')

ax[1].plot(data_u_arr[:,0], rec_u_arr[:,0], '.k')
xabsmax = np.max(np.abs(data_u_arr[:,0]))
ax[1].set_xlim([60, 410])
ax[1].set_ylim([60, 410])
ax[1].set_aspect('equal')

ax[2].plot(data_u_arr[:,1], rec_u_arr[:,1], '.k')
xabsmax = np.max(np.abs(rec_u_arr[:,1]))
ax[2].set_xlim([-100, 200])
ax[2].set_ylim([-100, 200])
ax[2].set_aspect('equal')

ax[3].plot(data_u_arr[:,2], -rec_u_arr[:,2], '.k')
xabsmax = np.max(np.abs(rec_u_arr[:,2]))
ax[3].set_xlim([-180, 100])
ax[3].set_ylim([-180, 100])
ax[3].set_aspect('equal')

moment_label = [r'$\rho$', r'$U_x$', r'$U_y$', r'$U_z$']

for ax_idx, axs in enumerate(ax):
    lims = [
        np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
        np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    axs.plot(lims, lims, 'r--', alpha=0.75, zorder=100, lw=2)
    axs.text(0.04, 0.8, f'{moment_label[ax_idx]}', transform=axs.transAxes,
             va='bottom', ha='left', color='black', fontsize=20, fontweight='bold')

fig.text(0.55, 0.04, 'Data moments', ha='center', fontweight='bold')
fig.text(0.001, 0.5, 'Reconstructed moments', va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(left=0.06, right=0.98, wspace=0.35, top=1.0, bottom=0.1)

plt.savefig('moment_rec_comparison.pdf')