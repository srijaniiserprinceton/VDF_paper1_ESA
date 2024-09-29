import numpy as np; NAX=np.newaxis
from scipy.integrate import simpson as simps
from scipy.interpolate import griddata
import pickle, sys
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')
from matplotlib.widgets import Slider, Button, RadioButtons
from astropy.coordinates import cartesian_to_spherical as c2s
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def plot_diagnostic_panels(ax, e, pp, tt, vv):
    vmin, vmax = 0, 6
    im = ax.pcolormesh(pp, tt, vv, cmap='inferno', rasterized=True, vmin=vmin, vmax=vmax)
    # im = ax.contourf(pp, tt, np.log10(vv), cmap='BuPu', rasterized=True)#, vmin=vmin, vmax=vmax)
    ax.text(0.05, 0.05, f'{e:.2f} [eV]', transform=ax.transAxes,
                    va='bottom', ha='left', color='white', fontweight='bold')
    ax.set_aspect('equal')

StepII_bundle = read_pickle('StepII_bundle')

time_idx = 0
E = StepII_bundle.rec_dict.ENERGY[time_idx,:,0,0]
PP, TT = StepII_bundle.lon_hr, 90-StepII_bundle.lat_hr
PPR, TTR = np.radians(PP), np.radians(TT)

Vmag = 13.8 * np.sqrt(E)
VXX = Vmag[:,NAX,NAX] * np.sin(TTR)[NAX,:,:] * np.cos(PPR)[NAX,:,:]
VYY = Vmag[:,NAX,NAX] * np.sin(TTR)[NAX,:,:] * np.sin(PPR)[NAX,:,:]
VZZ = Vmag[:,NAX,NAX] * np.cos(TTR)[NAX,:,:]

# the index by which we want to roll in phi
phi_roll_idx = 0

# VDF_3D = StepII_bundle.fine_from_fine * 1.0
# VDF_3D = np.roll(VDF_3D, phi_roll_idx, axis=2)

# rolling the basis functions as well since they need to be gyrotropic
StepII_bundle.G_all_coarse = {}
for L in range(StepII_bundle.Lmin, StepII_bundle.Lmax+1):
    StepII_bundle.G_all[f'{L}'] = np.roll(StepII_bundle.G_all[f'{L}'], phi_roll_idx, axis=2)
    StepII_bundle.G_all_coarse[f'{L}'] = np.zeros((len(StepII_bundle.G_all[f'{L}']), StepII_bundle.N_lat_lr, StepII_bundle.N_lon_lr))
    for bidx in range(len(StepII_bundle.G_all[f'{L}'])):
        StepII_bundle.G_all_coarse[f'{L}'][bidx] = griddata((StepII_bundle.lat_hr.flatten(), StepII_bundle.lon_hr.flatten()),
                                                             StepII_bundle.G_all[f'{L}'][bidx].flatten(),
                                                            (StepII_bundle.lat_lr.T, StepII_bundle.lon_lr.T), method='linear')


# interpolating the VDF_3D data
VDF_3D = griddata((StepII_bundle.lat_hr.flatten(), StepII_bundle.lon_hr.flatten()),
                   StepII_bundle.fine_from_fine.flatten(),
                   (StepII_bundle.lat_lr.T, StepII_bundle.lon_lr.T), method='linear')
VDF_3D = np.roll(VDF_3D, phi_roll_idx, axis=2)

Nlat, Nlon = PP.shape


# # making coeffs to get VDF_3D
# coeffs = {}
# # VDF_3D *= 0.0
# VDF_3D = np.zeros_like(StepII_bundle.fine_from_fine)
# for L in range(StepII_bundle.Lmin, StepII_bundle.Lmin+1):
#     coeffs[f'{L}'] = np.random.rand(len(StepII_bundle.G_all[f'{L}']))
#     VDF_3D += np.dot(np.moveaxis(StepII_bundle.G_all[f'{L}'], 0, -1), coeffs[f'{L}'])

# plotting
plt.figure()
plt.contourf(VXX[:,90], VYY[:,90], VDF_3D[:,90], cmap='inferno', levels=10, rasterized=True, vmin=1, vmax=7)
# plt.pcolormesh(VXX[:,90], VYY[:,90], VDF_3D[:,90], cmap='inferno', rasterized=True)
plt.gca().set_aspect('equal')
plt.xlim([-800,0])
plt.ylim([-500,500])
plt.axhline(0.0, color='white', ls='--')

# plotting the energy slices
Nrows, Ncols = 4, 8
fig, ax = plt.subplots(Nrows, Ncols, figsize=(16,8), sharex=True, sharey=True)
for i, E_idx in enumerate(np.arange(0, Nrows * Ncols)):
    # finding the row and the column of the subplots
    row, col = i//Ncols, i%Ncols
    plot_diagnostic_panels(ax[row,col], E[E_idx], PP, TT, VDF_3D[E_idx])

sys.exit()

# SPAN grid region
ESA_PHIMIN, ESA_PHIMAX = 95.625, 174.375
ESA_THETAMIN, ESA_THETAMAX = 38.22, 142.47
ESA_PHIMIN_IDX, ESA_PHIMAX_IDX = np.argmin(np.abs(PP[0]-ESA_PHIMIN)), np.argmin(np.abs(PP[0]-ESA_PHIMAX))
ESA_THETAMIN_IDX, ESA_THETAMAX_IDX = np.argmin(np.abs(TT[:,0]-ESA_THETAMIN)), np.argmin(np.abs(TT[:,0]-ESA_THETAMAX))

# SPC cup window
PHI_SPC, THETA_SPC = 190, 180 - StepII_bundle.theta0
PHI_SPC_R, THETA_SPC_R = np.radians(PHI_SPC), np.radians(THETA_SPC)
ANGWIN_SPC = 40
tcirc = np.linspace(0, 2*np.pi, 100)
xcirc_spc, ycirc_spc = ANGWIN_SPC * np.cos(tcirc) + PHI_SPC, ANGWIN_SPC * np.sin(tcirc) + THETA_SPC

# function to integrate the 3D VDF inside the SPC cone
IN_SPC_MASK = np.arccos(np.cos(TTR) * np.cos(THETA_SPC_R) +\
              np.sin(TTR) * np.sin(THETA_SPC_R) * np.cos(PPR - PHI_SPC_R)) < (40 * np.pi / 180)

# plotting one of the energy slices showing the SPAN and SPC coverages
def VDF_shift(eshell_idx, phi_shift): 
    return np.roll(StepII_bundle.fine_from_fine[int(eshell_idx)], int(phi_shift), axis=1)

VDF_time_sinTT = VDF_3D * np.sin(TTR)[NAX,:,:]

# def SPC_1D(phi_shift):
#     SPC_RVDF = np.zeros_like(E)
#     for E_idx in range(len(E)):
#         SPC_VDF = np.roll(StepII_bundle.fine_from_fine, int(phi_shift), axis=2)[E_idx]
#         SPC_VDF[~IN_SPC_MASK] *= 0.0
#         SPC_RVDF[E_idx] = simps(simps(SPC_VDF * np.sin(TTR), x=TTR[:,0], axis=0), x=PPR[0], axis=0)

#     return SPC_RVDF

def SPC_1D():
    SPC_RVDF = np.zeros_like(E)
    for E_idx in range(len(E)):
        SPC_VDF = VDF_time_sinTT[E_idx] * 1.0
        SPC_VDF[~IN_SPC_MASK] *= 0.0
        SPC_RVDF[E_idx] = simps(simps(SPC_VDF, x=TTR[:,0], axis=0), x=PPR[0], axis=0)

    return SPC_RVDF



# inversion using SPAN and SPC
# SPC_data = SPC_1D(phi_roll_idx)
SPC_data = SPC_1D()


# generating the sin(theta) weighted polar cap Slepians for faster computation
G_all_sinTT = {}

for L in range(StepII_bundle.Lmin, StepII_bundle.Lmax+1):
    G_all_sinTT[f'{L}'] = StepII_bundle.G_all[f'{L}'] * np.sin(TTR)[NAX,:,:]


# G_all_sinTT = G_all * np.sin(TTR)[NAX,:,:]

# integrating the Slepians on polar cap according to the relative angle between SPC direction and gyroaxis
# we shall call these theta, phi integrated Slepians as Reduced Slepians
RSlep_dict = {}


for L in range(StepII_bundle.Lmin, StepII_bundle.Lmax+1):
    RSlep_dict[f'{L}'] = np.zeros(len(G_all_sinTT[f'{L}']))
    G_L_sinTT = G_all_sinTT[f'{L}']
    G_L_sinTT[:,~IN_SPC_MASK] *= 0.0
    RSlep_dict[f'{L}'] = simps(simps(G_L_sinTT, x=TTR[:,0], axis=1), x=PPR[0], axis=1)
    RSlep_dict[f'{L}'] = np.reshape(RSlep_dict[f'{L}'], (len(RSlep_dict[f'{L}']), -1))


# G_all_sinTT[:,~IN_SPC_MASK] *= 0.0
# RSlep = simps(simps(G_all_sinTT, x=TTR[:,0], axis=1), x=PPR[0], axis=1)
# RSlep = np.reshape(RSlep, (len(RSlep), -1))

# finding the coefficients for the polar cap slepians
CSlep_dict = {}

# fitting the polar Slepians
VDF_polar_rec = np.zeros((StepII_bundle.N_Eshells, StepII_bundle.N_lat_hr, StepII_bundle.N_lon_hr))
RVDF_rec = np.zeros_like(SPC_data)

rcond = 1e-12

# clipping off the SPAN data
SPAN_data = np.zeros_like(VDF_3D) + np.nan
SPAN_data[:,ESA_THETAMIN_IDX:ESA_THETAMAX_IDX, ESA_PHIMIN_IDX:ESA_PHIMAX_IDX] =\
    VDF_3D[:,ESA_THETAMIN_IDX:ESA_THETAMAX_IDX, ESA_PHIMIN_IDX:ESA_PHIMAX_IDX] * 1.0
# SPAN_data += np.random.rand(Nlat, Nlon) * 1.0


for L in range(StepII_bundle.Lmin, StepII_bundle.Lmin+1):
    for E_idx in range(len(E)):
        print(E_idx)
        # removing the previously fitted part for the SPAN part
        img_hr = SPAN_data[E_idx] * 1.0
        img_hr = img_hr - VDF_polar_rec[E_idx]

        # removing the previosly fitted part for the SPC part
        SPC_val = SPC_data[E_idx] * 1.0
        SPC_val = SPC_val - RVDF_rec[E_idx]

        # masking out the nan entries
        nan_mask_hr = np.isnan(img_hr)
        G_nonan_hr = StepII_bundle.G_all[f'{L}'][:,~nan_mask_hr] * 1.0
        data_nonan = img_hr[~nan_mask_hr]

        # appending RSlepians row in the operator
        if(np.isnan([SPC_val])): pass
        else:
            print('Using SPC')
            G_nonan_hr = np.append(G_nonan_hr, RSlep_dict[f'{L}'] * 1e10, axis=1)
            # appending the SPC data in the img_hr 
            data_nonan = np.append(data_nonan, SPC_val * 1e10)
            # data_nonan = img_hr[~nan_mask_hr]

        M_hr = G_nonan_hr @ G_nonan_hr.T 
        __, S_hr, __ = np.linalg.svd(M_hr)
        I_hr = np.identity(M_hr.shape[0])
        coeffs_hr = np.linalg.inv(M_hr +  S_hr.max() * rcond * I_hr) @ G_nonan_hr @ data_nonan

        # reconstructing from the polar Slepians and plotting 
        Eshell_VDF_increment = np.dot(np.moveaxis(StepII_bundle.G_all[f'{L}'], 0, -1), coeffs_hr)
        VDF_polar_rec[E_idx] += Eshell_VDF_increment

        # building the SPC increment
        # SPC_coverage_VDF = np.zeros_like(Eshell_VDF_increment)
        # SPC_coverage_VDF[IN_SPC_MASK] += Eshell_VDF_increment[IN_SPC_MASK]
        # RVDF_rec[E_idx] += simps(simps(SPC_coverage_VDF * np.sin(TTR), x=TTR[:,0], axis=0), x=PPR[0], axis=0)

        SPC_coverage_VDF = np.zeros_like(Eshell_VDF_increment)
        SPC_coverage_VDF[IN_SPC_MASK] += VDF_polar_rec[E_idx, IN_SPC_MASK]
        RVDF_rec[E_idx] = simps(simps(SPC_coverage_VDF * np.sin(TTR), x=TTR[:,0], axis=0), x=PPR[0], axis=0)
'''


for E_idx in range(len(E)):
    print(E_idx)
    # removing the previously fitted part for the SPAN part
    img_hr = SPAN_data[E_idx] * 1.0
    img_hr = img_hr - VDF_polar_rec[E_idx]

    # removing the previosly fitted part for the SPC part
    SPC_val = SPC_data[E_idx] * 1.0
    SPC_val = SPC_val - RVDF_rec[E_idx]

    # masking out the nan entries
    nan_mask_hr = np.isnan(img_hr)
    G_nonan_hr = G_all[:,~nan_mask_hr] * 1.0
    data_nonan = img_hr[~nan_mask_hr]

    # appending RSlepians row in the operator
    if(np.isnan([SPC_val])): pass
    else:
        print('Using SPC')
        G_nonan_hr = np.append(G_nonan_hr, RSlep * 1e5, axis=1)
        # appending the SPC data in the img_hr 
        data_nonan = np.append(data_nonan, SPC_val * 1e5)
        # data_nonan = img_hr[~nan_mask_hr]

    M_hr = G_nonan_hr @ G_nonan_hr.T 
    __, S_hr, __ = np.linalg.svd(M_hr)
    I_hr = np.identity(M_hr.shape[0])
    coeffs_hr = np.linalg.inv(M_hr +  S_hr.max() * rcond * I_hr) @ G_nonan_hr @ data_nonan

    # reconstructing from the polar Slepians and plotting 
    Eshell_VDF_increment = np.dot(np.moveaxis(G_all, 0, -1), coeffs_hr)
    VDF_polar_rec[E_idx] += Eshell_VDF_increment

    # building the SPC increment
    # SPC_coverage_VDF = np.zeros_like(Eshell_VDF_increment)
    # SPC_coverage_VDF[IN_SPC_MASK] += Eshell_VDF_increment[IN_SPC_MASK]
    # RVDF_rec[E_idx] += simps(simps(SPC_coverage_VDF * np.sin(TTR), x=TTR[:,0], axis=0), x=PPR[0], axis=0)

    SPC_coverage_VDF = np.zeros_like(Eshell_VDF_increment)
    SPC_coverage_VDF[IN_SPC_MASK] += VDF_polar_rec[E_idx, IN_SPC_MASK]
    RVDF_rec[E_idx] = simps(simps(SPC_coverage_VDF * np.sin(TTR), x=TTR[:,0], axis=0), x=PPR[0], axis=0)
'''
# comparing the true data to the inferred data
vmin, vmax = 1, 7
fig, ax = plt.subplots(3, 1, figsize=(8,8))
ax[0].pcolormesh(PP, TT, VDF_3D[15], vmin=vmin, vmax=vmax)
ax[1].pcolormesh(PP, TT, SPAN_data[15], vmin=vmin, vmax=vmax)
ax[2].pcolormesh(PP, TT, VDF_polar_rec[15], vmin=vmin, vmax=vmax)
ax[0].plot(xcirc_spc, ycirc_spc, ls='--', color='red')
ax[1].plot(xcirc_spc, ycirc_spc, ls='--', color='red')
ax[2].plot(xcirc_spc, ycirc_spc, ls='--', color='red')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')

sys.exit()
#------------------------------- interactive plotting starts here --------------------------------#

fig = plt.figure(figsize=(16,8))

# making the first subplot with FOV demonstration
ax = fig.add_subplot(121)
fig.subplots_adjust(bottom=0.25)
vmin, vmax = 1, 6
phi_shift0 = 8
eshell_idx0 = 15
im1 = ax.pcolormesh(PP, TT, VDF_shift(eshell_idx0, phi_shift0), cmap='binary_r', rasterized=True, vmin=vmin, vmax=vmax)
im2 = ax.pcolormesh(PP[ESA_THETAMIN_IDX:ESA_THETAMAX_IDX, ESA_PHIMIN_IDX:ESA_PHIMAX_IDX],
            TT[ESA_THETAMIN_IDX:ESA_THETAMAX_IDX, ESA_PHIMIN_IDX:ESA_PHIMAX_IDX],
            VDF_shift(eshell_idx0, phi_shift0)[ESA_THETAMIN_IDX:ESA_THETAMAX_IDX, ESA_PHIMIN_IDX:ESA_PHIMAX_IDX],
            cmap='plasma', rasterized=True, vmin=vmin, vmax=vmax)
ax.plot(xcirc_spc, ycirc_spc, ls='--', color='yellow')
im3=ax.text(0.05, 0.05, f'{E[eshell_idx0]:.2f} [eV]', transform=ax.transAxes,
            va='bottom', ha='left', color='white', fontweight='bold')
ax.text(0.35, 0.75, f'SPAN-Ai', transform=ax.transAxes,
        va='bottom', ha='left', color='white', fontweight='bold')
ax.text(0.63, 0.6, f'SPC', transform=ax.transAxes,
        va='bottom', ha='left', color='yellow', fontweight='bold')
ax.set_aspect('equal')

# making the second plot with SPC integral
ax = fig.add_subplot(122)
[line_SPC] = ax.plot(Vmag, SPC_1D(phi_shift0), '.-', color='yellow')
ax.set_xlabel('Vmag [km/s]')
ax.set_ylabel('Reduced VDF in arbitrary units')
ax.set_xlim([0, 1000])

# Define an axes area and draw a slider in it
axis_color = 'white'
muphi_slider_ax  = fig.add_axes([0.4, 0.15, 0.2, 0.03], facecolor=axis_color)
muphi_slider = Slider(muphi_slider_ax, r'$\mu_{\phi}$', -100, 100, valinit=8)

eshell_slider_ax  = fig.add_axes([0.4, 0.05, 0.2, 0.03], facecolor=axis_color)
eshell_slider = Slider(eshell_slider_ax, 'Eshell index', 0, 31, valinit=15)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    im1.set_array(VDF_shift(eshell_slider.val, muphi_slider.val))
    im2.set_array(VDF_shift(eshell_slider.val, muphi_slider.val)[ESA_THETAMIN_IDX:ESA_THETAMAX_IDX, ESA_PHIMIN_IDX:ESA_PHIMAX_IDX])
    im3.set_text(f'{E[int(eshell_slider.val)]:.2f} [eV]')
    line_SPC.set_ydata(SPC_1D(muphi_slider.val))
    fig.canvas.draw_idle()
muphi_slider.on_changed(sliders_on_changed)
eshell_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color='black', hovercolor='0.1')
def reset_button_on_clicked(mouse_event):
    muphi_slider.reset()
    eshell_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)
