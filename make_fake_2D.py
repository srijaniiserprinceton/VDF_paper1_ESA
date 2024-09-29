import numpy as np
from scipy.interpolate import LinearNDInterpolator as interp
import pickle, sys
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')
from matplotlib.widgets import Slider, Button, RadioButtons

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def get_SPAN_FOV(rotate_angle):
    phimin, phimax = PHIMIN_SPAN + rotate_angle, PHIMAX_SPAN + rotate_angle

    # now changing to the angle convention of our coordinate system
    phimin, phimax = 180 - phimin, 180 - phimax

    # making the SPAN FOV lines
    ymin = np.tan(np.radians(phimin)) * StepII_bundle.XX[0]
    ymax = np.tan(np.radians(phimax)) * StepII_bundle.XX[0]

    return ymin, ymax

def get_SPC_FOV(rotate_angle):
    phicen = PHICEN_SPC + rotate_angle
    phimin, phimax = phicen - PHIWIN_SPC, phicen + PHIWIN_SPC

    # now changing to the angle convention of our coordinate system
    phimin, phicen, phimax = 180 - phimin, 180 - phicen, 180 - phimax

    # making the SPC FOV lines
    ymin = np.tan(np.radians(phimin)) * XX[0]
    ycen = np.tan(np.radians(phicen)) * XX[0]
    ymax = np.tan(np.radians(phimax)) * XX[0]

    return ymin, ycen, ymax

StepII_bundle = read_pickle('StepII_bundle')

# extracting the 2D VDF
VDF_2D = StepII_bundle.VDF_Sleprec
XX = StepII_bundle.XX
YY = StepII_bundle.YY
NX, NY = XX.shape

# extracting the localized basis functions
G = StepII_bundle.G
N2D = int(np.sum(StepII_bundle.V))

G_sym = []

# retaining only the axisymmetric basis functions
for slep_idx in range(100):
    # print(slep_idx, np.sum(np.abs(StepII_bundle.G[::-1,:,slep_idx] + StepII_bundle.G[:,:,slep_idx])) / np.sum(np.abs(StepII_bundle.G[::-1,:,slep_idx]))/2)
    asym_frac = np.sum(np.abs(G[::-1,:,slep_idx] + G[:,:,slep_idx])) / np.sum(np.abs(G[::-1,:,slep_idx]))/2
    print(slep_idx, asym_frac)
    if(asym_frac > 0.5): G_sym.append(G[:,:,slep_idx])

G = np.asarray(G_sym)
G = np.moveaxis(G, 0, 1)
G = np.moveaxis(G, 1, 2)

def gyrotropic_recon_2D_VDF(VDF_data, rcond=0.0):
    # reconstructing the interpolated VDF in Cartesian Slepians
    nan_mask = np.isnan(VDF_data)
    G_nonan = G[~nan_mask,:]
    M = G_nonan.T @ G_nonan
    __, S, __ = np.linalg.svd(M)
    I = np.identity(M.shape[0])
    coeffs = np.linalg.inv(M +  S.max() * rcond * I) @ G_nonan.T @ VDF_data[~nan_mask]
    VDF_Sleprec = np.dot(G, coeffs)

    return VDF_Sleprec

# getting a purely gyrotropized 2D VDF representation
VDF_2D_rec = gyrotropic_recon_2D_VDF(VDF_2D)

# default SPAN and SPC directions
PHIMIN_SPAN, PHIMAX_SPAN = 95.625, 174.375 #- 25
PHICEN_SPC, PHIWIN_SPC = 180, 40

# making fake data
# SPAN FOV
ymin_span, ymax_span = get_SPAN_FOV(0)
# making the vertical max and min st lines for plotting 
yvertmax = np.tan(np.pi/2 * 0.999) * StepII_bundle.XX[0]
yvertmin = np.tan(-np.pi/2 * 0.999) * StepII_bundle.XX[0]

# SPC FOV
ymin_spc, ycen_spc, ymax_spc = get_SPC_FOV(0)

def make_SPAN_data(rotate_angle, NSR=0.0):
    ymin_span, ymax_span = get_SPAN_FOV(rotate_angle)
    YY_gt = YY - ymax_span[np.newaxis,:]
    # YY_lt = YY - ymin_span[np.newaxis,:]
    data_mask = (YY_gt > 0) #* (YY_lt < 0)
    data = np.zeros_like(VDF_2D_rec) + np.nan
    data[data_mask] = VDF_2D_rec[data_mask]
    data += np.random.rand(NX,NY) * NSR
    return data

def f_interpolate_2D_to_1D(arr):
    f = interp(list(zip(XX.flatten(), YY.flatten())), arr.flatten())
    return f

# generating the interpolated functions for all the Slepians
Slep_1D_f = {}
Slep_equator = np.zeros((NY, G.shape[-1]))
for Slep_idx in range(G.shape[-1]):
    Slep_1D_f[Slep_idx] = f_interpolate_2D_to_1D(G[:,:,Slep_idx])

    # generating these 1D Slepian profiles at the y=0 line (where the core is located)
    Slep_equator[:,Slep_idx] = Slep_1D_f[Slep_idx](XX[0], ycen_spc)

# generating interpolation function for generating SPC data
SPC_f = f_interpolate_2D_to_1D(VDF_2D_rec)
spc_equator_data = SPC_f(XX[0], ycen_spc)

def gyrotropic_joint_recon_2D_VDF(VDF_data, rcond=0.0):
    # reconstructing the interpolated VDF in Cartesian Slepians
    nan_mask = np.isnan(VDF_data)
    G_nonan = G[~nan_mask,:]

    # adding the 1D Slepians
    G_nonan = np.append(G_nonan, Slep_equator, axis=0)

    # adding the SPC data
    data_nonan = VDF_data[~nan_mask]
    data_nonan = np.append(data_nonan, spc_equator_data)

    M = G_nonan.T @ G_nonan
    __, S, __ = np.linalg.svd(M)
    I = np.identity(M.shape[0])
    coeffs = np.linalg.inv(M +  S.max() * rcond * I) @ G_nonan.T @ data_nonan
    VDF_Sleprec = np.dot(G, coeffs)

    return VDF_Sleprec

# recovering from SPAN_data
SPAN_data = make_SPAN_data(0)
# VDF_fakeSPAN_rec = gyrotropic_recon_2D_VDF(SPAN_data)
VDF_fakeSPANSPC_rec = gyrotropic_joint_recon_2D_VDF(SPAN_data)

plt.figure()
plt.contourf(XX, YY, SPAN_data, vmin=1, vmax=7)
plt.figure()
# plt.contourf(XX, YY, VDF_fakeSPAN_rec, vmin=1, vmax=7)
plt.contourf(XX, YY, VDF_fakeSPANSPC_rec, vmin=1, vmax=7)
plt.figure()
# plt.contourf(XX, YY, VDF_fakeSPAN_rec - np.nan_to_num(SPAN_data), vmin=1, vmax=7)
plt.contourf(XX, YY, VDF_fakeSPANSPC_rec - np.nan_to_num(SPAN_data), vmin=1, vmax=7)
plt.figure()
plt.plot(XX[0], SPC_f(XX[0], ycen_spc))

# sys.exit()

#----------------- interactive plotting --------------------#
fill_color = 'black'
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(221)
fig.subplots_adjust(bottom=0.25)

im1 = ax1.pcolormesh(XX, YY, VDF_2D_rec, cmap='inferno', vmin=1, vmax=7)

[spanmin] = ax1.plot(XX[0], ymin_span, '-w', alpha=0.5)
[spanmax] = ax1.plot(XX[0], ymax_span, '-w', alpha=0.5)
# spanfill1 = ax.fill_between(XX[0], ymin_span, yvertmax, color=fill_color)
spanfill2 = ax1.fill_between(XX[0], yvertmin, ymax_span, color=fill_color)

[spcmin] = ax1.plot(XX[0], ymin_spc, '-y', alpha=0.5)
[spccen] = ax1.plot(XX[0], ycen_spc, '--y', alpha=0.5)
[spcmax] = ax1.plot(XX[0], ymax_spc, '-y', alpha=0.5)

ax1.set_xlim([0,1000])
ax1.set_ylim([-600,600])
ax1.set_aspect('equal')

ax2 = fig.add_subplot(222)
span_data = make_SPAN_data(0)
im2 = ax2.pcolormesh(XX, YY, gyrotropic_recon_2D_VDF(span_data), cmap='inferno', vmin=1, vmax=7)
ax2.set_aspect('equal')
ax2.set_xlim([0,1000])
ax2.set_ylim([-600,600])

ax3 = fig.add_subplot(223)
spc_data = SPC_f(XX[0], ycen_spc)
[spcdata] = ax3.plot(XX[0], spc_data, color='yellow')
ax3.set_xlim([0,1000])

ax4 = fig.add_subplot(224)
im4 = ax4.pcolormesh(XX, YY, gyrotropic_joint_recon_2D_VDF(span_data), cmap='inferno', vmin=1, vmax=7)
ax4.set_aspect('equal')
ax4.set_xlim([0,1000])
ax4.set_ylim([-600,600])

# Define an axes area and draw a slider in it
axis_color = 'white'
angle_slider_ax  = fig.add_axes([0.4, 0.12, 0.2, 0.03], facecolor=axis_color)
angle_slider = Slider(angle_slider_ax, r'$\mu_{\phi}$', -45, 45, valinit=0)

NSR_slider_ax  = fig.add_axes([0.4, 0.08, 0.2, 0.03], facecolor=axis_color)
NSR_slider = Slider(NSR_slider_ax, 'Noise', 0, 1, valinit=0.0)

rcond_slider_ax  = fig.add_axes([0.4, 0.04, 0.2, 0.03], facecolor=axis_color)
rcond_slider = Slider(rcond_slider_ax, '-log(rcond)', -16, 0, valinit=-16)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    ymin_span, ymax_span = get_SPAN_FOV(angle_slider.val)
    ymin_spc, ycen_spc, ymax_spc = get_SPC_FOV(angle_slider.val)

    spanmin.set_ydata(ymin_span)
    spanmax.set_ydata(ymax_span)
    spcmin.set_ydata(ymin_spc)
    spccen.set_ydata(ycen_spc)
    spcmax.set_ydata(ymax_spc)
    spcdata.set_ydata(SPC_f(XX[0], ycen_spc))

    span_data = make_SPAN_data(angle_slider.val, NSR=NSR_slider.val)
    im1.set_array(span_data)

    dummy = ax1.fill_between(XX[0], yvertmin, ymax_span, alpha=0)
    dp = dummy.get_paths()[0]
    dummy.remove()
    spanfill2.set_paths([dp.vertices])

    rcond = 10**(rcond_slider.val)
    im2.set_array(gyrotropic_recon_2D_VDF(span_data, rcond=rcond))
    im4.set_array(gyrotropic_joint_recon_2D_VDF(span_data, rcond=rcond))

    fig.canvas.draw_idle()

angle_slider.on_changed(sliders_on_changed)
NSR_slider.on_changed(sliders_on_changed)
rcond_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color='black', hovercolor='0.1')
def reset_button_on_clicked(mouse_event):
    angle_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)
