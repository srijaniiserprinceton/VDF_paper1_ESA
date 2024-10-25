import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 14})
import pickle

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

# loading the saved dictionaries
tstamp = 533
E_idx = 16

# making a circle to show polar cap domain
ctheta = np.linspace(0, 2 * np.pi, 100)
xcirc = 85 * np.cos(ctheta) + 180
ycirc = 85 * np.sin(ctheta)

StepII_bundle = read_pickle(f'StepIIbundle_{tstamp}_MMSplot')
VDF_rec_dict = read_pickle(f'VDF3Dbundle_{tstamp}_MMSplot')

# plotting the before and after reconstruction along with the Shannon number plot
fig, ax = plt.subplots(1, 3, figsize=(12, 2.5))

cmap = 'inferno'
vmin, vmax = 1, 7

# note that for FPI, theta array stays the same at different times but phi array changes
tt_orig, pp_orig, vv = StepII_bundle['ESA_THETA'][tstamp, E_idx, :, :],\
                       StepII_bundle['ESA_PHI'][tstamp, E_idx, :, :],\
                       StepII_bundle['VDF'][tstamp, E_idx, :, :]
# calculating the log and setting +\- inf to nan
logvv = np.log10(vv)
logvv[logvv==0.0] = np.nan

E = StepII_bundle['ENERGY'][tstamp, E_idx, 0, 0]

ax[0].pcolormesh(pp_orig, 90 - tt_orig, logvv, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
ax[0].plot(xcirc, ycirc, 'r--')
ax[0].set_xlim([0, 360])
ax[0].set_ylim([-90, 90])
ax[0].set_aspect('equal')
ax[0].set_xlabel(r'Aziumth ($\phi$)')
ax[0].set_ylabel(r'Elevation ($\theta$)')
ax[0].set_title(f'MMS-FPI at {E:.2f} [eV]')

ax[1].pcolormesh(StepII_bundle['SLEP_PHI'], StepII_bundle['SLEP_THETA'], StepII_bundle['fine_from_fine'][E_idx],
                 cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
ax[1].plot(xcirc, ycirc, 'r--')
ax[1].set_xlim([0, 360])
ax[1].set_ylim([-90, 90])
ax[1].set_aspect('equal')
ax[1].set_xlabel(r'Aziumth ($\phi$)')
ax[1].set_ylabel(r'Elevation ($\theta$)')
ax[1].set_title('Slepian reconstruction')

ax[2].plot(StepII_bundle['V'], '.k')
N2D = int(np.sum(StepII_bundle['V']))
ax[2].axvline(N2D, color='red', ls='-.')
ax[2].set_aspect(43)
ax[2].set_xlabel(r'$\alpha$')
ax[2].set_ylabel(r'$\lambda_{\alpha}$')
ax[2].set_title('Polar cap localization')

plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.1, wspace=0.3)
plt.savefig('Slepian_recdemo_MMS.pdf')

# plotting the first 8 Slepian functions and the spectral domain confinement
def plot_basis_functions(E, SE, V):
    sumSEsq_V = np.abs(SE)**2 @ V
    Nbasis = int(np.sum(V))

    nrow = 2
    if((Nbasis+1)%2 == 0):
        # ncol = Nbasis // 2 + 1
        ncol = 8
    else:
        # ncol = Nbasis // 2
        ncol = 8

    # adding two extra columns to plot the spectral plot
    fig, ax = plt.subplots(2, ncol, figsize=(15,2.7), sharex=True, sharey=True)

    # finding the colorscale
    vabsmax = np.max(np.abs(E)) / 2

    for row in range(nrow):
        for col in range(ncol):
            basis_num = row * ncol + col
            im = ax[row,col].pcolormesh(StepII_bundle['SLEP_PHI'], StepII_bundle['SLEP_THETA'], E[:,:,basis_num], rasterized=True,
                                   vmin=-vabsmax, vmax=vabsmax, cmap='seismic')
            ax[row,col].plot(xcirc, ycirc, 'k--')
            ax[row,col].set_aspect('equal')
            ax[row,col].text(0.02, 0.75, r'$g_{%i}$'%basis_num, transform=ax[row,col].transAxes,
                             va='bottom', ha='left', color='black', fontsize=14, fontweight='bold')

    fig.text(0.5, 0.04, r'Azimuth ($\phi$)', ha='center')
    fig.text(0.001, 0.5, r'Elevation ($\theta$)', va='center', rotation='vertical')

    fig.subplots_adjust(right=0.965)
    cbar_ax = fig.add_axes([0.965, 0.25, 0.005, 0.65])
    fig.colorbar(im, cax=cbar_ax)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.05,
                        bottom=0.18, 
                        right=0.96, 
                        top=1.0, 
                        wspace=0.05, 
                        hspace=0.05)

# finding the spectral transforms
StepII_bundle['G'] = np.moveaxis(StepII_bundle['G'], 0, -1)
SH = np.zeros_like(StepII_bundle['G'])
for i in range(StepII_bundle['G'].shape[2]):
    SH[:,:,i] = np.fft.fftshift(np.fft.fft2(StepII_bundle['G'][:,:,i]))

plot_basis_functions(StepII_bundle['G'], SH, StepII_bundle['V'])
plt.savefig('Slepian_basis.pdf')