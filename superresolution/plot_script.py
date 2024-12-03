import spherepy as sp
import numpy as np
from scipy.interpolate import griddata
from scipy.special import sph_harm
from skimage import io
import matplotlib.pyplot as plt; plt.ion()
import sys

plt.rcParams.update({'font.size': 10})

def gen_SH_scipy(L, NPHI, NTHETA, PHI, THETA):
    sh_basis = np.zeros(((L+1)**2, NTHETA, NPHI), dtype='complex128')

    basis_count = 0
    for ell in range(L+1):
        for m in range(-ell, ell+1):
            sh_basis[int(basis_count)] = sph_harm(m, ell, PHI, THETA)
            basis_count += 1

    return sh_basis

def plot_image(ax, image, xgrid, ygrid, title, text):
    ax.pcolormesh(xgrid, ygrid, image, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    ax.text(0.02, 0.8, title, transform=ax.transAxes,
            va='bottom', ha='left', color='white', fontweight='bold')
    ax.text(0.02, 0.02, text, transform=ax.transAxes,
            va='bottom', ha='left', color='white', fontweight='bold')
    ax.set_aspect('equal')


if __name__=='__main__':
    # loadging in the refernce image
    img = io.imread('Black_hole.jpg', as_gray=True)

    # downsampling it to reduce computational time
    img = np.flip(img[::15,::15], axis=0)
    vmin, vmax = img.min(), img.max()
    Nphi, Ntheta = img.shape

    # making the high resolution grid for super resolution plots
    p, t = np.linspace(0, 360, Nphi), np.linspace(0, 180, Ntheta)
    pp, tt = np.meshgrid(p, t)

    # setting the colormap
    cmap = 'inferno'

    # creating a new figure
    fig, ax = plt.subplots(2, 3, figsize=(10,4.2), sharex=True, sharey=True)

    # plotting the original reference image with high resolution
    plot_image(ax[0,0], img, pp, tt, '(A) Highres reference image', '(288 x 288)')

    # interpolating to a low resolution grid (same resolution as the MMS angular bins)
    Ntheta_lr, Nphi_lr = 16, 32
    p_lr, t_lr = np.linspace(0, 360, Nphi_lr), np.linspace(0, 180, Ntheta_lr)
    pp_lr, tt_lr = np.meshgrid(p_lr, t_lr)
    img_lr = griddata((pp.flatten(), tt.flatten()), img.flatten(), (pp_lr, tt_lr), fill_value=0.0)

    # plotting the low resolution image
    plot_image(ax[1,0], img_lr, pp_lr, tt_lr, '(B) Lowres input image', '(32 x 16)')

    # the Lmax Nyquist
    Lmax_Nyq = int(np.min([(Nphi_lr - 2)//2, (Ntheta_lr-2)]))


    # generating lowres spherical harmonics for Nyquist limit Lmax degree
    Lmax = Lmax_Nyq
    SH_basis_lr = gen_SH_scipy(Lmax, Nphi_lr, Ntheta_lr, np.radians(pp_lr), np.radians(tt_lr))

    nan_mask = ~np.isnan(img_lr)
    SH_nonan_lr = SH_basis_lr[:,nan_mask]
    M_lr = np.conjugate(SH_nonan_lr) @ SH_nonan_lr.T 
    I = np.identity(M_lr.shape[0])
    coeffs_lr = np.linalg.inv(M_lr + 1e-12 * I) @ np.conjugate(SH_nonan_lr) @ img_lr[nan_mask]

    img_lr_rec = np.dot(np.moveaxis(SH_basis_lr, 0, -1), coeffs_lr).real

    # plotting the lowres reconstruction from Nyquist Lmax
    plot_image(ax[0,1], img_lr_rec, pp_lr, tt_lr, r'(C) Lowres recon ($L_{\rm{max}}$ = 16)', '(32 x 16)')

    # generating highres reconstruction using lowres coefficients found
    SH_basis_hr = gen_SH_scipy(Lmax, Nphi, Ntheta, np.radians(pp), np.radians(tt))
    nan_mask = ~np.isnan(img)
    SH_nonan_hr = SH_basis_hr[:,nan_mask]
    M_hr = np.conjugate(SH_nonan_hr) @ SH_nonan_hr.T 

    img_hr_rec = np.dot(np.moveaxis(SH_basis_hr, 0, -1), coeffs_lr).real

    # plotting the highres reconstruction from Nyquist Lmax
    plot_image(ax[1,1], img_hr_rec, pp, tt, r'(D) Super resolved ($L_{\rm{max}}$ = 16)', '(288 x 288)')


    # generating lowres spherical harmonics for Nyquist limit Lmax degree
    Lmax = 17
    SH_basis_lr = gen_SH_scipy(Lmax, Nphi_lr, Ntheta_lr, np.radians(pp_lr), np.radians(tt_lr))

    nan_mask = ~np.isnan(img_lr)
    SH_nonan_lr = SH_basis_lr[:,nan_mask]
    M_lr = np.conjugate(SH_nonan_lr) @ SH_nonan_lr.T 
    I = np.identity(M_lr.shape[0])
    coeffs_lr = np.linalg.inv(M_lr + 1e-12 * I) @ np.conjugate(SH_nonan_lr) @ img_lr[nan_mask]

    img_lr_rec = np.dot(np.moveaxis(SH_basis_lr, 0, -1), coeffs_lr).real

    # plotting the lowres reconstruction from Nyquist Lmax
    plot_image(ax[0,2], img_lr_rec, pp_lr, tt_lr, r'(E) Lowres recon ($L_{\rm{max}}$ = 17)', '(32 x 16)')

    # generating highres reconstruction using lowres coefficients found
    SH_basis_hr = gen_SH_scipy(Lmax, Nphi, Ntheta, np.radians(pp), np.radians(tt))
    nan_mask = ~np.isnan(img)
    SH_nonan_hr = SH_basis_hr[:,nan_mask]
    M_hr = np.conjugate(SH_nonan_hr) @ SH_nonan_hr.T 

    img_hr_rec = np.dot(np.moveaxis(SH_basis_hr, 0, -1), coeffs_lr).real

    # plotting the highres reconstruction from Nyquist Lmax
    plot_image(ax[1,2], img_hr_rec, pp, tt, r'(F) Super resolved ($L_{\rm{max}}$ = 17)', '(288 x 288)')

    fig.text(0.55, 0.04, r'Azimuth ($\phi$)', ha='center', fontsize=14)
    fig.text(0.001, 0.5, r'Elevation ($\theta$)', va='center', rotation='vertical', fontsize=14)

    for axs in ax.flatten():
        axs.set_xlim([0,360])
        axs.set_ylim([0,180])

    # adjusting the figure whitespace
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.07, right=0.98, hspace=0.02, wspace=0.1)

    plt.savefig('Nyquist_demo.pdf')
