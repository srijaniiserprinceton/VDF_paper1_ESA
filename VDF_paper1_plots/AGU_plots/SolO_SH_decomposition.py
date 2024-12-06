import spherepy as sp
import numpy as np
from skimage import io
import matplotlib.pyplot as plt; plt.ion()
from scipy.interpolate import griddata
import os, sys
package_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(package_dir)
from source_scripts import misc_functions as misc_funcs

cmap = 'inferno'

# generating Slepian functions
class sh_dict:
    def __init__(self, rcond=1e-6):
        self.instrument = 'SolO'
        self.rcond = 1e-6

    def gen_SH(self, Lmax, NPHI, NTHETA):
        self.Lmax = Lmax
        self.G = None
        self.NPHI, self.NTHETA = NPHI, NTHETA
        # Swap the order of the basis function.
        sh_basis = np.zeros((NPHI, NTHETA, (self.Lmax+1)**2), dtype='complex128')
        sh_coefs = sp.zeros_coefs(nmax=self.Lmax, mmax=self.Lmax)
        basis_count = 0
        for ell in range(self.Lmax+1):
            for m in range(-ell, ell+1):
                sh_coefs[ell, m] += 1.0
                sh_basis[:,:,int(basis_count)] = sp.ispht(sh_coefs, nrows=NPHI, ncols=NTHETA).array
                sh_coefs[ell, m] *= 0.0j
                basis_count += 1
        self.G = sh_basis

    def sh_rec_scipy(self, arr):
        self.fine_from_fine = np.zeros((self.NTHETA, self.NPHI))
        # self.gen_Slepians_on_polarcap(self.Lmax)
        self.SLEP_coeffs = np.zeros(len(self.G))

        nan_mask = np.isnan(arr)
        G = self.G * 1.0
        G_mask = G[~nan_mask,:]
        M = np.conjugate(G_mask).T @ G_mask 
        __, self.S_hr, __ = np.linalg.svd(M)
        I = np.identity(M.shape[0])
        inverted_M = np.linalg.inv(M + self.S_hr.max() * self.rcond * I)
        self.SLEP_coeffs = inverted_M @ np.conjugate(G_mask).T @ arr[~nan_mask]

        # reconstructing from the polar Slepians and plotting
        rec_coefs = np.dot(G, self.SLEP_coeffs)
        self.fine_from_fine = rec_coefs.real

    def sh_rec_spherepy(self, arr):
        # decomposing into spherepy
        # first step is to convert image into ScalarPatternUniform (this is a datatype that spherepy uses and defines)
        img_spu = sp.ScalarPatternUniform(arr)

        # decomposing into coefficients (forward transform)
        img_sph_coefs = sp.spht(img_spu, nmax=self.Lmax, mmax=self.Lmax)

        # now converting the spherical harmonic coefficients into an image (inverse transform)
        self.fine_from_fine = sp.ispht(img_sph_coefs, nrows=Nrows, ncols=Ncols).array.real


# generating Slepian functions
class slep_dict:
    def __init__(self):
        self.instrument = 'SolO'
        self.slep_dir = self.read_config(package_dir)[0]

    def slep_gen(self, Lmax):
        self.Lmax = Lmax
        self.G, self.V, self.SLEP_THETA, self.SLEP_PHI = None, None, None, None
        misc_funcs.gen_SLEP(self, N2D_restrict=True)
        self.NTHETA_SLEP, self.NPHI_SLEP = self.G.shape[1], self.G.shape[2]

    def read_config(self, package_dir):
        with open(f"{package_dir}/.config", "r") as f:
            dirnames = f.read().splitlines()

        return dirnames

    def slep_rec(self, arr):
        self.fine_from_fine = np.zeros((self.NTHETA_SLEP, self.NPHI_SLEP))
        # self.gen_Slepians_on_polarcap(self.Lmax)
        self.SLEP_coeffs = np.zeros(len(self.G))

        nan_mask_hr = np.isnan(arr)
        G_nonan_hr = self.G[:,~nan_mask_hr]
        M = G_nonan_hr @ G_nonan_hr.T 
        __, self.S, __ = np.linalg.svd(M)

        I = np.identity(M.shape[0])
        self.SLEP_coeffs = np.linalg.inv(M) @ G_nonan_hr @ arr[~nan_mask_hr]

        # reconstructing from the polar Slepians and plotting
        fine_from_finecoefs = np.dot(np.moveaxis(self.G, 0, -1), self.SLEP_coeffs)
        self.fine_from_fine += fine_from_finecoefs


#--------------------------loadging the SolO data--------------------------------#
img_orig = np.load('vdf_12_50.npy')
img_orig[np.isnan(img_orig)] = np.nanmin(img_orig)
img_orig = img_orig / np.nanmin(img_orig)

# loading the grids
theta, phi = np.load('theta.npy'), np.load('phi.npy')
dtheta_avg, dphi_avg = np.mean(np.diff(theta)), np.mean(np.diff(phi))
Ntheta, Nphi = int(180 // dtheta_avg), int(360 // dphi_avg) + 1

# storing the original grids
PP_orig, TT_orig = np.meshgrid(phi, theta, indexing='ij')

#---------------------------The Slepian Reconstruction----------------------------#
# setting the maximum angular degree upto which we want to discretize
Lmax = 15
slep = slep_dict()
slep.slep_gen(Lmax)

# interpolating to a regular grid to match the Slepian functions grids
img = griddata((PP_orig.flatten(), TT_orig.flatten()), img_orig.flatten(),
               (slep.SLEP_PHI,slep.SLEP_THETA+90), fill_value=np.nanmin(img_orig))
img = np.nan_to_num(np.log10(img))

# filling in nans outside a 8 x 8 grid
nan_mask_img = np.zeros_like(img, dtype='bool')
nan_mask_img[15:23,27:35] = True
img[~nan_mask_img] = np.nan


img_orig_embedded = np.zeros_like(img)
img_orig_embedded[14:23, 26:37] = np.log10(img_orig.T)

fig, ax = plt.subplots(1, 3, figsize=(15,3.5))
# plotting the raw image
ax[0].pcolormesh(slep.SLEP_PHI, slep.SLEP_THETA+90, img, vmin=0, vmax=np.nanmax(np.log10(img_orig)), rasterized=True, cmap=cmap)
# ax[0].pcolormesh(PP_orig, TT_orig, np.log10(img_orig), vmin=0, vmax=np.nanmax(np.log10(img_orig)), rasterized=True, cmap=cmap)
ax[0].set_xlim([0,360])
ax[0].set_ylim([0,180])
ax[0].set_title('Data grid at E=1315 eV')
ax[0].set_aspect('equal')

# decomposing in Slepian functions
slep.slep_rec(img)

# plotting the reconstructed image
ax[2].pcolormesh(slep.SLEP_PHI, slep.SLEP_THETA, slep.fine_from_fine, vmin=0, vmax=np.nanmax(img), rasterized=True, cmap=cmap)
ax[2].set_title(f'{len(slep.G)} Slepian functions')
ax[2].set_aspect('equal')
ax[2].set_yticks([])

# setting common x and ylabels
fig.text(0.5, 0.04, r'Aximuth $\phi [{}^{\circ}]$', ha='center', fontsize=16)
fig.text(0.01, 0.5, r'Elevation $\theta [{}^{\circ}]$', va='center', rotation='vertical', fontsize=16)

plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.05, hspace=0.1, wspace=0.1)

TT, PP = np.meshgrid(np.linspace(0,180,Ntheta), np.linspace(0,360,Nphi), indexing='ij')

# interpolating to a regular grid
img = griddata((PP_orig.flatten(), TT_orig.flatten()), img_orig.flatten(), (PP,TT), fill_value=np.nanmin(img_orig))
img = np.nan_to_num(np.log10(img))

# filling in nans outside a 8 x 8 grid
nan_mask_img = np.zeros_like(img, dtype='bool')
nan_mask_img[15:23,27:35] = True
img[~nan_mask_img] = np.nan


Nrows, Ncols = img.shape

# # plotting some of the Slepian functions
# fig, ax = plt.subplots(4, 8, figsize=(15,8))
# ax = ax.flatten()
# for slep_idx in range(32):
#     vmin, vmax = slep.G[slep_idx].min(), slep.G[slep_idx].max()
#     ax[slep_idx].pcolormesh(slep.SLEP_PHI, slep.SLEP_THETA, slep.G[slep_idx], cmap='seismic', rasterized=True)

# generating and reconstructing in spherical harmonics basis
sh = sh_dict(rcond=0.0)

for Lmax_idx, Lmax in enumerate(np.arange(1, 30)):
    sh.gen_SH(int(Lmax), Nrows, Ncols)
    sh.sh_rec_scipy(img)
    ax[1].cla()
    # plotting the reconstructed image
    ax[1].pcolormesh(PP, TT, sh.fine_from_fine, vmin=0, vmax=np.nanmax(img), rasterized=True, cmap=cmap)
    ax[1].set_title(f'{((Lmax+1)**2)} Spherial harmonics')
    ax[1].set_aspect('equal')
    plt.savefig(f'span_grid/{Lmax_idx}.png')

    ax[1].set_yticks([])