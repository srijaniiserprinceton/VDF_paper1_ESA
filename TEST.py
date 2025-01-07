import spherepy as sp
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.special import sph_harm
plt.ion()


def gen_SH(NPHI, NTHETA, Lmax): # L, NPHI, NTHETA):

    # Swap the order of the basis function.
    sh_basis = np.zeros((NTHETA, NPHI, (Lmax+1)**2), dtype='complex128')
    sh_coefs = sp.zeros_coefs(nmax=Lmax, mmax=Lmax)

    basis_count = 0
    for ell in range(Lmax+1):
        for m in range(-ell, ell+1):
            sh_coefs[ell, m] += 1.0
            sh_basis[:,:,int(basis_count)] = sp.ispht(sh_coefs, nrows=NTHETA, ncols=NPHI).array
            sh_coefs[ell, m] *= 0.0j
            basis_count += 1

    return(sh_basis)

def gen_SH_scipy(NPHI, NTHETA, Lmax):
    sh_basis = np.zeros((NTHETA, NPHI, (Lmax+1)**2), dtype='complex128')

    PHI   = np.linspace(3.9375, 352.6875, NPHI)
    THETA = np.linspace(5.625, 174.375, NTHETA)

    PP, TT = np.meshgrid(PHI, THETA, indexing='ij')
    basis_count = 0
    for ell in range(Lmax+1):
        for m in range(-ell, ell+1):
            # norm = np.sqrt( ((2 * ell + 1)/(4*np.pi)) * (np.math.factorial(ell - m)/np.math.factorial(ell + m))  )
            sh_basis[:,:,int(basis_count)] = sph_harm(m, ell, np.radians(PP), np.radians(TT)).T
            basis_count += 1

    return(sh_basis)

def recon_vdf(G, arr, rcond=0):
    nan_mask = np.isnan(arr)
    G_mask = G[~nan_mask, :]
    M = np.conjugate(G_mask).T @ G_mask
    _, S, _ = np.linalg.svd(M)
    I = np.identity(M.shape[0])

    inverted_M = np.linalg.inv(M + rcond * S.max() * I )

    coeffs = inverted_M @ np.conjugate(G_mask).T @ arr[~nan_mask]

    rec = np.dot(G, coeffs)

    return(rec)


img = np.load('0_16.npy')
img[img == 0] = np.nan
img = img/np.nanmin(img)
img = np.log10(img)
img = np.nan_to_num(img, nan=0, posinf=0, neginf=0)
img = img.T

Nrows, Ncols = img.shape

# plotting the raw image
plt.figure()
plt.pcolormesh(img, cmap='inferno', vmin=1, vmax=4)
plt.colorbar()
# decomposing into spherepy
# first step is to convert image into ScalarPatternUniform (this is a datatype that spherepy uses and defines)
img_spu = sp.ScalarPatternUniform(img)

# this is not in the form that spherepy can decompose into coefficients of spherical harmonics
# setting the maximum angular degree upto which we want to discretize
Lmax = 14

# decomposing into coefficients (forward transform)
img_sph_coefs = sp.spht(img_spu, nmax=Lmax, mmax=Lmax)

# now converting the spherical harmonic coefficients into an image (inverse transform)
img_rec = sp.ispht(img_sph_coefs, nrows=Nrows, ncols=Ncols).array.real


# Apply recon_Vdf
G = gen_SH(Ncols, Nrows, Lmax)
img_new_rec = recon_vdf(G, img, 0)

Gscipy = gen_SH_scipy(Ncols, Nrows, Lmax)
img_new_rec2 = recon_vdf(Gscipy, img, 0)


# plotting the reconstructed image
plt.figure()
plt.pcolormesh(img_rec, cmap='inferno', vmin=1, vmax=4)
plt.colorbar()

plt.figure()
plt.pcolormesh(img_new_rec.real, cmap='inferno', vmin=1, vmax=4)
plt.colorbar()

plt.figure()
plt.pcolormesh(img_new_rec2.real, cmap='inferno', vmin=1, vmax=4)
plt.colorbar()