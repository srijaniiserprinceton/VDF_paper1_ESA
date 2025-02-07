import numpy as np
import spherepy as sp
from scipy.interpolate import griddata
from scipy.io import savemat
import matplotlib.pyplot as plt
plt.ion()

# import matlab.engine as matlab
# # generating the low and high resolution Slepians-on-polar-cap
# eng = matlab.start_matlab()
# s = eng.genpath('/Users/srijanbharatidas/Documents/Research/Codes/Helioseismology/Slepians/Slepian_Git')
# eng.addpath(s, nargout=0)

# import generate_2D_contour as gen_contour

class VDF_rec_polarcaps_Slepians:
    def __init__(self, rec_dict, time_idx, rcond=0.0):
        self.time_idx = time_idx
        self.__dict__.update(rec_dict.__dict__)
        self.rcond = rcond
        self.S = None

        # changing the nan location to unity before fitting using polar Slepians (will make them zero when taking log)
        self.VDF[np.isnan(self.VDF)] = 1e0

        # gyrotropized 2D VDF on a plane
        self.fine_from_fine = np.zeros((self.NENERGY, self.NTHETA_SLEP, self.NPHI_SLEP))

        # self.gen_Slepians_on_polarcap(self.Lmax)
        self.SLEP_coeffs = np.zeros((self.NENERGY, len(self.G)))

        self.recon_3D_VDF_MMS()

        # the final total fitted plot
        if(self.makeplot):
            mu_phi, mu_theta = self.ESA_PHI[time_idx,0,self.PHI_CEN_IDX,0],\
                               self.ESA_THETA[time_idx,0,0,self.THETA_CEN_IDX]

            # plotting the circle around the domain
            theta_circ = np.linspace(0, 2*np.pi, 100)
            xcirc, ycirc = self.TH * np.cos(theta_circ) + mu_phi, self.TH * np.sin(theta_circ) + mu_theta

            fig, ax = plt.subplots(4, 8, figsize=(16,8), sharex=True, sharey=True)
            for E_idx in range(self.NENERGY):
                self.plot_polar_rec_VDF(E_idx, ax[E_idx//8, E_idx%8], self.fine_from_fine[E_idx], xcirc, ycirc)

            plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
            # to put common x and y labels
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
            plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)
            plt.suptitle(f'{time_idx}')

            for axs in ax.flatten(): 
                axs.set_xlim([0, 360])
                axs.set_ylim([-90, 90])
                axs.set_aspect('equal')
            
            plt.savefig(f'VDF_paper1_plots/VDF_rec_polar_plot_MMS/{time_idx}.png')
            plt.close()


    # def gen_Slepians_on_polarcap(self, L, zonal_only=True):
    #     '''
    #     # generating the low resolution Slepians (NOT USED IN CURRENT IMPLEMENTATION)
    #     [G_lr, V_lr, lon_lr, lat_lr] = eng.glmalphapto('VDF_polarcap', self.Lmax, self.instrument, nargout=4)
    #     self.G_lr = np.asarray(G_lr)
    #     self.V_lr = np.asarray(V_lr)
    #     self.lon_lr = np.asarray(lon_lr)
    #     self.lat_lr = np.asarray(lat_lr)
    #     '''

    #     # generating the high resolution Slepians (USED IN CURRENT IMPLEMENTATION)
    #     # [G_hr, V_hr, lon_hr, lat_hr] = eng.glmalphapto('VDF_polarcap', self.Lmax, 'HIGHRES', nargout=4)
    #     [G_hr, V_hr, lon_hr, lat_hr] = eng.glmalphapto('VDF_polarcap_MMS', L, 'HIGHRES', nargout=4)
    #     self.G_hr = np.asarray(G_hr)
    #     self.V_hr = np.asarray(V_hr).squeeze()
    #     self.lon_hr = np.asarray(lon_hr)
    #     self.lat_hr = np.asarray(lat_hr)

        
    #     '''
    #     # keeping only until the Shannon number
    #     N2D = np.argmin(np.abs(self.V_hr - 0.5))
    #     self.G_hr = self.G_hr[:N2D]
    #     self.V_hr = self.V_hr[:N2D]
    #     '''


    def recon_3D_VDF_MMS(self):
        # looping over energy shells -> fitting polar Slepians
        for E_idx in range(self.NENERGY):
            E = self.ENERGY[self.time_idx, E_idx, 0, 0]
            vv = self.VDF[self.time_idx, E_idx, :, :] 
            logvv = np.log10(vv)
            logvv = np.nan_to_num(logvv, posinf=np.nan, neginf=np.nan)

            # # interpolating the data to Slepian grid before fitting polar Slepians (minor adjustments)
            # orig_phi_grid = self.ESA_PHI[self.time_idx, E_idx] - self.ESA_PHI[self.time_idx, E_idx, 0, 0]
            # orig_theta_grid = self.ESA_THETA[self.time_idx, E_idx]
            # img_hr = griddata((orig_phi_grid.flatten(), orig_theta_grid.flatten()), logvv.flatten(),
            #                   (self.SLEP_PHI, self.SLEP_THETA+90), method='linear')     # Possibly -90 not +90

            img_hr = np.flip(logvv.T, axis=0)

            # fitting the polar Slepians
            nan_mask_hr = np.isnan(img_hr)
            # img_hr[nan_mask_hr] = 0
            # nan_mask_hr = np.isnan(img_hr)
            G_nonan_hr = self.G[:,~nan_mask_hr]
            M = G_nonan_hr @ G_nonan_hr.T 
            __, self.S, __ = np.linalg.svd(M)
            I = np.identity(M.shape[0])
            self.SLEP_coeffs[E_idx] = np.linalg.inv(M +  self.S.max() * self.rcond * I) @ G_nonan_hr @ img_hr[~nan_mask_hr]

            # reconstructing from the polar Slepians and plotting
            fine_from_finecoefs = np.dot(np.moveaxis(self.G, 0, -1), self.SLEP_coeffs[E_idx])
            self.fine_from_fine[E_idx] += fine_from_finecoefs
        
        # adjusting the reconstruction to account for a flipped theta convention
        self.fine_from_fine = self.fine_from_fine[:,::-1,:]

    def plot_polar_rec_VDF(self, E_idx, ax, fine_from_finecoefs, xcirc, ycirc):
        vmin, vmax = 1, 5
        E = self.ENERGY[self.time_idx, E_idx, 0, 0]
        ax.pcolormesh(self.SLEP_PHI, self.SLEP_THETA, fine_from_finecoefs,
                      cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
        ax.plot(xcirc, ycirc, '--r')
        ax.set_aspect('equal')
        ax.set_xlim([0, 360])
        ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
                va='bottom', ha='left', color='white', fontweight='bold')

class VDF_rec_polarcaps_SphericalHarmonics:
    def __init__(self, rec_dict, time_idx, rcond=0.0):
        self.time_idx = time_idx
        self.__dict__.update(rec_dict.__dict__)
        self.rcond = rcond

        # these get flipped somehow when the Slepians are generated in Matlab
        self.N_lat_lr, self.N_lon_lr = self.NTHETA_ESA, self.NPHI_ESA           # StepI_bundle.lon_lr.T.shape
        self.N_lat_hr, self.N_lon_hr = self.NTmesh, self.NPmesh                 # StepI_bundle.lon_hr.T.shape

        self.lat_lr, self.lon_lr = np.meshgrid(np.linspace(0, 180, self.NTHETA_ESA), np.linspace(0, 360, self.NPHI_ESA), indexing='ij')
        self.lat_hr, self.lon_hr = np.meshgrid(np.linspace(0, 180, self.N_lat_hr), np.linspace(0, 360, self.N_lon_hr), indexing='ij')

        # changing the nan location to unity before fitting using polar Slepians (will make them zero when taking log)
        self.VDF[np.isnan(self.VDF)] = 1e0

        # gyrotropized 2D VDF on a plane
        self.fine_from_fine = np.zeros((self.NENERGY, self.NPHI_ESA, self.NTHETA_ESA))
        self.smooth_vdf = None

        self.SLEP_coeffs = np.zeros((self.NENERGY, (self.Lmax+1)**2), dtype='complex128')
        self.angular_extent_mask()

        self.recon_3D_VDF_MMS()

        # the final total fitted plot
        if(self.makeplot):
            fig, ax = plt.subplots(4, 8, figsize=(16,8), sharex=True, sharey=True)
            for E_idx in range(self.NENERGY):
                self.plot_polar_rec_VDF(E_idx, ax[E_idx//8, E_idx%8], self.fine_from_fine[E_idx])

            plt.subplots_adjust(top=0.96, bottom=0.05, left=0.03, right=0.99, wspace=0.05, hspace=0.05)
            # to put common x and y labels
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel(r'$v_{\phi} [{}^{\circ}]$', labelpad=0.01, fontsize=16)
            plt.ylabel(r'$v_{\theta} [{}^{\circ}]$', fontsize=16)
            plt.suptitle(f'{time_idx}')
            if(self.instrument=='MMS'):
                for axs in ax.flatten(): 
                    axs.set_xlim([0, 360])
                    axs.set_ylim([-90,90])
                    axs.set_aspect('equal')
            
            plt.savefig(f'VDF_paper1_plots/VDF_rec_polar_plot_MMS/{time_idx}.png')
            plt.close()

    def angular_extent_mask(self):
        # This function will apply a tanh filter at a set angular extent about the phi centroid identified in locate axis.
        # Take in the defined angular extent
        angular_extent = self.TH

        # Get the phi centroid
        phi_idx = self.PHI_CEN_IDX
        theta_idx = self.THETA_CEN_IDX

        theta_center = self.ESA_THETA[self.time_idx, 0, 0, theta_idx]
        phi_center = self.ESA_PHI[self.time_idx, 0, phi_idx, 0]

        # Now get the theta and phi for the given array
        Theta = self.ESA_THETA[self.time_idx, 0]        # 32 x 16 array
        Phi = self.ESA_PHI[self.time_idx, 0]            # 32 x 16 array

        # Define the OMEGA
        Omega = np.sqrt((Phi - phi_center)**2 + (Theta - theta_center)**2)

        # Get the unique values
        unique_omega = np.unique(Omega)

        filter = np.ones(len(unique_omega))
        filter[unique_omega > angular_extent] = np.tanh(np.radians(90 - unique_omega[unique_omega > angular_extent]))
        filter[filter < 0] = 0      # Kill off all power after this point.


        VDF_MASK = np.zeros_like(self.VDF[self.time_idx])
        for E_idx in range(self.NENERGY):
            VDF = self.VDF[self.time_idx, E_idx, :, :].copy()

            for i in range(len(unique_omega)):
                VDF[Omega == unique_omega[i]] = VDF[Omega == unique_omega[i]] * filter[i]

            VDF_MASK[E_idx] = VDF

        self.VDF_MASKED = VDF_MASK

        return(0)

    def recon_3D_VDF_MMS(self):
        # looping over energy shells -> fitting Spherical Harmonics
        for E_idx in range(self.NENERGY):
            E = self.ENERGY[self.time_idx, E_idx, 0, 0]
            vv = self.VDF[self.time_idx, E_idx, :, :]
            data = np.log10(vv)
            data = np.nan_to_num(data, posinf=np.nan, neginf=np.nan)
            
            img_hr = data * 1.0
            # interpolating the data to Slepian grid before fitting polar Slepians (minor adjustments)
            # orig_phi_grid = self.ESA_PHI[self.time_idx, E_idx] - self.ESA_PHI[self.time_idx, E_idx, 0, 0]
            # orig_theta_grid = self.ESA_THETA[self.time_idx, E_idx]
            # img_hr = griddata((orig_phi_grid.flatten(), orig_theta_grid.flatten()), data.flatten(),
            #                   (self.SLEP_PP, self.SLEP_TT+90), method='linear').T  
            
            # fitting the Spherical Harmonics
            nan_mask = np.isnan(img_hr)
            G = self.G * 1.0                    # This will be replaced with self.G in upcoming version
            G_mask = G[~nan_mask,:]
            M = np.conjugate(G_mask).T @ G_mask 
            __, self.S_hr, __ = np.linalg.svd(M)
            I = np.identity(M.shape[0])
            inverted_M = np.linalg.inv(M + self.S_hr.max() * self.rcond * I)
            self.SLEP_coeffs[E_idx] = inverted_M @ np.conjugate(G_mask).T @ img_hr[~nan_mask]

            # reconstructing from the polar Slepians and plotting
            rec_coefs = np.dot(G, self.SLEP_coeffs[E_idx])
            self.fine_from_fine[E_idx] = rec_coefs       

    def plot_polar_rec_VDF(self, E_idx, ax, fine_from_finecoefs):
        vmin, vmax = 1, 7
        E = self.ENERGY[self.time_idx, E_idx, 0, 0]
        ax.pcolormesh(self.SLEP_PP, self.SLEP_TT, fine_from_finecoefs,
                      cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
        # ax.plot(xcirc, ycirc, '--r')
        ax.set_aspect('equal')
        ax.set_xlim([0, 360])
        ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
                va='bottom', ha='left', color='white', fontweight='bold')