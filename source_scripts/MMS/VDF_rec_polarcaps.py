import numpy as np
from scipy.interpolate import griddata
from scipy.io import savemat
import matplotlib.pyplot as plt
plt.ion()

import matlab.engine as matlab
# generating the low and high resolution Slepians-on-polar-cap
eng = matlab.start_matlab()
s = eng.genpath('/Users/srijanbharatidas/Documents/Research/Codes/Helioseismology/Slepians/Slepian_Git')
eng.addpath(s, nargout=0)

import generate_2D_contour as gen_contour

class VDF_rec_polarcaps:
    def __init__(self, DATA, StepI_bundle, time_idx, instrument='MMS', Lmin=8, Lmax=12, rcond=0.0, iterative_fit=True, makeplot=True):
        self.DATA = DATA
        self.time_idx = time_idx
        self.__dict__.update(StepI_bundle.__dict__)
        self.Lmin, self.Lmax = Lmin, Lmax
        self.rcond = rcond
        self.instrument = instrument
        self.makeplot = makeplot
        self.G_lr, self.V_lr = None, None
        self.G_hr, self.V_hr = None, None
        self.S_hr = None

        # these get flipped somehow when the Slepians are generated in Matlab
        self.N_lat_lr, self.N_lon_lr = StepI_bundle.lon_lr.T.shape
        self.N_lat_hr, self.N_lon_hr = StepI_bundle.lon_hr.T.shape

        self.tt_lr_idx, self.pp_lr_idx = np.meshgrid(np.linspace(0, 180, self.N_lat_lr), np.linspace(0, 360, self.N_lon_lr), indexing='ij')
        self.tt_hr_idx, self.pp_hr_idx = np.meshgrid(np.linspace(0, 180, self.N_lat_hr), np.linspace(0, 360, self.N_lon_hr), indexing='ij')

        # changing the nan location to unity before fitting using polar Slepians (will make them zero when taking log)
        self.DATA.VDF[np.isnan(self.DATA.VDF)] = 1e0
        self.N_Eshells = self.DATA.VDF.shape[0]

        # gyrotropized 2D VDF on a plane
        self.VDF_2D = np.zeros((self.N_Eshells, self.N_lat_hr))
        self.fine_from_fine = np.zeros((self.N_Eshells, self.N_lat_hr, self.N_lon_hr))

        # performing iterative fitting
        if(iterative_fit):
            for L in range(self.Lmin, self.Lmax + 1):
                self.gen_Slepians_on_polarcap(L)
                if(L == self.Lmin):
                    self.G_hr = np.reshape(self.G_hr[0,:,:], (1, self.N_lat_hr, self.N_lon_hr))
                else:
                    self.G_hr = self.G_hr[1:,:,:]
                # performing the iterative fitting with the chosen eigenfunctions
                self.gyrotropic_recon_3D_VDF()

        else:
            self.gen_Slepians_on_polarcap(self.Lmax)
            print(self.lon_hr, self.lat_hr)
            plt.figure()
            plt.pcolormesh(self.lon_hr, self.lat_hr, self.G_hr[1])
            # plotting the circle around the domain
            theta_circ = np.linspace(0, 2*np.pi, 100)
            xcirc, ycirc = self.TH * np.cos(theta_circ) + self.phi0, self.TH * np.sin(theta_circ) + (self.theta0-90)
            plt.plot(xcirc, ycirc, '--r')
            plt.scatter(self.phi0, self.theta0 - 90, marker='x', color='white')
            plt.text(0.05, 0.05, f'({self.phi0:.2f}, {self.theta0:.2f}) [eV]', transform=plt.gca().transAxes,
                     va='bottom', ha='left', color='black', fontweight='bold')
            plt.gca().set_aspect('equal')
            plt.colorbar()
            self.gyrotropic_recon_3D_VDF_MMS()

        # the final total fitted plot
        if(self.makeplot):
            fig, ax = plt.subplots(4, 8, figsize=(16,8), sharex=True, sharey=True)
            for E_idx in range(self.N_Eshells):
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
                    axs.set_ylim([-90, 90])
                    axs.set_aspect('equal')
            
            plt.savefig(f'VDF_paper1_plots/VDF_rec_polar_plot_MMS/{time_idx}.png')
            # plt.close()

        # finding the nearest index for phi0 to store the gyrotropized VDF in a plane
        phi0_idx = np.argmin(np.abs(self.lon_hr[0] - self.phi0))
        # saving the 2D VDF by taking a slice along the nearest phi grid to phi0
        self.VDF_2D = self.fine_from_fine[:, :, phi0_idx]


    def gen_Slepians_on_polarcap(self, L, zonal_only=True):
        '''
        # generating the low resolution Slepians (NOT USED IN CURRENT IMPLEMENTATION)
        [G_lr, V_lr, lon_lr, lat_lr] = eng.glmalphapto('VDF_polarcap', self.Lmax, self.instrument, nargout=4)
        self.G_lr = np.asarray(G_lr)
        self.V_lr = np.asarray(V_lr)
        self.lon_lr = np.asarray(lon_lr)
        self.lat_lr = np.asarray(lat_lr)
        '''

        # generating the high resolution Slepians (USED IN CURRENT IMPLEMENTATION)
        # [G_hr, V_hr, lon_hr, lat_hr] = eng.glmalphapto('VDF_polarcap', self.Lmax, 'HIGHRES', nargout=4)
        [G_hr, V_hr, lon_hr, lat_hr] = eng.glmalphapto('VDF_polarcap_MMS', L, 'HIGHRES', nargout=4)
        self.G_hr = np.asarray(G_hr)
        self.V_hr = np.asarray(V_hr).squeeze()
        self.lon_hr = np.asarray(lon_hr)
        self.lat_hr = np.asarray(lat_hr)

        '''
        # keeping only until the Shannon number
        N2D = np.argmin(np.abs(self.V_hr - 0.5))
        self.G_hr = self.G_hr[:N2D]
        self.V_hr = self.V_hr[:N2D]
        '''

    def gyrotropic_recon_3D_VDF_MMS(self):
        # if(self.makeplot): fig, ax = plt.subplots(4, 8, figsize=(16,8), sharex=True, sharey=True)

        # looping over energy shells -> fitting polar Slepians
        for E_idx in range(self.N_Eshells):
            # E_idx = 15
            # print(E_idx)
            E = self.DATA.ENERGY[E_idx, 0, 0]
            vv = self.DATA.VDF[E_idx, :, :] 
            data_vv = np.log10(vv)
            data_vv = np.nan_to_num(data_vv, posinf=np.nan, neginf=np.nan)
            data = np.zeros((self.N_lat_lr, self.N_lon_lr))# + np.nan
            # tiling the MMS-ion data in the correct location
            data = data_vv

            # interpolating the data to higher resolution before fitting polar Slepians
            # img_hr = griddata((self.tt_lr_idx.flatten(), self.pp_lr_idx.flatten()), data.flatten(),
            #                   (self.tt_hr_idx, self.pp_hr_idx), method='linear')
            img_hr = griddata((self.lon_lr.flatten(), self.lat_lr.flatten()), data.flatten(),
                              (self.lon_hr, self.lat_hr), method='linear')

            # removing the previously fitting part 
            img_hr = img_hr - self.fine_from_fine[E_idx]

            # fitting the polar Slepians
            nan_mask_hr = np.isnan(img_hr)
            G_nonan_hr = self.G_hr[:,~nan_mask_hr]
            M_hr = G_nonan_hr @ G_nonan_hr.T 
            __, self.S_hr, __ = np.linalg.svd(M_hr)
            I_hr = np.identity(M_hr.shape[0])
            coeffs_hr = np.linalg.inv(G_nonan_hr @ G_nonan_hr.T +  self.S_hr.max() * self.rcond * I_hr) @ G_nonan_hr @ img_hr[~nan_mask_hr]

            # reconstructing from the polar Slepians and plotting
            fine_from_finecoefs = np.dot(np.moveaxis(self.G_hr, 0, -1), coeffs_hr)
            self.fine_from_fine[E_idx] += fine_from_finecoefs

            # if(self.makeplot): self.plot_polar_rec_VDF(E_idx, ax[E_idx//8, E_idx%8], fine_from_finecoefs)

    def plot_polar_rec_VDF(self, E_idx, ax, fine_from_finecoefs):
        vmin, vmax = 1, 7
        E = self.DATA.ENERGY[E_idx, 0, 0]
        ax.pcolormesh(self.lon_hr, self.lat_hr, fine_from_finecoefs,
                      cmap='plasma', vmin=vmin, vmax=vmax, rasterized=True)
        theta_circ = np.linspace(0, 2*np.pi, 100)
        xcirc, ycirc = self.TH * np.cos(theta_circ) + self.phi0, self.TH * np.sin(theta_circ) + (self.theta0-90)
        ax.plot(xcirc, ycirc, '--r')
        ax.scatter(self.phi0, self.theta0 - 90, marker='x', color='black')
        ax.set_aspect('equal')
        ax.set_xlim([0, 360])
        ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
                va='bottom', ha='left', color='black', fontweight='bold')

