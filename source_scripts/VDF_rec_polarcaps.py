import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
plt.ion()

import matlab.engine as matlab
eng = matlab.start_matlab()
s = eng.genpath('/Users/srijanbharatidas/Documents/Research/Codes/Helioseismology/Slepians/Slepian_Git')
eng.addpath(s, nargout=0)

class VDF_rec_polarcaps:
    def __init__(self, DATA, StepI_bundle, Lmax=12, rcond=0.0):
        self.DATA = DATA
        self.__dict__.update(StepI_bundle.__dict__)
        self.Lmax = Lmax
        self.rcond = rcond
        self.G_lr, self.V_lr, self.lat_lr, self.lon_lr = None, None, None, None
        self.G_hr, self.V_hr, self.lat_hr, self.lon_hr = None, None, None, None

        # changing the nan location to unity before fitting using polar Slepians (will make them zero when taking log)
        self.DATA.VDF[np.isnan(self.DATA.VDF)] = 1e0
        self.N_Eshells = self.DATA.VDF.shape[0]

        # generating the low and high resolution Slepians-on-polar-cap
        self.gen_Slepians_on_polarcap()

        __, self.N_lat_lr, self.N_lon_lr = self.G_lr.shape
        __, self.N_lat_hr, self.N_lon_hr = self.G_hr.shape

        self.tt_lr_idx, self.pp_lr_idx = np.meshgrid(np.linspace(0, 180, self.N_lat_lr), np.linspace(0, 360, self.N_lon_lr), indexing='ij')
        self.tt_hr_idx, self.pp_hr_idx = np.meshgrid(np.linspace(0, 180, self.N_lat_hr), np.linspace(0, 360, self.N_lon_hr), indexing='ij')

        # gyrotropized 2D VDF on a plane
        self.VDF_2D = np.zeros((self.N_Eshells, self.N_lat_hr))
        self.gyrotropic_recon_3D_VDF()

        # generating the 2D velocity grid 
        self.V1, self.V2 = None, None
        self.generate_2D_Vgrid()

    def gen_Slepians_on_polarcap(self):
        # generating the low resolution Slepians (NOT USED IN CURRENT IMPLEMENTATION)
        [G_lr, V_lr, lon_lr, lat_lr] = eng.glmalphapto('VDF_polarcap', self.Lmax, self.instrument, nargout=4)
        self.G_lr = np.asarray(G_lr)
        self.V_lr = np.asarray(V_lr)
        self.lon_lr = np.asarray(lon_lr)
        self.lat_lr = np.asarray(lat_lr)
        

        # generating the high resolution Slepians (USED IN CURRENT IMPLEMENTATION)
        [G_hr, V_hr, lon_hr, lat_hr] = eng.glmalphapto('VDF_polarcap', self.Lmax, 'HIGHRES', nargout=4)
        self.G_hr = np.asarray(G_hr)
        self.V_hr = np.asarray(V_hr)
        self.lon_hr = np.asarray(lon_hr)
        self.lat_hr = np.asarray(lat_hr)

    def gyrotropic_recon_3D_VDF(self):
        # finding the nearest index for phi0 to store the gyrotropized VDF in a plane
        phi0_idx = np.argmin(np.abs(self.lon_hr[0] - self.phi0))

        # looping over energy shells -> fitting polar Slepians
        for E_idx in range(self.N_Eshells):
            E = self.DATA.ENERGY[E_idx, 0, 0]
            vv = self.DATA.VDF[E_idx, :, :] 
            data_vv = np.log10(vv)
            data = np.zeros((self.N_lat_lr, self.N_lon_lr)) + np.nan
            # tiling the SPAN-Ai data in the correct location
            data[2:10, 8:16] = data_vv.T

            # interpolating the data to higher resolution before fitting polar Slepians
            img_hr = griddata((self.tt_lr_idx.flatten(), self.pp_lr_idx.flatten()), data.flatten(),
                              (self.tt_hr_idx, self.pp_hr_idx), method='linear')

            # fitting the polar Slepians
            nan_mask_hr = np.isnan(img_hr)
            G_nonan_hr = self.G_hr[:,~nan_mask_hr]
            M_hr = G_nonan_hr @ G_nonan_hr.T 
            __, S_hr, __ = np.linalg.svd(M_hr)
            I_hr = np.identity(M_hr.shape[0])
            coeffs_hr = np.linalg.inv(G_nonan_hr @ G_nonan_hr.T +  S_hr.max() * self.rcond * I_hr) @ G_nonan_hr @ img_hr[~nan_mask_hr]

            # reconstructing from the polar Slepians and plotting
            fine_from_finecoefs = np.dot(np.moveaxis(self.G_hr, 0, -1), coeffs_hr)

            # saving the 2D VDF by taking a slice along the nearest phi grid to phi0
            self.VDF_2D[E_idx] = fine_from_finecoefs[:, phi0_idx]

        # rolling the VDF in theta to adjust the theta center for gyrotropy in Cartesian
        roll_theta_idx = -int(self.theta0 - 90)
        self.VDF_2D = np.roll(self.VDF_2D, roll_theta_idx, axis=1)

    def generate_2D_Vgrid(self):
        # converting grids to velocity space
        m_p = 0.010438870      #eV/c^2 where c = 299792 km/s
        q_p = 1 
        vmag = np.sqrt(2 * q_p * self.DATA.ENERGY[:, 0, 0] / m_p)   # in km/s

        theta_hr = self.lat_hr[:,0]
        self.V1 = vmag[:, np.newaxis] * np.cos(theta_hr[np.newaxis,:] * np.pi/180)
        self.V2 = vmag[:, np.newaxis] * np.sin(theta_hr[np.newaxis:,] * np.pi/180)
