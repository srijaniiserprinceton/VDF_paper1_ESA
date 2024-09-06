import spherepy as sp
import numpy as np
from scipy.io import savemat

class sph2slep:
    def __init__(self, Slep_dict):
        self.G = Slep_dict['G_slep']
        self.V = Slep_dict['V']
        self.N = Slep_dict['N']
        self.EL = Slep_dict['EL']
        self.EM = Slep_dict['EM']
        self.ell_arr = Slep_dict['EL'].astype('int')
        self.m_arr = Slep_dict['EM'].astype('int')
        self.Lmax = int(np.max(self.ell_arr))
        self.Nsleps = self.G.shape[0]

        sortidx = np.argsort(self.V)[::-1]
        self.G = self.G[sortidx]
        self.V = self.V[sortidx]
        self.EL = self.EL[sortidx]
        self.EM = self.EM[sortidx]

class get_StepI_dict:
    def __init__(self, phi0, theta0, TH, phi_ESA, theta_ESA, instrument='MMS', save_odd_grid=True):
        self.phi0, self.theta0, self.TH = phi0, theta0, TH
        self.phi_ESA, self.theta_ESA = phi_ESA, theta_ESA
        self.instrument = instrument
        self.save_odd_grid = save_odd_grid

        # the low resolution and high resolution grids
        self.lon_lr, self.lat_lr = None, None
        self.lon_hr, self.lat_hr = None, None

        self.save_Slepian_grid_MMS()

    def save_Slepian_grid_MMS(self):
        # make theta range 90 -> -90
        phi_Slepian, theta_Slepian = self.get_Slepian_grid_MMS()
        pp_Slep, tt_Slep = np.meshgrid(phi_Slepian, theta_Slepian, indexing='ij')
        # since the Slepian code takes the grids as flattened point arrays
        pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
        Nphi, Ntheta = pp_Slep.shape

        self.lon_lr, self.lat_lr = pp_Slep, tt_Slep

        #----------------------making a higher resolution theta phi grid----------------------------------#
        # phi_Slepian, theta_Slepian = np.linspace(0, 360, 361), np.linspace(0, 180, 181)
        phi_Slepian, theta_Slepian = np.linspace(0, 360, 32), np.linspace(0, 180, 16)
        # reversing the order of theta_Slepian since the matlab code wants latitude from [90,-90]
        theta_Slepian = theta_Slepian - 90
        pp_Slep, tt_Slep = np.meshgrid(phi_Slepian, theta_Slepian, indexing='ij')
        pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
        Nphi, Ntheta = pp_Slep.shape

        self.lon_hr, self.lat_hr = pp_Slep, tt_Slep

        # saving these files as matlab readable arrays
        mdict = {'phi0': self.phi0, 'theta0': self.theta0, 'cap_extent': self.TH, 'phi_grid': pp_Slep_flat,
                'theta_grid': tt_Slep_flat, 'Nphi': Nphi, 'Ntheta': Ntheta}
        savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{self.instrument}_HIGHRES.mat', mdict)

    def get_Slepian_grid_MMS(self):
        # reversing the order of theta_Slepian since the matlab code wants latitude from [90,-90]
        return self.phi_ESA, self.theta_ESA[::-1] - 90




