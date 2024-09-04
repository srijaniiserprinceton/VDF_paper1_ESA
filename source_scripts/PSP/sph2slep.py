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
    def __init__(self, phi0, theta0, TH, phi_ESA, theta_ESA, instrument='PSP-SPAN', save_odd_grid=True):
        self.phi0, self.theta0, self.TH = phi0, theta0, TH
        self.phi_ESA, self.theta_ESA = phi_ESA, theta_ESA
        self.instrument = instrument
        self.save_odd_grid = save_odd_grid

        # the low resolution and high resolution grids
        self.lon_lr, self.lat_lr = None, None
        self.lon_hr, self.lat_hr = None, None

        if(self.instrument=='PSP-SPAN'): 
            self.get_Slepian_grid = self.get_Slepian_grid_SPAN
            self.save_Slepian_grid_SPAN()

    def save_Slepian_grid_SPAN(self):
        # padding the raw SPAN grid to make phi range 0 -> 360 and theta range 90 -> -90
        phi_Slepian, theta_Slepian = self.get_Slepian_grid()
        pp_Slep, tt_Slep = np.meshgrid(phi_Slepian, theta_Slepian, indexing='ij')
        # since the Slepian code takes the grids as flattened point arrays
        pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
        Nphi, Ntheta = pp_Slep.shape

        # generating an odd grid using interpolation (NOTETHAT: DATA TO GRID MAPPING NOT EXACT)
        if(self.save_odd_grid):
            phi_grid_odd = np.linspace(phi_Slepian.min(), phi_Slepian.max(), Nphi + 1)
            theta_grid_odd = np.linspace(theta_Slepian.min(), theta_Slepian.max(), Ntheta + 1)[::-1] # since it goes from 90 -> -90
            pp_Slep, tt_Slep = np.meshgrid(phi_grid_odd, theta_grid_odd, indexing='ij')
            pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
            Nphi, Ntheta = pp_Slep.shape

        self.lon_lr, self.lat_lr = pp_Slep, tt_Slep

        # saving the SPAN resolution grid for Slepian generation in matlab readable arrays
        mdict = {'phi0': self.phi0, 'theta0': self.theta0, 'cap_extent': self.TH, 'phi_grid': pp_Slep_flat,
                'theta_grid': tt_Slep_flat, 'Nphi': Nphi, 'Ntheta': Ntheta}
        savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{self.instrument}.mat', mdict)

        #----------------------making a higher resolution theta phi grid----------------------------------#
        phi_Slepian, theta_Slepian = np.linspace(0, 360, 361), np.linspace(0, 180, 181)
        # reversing the order of theta_Slepian since the matlab code wants latitude from [90,-90]
        theta_Slepian = 90 - theta_Slepian
        pp_Slep, tt_Slep = np.meshgrid(phi_Slepian, theta_Slepian, indexing='ij')
        pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
        Nphi, Ntheta = pp_Slep.shape

        self.lon_hr, self.lat_hr = pp_Slep, tt_Slep

        # saving these higher resolution grid files as matlab readable arrays
        mdict = {'phi0': self.phi0, 'theta0': self.theta0, 'cap_extent': self.TH, 'phi_grid': pp_Slep_flat,
                'theta_grid': tt_Slep_flat, 'Nphi': Nphi, 'Ntheta': Ntheta}
        savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{self.instrument}_HIGHRES.mat', mdict)

    def get_Slepian_grid_SPAN(self):
        # trying to find the average spacing of the grids
        phi_diff_avg = np.mean(np.diff(self.phi_ESA))
        theta_diff_avg = np.mean(np.diff(self.theta_ESA))

        # now building the adjacent grids in phi: (0, phi_min) + (phi_max, 180)
        num_phi_before = int(self.phi_ESA[0] / phi_diff_avg) + 1
        phi_before = np.linspace(0, self.phi_ESA[0], num_phi_before)

        num_phi_after = int((360 - self.phi_ESA[-1]) / phi_diff_avg) + 1
        phi_after = np.linspace(self.phi_ESA[-1], 360, num_phi_after)

        phi_Slepian = np.append(phi_before[:-1], self.phi_ESA)
        phi_Slepian = np.append(phi_Slepian, phi_after[1:])

        # now building the adjacent grids in theta: (0, theta_min) + (theta_max, 180)
        num_theta_before = int(self.theta_ESA[0] / theta_diff_avg) + 1
        theta_before = np.linspace(0, self.theta_ESA[0], num_theta_before)

        num_theta_after = int((180 - self.theta_ESA[-1]) / theta_diff_avg) + 1
        theta_after = np.linspace(self.theta_ESA[-1], 180, num_theta_after)

        theta_Slepian = np.append(theta_before[:-1], self.theta_ESA)
        theta_Slepian = np.append(theta_Slepian, theta_after[1:])

        # reversing the order of theta_Slepian since the matlab code wants latitude from [90,-90]
        theta_Slepian = 90 - theta_Slepian

        return phi_Slepian, theta_Slepian




