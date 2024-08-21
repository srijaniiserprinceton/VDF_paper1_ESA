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

def save_Slepian_grid(mu_phi, mu_theta, phi_ESA, theta_ESA, instrument='SPAN', save_odd_grid=True):
    # padding the raw SPAN grid to make phi range 0 -> 360 and theta range 90 -> -90
    phi_Slepian, theta_Slepian = get_Slepian_grid(phi_ESA, theta_ESA, instrument)
    pp_Slep, tt_Slep = np.meshgrid(phi_Slepian, theta_Slepian, indexing='ij')
    # since the Slepian code takes the grids as flattened point arrays
    pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
    Nphi, Ntheta = pp_Slep.shape

    # generating an odd grid using interpolation (NOTETHAT: DATA TO GRID MAPPING NOT EXACT)
    if(save_odd_grid):
        phi_grid_odd = np.linspace(phi_Slepian.min(), phi_Slepian.max(), Nphi + 1)
        theta_grid_odd = np.linspace(theta_Slepian.min(), theta_Slepian.max(), Ntheta + 1)[::-1] # since it goes from 90 -> -90
        pp_Slep, tt_Slep = np.meshgrid(phi_grid_odd, theta_grid_odd, indexing='ij')
        pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
        Nphi, Ntheta = pp_Slep.shape

    # saving the SPAN resolution grid for Slepian generation in matlab readable arrays
    mdict = {'phi0': mu_phi, 'theta0': mu_theta, 'cap_extent': 45, 'phi_grid': pp_Slep_flat,
            'theta_grid': tt_Slep_flat, 'Nphi': Nphi, 'Ntheta': Ntheta}
    savemat(f'./input_data_files/Slepian_functions/slepgen_grid_{instrument}.mat', mdict)

    # making a higher density theta phi grid
    phi_Slepian, theta_Slepian = np.linspace(0, 360, 361), np.linspace(0, 180, 181)
    # reversing the order of theta_Slepian since the matlab code wants latitude from [90,-90]
    theta_Slepian = 90 - theta_Slepian
    pp_Slep, tt_Slep = np.meshgrid(phi_Slepian, theta_Slepian, indexing='ij')
    pp_Slep_flat, tt_Slep_flat = pp_Slep.flatten(), tt_Slep.flatten()
    Nphi, Ntheta = pp_Slep.shape

    # saving these files as matlab readable arrays
    mdict = {'phi0': mu_phi, 'theta0': mu_theta, 'cap_extent': 45, 'phi_grid': pp_Slep_flat,
             'theta_grid': tt_Slep_flat, 'Nphi': Nphi, 'Ntheta': Ntheta}
    savemat(f'./input_data_files/Slepian_functions/slepgen_grid_HIGHRES.mat', mdict)

def get_Slepian_grid(phi_ESA, theta_ESA, instrument):
    if(instrument == 'SPAN'): return get_Slepian_grid_SPAN(phi_ESA, theta_ESA)

def get_Slepian_grid_SPAN(phi_ESA, theta_ESA):
    # trying to find the average spacing of the grids
    phi_diff_avg = np.mean(np.diff(phi_ESA))
    theta_diff_avg = np.mean(np.diff(theta_ESA))

    # now building the adjacent grids in phi: (0, phi_min) + (phi_max, 180)
    num_phi_before = int(phi_ESA[0] / phi_diff_avg) + 1
    phi_before = np.linspace(0, phi_ESA[0], num_phi_before)

    num_phi_after = int((360 - phi_ESA[-1]) / phi_diff_avg) + 1
    phi_after = np.linspace(phi_ESA[-1], 360, num_phi_after)

    phi_Slepian = np.append(phi_before[:-1], phi_ESA)
    phi_Slepian = np.append(phi_Slepian, phi_after[1:])

    # now building the adjacent grids in theta: (0, theta_min) + (theta_max, 180)
    num_theta_before = int(theta_ESA[0] / theta_diff_avg) + 1
    theta_before = np.linspace(0, theta_ESA[0], num_theta_before)

    num_theta_after = int((180 - theta_ESA[-1]) / theta_diff_avg) + 1
    theta_after = np.linspace(theta_ESA[-1], 180, num_theta_after)

    theta_Slepian = np.append(theta_before[:-1], theta_ESA)
    theta_Slepian = np.append(theta_Slepian, theta_after[1:])

    # reversing the order of theta_Slepian since the matlab code wants latitude from [90,-90]
    theta_Slepian = 90 - theta_Slepian

    return phi_Slepian, theta_Slepian




