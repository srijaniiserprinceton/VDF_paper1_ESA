import spherepy as sp
import numpy as np

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

def get_Slepian_grid(phi_ESA, theta_ESA):
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

    return phi_Slepian, theta_Slepian




