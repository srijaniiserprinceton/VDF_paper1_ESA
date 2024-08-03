import spherepy as sp
import numpy as np

class sph2slep:
    def __init__(self, Slep_dict):
        self.G = Slep_dict['G_slep']
        self.V = Slep_dict['V']
        self.N = Slep_dict['N']
        self.ell_arr = Slep_dict['EL'].astype('int')
        self.m_arr = Slep_dict['EM'].astype('int')
        self.Lmax = int(np.max(self.ell_arr))
        self.Nsleps = self.G.shape[0]

        sortidx = np.argsort(self.V)[::-1]
        self.G = self.G[sortidx]
        self.V = self.V[sortidx]

    def get_spherepy_coefs(self, alpha):
        # initializing the spherepy coefficient object with zeros
        coefs = sp.zeros_coefs(nmax=self.Lmax, mmax=self.Lmax)

        # looping over the ell and m to fill in the coefficients
        for sph_idx in range(len(self.ell_arr)):
            ell, m = self.ell_arr[sph_idx], self.m_arr[sph_idx]
            coefs[int(ell), int(m)] = self.Glm_alpha[sph_idx, alpha]
        return coefs

    def get_slep(self, Ntheta, Nphi):
        # the array to store the Slepian functions 
        G_sleps = np.zeros((self.Nsleps, Ntheta, Nphi))
        Lmax = np.max(self.ell_arr)
        Y_real = self.get_real_sph(Ntheta, Nphi).real

        # moving the axis of Y_real to facilitate matrix product
        Y_real = np.moveaxis(Y_real, 0, 1)

        # looping over different Slepian functions
        for alpha in range(self.Nsleps):
            G_sleps[alpha] = self.Glm_alpha[:,alpha] @ Y_real
            # coefs_alpha = self.get_spherepy_coefs(alpha)
            # G_sleps[alpha] = sp.ispht(coefs_alpha, nrows=Ntheta, ncols=Nphi).array
        
        # reordering according to eigenvalues
        sort_idx = np.argsort(self.V)[::-1]
        V_ordered = self.V[sort_idx]
        G_ordered = G_sleps[sort_idx]

        return G_ordered, V_ordered

    def get_real_sph(self, Ntheta, Nphi):
        # creating the array for storing complex spherical harmonics
        sph_real = np.zeros((self.Nsleps, Ntheta, Nphi), dtype='complex128')

        # first constructing the complex spherical harmonics
        # making it in the form of a dictionary
        Y = {}
        for sph_idx in range(self.Nsleps):
            coefs_arr = sp.zeros_coefs(nmax=self.Lmax, mmax=self.Lmax)
            ell, m = self.ell_arr[sph_idx], self.m_arr[sph_idx]
            coefs_arr[int(ell), int(m)] = 1.0
            Y[f'{ell},{m}'] = sp.ispht(coefs_arr, ncols=Nphi, nrows=Ntheta).array
        

        for sph_idx in range(self.Nsleps):
            ell, m = self.ell_arr[sph_idx], self.m_arr[sph_idx]
            if(m < 0):
                sph_real[sph_idx] = 1j/np.sqrt(2) *\
                                    (Y[f'{ell},{m}'] - np.power(-1, np.abs(m)) * Y[f'{ell},{-m}'])
            elif(m == 0):
                sph_real[sph_idx] = Y[f'{ell},{m}']
            else:
                sph_real[sph_idx] = 1/np.sqrt(2) *\
                                    (Y[f'{ell},{-m}'] + np.power(-1, np.abs(m)) * Y[f'{ell},{m}'])
            
        return sph_real

                                




