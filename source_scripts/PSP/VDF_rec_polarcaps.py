import sys
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
    def __init__(self, rec_dict, StepI_bundle, time_idx, instrument='PSP-SPAN', Lmin=8, Lmax=12, rcond=0.0, iterative_fit=True, makeplot=True):
        self.rec_dict = rec_dict
        self.time_idx = time_idx
        self.__dict__.update(StepI_bundle.__dict__)
        self.Lmin, self.Lmax = Lmin, Lmax
        self.rcond = rcond
        self.instrument = instrument
        self.makeplot = makeplot
        self.G_lr, self.V_lr = None, None
        self.G_hr, self.V_hr = None, None
        self.S_hr = None

        # making a dictionary of all Slepians used
        self.G_all = {}

        # these get flipped somehow when the Slepians are generated in Matlab
        self.N_lat_lr, self.N_lon_lr = StepI_bundle.lon_lr.T.shape
        self.N_lat_hr, self.N_lon_hr = StepI_bundle.lon_hr.T.shape

        self.tt_lr_idx, self.pp_lr_idx = np.meshgrid(np.linspace(0, 180, self.N_lat_lr), np.linspace(0, 360, self.N_lon_lr), indexing='ij')
        self.tt_hr_idx, self.pp_hr_idx = np.meshgrid(np.linspace(0, 180, self.N_lat_hr), np.linspace(0, 360, self.N_lon_hr), indexing='ij')

        # changing the nan location to unity before fitting using polar Slepians (will make them zero when taking log)
        self.rec_dict.VDF[time_idx, np.isnan(self.rec_dict.VDF[time_idx])] = 1e0
        self.N_Eshells = self.rec_dict.VDF[time_idx].shape[0]

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

                # storing the Slepians used for SPC joint inversion
                self.G_all[f'{L}'] = self.G_hr * 1.0 

                # performing the iterative fitting with the chosen eigenfunctions
                self.gyrotropic_recon_3D_VDF()

        else:
            self.gen_Slepians_on_polarcap(self.Lmax)
            self.gyrotropic_recon_3D_VDF()

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
            
            plt.savefig(f'VDF_paper1_plots/VDF_rec_polar_plot_{self.instrument}/{time_idx}.png')
            # plt.close()

        # finding the nearest index for phi0 to store the gyrotropized VDF in a plane
        phi0_idx = np.argmin(np.abs(self.lon_hr[0] - self.phi0))
        # saving the 2D VDF by taking a slice along the nearest phi grid to phi0
        self.VDF_2D = self.fine_from_fine[:, :, phi0_idx]

        
        # rolling the VDF in theta to adjust the theta center for gyrotropy in Cartesian
        roll_theta_idx = int(self.theta0 - 90)
        self.VDF_2D = np.roll(self.VDF_2D, roll_theta_idx, axis=1)
        

        # generating the 2D velocity grid 
        self.V1, self.V2 = None, None
        self.generate_2D_Vgrid()
        # generating the contour for Cartesian Slepians
        self.generate_cartesian_contour()

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
        [G_hr, V_hr, lon_hr, lat_hr] = eng.glmalphapto('VDF_polarcap', L, f'{self.instrument}_HIGHRES', nargout=4)
        self.G_hr = np.asarray(G_hr)
        self.V_hr = np.asarray(V_hr)
        self.lon_hr = np.asarray(lon_hr)
        self.lat_hr = np.asarray(lat_hr)

        '''
        # only retaining the axisymmetric Slepians-on-a-polar-cap
        if(zonal_only):
            zonal_idx = np.array([0, 5, 14, 29])
            self.G_hr = self.G_hr[zonal_idx]
        '''

    def gyrotropic_recon_3D_VDF(self):
        # looping over energy shells -> fitting polar Slepians
        for E_idx in range(self.N_Eshells):
            E = self.rec_dict.ENERGY[self.time_idx, E_idx, 0, 0]
            vv = self.rec_dict.VDF[self.time_idx, E_idx, :, :] 
            data_vv = np.log10(vv)
            data = np.zeros((self.N_lat_lr, self.N_lon_lr)) + np.nan
            # tiling the SPAN-Ai data in the correct location
            data[2:10, 8:16] = data_vv.T

            # interpolating the data to higher resolution before fitting polar Slepians
            img_hr = griddata((self.tt_lr_idx.flatten(), self.pp_lr_idx.flatten()), data.flatten(),
                              (self.tt_hr_idx, self.pp_hr_idx), method='linear')

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


    def generate_2D_Vgrid(self):
        # converting grids to velocity space
        m_p = 0.010438870      #eV/c^2 where c = 299792 km/s
        q_p = 1 
        vmag = np.sqrt(2 * q_p * self.rec_dict.ENERGY[self.time_idx, :, 0, 0] / m_p)   # in km/s

        theta_hr = self.lat_hr[:,0]
        self.V1 = vmag[:, np.newaxis] * np.cos(theta_hr[np.newaxis,:] * np.pi/180)
        self.V2 = vmag[:, np.newaxis] * np.sin(theta_hr[np.newaxis:,] * np.pi/180)

    def generate_cartesian_contour(self):
        '''
        # setting up the grid and interpolating to compare the fitting with
        x, y = np.ravel(self.V1, 'F'), np.ravel(self.V2, 'F')
        z = np.ravel(self.VDF_2D, 'F')

        X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                           np.linspace(y.min(), y.max(), 100))
        Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

        #  setting up the model and indicating independent variables
        gencontourdemo = gen_contour.gen_contour(z, x, y, 'Gaussian_ycentered')


        # fitting the model with the data
        result = gencontourdemo.fit_2D_VDF()
        fit = gencontourdemo.model.func(X, Y, **result.best_values)
        '''

        plt.figure()
        self.VDF_2D[:6] = np.nan
        # img = plt.contourf(X, Y, fit, cmap='gnuplot2', vmin=-1e-3, vmax=6, levels=[1.0, 6.0])
        img = plt.contourf(self.V1, self.V2, self.VDF_2D, cmap='gnuplot2', vmin=0.0, vmax=100, levels=[1.0, 100.0])
        plt.close()
        p = img.collections[0].get_paths()[0]
        v = p.vertices
        x = v[:,0]
        y = v[:,1]

        # because we symmetrize in theta, we make a mandatorily symmetric domain
        Xsym, Ysym = x[y>0] * 1.0, y[y>0] * 1.0

        roll_idx = np.argmax(np.diff(Xsym)) + 1
        Xsym = np.roll(Xsym, -roll_idx)
        Ysym = np.roll(Ysym, -roll_idx)

        # adding the symmetric counterpart
        Xsym = np.append(Xsym, Xsym[::-1])
        Ysym = np.append(Ysym, -1. * Ysym[::-1])

        # to complete the curve
        Xsym = np.append(Xsym, Xsym[0])
        Ysym = np.append(Ysym, Ysym[0])

        plt.figure(); plt.plot(Xsym, Ysym, 'k')
        plt.plot(Xsym, Ysym, '--r')

        # storing the contour
        # curve2storeXY = {'X': v.T[0], 'Y': v.T[1]}
        curve2storeXY = {'X': Xsym, 'Y': Ysym}
        savemat('XY_pts.mat', curve2storeXY)

        '''
        plt.figure()
        plt.contourf(self.V1, self.V2, self.VDF_2D, cmap='gnuplot2', vmin=0.0, vmax=100, levels=[1.0, 100.0])
        np.save('VDF_2D.npy', self.VDF_2D)
        np.save('V1.npy', self.V1)
        np.save('V2.npy', self.V2)
        plt.plot(v.T[0], v.T[1], 'k')
        plt.savefig(f'VDF_paper1_plots/VDF_rec_polar_plot/contour_{self.time_idx}.png')
        plt.close()
        '''

        # storing the evaluation points
        evalpts2storeXY = {'XP': np.ravel(self.V1,'F'), 'YP': np.ravel(self.V2,'F')}
        savemat('XYP.mat', evalpts2storeXY)

    def plot_polar_rec_VDF(self, E_idx, ax, fine_from_finecoefs):
        vmin, vmax = 0, 7
        E = self.rec_dict.ENERGY[self.time_idx, E_idx, 0, 0]
        ax.pcolormesh(self.lon_hr, self.lat_hr + 90, fine_from_finecoefs,
                      cmap='BuPu', vmin=vmin, vmax=vmax, rasterized=True)
        ax.scatter(self.phi0, self.theta0-90, marker='o', color='orange', s=2)
        ax.set_aspect('equal')
        ax.set_xlim([75, 275])
        ax.text(0.05, 0.05, f'{E:.2f} [eV]', transform=ax.transAxes,
                va='bottom', ha='left', color='black', fontweight='bold')

