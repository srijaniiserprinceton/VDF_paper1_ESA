import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import griddata

#---------------operating Matlab from within Python------------------#
import matlab.engine as matlab
eng = matlab.start_matlab()
s = eng.genpath('/Users/srijanbharatidas/Documents/Research/Codes/Helioseismology/Slepians/Slepian_Git')
eng.addpath(s, nargout=0)

class VDF_rec_cartesian:
    def __init__(self, StepII_bundle, N=30, Vmin_shell=350, rcond=1e-4, makeplot=True):
        self.N, self.Vmin_shell, self.rcond = N, Vmin_shell, rcond
        self.__dict__.update(StepII_bundle.__dict__)
        [G, H, V, K, XYP, XY] = eng.localization2D('VDF_cartesian', self.N, nargout=6)
        self.G = np.asarray(G)
        self.H = np.asarray(H)
        self.V = np.asarray(V)
        self.K = np.asarray(K)
        XYP = np.asarray(XYP)
        self.XY = np.asarray(XY)

        # self.G = self.H * 1.0

        # locating the Shannon number
        self.Nshannon = np.argmin(np.abs(self.V-0.5))

        # converting XYP from a flattened array into a meshgrid
        NX, NY, __ = self.G.shape
        self.XX, self.YY =  np.reshape(XYP[:,0], (NX, NY), 'F'), np.reshape(XYP[:,1], (NX, NY), 'F')

        # interpolating the 2D raw VDF into the Slepian domain
        self.VDF_interp = griddata((np.ravel(self.V1,'F'), np.ravel(self.V2,'F')),
                                    np.ravel(self.VDF_2D, 'F'), (self.XX, self.YY), method='linear')
        # setting shells under Vmin_shell [km/s] to nan
        nan_mask_data = np.sqrt(self.XX**2 + self.YY**2) < self.Vmin_shell
        self.VDF_interp[nan_mask_data] = np.nan

        # performing the 2D Slepian reconstruction
        self.gyrotropic_recon_2D_VDF()

        # making the comparison plot (if required)
        if(makeplot): self.Cartesian_comparison_plot()


    def gyrotropic_recon_2D_VDF(self):
        # reconstructing the interpolated VDF in Cartesian Slepians
        nan_mask = np.isnan(self.VDF_interp)
        G_nonan = self.G[~nan_mask,:]
        M = G_nonan.T @ G_nonan
        __, S, __ = np.linalg.svd(M)
        I = np.identity(M.shape[0])
        coeffs = np.linalg.inv(M +  S.max() * self.rcond * I) @ G_nonan.T @ self.VDF_interp[~nan_mask]
        self.VDF_Sleprec = np.dot(self.G, coeffs)

    def Cartesian_comparison_plot(self):
        # plotting limits
        xmin, xmax = self.XX.min(), self.XX.max()
        ymin, ymax = self.YY.min(), self.YY.max()

        # plotting the 2D raw VDF from gyrotropization step
        fig, ax = plt.subplots(2, 2, figsize=(8, 9), sharex=True, sharey=True)
        ax[0,0].pcolormesh(self.V1, self.V2, self.VDF_2D, vmin=0, vmax=6, cmap='hot', rasterized=True)
        ax[0,0].contour(self.V1, self.V2, self.VDF_2D, levels=10, cmap='hot')
        ax[0,0].plot(self.XY[:,0], self.XY[:,1], '--k')
        ax[0,0].axhline(0, color='white', ls='dashed')
        ax[0,0].set_xlim([xmin, xmax])
        ax[0,0].set_ylim([ymin, ymax])
        ax[0,0].set_aspect('equal')
        ax[0,0].set_title('VDF 2D raw')

        # plotting the interpolated VDF
        ax[0,1].pcolormesh(self.XX, self.YY, self.VDF_interp, vmin=0, vmax=6, cmap='hot', rasterized=True)
        ax[0,1].contour(self.XX, self.YY, self.VDF_interp, levels=10, cmap='hot')
        ax[0,1].plot(self.XY[:,0], self.XY[:,1], '--k')
        ax[0,1].axhline(0, color='white', ls='dashed')
        ax[0,1].set_xlim([xmin, xmax])
        ax[0,1].set_ylim([ymin, ymax])
        ax[0,1].set_aspect('equal')
        ax[0,1].set_title('VDF 2D interpolated')

        # plotting the reconstructed distribution
        ax[1,0].pcolormesh(self.XX, self.YY, self.VDF_Sleprec, vmin=0, vmax=6, cmap='hot', rasterized=True)
        ax[1,0].contour(self.XX, self.YY, self.VDF_Sleprec, levels=10, cmap='hot')
        ax[1,0].plot(self.XY[:,0], self.XY[:,1], '--k')
        ax[1,0].axhline(0, color='white', ls='dashed')
        ax[1,0].set_xlim([xmin, xmax])
        ax[1,0].set_ylim([ymin, ymax])
        ax[1,0].set_aspect('equal')
        ax[1,0].set_title('Cartesian Slepian reconstruction')

        # plotting the reconstructed distribution vs. the raw 2D VDF from gyrotropization
        ax[1,1].pcolormesh(self.V1, self.V2, self.VDF_2D, vmin=0, vmax=6, cmap='hot', rasterized=True)
        ax[1,1].pcolormesh(self.XX[81:], self.YY[81:], self.VDF_Sleprec[81:], vmin=0, vmax=6, cmap='hot', rasterized=True)
        ax[1,1].set_xlim([xmin, xmax])
        ax[1,1].set_ylim([ymin, ymax])
        ax[1,1].set_aspect('equal')
        ax[1,1].set_title('2D raw vs. Reconstructed')

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.97)