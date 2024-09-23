import numpy as np

class MMS:
    def __init__(self, data_ESA, TH, PHI_CEN_IDX=16, THETA_CEN_IDX=0):
        self.instrument = 'MMS'

        NTIME, NENERGY, NPHI, NTHETA = data_ESA.vdf.data.shape
        VDF = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        VDF_ERR = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        ENERGY = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        THETA = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))
        PHI = np.zeros((NTIME, NENERGY, NPHI+1, NTHETA))

        VDF[:,:,:NPHI,:] = data_ESA.vdf.data
        VDF_ERR[:,:,:NPHI,:] = data_ESA.vdf_err.data
        ENERGY[:,:,:NPHI,:] = data_ESA.energy.data
        THETA[:,:,:NPHI,:] = data_ESA.theta.data
        PHI[:,:,:NPHI,:] = data_ESA.phi.data

        # we want to scale VDF such that the lowest non-zero entry is 1.0
        VDF[VDF == 0] = np.nan
        self.VDF_minval_true = np.nanmin(VDF)
        VDF = VDF / self.VDF_minval_true

        # populating the ghost cell in phi
        VDF[:,:,NPHI,:] = (VDF[:,:,0,:] + VDF[:,:,NPHI-1,:])/2.
        VDF_ERR[:,:,NPHI,:] = (VDF_ERR[:,:,0,:] + VDF_ERR[:,:,NPHI-1,:])/2.
        ENERGY[:,:,NPHI,:] = ENERGY[:,:,0,:]
        THETA[:,:,NPHI,:] = THETA[:,:,0,:]
        PHI[:,:,NPHI,:] = PHI[:,:,NPHI-1,:] + 11.25   # IN DEGREES

        self.VDF = VDF * 1.0
        self.VDF_ERR = VDF_ERR * 1.0
        self.ENERGY = ENERGY * 1.0

        # the true MMS FPI grid
        self.ESA_THETA = THETA * 1.0
        self.ESA_PHI = PHI * 1.0

        self.NENERGY = NENERGY
        self.NPHI = NPHI + 1
        self.NTHETA = NTHETA

        # the grid to be used for Slepian reconstruction
        self.SLEP_THETA = np.linspace(0, 180, self.NTHETA) - 90
        self.SLEP_PHI = np.linspace(0, 360, self.NPHI)

        self.SLEP_PP, self.SLEP_TT = np.meshgrid(self.SLEP_PHI, self.SLEP_THETA, indexing='ij')

        # to be initialized later in the workflow
        self.G = None
        self.V = None
        self.TH = TH

        # the location of the phi and theta center
        self.PHI_CEN_IDX = PHI_CEN_IDX
        self.THETA_CEN_IDX = THETA_CEN_IDX