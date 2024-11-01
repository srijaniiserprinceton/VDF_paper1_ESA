import numpy as np

from source_scripts import misc_functions

class PSP:
    def __init__(self, data_ESA, TH):
        self.instrument = 'PSP'

        NTIME, NENERGY, NPHI, NTHETA = data_ESA.vdf.data.shape

        VDF = data_ESA.vdf.data[:,:,::-1,:]
        ENERGY = data_ESA.energy.data[:,:,::-1,:]
        THETA = data_ESA.theta.data[:,:,::-1,:] + 90
        PHI = data_ESA.phi.data[:,:,::-1,:]

        # we want to scale VDF such that the lowest non-zero entry is 1.0
        VDF[VDF == 0] = np.nan
        self.VDF_minval_true = np.nanmin(VDF)
        VDF = VDF / self.VDF_minval_true

        self.VDF = VDF * 1.0
        self.ENERGY = ENERGY * 1.0

        # the true PSP-SPAN grid
        self.ESA_THETA = THETA * 1.0
        self.ESA_PHI = PHI * 1.0

        self.NENERGY = NENERGY
        self.NPHI_ESA = NPHI + 1
        self.NTHETA_ESA = NTHETA

        # the grid to be used for Slepian reconstruction
        self.SLEP_THETA = np.linspace(0, 180, self.NTHETA_ESA) - 90
        self.SLEP_PHI = np.linspace(0, 360, self.NPHI_ESA)

        self.NPHI_SLEP = NPHI + 1
        self.NTHETA_SLEP = NTHETA

        self.SLEP_PP, self.SLEP_TT = np.meshgrid(self.SLEP_PHI, self.SLEP_THETA, indexing='ij')

        # to be initialized later in the workflow
        self.G = None
        self.V = None
        self.TH = TH


class MMS:
    def __init__(self, data_ESA, TH, Lmax=None, Nmesh=(100, 201, 101), Espline_order=3, PHI_CEN_IDX=16, THETA_CEN_IDX=0, makeplot=True):
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
        self.NPHI_ESA = NPHI + 1
        self.NTHETA_ESA = NTHETA

        # the grid to be used for Slepian reconstruction
        self.SLEP_THETA = np.linspace(0, 180, self.NTHETA_ESA) - 90
        self.SLEP_PHI = np.linspace(0, 360, self.NPHI_ESA)

        self.NPHI_SLEP = NPHI + 1
        self.NTHETA_SLEP = NTHETA

        self.SLEP_PP, self.SLEP_TT = np.meshgrid(self.SLEP_PHI, self.SLEP_THETA, indexing='ij')

        # Setup slepian directory
        self.slep_dir = misc_functions.read_config()[0]

        # Check Lmax
        self.Lmax_Nyq = int(self.NTHETA_ESA / 2)
        if Lmax is None: 
            self.Lmax = self.Lmax_Nyq
        else:
            if Lmax > self.Lmax_Nyq: print("Lmax exceeds Nyquist. Resetting to Nyquist.")
            self.Lmax = min(Lmax, self.Lmax_Nyq)
        
        # to be initialized later in the workflow
        self.G = None
        self.V = None
        self.TH = TH

        # the location of the phi and theta center
        self.PHI_CEN_IDX = PHI_CEN_IDX
        self.THETA_CEN_IDX = THETA_CEN_IDX

        # Init parameters for Energy Shell interpolation. 
        self.NEmesh, self.NPmesh, self.NTmesh = Nmesh
        self.Espline_order = Espline_order


        self.makeplot = makeplot

        # finding the vmin and vmax according to the time 
        self.vmax_t = np.nanmax(np.log10(self.VDF), axis=(1,2,3)).astype('int')
        self.vmax_t = np.nan_to_num(nan = 1.1, posinf=1.0, neginf=1.0)
        self.vmin_t = np.ones_like(vmax_t)

class SolO:
    def __init__(self, data_ESA, TH=45, Lmax=None, Nmesh=(100, 201, 101), Espline_order=3, makeplot = True):
        self.instrument = 'SolO'

        NTIME, NENERGY, NPHI, NTHETA = data_ESA.vdf.data.shape
        VDF = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        ENERGY = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        THETA = np.zeros((NTIME, NENERGY, NPHI, NTHETA))
        PHI = np.zeros((NTIME, NENERGY, NPHI, NTHETA))

        VDF[:,:,:NPHI,:] = data_ESA.vdf.data
        # energy goes from high to low in ESA data
        ENERGY[:,:,:NPHI,:] = data_ESA.energy.data[:,::-1,:,:]
        THETA[:,:,:NPHI,:] = data_ESA.theta.data
        PHI[:,:,:NPHI,:] = data_ESA.phi.data

        # we want to scale VDF such that the lowest non-zero entry is 1.0
        VDF[VDF == 0] = np.nan
        self.VDF_minval_true = np.nanmin(VDF)
        VDF = VDF / self.VDF_minval_true

        # changing the nan location to unity before fitting using polar Slepians (will make them zero when taking log)
        self.nanval = np.nan #1e-5
        VDF[np.isnan(VDF)] = self.nanval

        self.VDF = VDF * 1.0
        self.ENERGY = ENERGY * 1.0

        # since phi in the original grid goes from (-180 to 180) and we want it to be from (0 to 360)
        # phi = 0 in SolO grid corresponds to phi = 180 in PSP grid.
        self.ESA_THETA = THETA * 1.0 + 90
        self.ESA_PHI = PHI * 1.0 + 180

        self.NENERGY = NENERGY
        self.NPHI_ESA = NPHI
        self.NTHETA_ESA = NTHETA

        # estimating number of gridpoints for Slepians to maintain a similar resolution of SolO grid
        self.NPHI_SLEP = int(360 // np.mean(np.diff(self.ESA_PHI[0,0,:,0])))
        self.NTHETA_SLEP = int(180 // np.mean(np.diff(self.ESA_THETA[0,0,0,:])))

        # the grid to be used for Slepian reconstruction
        self.SLEP_PHI = np.linspace(0, 360, self.NPHI_SLEP)
        self.SLEP_THETA = np.linspace(0, 180, self.NTHETA_SLEP) - 90

        self.SLEP_PP, self.SLEP_TT = np.meshgrid(self.SLEP_PHI, self.SLEP_THETA, indexing='ij')
        
        # Setup slepian directory
        self.slep_dir = misc_functions.read_config()[0]

        # Check Lmax
        dtheta_Nyq = np.max(np.diff(self.ESA_THETA[:,0,0,:]))
        dphi_Nyq = np.max(np.diff(self.ESA_PHI[:,0,:,0]))
        self.Lmax_Nyq = int(np.min([np.ceil(180/dtheta_Nyq), np.ceil(180/dphi_Nyq)]))
        if Lmax is None: 
            self.Lmax = self.Lmax_Nyq
        else:
            if Lmax > self.Lmax_Nyq: print(f"Lmax exceeds Nyquist. Resetting to Lmax_Nyquist = {self.Lmax_Nyq}.")
            self.Lmax = min(Lmax, self.Lmax_Nyq)

        # to store the gyro center in phi and theta
        self.mu_phi = np.zeros(NTIME)
        self.mu_theta = np.zeros(NTIME)
        self.mu_sigma = np.zeros(NTIME)

        # setting the range in energy indices which will be used for centroid finding and VDF width determination
        self.E_minidx, self.E_maxidx = None, None

        # array to store the Eshell info for plotting (in datascan_mode)
        self.Eshell_info = None
        
        # to be initialized later in the workflow
        self.G = None
        self.V = None
        self.TH = TH

        # Init parameters for Energy Shell interpolation. 
        self.NEmesh, self.NPmesh, self.NTmesh = Nmesh
        self.Espline_order = Espline_order


        self.makeplot = makeplot

        # finding the vmin and vmax according to the time 
        self.vmax_t = np.nanmax(np.log10(self.VDF), axis=(1,2,3)).astype('int')
        self.vmax_t = np.nan_to_num(self.vmax_t, nan = 1.1, posinf=1.0, neginf=1.0)
        self.vmin_t = np.ones_like(self.vmax_t)

