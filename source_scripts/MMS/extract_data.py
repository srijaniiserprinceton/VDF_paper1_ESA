import numpy as np

class extract_VDF_data:
    def __init__(self, data, time_idx, instrument):
        self.data = data
        self.time_idx = time_idx
        self.instrument = instrument

        self.ENERGY = None
        self.THETA = None
        self.PHI = None
        self.VDF = None
        self.minval_true = None    # to be filled in while scaling the data 

        self.extract_MMS_data()

    def extract_MMS_data(self):
        '''
        - Each array should be in the shape of [Ntime, dim1, dim2, dim3].
        - Ntime is the number of time stamps.
        - dim1: energy dimension, dim2: phi dimension, and dim3: theta dimension.
        - flipping the phi axis to have azimuthal angle to be monotonically increasing.
        '''
        self.ENERGY = self.data.energy.data[self.time_idx,:,:,:]
        self.THETA = self.data.theta.data[self.time_idx,:,:,:]
        self.PHI = self.data.phi.data[self.time_idx,:,:,:]
        self.VDF = self.data.vdf.data[self.time_idx,:,:,:]
