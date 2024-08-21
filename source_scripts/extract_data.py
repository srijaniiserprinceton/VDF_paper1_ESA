import numpy as np

class extract_VDF_data:
    def __init__(self, data, instrument='SPAN'):
        self.data = data
        self.instrument = instrument

        self.ENERGY = None
        self.THETA = None
        self.PHI = None
        self.VDF = None 

        if(self.instrument == 'SPAN'):
            self.extract_SPAN_data()
    
    def extract_SPAN_data(self):
        '''
        - Each array should be in the shape of [Ntime, dim1, dim2, dim3].
        - Ntime is the number of time stamps.
        - dim1: energy dimension, dim2: phi dimension, and dim3: theta dimension.
        - flipping the phi axis to have azimuthal angle to be monotonically increasing.
        '''
        self.ENERGY = self.data.energy.data[:,:,::-1,:]
        self.THETA = self.data.theta.data[:,:,::-1,:] + 90
        self.PHI = self.data.phi.data[:,:,::-1,:]
        self.VDF = self.data.vdf.data[:,:,::-1,:]
