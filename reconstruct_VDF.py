'''
---------------------------------------------------------------------------
This is the driver script that should be run to get the reconstructed VDFs.
---------------------------------------------------------------------------
'''

# import statements
import cdflib, sys
import numpy as np

# imports from our custom package
from source_scripts import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps

if __name__=='__main__':
    #----------------------READING THE SOURCE FILE----------------------------------#
    filename = './input_data_files/2020-01-26_VDFs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    #------------------USER SPECIFIED PARAMETERS------------------------------------#
    time_idx = 1300        # time index of VDF to be reconstructed
    makeplot = True        # whether we want to save the diagnostic plots
    TH = 45                # the angular radius of the polar cap
    Lmax = 12              # maximum angular degree for polar Slepian generation

    # extracting the required timestamp
    DATA = extract_data.extract_VDF_data(data, time_idx, instrument='SPAN')
    # we want to scale VDF such that the lowest non-zero entry is 1.0
    DATA.VDF[DATA.VDF == 0] = np.nan
    DATA.VDF = DATA.VDF / np.nanmin(DATA.VDF)

    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta = locate_axis.find_gyroaxis(DATA, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=makeplot)

    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    StepI_bundle = sph2slep.get_StepI_dict(mu_phi, mu_theta, TH, DATA.PHI[0,:,0], DATA.THETA[0,0], instrument='SPAN')

    #=============STEP II: Decomposing 3D measured VDF into Slepians on polar caps (gyrotropic)======================#
    VDF_2D = VDF_rec_polarcaps.VDF_rec_polarcaps(DATA, StepI_bundle, Lmax=Lmax, rcond=0.0)