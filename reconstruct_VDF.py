'''
---------------------------------------------------------------------------
This is the driver script that should be run to get the reconstructed VDFs.
---------------------------------------------------------------------------
'''

# import statements
import cdflib, sys
import numpy as np

# imports from our custom package
from source_scripts import sph2slep, extract_data, locate_axis

if __name__=='__main__':
    #----------------------READING THE SOURCE FILE----------------------------------#
    filename = './input_data_files/2020-01-26_VDFs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)
    DATA = extract_data.extract_VDF_data(data, instrument='SPAN')

    #------------------USER SPECIFIED PARAMETERS------------------------------------#
    time_idx = 1300        # time index of VDF to be reconstructed
    makeplot = True        # whether we want to save the diagnostic plots
    TH = 45                # the angular radius of the polar cap

    # we want to scale VDF such that the lowest non-zero entry is 1.0
    DATA.VDF[DATA.VDF == 0] = np.nan
    DATA.VDF = DATA.VDF / np.nanmin(DATA.VDF)

    #=============STEP I: Finding effective axis of gyrotropic across all relevant shells===========================#
    mu_phi, mu_theta = locate_axis.find_gyroaxis(DATA, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=makeplot)

    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    sph2slep.save_Slepian_grid(mu_phi, mu_theta, DATA.PHI[0,0,:,0], DATA.THETA[0,0,0], instrument='SPAN')