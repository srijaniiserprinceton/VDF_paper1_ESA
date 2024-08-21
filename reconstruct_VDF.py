'''
---------------------------------------------------------------------------
This is the first script that should be run so that we know the range of
energy shells that contain valid data. Use the visual results from the 
diagnostic plot to find E_idx_min and E_idx_max for making histogram plots
to find the effect axis of gyrotropy.
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
    time_stamp = '2020-01-26'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)
    DATA = extract_data.extract_VDF_data(data, instrument='SPAN')

    #------------------USER SPECIFIED PARAMETERS------------------------------------#
    time_idx = 1300        # time index of VDF to be reconstructed
    makeplot = True        # whether we want to save the diagnostic plots
    TH = 45                # the angular radius of the polar cap

    # we want to scale VDF such that the lowest non-zero entry is 1.0
    DATA.VDF[DATA.VDF == 0] = np.nan
    DATA.VDF = DATA.VDF / np.nanmin(DATA.VDF)

    # effective axis of gyrotropic across all relevant shells
    mu_phi, mu_theta = locate_axis.find_gyroaxis(DATA, time_idx, TH=TH, Nrows=4, Ncols=8, makeplot=makeplot)

    #------------------------saving the theta and phi grid for generating Slepians-on-polar-cap---------------------#
    sph2slep.save_Slepian_grid(mu_phi, mu_theta, DATA.PHI[0,0,:,0], DATA.THETA[0,0,0], instrument='SPAN')