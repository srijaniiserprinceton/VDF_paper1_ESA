'''
---------------------------------------------------------------------------
This is the driver script that should be run to get the reconstructed VDFs.
---------------------------------------------------------------------------
'''

# import statements
import cdflib, sys
import numpy as np

# imports from our custom package
from source_scripts import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian

if __name__=='__main__':
    #----------------------READING THE SOURCE FILE----------------------------------#
    filename = './input_data_files/2020-01-26_VDFs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)

    #------------------USER SPECIFIED PARAMETERS------------------------------------#
    time_idx = 12300        # time index of VDF to be reconstructed
    makeplot = True        # whether we want to save the diagnostic plots
    TH = 45                # the angular radius of the polar cap [in degrees]
    Lmax = 12              # maximum angular degree for polar Slepian generation
    Ncart = 30             # effective Shannon number of 2D Cartesian Slepian functions
    Vmin_shell = 350       # Minimum reliable energy shell [in km/s]
    rcond_polcap = 0.0     # Condition number for the inversion in polar caps
    rcond_cart = 1e-4      # Condition number for the inversion on a 2D plane

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
    StepII_bundle = VDF_rec_polarcaps.VDF_rec_polarcaps(DATA, StepI_bundle, Lmax=Lmax, rcond=rcond_polcap)

    #=============STEP III: Decomposing 2D gyrotropized VDF into Slepians in 2D (V{perp} vs V{||})===================#
    VDF_2D_rec = VDF_rec_cartesian.VDF_rec_cartesian(StepII_bundle, N=Ncart, Vmin_shell=Vmin_shell,
                                                     rcond=rcond_cart, makeplot=makeplot)