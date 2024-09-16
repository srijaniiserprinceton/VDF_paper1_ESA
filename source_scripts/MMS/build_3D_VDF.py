import numpy as np
import scipy.interpolate as interpolate

def get_3D_VDF(vdf_rec_dict, NEmesh=100, spline_order=3):
    # B-spline interpolation for each Slepian basis function
    lnE = np.log10(vdf_rec_dict.DATA.ENERGY[:,0,0])

    # constructing finer mesh in lnE
    lnEmin, lnEmax = lnE.min(), lnE.max()
    lnE_mesh = np.log10(np.logspace(lnEmin, lnEmax, NEmesh))

    # interpolation in log-energy
    NSlepians = vdf_rec_dict.G_hr.shape[0]
    coeffs_hr_interp = np.zeros((NEmesh, NSlepians))

    # interpolating the Slepian functions in a finer grid in theta and phi
    p_old, t_old = vdf_rec_dict.lon_hr, vdf_rec_dict.lat_hr
    thetamin, thetamax = t_old.min(), t_old.max()
    phimin, phimax = p_old.min(), p_old.max()
    p, t = np.linspace(phimin, phimax, 201), np.linspace(thetamin, thetamax, 101)
    pp, tt = np.meshgrid(p, t)
    points = np.asarray(list((p_old.flatten(), t_old.flatten()))).T

    G_hr_interp = np.zeros((NSlepians, 101, 201))

    for Slep_idx in range(NSlepians):
        t, c, k = interpolate.splrep(lnE, vdf_rec_dict.coeffs_hr[:, Slep_idx], s=0, k=spline_order)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        # B-spline interpolation
        coeffs_hr_interp[:, Slep_idx] = spline(lnE_mesh)

        
        G_hr_interp[Slep_idx] = interpolate.griddata(points, vdf_rec_dict.G_hr[Slep_idx].flatten(),
                                                    (pp, tt))
        

    # inner product with Slepian function basis to make full 3D structure
    # VDF_3D_polar = coeffs_hr_interp @ np.moveaxis(vdf_rec_dict.G_hr, 0, 1)
    VDF_3D_polar = coeffs_hr_interp @ np.moveaxis(G_hr_interp, 0, 1)
    VDF_3D_polar = np.moveaxis(VDF_3D_polar, 0, 1)

    # converting the theta mesh to go from 0 to 180 when returning instead of -90 to 90
    return lnE_mesh, (tt + 90) * np.pi / 180, pp * np.pi / 180, VDF_3D_polar

