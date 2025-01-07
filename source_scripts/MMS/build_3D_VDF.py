import numpy as np
import scipy.interpolate as interpolate

def get_3D_VDF(vdf_rec_dict):
    # B-spline interpolation for each Slepian basis function
    lnE = np.log10(vdf_rec_dict.ENERGY[0,:,0,0])

    # constructing finer mesh in lnE
    lnEmin, lnEmax = lnE.min(), lnE.max()
    lnE_mesh = np.log10(np.logspace(lnEmin, lnEmax, vdf_rec_dict.NEmesh))

    # interpolation in log-energy
    NSlepians = vdf_rec_dict.G.shape[0]
    SLEP_coeffs_interp = np.zeros((vdf_rec_dict.NEmesh, NSlepians))

    # interpolating the Slepian functions in a finer grid in theta and phi
    p_old, t_old = vdf_rec_dict.SLEP_PP, vdf_rec_dict.SLEP_TT
    thetamin, thetamax = t_old.min(), t_old.max()
    phimin, phimax = p_old.min(), p_old.max()
    p, t = np.linspace(phimin, phimax, vdf_rec_dict.NPmesh), np.linspace(thetamin, thetamax, vdf_rec_dict.NTmesh)
    pp, tt = np.meshgrid(p, t)
    points = np.asarray(list((p_old.flatten(), t_old.flatten()))).T

    G_interp = np.zeros((NSlepians, vdf_rec_dict.NTmesh, vdf_rec_dict.NPmesh))

    for Slep_idx in range(NSlepians):
        t, c, k = interpolate.splrep(lnE, vdf_rec_dict.SLEP_coeffs[:, Slep_idx], s=0, k=vdf_rec_dict.Espline_order)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        # B-spline interpolation
        SLEP_coeffs_interp[:, Slep_idx] = spline(lnE_mesh)
        
        if vdf_rec_dict.angular_basis == 'Slepians':
            G_interp[Slep_idx] = interpolate.griddata(points, vdf_rec_dict.G[Slep_idx].flatten(), (pp, tt))
        else:
            G_interp[Slep_idx] = interpolate.griddata(points, vdf_rec_dict.G[:,:,Slep_idx].flatten(), (pp, tt))
    # inner product with Slepian function basis to make full 3D structure
    # VDF_3D_polar = coeffs_hr_interp @ np.moveaxis(vdf_rec_dict.G, 0, 1)
    VDF_3D_polar = SLEP_coeffs_interp @ np.moveaxis(G_interp, 0, 1)
    VDF_3D_polar = np.moveaxis(VDF_3D_polar, 0, 1)

    # converting the theta mesh to go from 0 to 180 when returning instead of -90 to 90
    return lnE_mesh, (tt + 90) * np.pi / 180, pp * np.pi / 180, VDF_3D_polar

