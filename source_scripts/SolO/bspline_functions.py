import bspline
import bspline.splinelab as splinelab

def make_bsplines(xgrid, knots, second_derivaive=False):
    """
    Function to make the basis of cubic Bsplines at desired knot locations.
    """

    p = 3                           # order of spline (as-is; 3 = cubic)
    k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, p)       # create spline basis of order p on knots k

    # extracting the Bspline basis elements
    bsp_basis = np.array([B(i) for i in xgrid]).T
    # making a small adjustment in the right most B-spline
    bsp[-1,-1] = bsp[0,0]

    if(second_derivative):
        d2bsp_basis = np.gradient(np.gradient(bsp, axis=1), axis=1)

        return bsp_basis, d2bsp_basis

    return bsp_basis