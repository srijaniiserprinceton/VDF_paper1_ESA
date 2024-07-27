import numpy as np
from astropy import coordinates as coor
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator
# import spherical
import spherepy as sp
import matplotlib.pyplot as plt
from math import factorial as factorial

fac = np.vectorize(factorial)

def Rx(betax):
	'''
	Matrix for rotating about the x-axis.

	Parameters:
	-----------
	betax : float
			Angle of rotation in degrees.
	'''
	# converting to radians
	betax *= np.pi/180.0

	return np.array([[1, 0, 0],
					[0, np.cos(betax), -np.sin(betax)],
					[0, np.sin(betax), np.cos(betax)]])

def Ry(betaz):
	'''
	Matrix for rotating about the z-axis.

	Parameters:
	-----------
	betaz : float
			Angle of rotation in degrees.
	'''
	# converting to radians
	betaz *= np.pi/180.0

	return np.array([[np.cos(betaz), 0, np.sin(betaz)],
					[0, 1, 0],
					[-np.sin(betaz), 0, np.cos(betaz)]])

def Rz(betay):
	'''
	Matrix for rotating about the y-axis.

	Parameters:
	-----------
	betay : float
			Angle of rotation in degrees.
	'''
	# converting to radians
	betay *= np.pi/180.0

	return np.array([[np.cos(betay), -np.sin(betay), 0],
					[np.sin(betay), np.cos(betay), 0],
					[0, 0, 1]])

def rotate_cartesian(r, rot_or_derot, *beta):
	'''
	Function to rotate an input Cartesian grid.

	Paramters:
	----------
	r : array_like of shape (3, Ntheta, Nphi)
		Array of the original Cartesian coordinates before rotation.
	rot_or_derot : string
				   To rotate or derotate (essentially setting the order of the Euler rotation).
	beta : tuple of floats (betax, betay, betaz)
			Rotation angles in degrees.

	Returns:
	--------
	r_ : array_like of shape (3, Ntheta, Nphi)
			Array the rotated Cartesian grid.
	'''
	# unpacking the rotation angles from the tuple beta
	betax, betay, betaz = beta

	# constructing the rotation matrix for the required angles
	R_x, R_y, R_z = Rx(betax), Ry(betay), Rz(betaz)

	# building the total rotation matrix
	if(rot_or_derot == 'rot'):
		R = R_z @ R_y @ R_x
	else:
		R = R_x @ R_y @ R_z

	# the new set of rotated points
	# but first swapping the axes in r to make matrix multiplication easier
	if(r.ndim > 2):
		r = np.moveaxis(r, 0, 1)  # new shape (Ntheta, 3, Nphi)
		r_ = R @ r
		# returns the grid of rotated points (to be used by interpolator)
		# restoring the shape to (3, Ntheta, Nphi)
		r_ = np.moveaxis(r_, 0, 1)

	else: 
		r_ = R @ r

	return r_

def interpolate_on_rotated_grid(theta, phi, pattern, beta, rot_or_derot='rot'):
	'''
	Returns the interpolated pattern on the regular grid of theta and phi
	using the rotated Cartesian coordinates.

	Parameters:
	-----------
	beta : tuple of ints of shape (betax, betay, betaz)
		   Euler angles to rotate the grid. Produces a rotated grid (X_, Y_, Z_).
	theta : array_like of shape (Ntheta,)
		    Array of values of the regular theta grid where we want interpolated values.
	phi : array_like of shape (Nphi,)
		  Array of values of the regular phi grid where we want interpolated values.
	pattern : array_like of shape (Ntheta, Nphi)
		      2D array used as input for the interpolation to find the pattern on
			  our desired regular grid of (theta, phi).
	rot_or_derot : string
				   To rotate or derotate (essentially setting the order of the Euler rotation).
	'''
	# creating the X, Y, Z from theta and phi where we want our interpolated pattern
	THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
	X = np.sin(THETA) * np.cos(PHI)
	Y = np.sin(THETA) * np.sin(PHI)
	Z = np.cos(THETA)

	# unrotated original grid
	r = np.array([X, Y, Z])

	# finding the rotated grid
	r_ = rotate_cartesian(r, rot_or_derot, *beta)

	# using astropy to get the latitude and longitude of the 
	# rotated coordinate
	X_, Y_, Z_ = r_
	__, lat_, lon_ = coor.cartesian_to_spherical(X_, Y_, Z_)

	# converting from latitude to co-latitude (which is theta_ for us)
	theta_ = -(np.asarray(lat_) - np.pi/2)
	phi_ = np.asarray(lon_)

	# interpolating to find pattern on the regular grid
	theta_, phi_ = theta_.flatten(), phi_.flatten()
	points = list(zip(theta_, phi_))

	print(theta_, phi_)
	# pattern_interp = griddata(points, pattern.flatten(),
	# 						 (THETA, PHI), method='cubic')
	pattern_interp = CloughTocher2DInterpolator(points, pattern.flatten(), fill_value=np.nan)(THETA * np.pi/180, PHI * np.pi/180)

	return pattern_interp

def gen_wignerd_matrix(betay, Lmax=2, only_zonal=True):
	'''
	Function to compute the components of the Wigner d-matrix
	needed for finding the rotation axis to axisymmetrize the VDF.

	Parameters:
	-----------
	betay : float in degrees
			The angle of rotation taken as argument by the Wigner d-matrix.
	Lmax : int
		   The maximum angular degree for which we want the Wigner d elements.
		   This is the maximum ell which we use to find the first estimate of beta
		   from the non-axisymmetric data.

	Returns:
	--------
	dmatrix : dictionary of floats of type real
		      The matrix elements corresponding to the m=0 component 
			  of the Wigner d-matrix arrange by angular degrees as keys.
			  So, dmatrix['1'] will contain the three elements for ell=1,
			  dmatrix['2'] will contain the five elements for ell=2.
	'''

	# wigner = spherical.Wigner(Lmax)
	# d = wigner.d(np.exp(1j * betay * np.pi/180.0))

	dmatrix = {}

	# # the starting indices for each ell
	# start_ind_arr = np.cumsum((np.arange(Lmax) * 2 + 1)**2)
	# start_ind_arr = np.append(np.array([0]), start_ind_arr)

	# returning only the m=0 component for zonal cases (when optimizing beta)
	if(only_zonal):
		for ell in range(Lmax+1):
			d = Wigner_d_matrix(ell, betay * np.pi / 180)
			# dmatrix[ell] = np.reshape(d[start_ind_arr[ell]:start_ind_arr[ell] + (2*ell+1)**2],
			# 						(2*ell+1, 2*ell+1))[:, ell]
			dmatrix[ell] = d[:, ell]
		return dmatrix
	
	# returning all the (mxm) elements for the non-zonal case 
	else: 
		for ell in range(Lmax+1):
			d = Wigner_d_matrix(ell, betay * np.pi / 180)
			# dmatrix[ell] = np.reshape(d[start_ind_arr[ell]:start_ind_arr[ell] + (2*ell+1)**2],
			# 						(2*ell+1, 2*ell+1))
			dmatrix[ell] = d
		
		return dmatrix
		

def get_rotated_coefs_from_Wigd(c_unrotated, d):
	'''
	Function to generate the spherical harmonic coefficients in the rotated coordinate frame.

	Parameters:
	-----------
	c_unrotated : object of type spherepy.ScalarCoefs
			      The zonal coefficients to be rotated.
	
	d : dictionary
		A dictionary of Wigner d-matrix elements with the desired angle of rotation.
		Maybe for m=0 with only_zonal=True in gen_wignerd_matrix().

	Returns:
	--------
	c_rotated : object of type spherepy.ScalarCoefs
			 	The coefficients after rotating using Wigner d-matrix
	'''
	# initializing the spherepy object to store the coefficients in the rotated frame
	c_rotated = sp.zeros_coefs(c_unrotated.nmax, c_unrotated.mmax, coef_type = sp.scalar)

	for ell in d.keys():
		d_ell_mp = d[ell]
		for mpind, mp in enumerate(np.arange(-ell, ell+1)):
			c_rotated[int(ell), int(mp)] = np.sum(c_unrotated[int(ell), :] * d_ell_mp[mpind, :])
	
	return c_rotated

def rotate_using_Wigd(c_synthetic, betay, Lmax, Ntheta, Nphi):
	'''
	Function to rotate a 2D pattern by using the coefficients of the 
	unrotated coordinate and the angle beta for constructing the Wigner d-matrix.
	Note that as of now this is only build for rotating from an originally 
	axisymmetric 2D pattern.

	Parameters:
	-----------
	c_synthetic : Coefficients of type spherepy.ScalarCoefs
				  The set of coefficients c_{ell,m} obtained from a 
				  spherical harmonic transform of the original synthetic 
				  2D map (axisymmetric).

	betay : float
			Angle to rotate using Wigner d-matrices, in degrees.
	
	Lmax : int
		   Maximum angular degree which we are interested.
	
	Ntheta : int
			 Number of grid points in theta.
	
	Nphi : int
		   Number of grid points in phi. spherepy requires this to be even.

	Returns:
	--------
	pattern_rot_Wigd : array_like of floats of type spherepy.ScalarPatternUniform
					   The (theta, phi) map obtained from the modified coefficients
					   to obtain the rotated pattern.
	'''

	# getting the Wigner d-matrix elements needed for multiplying with the 
	# axisymmetric coefficients c_synthetic.
	dmatrix = gen_wignerd_matrix(betay, Lmax=Lmax, only_zonal=False)

	# get coefficients in rotated frame
	c_syn_rotated = get_rotated_coefs_from_Wigd(c_synthetic, dmatrix)

	# generate the 2D map of type spherepy.ScalarPattermUniform
	pattern_rot_Wigd = sp.ispht(c_syn_rotated, nrows=Ntheta, ncols=Nphi)

	return pattern_rot_Wigd

def find_Yell0_in_old_coor(beta_est, Ntheta, Nphi, ell=1):
	# finding the dmatrix for only getting the zonal Yell0
	d = gen_wignerd_matrix(beta_est, Lmax=ell, only_zonal=True)

	# making the coefficients to be filled by d-matrix
	c = sp.zeros_coefs(nmax=ell, mmax=ell)
	
	# filling in the elements with d-matrix elements
	for m_ind, m in enumerate(np.arange(-ell, ell+1)):
		c[int(ell), int(m)] = d[ell][m_ind]

	# finding the 2D pattern of Yell0 in the old coordinate
	Yell0_old_coor = sp.ispht(c, nrows=Ntheta, ncols=Nphi)

	return Yell0_old_coor.array.real

def Wigner_d_matrix(ell,beta):
    """
    Function to calculate the Wigner d-matrix for a certain specified $\ell$ and $\\beta$.
    
    Parameters:
    -----------
    ell : int, scalar
          The angular degree of spherical harmonic or the mode of interest.
    
    beta : float, scalar
           Relative angle of inclination between the two coordinate axes in radians.

    Returns:
    --------
    wigner_d_matrix : ndarray, shape (2*ell+1 x 2*ell+1)
                      The Wigner d-matrix.
    """
    # investigating the factorials, it is easy to see that
    # the lowest value of s can be 0 (otherwise the s! term has a negatve factorial)
    # and the highest possible value of s can be 2 * ell
    s_arr = np.arange(0, 2*ell+1)

    # making the m and m_ arrays and meshes
    m_arr = np.arange(-ell, ell+1)

    mm, mm_, ss = np.meshgrid(m_arr, m_arr, s_arr, indexing='ij')

    # making the matrix with an extra s dimension which will be summed over at the end
    d_matrix = np.zeros((2*ell+1, 2*ell+1, len(s_arr)))

    # making the factorial terms arguments
    fac1_arg = ell + mm_ - ss
    fac2_arg = ss
    fac3_arg = mm - mm_ + ss
    fac4_arg = ell - mm - ss

    # finding masks for where any of these factorial arguments is negative
    mask_neg_fac_arg = (fac1_arg < 0) + (fac2_arg < 0) + (fac3_arg < 0) + (fac4_arg < 0)

    # extracting only the valid entries for the factorial
    fac1_arg = fac1_arg[~mask_neg_fac_arg]
    fac2_arg = fac2_arg[~mask_neg_fac_arg]
    fac3_arg = fac3_arg[~mask_neg_fac_arg]
    fac4_arg = fac4_arg[~mask_neg_fac_arg]

    # computing on only the non-negative factorial arguments
    d_matrix[~mask_neg_fac_arg] = (np.power(-1, np.abs(mm - mm_ + ss)) * np.power(np.cos(0.5 * beta), np.abs(2*ell + mm_ - mm - 2*ss)) *\
                                   np.power(np.sin(0.5 * beta), np.abs(mm - mm_ + 2*ss)))[~mask_neg_fac_arg] \
                                   / (fac(fac1_arg) * fac(fac2_arg) * fac(fac3_arg) * fac(fac4_arg))
    
    # summing the s dimension
    d_s_summed = np.sum(d_matrix, axis=2)

    # multiplying the s-independent prefactor
    # redefining the mm and mm_ meshgrid without an s dimension this time
    mm, mm_ = np.meshgrid(m_arr, m_arr, indexing='ij')
    s_indep_prefac = np.sqrt(fac(ell+mm) * fac(ell-mm) * fac(ell+mm_) * fac(ell-mm_))

    # total Wigner d-matrix for a specific ell and beta. Shape (2*ell+1 x 2*ell+1)
    wigner_d_matrix = s_indep_prefac * d_s_summed
    
    return wigner_d_matrix