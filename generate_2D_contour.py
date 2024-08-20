import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy.interpolate import griddata

import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian

class gen_contour:
    def __init__(self, VDF_2D, x, y, model_func_name='Gaussian'):
        self.z = VDF_2D
        self.X = x
        self.Y = y

        self.model = None
        self.params = None

        #  setting up the model and indicating independent variables
        if(len(VDF_2D) == 1): pass
        elif(model_func_name == 'Gaussian'):
            self.model = lmfit.Model(self.doubleGaussian2d, independent_vars=['x', 'y'])

            # setting up the initial guesses
            self.params = self.model.make_params()

            # mode initial guesses and min-max ranges of fitting parameters
            self.params['centerx'].set(value=x[np.argmax(self.z)], min=x.min(), max=x.max())
            self.params['centery'].set(value=y[np.argmax(self.z)], min=y.min(), max=y.max())
            self.params['amplitude'].set(value=10, min=0)
            self.params['rotation'].set(value=.1, min=0, max=np.pi/2)
            self.params['sigmax'].set(value=(x.max() - x.min()) * 0.5, min=0)
            self.params['sigmay'].set(value=(y.max() - y.min()) * 0.5, min=0)
            self.params['centerx_'].set(value=x[np.argmax(self.z)]*1.5, min=x.min(), max=x.max())
            self.params['centery_'].set(value=y[np.argmax(self.z)]*1.5, min=y.min(), max=y.max())
            self.params['amplitude_'].set(value=10*0.5, min=0)
            self.params['rotation_'].set(value=.1*0.5, min=0, max=np.pi/2)
            self.params['sigmax_'].set(value=(x.max() - x.min()) * 0.1, min=0)
            self.params['sigmay_'].set(value=(y.max() - y.min()) * 0.1, min=0)
        
        elif(model_func_name == 'Gaussian_ycentered'):
            self.model = lmfit.Model(self.doubleGaussian2d_ycentered, independent_vars=['x', 'y'])

            # setting up the initial guesses
            self.params = self.model.make_params()

            # mode initial guesses and min-max ranges of fitting parameters
            self.params['centerx'].set(value=x[np.argmax(self.z)], min=x.min(), max=x.max())
            self.params['amplitude'].set(value=10, min=0)
            self.params['sigmax'].set(value=(x.max() - x.min()) * 0.5, min=0)
            self.params['sigmay'].set(value=(y.max() - y.min()) * 0.5, min=0)
            self.params['centerx_'].set(value=x[np.argmax(self.z)]*1.5, min=x.min(), max=x.max())
            self.params['amplitude_'].set(value=10*0.5, min=0)
            self.params['sigmax_'].set(value=(x.max() - x.min()) * 0.1, min=0)
            self.params['sigmay_'].set(value=(y.max() - y.min()) * 0.1, min=0)

        elif(model_func_name == 'Lorentzian'):
            self.model = lmfit.Model(self.doubleLorentzian2d, independent_vars=['x', 'y'])

            # setting up the initial guesses
            self.params = self.model.make_params()

            # mode initial guesses and min-max ranges of fitting parameters
            self.params['centerx'].set(value=x[np.argmax(self.z)], min=x.min(), max=x.max())
            self.params['centery'].set(value=y[np.argmax(self.z)], min=y.min(), max=y.max())
            self.params['amplitude'].set(value=10, min=0)
            self.params['rotation'].set(value=.1, min=0, max=np.pi/2)
            self.params['sigmax'].set(value=(x.max() - x.min()) * 0.5, min=0)
            self.params['sigmay'].set(value=(y.max() - y.min()) * 0.5, min=0)
            self.params['centerx_'].set(value=x[np.argmax(self.z)]*1.5, min=x.min(), max=x.max())
            self.params['centery_'].set(value=y[np.argmax(self.z)]*1.5, min=y.min(), max=y.max())
            self.params['amplitude_'].set(value=10*0.5, min=0)
            self.params['rotation_'].set(value=.1*0.5, min=0, max=np.pi/2)
            self.params['sigmax_'].set(value=(x.max() - x.min()) * 0.1, min=0)
            self.params['sigmay_'].set(value=(y.max() - y.min()) * 0.1, min=0)
    
    def doubleLorentzian2d(self, x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1., rotation=0,
                           amplitude_=1., centerx_=0., centery_=0., sigmax_=1., sigmay_=1., rotation_=0):
        """
        Return a two dimensional lorentzian.

        The maximum of the peak occurs at ``centerx`` and ``centery``
        with widths ``sigmax`` and ``sigmay`` in the x and y directions
        respectively. The peak can be rotated by choosing the value of ``rotation``
        in radians.
        """
        # setup corresponding to the first Lorentzian
        xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
        yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
        R = (xp/sigmax)**2 + (yp/sigmay)**2

        # setup corresponding to the second Lorentzian
        xp_ = (x - centerx_)*np.cos(rotation_) - (y - centery_)*np.sin(rotation_)
        yp_ = (x - centerx_)*np.sin(rotation_) + (y - centery_)*np.cos(rotation_)
        R_ = (xp_/sigmax_)**2 + (yp_/sigmay_)**2

        # return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay) +\
        #        2*amplitude_*lorentzian(R_)/(np.pi*sigmax_*sigmay_)

        return amplitude*gaussian2d(-R) + amplitude_*gaussian2d(-R_)

    def doubleGaussian2d(self, x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1., rotation=0,
                           amplitude_=1., centerx_=0., centery_=0., sigmax_=1., sigmay_=1., rotation_=0):
        """
        Return a two dimensional lorentzian.

        The maximum of the peak occurs at ``centerx`` and ``centery``
        with widths ``sigmax`` and ``sigmay`` in the x and y directions
        respectively. The peak can be rotated by choosing the value of ``rotation``
        in radians.
        """
        # setup corresponding to the first Lorentzian
        xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
        yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
        R = (xp/sigmax)**2 + (yp/sigmay)**2

        # setup corresponding to the second Lorentzian
        xp_ = (x - centerx_)*np.cos(rotation_) - (y - centery_)*np.sin(rotation_)
        yp_ = (x - centerx_)*np.sin(rotation_) + (y - centery_)*np.cos(rotation_)
        R_ = (xp_/sigmax_)**2 + (yp_/sigmay_)**2

        # return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay) +\
        #        2*amplitude_*lorentzian(R_)/(np.pi*sigmax_*sigmay_)

        return amplitude*gaussian2d(-R) + amplitude_*gaussian2d(-R_)

    def doubleGaussian2d_ycentered(self, x, y, amplitude=1., centerx=0., sigmax=1., sigmay=1.,
                                   amplitude_=1., centerx_=0., sigmax_=1., sigmay_=1.):
        """
        Return a two dimensional lorentzian.

        The maximum of the peak occurs at ``centerx`` and ``centery``
        with widths ``sigmax`` and ``sigmay`` in the x and y directions
        respectively. The peak can be rotated by choosing the value of ``rotation``
        in radians.
        """
        # setup corresponding to the first Lorentzian
        rotation = 0.0
        xp = (x - centerx)*np.cos(rotation) - (y - 0)*np.sin(rotation)
        yp = (x - centerx)*np.sin(rotation) + (y - 0)*np.cos(rotation)
        R = (xp/sigmax)**2 + (yp/sigmay)**2

        # setup corresponding to the second Lorentzian
        rotation_ = 0.0
        xp_ = (x - centerx_)*np.cos(rotation_) - (y - 0)*np.sin(rotation_)
        yp_ = (x - centerx_)*np.sin(rotation_) + (y - 0)*np.cos(rotation_)
        R_ = (xp_/sigmax_)**2 + (yp_/sigmay_)**2

        return amplitude*gaussian2d(-R) + amplitude_*gaussian2d(-R_)

    def fit_2D_VDF(self):
        # fitting the model with the data
        result = self.model.fit(self.z, x=self.X, y=self.Y, params=self.params, nan_policy='omit') #, weights=1/self.error)

        # printing the report
        lmfit.report_fit(result)

        return result

if __name__=='__main__':
    # the number of grid points to be used in data
    npoints = 10000
    x = np.random.rand(npoints)*10 - 4
    y = np.random.rand(npoints)*5 - 3

    gendata = gen_contour([0], [0], [0])

    # generating the data
    z = gendata.doublelorentzian2d(x, y, amplitude=50, centerx=0.0, centery=0.0, sigmax=1.5,
                                   sigmay=1.5, rotation=0*np.pi/180, amplitude_=6, centerx_=1.5, centery_=0.0, sigmax_=0.7,
                                   sigmay_=0.7, rotation_=0.0*np.pi/180)
    # adding noise
    z += 2*(np.random.rand(*z.shape)-.5)
    error = np.sqrt(z+1)

    # setting up the grid and interpolating to compare the fitting with
    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                       np.linspace(y.min(), y.max(), 100))
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

    #  setting up the model and indicating independent variables
    gencontourdemo = gen_contour(z, x, y)

    # fitting the model with the data
    result = gencontourdemo.fit_2D_VDF()

    # plot to compare the data with the fitting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    vmax = np.nanpercentile(Z, 99.9)

    ax = axs[0, 0]
    art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data')
    ax.set_aspect('equal')

    ax = axs[0, 1]
    fit = gencontourdemo.model.func(X, Y, **result.best_values)
    art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Fit')
    ax.set_aspect('equal')

    ax = axs[1, 0]
    fit = gencontourdemo.model.func(X, Y, **result.best_values)
    art = ax.pcolor(X, Y, Z-fit, vmin=0, vmax=10, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data - Fit')
    ax.set_aspect('equal')

    for ax in axs.ravel():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axs[1, 1].remove()