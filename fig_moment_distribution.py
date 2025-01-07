import os
import sys
import numpy as np
import scipy as sp
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    rec_moments = pickle.load(open('rec_moments_SLEP_TH85_Lmax14_update.pkl', 'rb'))
    data_moments = pickle.load(open('data_moments_SLEP_TH85_Lmax14_update.pkl', 'rb'))
    # rec_moments = pickle.load(open('rec_moments_Lmax12.pkl', 'rb'))
    # data_moments = pickle.load(open('data_moments_Lmax12.pkl', 'rb'))

    # Get the density data 
    rec_den = np.array([rec_moments[i][0]/1e6 for i in rec_moments.keys()])
    data_den = np.array([data_moments[i][0]/1e6 for i in data_moments.keys()])
    
    # Get the velocity vectors for the reconstruction and the original data
    rec_vel  = np.array([rec_moments[i][1]/1000 for i in rec_moments.keys()])
    data_vel = np.array([data_moments[i][1]/1000 for i in data_moments.keys()])

    # Get the Temperature information
    rec_temp = np.array([rec_moments[i][2] for i in rec_moments.keys()])
    data_temp = np.array([data_moments[i][2] for i in data_moments.keys()])

    # Define the ratios that we are interested in
    den_ratio = rec_den/data_den

    vvec_ratio = rec_vel/data_vel
    vmag_ratio = np.linalg.norm(rec_vel, axis=1) / np.linalg.norm(data_vel, axis=1)

    t_tens_ratio = rec_temp / data_temp
    t_trace_ratio = np.array([np.trace(rec_temp[i]) for i in rec_moments.keys()])/np.array([np.trace(data_temp[i]) for i in data_moments.keys()])

    fig, ax = plt.subplots(1,3,figsize=(12,4), layout='constrained')

    nbins = 25
    lmin = 0.9
    lmax = 1.1

    ax[0].hist(den_ratio, bins=nbins, range=(lmin, lmax))

    ax[1].hist(vmag_ratio, bins=nbins, range=(lmin,lmax),  label='|V|')

    ax[1].hist(vvec_ratio[:,0], bins=nbins, range=(lmin,lmax), alpha=0.4, label=r'$V_{x}$')
    ax[1].hist(vvec_ratio[:,1], bins=nbins, range=(lmin,lmax), alpha=0.4, label=r'$V_{y}$')
    ax[1].hist(vvec_ratio[:,2], bins=nbins, range=(lmin,lmax), alpha=0.4, label=r'$V_{z}$')

    ax[2].hist(t_trace_ratio, bins=nbins, range=(lmin, lmax), label=r'$T_{trace}$')
    ax[2].hist(t_tens_ratio[:,0,0], bins=nbins, range=(lmin, lmax), alpha=0.4, label=r'$T_{xx}$')
    ax[2].hist(t_tens_ratio[:,1,1], bins=nbins, range=(lmin, lmax), alpha=0.4, label=r'$T_{yy}$')
    ax[2].hist(t_tens_ratio[:,2,2], bins=nbins, range=(lmin, lmax), alpha=0.4, label=r'$T_{zz}$')

    ax[0].set_xlabel(r'$n_{rec}/n_{orig}$', fontsize=14)
    ax[1].set_xlabel(r'$v_{rec}/v_{orig}$', fontsize=14)
    ax[2].set_xlabel(r'$T_{rec}/T_{orig}$', fontsize=14)

    ax[1].set_title(r'MMS Slepians $L_{max} = 14$', fontsize=16)

    ax[0].set_ylabel('Counts', fontsize=14)

