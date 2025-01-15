import os
import sys
import numpy as np
import scipy as sp
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    rec_moments = pickle.load(open('/home/michael/Research/VDF_paper1_ESA/rec_moments_SLEP_TH85_Lmax14_update.pkl', 'rb'))
    data_moments = pickle.load(open('/home/michael/Research/VDF_paper1_ESA/data_moments_SLEP_TH85_Lmax14_update.pkl', 'rb'))
    # rec_moments = pickle.load(open('/home/michael/Research/VDF_paper1_ESA/rec_moments_solo1.pkl', 'rb'))
    # data_moments = pickle.load(open('/home/michael/Research/VDF_paper1_ESA/data_moments_solo1.pkl', 'rb'))

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

    fig, ax = plt.subplots(1,3, figsize=(12,3), sharex=True, sharey=True)

    nbins = 15
    lmin = 0.9
    lmax = 1.1
    htype = 'step'
    dflag = True

    ax[0].hist(den_ratio, bins=nbins, range=(lmin, lmax), histtype='bar', density=dflag, alpha=0.5, label=r'$n_{rec}/n_{data} = $'+f'${np.round(np.quantile(den_ratio, 0.5), 3)}^{{+{np.round(np.quantile(den_ratio - np.quantile(den_ratio, 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(den_ratio, 0.14) - np.quantile(den_ratio, 0.5), 4)}}}$')
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_xlabel(r'Density Ratio', fontsize=14)
    ax[0].set_ylim([0,100])

    ax[1].hist(vmag_ratio, bins=nbins, range=(lmin,lmax), histtype='bar', density=dflag, alpha=0.5, label=r'$|V_{rec}|/|V_{data}|$ = '+f'${np.round(np.quantile(vmag_ratio, 0.5), 3)}^{{+{np.round(np.quantile(vmag_ratio - np.quantile(vmag_ratio, 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(vmag_ratio, 0.14) - np.quantile(vmag_ratio, 0.5), 4)}}}$', color='tab:blue')

    ax[1].hist(vvec_ratio[:,0], bins=nbins, range=(lmin,lmax), histtype=htype, density=dflag, alpha=1, linewidth=2, label=r'$V_{x,rec}/V_{x,data}$ = '+f'${np.round(np.quantile(vvec_ratio[:,0], 0.5), 3)}^{{+{np.round(np.quantile(vvec_ratio[:,0] - np.quantile(vvec_ratio[:,0], 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(vvec_ratio[:,0], 0.14) - np.quantile(vvec_ratio[:,0], 0.5), 4)}}}$', color='k')
    ax[1].hist(vvec_ratio[:,1], bins=nbins, range=(lmin,lmax), histtype=htype, density=dflag, alpha=1, linewidth=2, label=r'$V_{y,rec}/V_{y,data}$ = '+f'${np.round(np.quantile(vvec_ratio[:,1], 0.5), 3)}^{{+{np.round(np.quantile(vvec_ratio[:,1] - np.quantile(vvec_ratio[:,1], 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(vvec_ratio[:,1], 0.14) - np.quantile(vvec_ratio[:,1], 0.5), 4)}}}$', color='tab:orange')
    ax[1].hist(vvec_ratio[:,2], bins=nbins, range=(lmin,lmax), histtype=htype, density=dflag, alpha=1, linewidth=2, label=r'$V_{z,rec}/V_{z,data}$ = '+f'${np.round(np.quantile(vvec_ratio[:,2], 0.5), 3)}^{{+{np.round(np.quantile(vvec_ratio[:,2] - np.quantile(vvec_ratio[:,2], 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(vvec_ratio[:,2], 0.14) - np.quantile(vvec_ratio[:,2], 0.5), 4)}}}$', color='tab:green')
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_xlabel(r'Velocity Ratios', fontsize=14)

    ax[2].hist(t_trace_ratio, bins=nbins, range=(lmin, lmax), histtype='bar', density=dflag, alpha=0.5, label=r'$Tr\{T_{rec}\}/Tr\{T_{data}\}$ = '+f'${np.round(np.quantile(t_trace_ratio, 0.5), 3)}^{{+{np.round(np.quantile(t_trace_ratio - np.quantile(t_trace_ratio, 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(t_trace_ratio, 0.14) - np.quantile(t_trace_ratio, 0.5), 4)}}}$', color='tab:blue')

    ax[2].hist(t_tens_ratio[:,0,0], bins=nbins, range=(lmin, lmax), histtype=htype, density=dflag, alpha=1, linewidth=2, label=r'$T_{xx,rec}/T_{xx,data}$ = '+f'${np.round(np.quantile(t_tens_ratio[:,0,0], 0.5), 3)}^{{+{np.round(np.quantile(t_tens_ratio[:,0,0] - np.quantile(t_tens_ratio[:,0,0], 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(t_tens_ratio[:,0,0], 0.14) - np.quantile(t_tens_ratio[:,0,0], 0.5), 4)}}}$', color='k')
    ax[2].hist(t_tens_ratio[:,1,1], bins=nbins, range=(lmin, lmax), histtype=htype, density=dflag, alpha=1, linewidth=2, label=r'$T_{yy,rec}/T_{yy,data}$ = '+f'${np.round(np.quantile(t_tens_ratio[:,1,1], 0.5), 3)}^{{+{np.round(np.quantile(t_tens_ratio[:,1,1] - np.quantile(t_tens_ratio[:,1,1], 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(t_tens_ratio[:,1,1], 0.14) - np.quantile(t_tens_ratio[:,1,1], 0.5), 4)}}}$', color='tab:orange')
    ax[2].hist(t_tens_ratio[:,2,2], bins=nbins, range=(lmin, lmax), histtype=htype, density=dflag, alpha=1, linewidth=2, label=r'$T_{zz,rec}/T_{zz,data}$ = '+f'${np.round(np.quantile(t_tens_ratio[:,2,2], 0.5), 3)}^{{+{np.round(np.quantile(t_tens_ratio[:,2,2] - np.quantile(t_tens_ratio[:,2,2], 0.5), 0.86), 4)}}}_{{{np.round(np.quantile(t_tens_ratio[:,2,2], 0.14) - np.quantile(t_tens_ratio[:,2,2], 0.5), 4)}}}$', color='tab:green')
    ax[2].legend()
    ax[2].set_xlabel(r'Temperature Ratios', fontsize=14)
    ax[2].legend(frameon=False, fontsize=9)

    # [ax[i].set_box_aspect(1) for i in range(3)]

    # ax[0].set_xlabel(r'$n_{rec}/n_{orig}$', fontsize=14)
    # ax[1].set_xlabel(r'$v_{rec}/v_{orig}$', fontsize=14)
    # ax[2].set_xlabel(r'$T_{rec}/T_{orig}$', fontsize=14)

    # ax[1].set_title(r'MMS Slepians $L_{max} = 14$', fontsize=16)

    # ax[0].set_ylabel('Counts', fontsize=14)

    # for ax_idx, axs in enumerate(ax):
    #     lims = [
    #         np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
    #         np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    #     ]

    #     # now plot both limits against eachother
    #     axs.plot(lims, lims, 'r--', alpha=0.75, zorder=100, lw=2)
    #     # axs.text(0.04, 0.8, f'{moment_label[ax_idx]}', transform=axs.transAxes,
    #             # va='bottom', ha='left', color='black', fontsize=20, fontweight='bold')

    # fig.text(0.55, 0.04, r'Moment Ratios', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.01, 0.55, r'Probability Density', va='center', rotation='vertical', fontsize=14)#, fontweight='bold')

    # plt.subplots_adjust(left=0.06, right=0.98, wspace=0.35, top=1.0, bottom=0.1)
    plt.subplots_adjust(left=0.08, right=0.97, hspace=0.05, wspace=0.05, top=0.95, bottom=0.2)
    # plt.show()
    plt.savefig('./moment_rec_distribution.pdf')
