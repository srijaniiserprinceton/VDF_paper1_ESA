import numpy as np
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')
import cdflib, pickle
from tqdm import tqdm
from datetime import datetime, timedelta
func = np.vectorize(datetime.utcfromtimestamp)

from calculations.calc_moments import calc_moments, spher_moments

def get_moments(t_idx):
    energy  = data_xr.energy.data[t_idx, :, 0, 0]
    theta   = data_xr.theta.data[t_idx, 0, 0, :]
    phi     = data_xr.phi.data[t_idx, 0, :, 0]
    vdf     = data_xr.vdf.data[t_idx, :, :, :] * 1e12    
    vdf_err = data_xr.vdf_err.data[t_idx, :, :, :] * 1e12

    err_mask = vdf_err/vdf < 0.7
    vdf_noise = vdf.copy()
    vdf_noise[err_mask] = 0

    n_noise, u_noise, p_noise = calc_moments(np.transpose(vdf_noise, [0, 2, 1]), 13.8*np.sqrt(energy) * 1000, np.radians(theta), np.radians(phi))
    n_sig, u_sig, p_sig       = calc_moments(np.transpose(vdf, [0, 2, 1]), 13.8*np.sqrt(energy) * 1000, np.radians(theta), np.radians(phi))

    return n_sig, u_sig, p_sig, n_noise, u_noise, p_noise

def plot_moments(n1, u1, p1, n2, u2, p2, time_idx=None, animate=False):
    lw, alpha = 1, 0.1

    if(animate==False):
        fig, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

        ax[0].plot(times, n1, 'white', lw=lw)
        ax[0].plot(times, n2, 'orange', lw=lw, alpha=0.7)
        ax[0].set_ylabel(r'Density [$\rm{cm}^{-3}$]')
        ax[0].grid(True, alpha=0.3, linestyle='--')

        ax[1].plot(times, u1[:,0], 'white', lw=lw, label=r'MMS data')
        ax[1].plot(times, u2[:,0], 'orange', lw=lw, alpha=0.7, label=r'Slepian reconstruction')
        ax[1].set_ylabel(r'$U_x$ [km/s]')
        ax[1].grid(True, alpha=0.3, linestyle='--')

        ax[2].plot(times, u1[:,1], 'white', lw=lw)
        ax[2].plot(times, u2[:,1], 'orange', lw=lw, alpha=0.7)
        ax[2].set_ylabel(r'$U_y$ [km/s]')
        ax[2].grid(True, alpha=0.3, linestyle='--')

        ax[3].plot(times, u1[:,2], 'white', lw=lw)
        ax[3].plot(times, -u2[:,2], 'orange', lw=lw, alpha=0.7)
        ax[3].set_ylabel(r'$U_z$ [km/s]')
        ax[3].grid(True, alpha=0.3, linestyle='--')

        ax[3].set_xlabel('Times [s]')

        # ax[2].plot(times, p1[:,0], label=r'$P_{xx}$')
        # ax[2].plot(times, p1[:,3], label=r'$P_{yy}$')
        # ax[2].plot(times, p1[:,5], label=r'$P_{zz}$')
        # ax[2].plot(times, p1[:,1], '--', label=r'$P_{xy}$')
        # ax[2].plot(times, p1[:,2], '--', label=r'$P_{xz}$')
        # ax[2].plot(times, p1[:,4], '--', label=r'$P_{yz}$')
        # ax[2].set_ylabel(r'Pressure [Pa]')
        
        lines = [] 
        labels = [] 
        
        for ax in fig.axes: 
            Line, Label = ax.get_legend_handles_labels() 
            # print(Label) 
            lines.extend(Line) 
            labels.extend(Label)

        fig.legend(lines, labels, loc='upper center', ncols=2) 

        plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.97)

    else:
        fig, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

        ax[0].plot(times, n1, 'white', lw=lw, alpha=alpha)
        ax[0].plot(times, n2, 'orange', lw=lw, alpha=alpha)
        ax[0].plot(times[:time_idx], n1[:time_idx], 'white', lw=lw, label=r'MMS data')
        ax[0].plot(times[:time_idx], n2[:time_idx], 'orange', lw=lw, alpha=0.7, label=r'Slepian reconstruction')
        ax[0].set_ylabel(r'Density [$\rm{cm}^{-3}$]')
        ax[0].grid(True, alpha=0.3, linestyle='--')

        ax[1].plot(times, u1[:,0], 'white', lw=lw, alpha=alpha)
        ax[1].plot(times, u2[:,0], 'orange', lw=lw, alpha=alpha)
        ax[1].plot(times[:time_idx], u1[:time_idx,0], 'white', lw=lw)
        ax[1].plot(times[:time_idx], u2[:time_idx,0], 'orange', lw=lw, alpha=0.7)
        ax[1].set_ylabel(r'$U_x$ [km/s]')
        ax[1].grid(True, alpha=0.3, linestyle='--')

        ax[2].plot(times, u1[:,1], 'white', lw=lw, alpha=alpha)
        ax[2].plot(times, u2[:,1], 'orange', lw=lw, alpha=alpha)
        ax[2].plot(times[:time_idx], u1[:time_idx,1], 'white', lw=lw)
        ax[2].plot(times[:time_idx], u2[:time_idx,1], 'orange', lw=lw, alpha=0.7)
        ax[2].set_ylabel(r'$U_y$ [km/s]')
        ax[2].grid(True, alpha=0.3, linestyle='--')

        ax[3].plot(times, u1[:,2], 'white', lw=lw, alpha=alpha)
        ax[3].plot(times, -u2[:,2], 'orange', lw=lw, alpha=alpha)
        ax[3].plot(times[:time_idx], u1[:time_idx,2], 'white', lw=lw)
        ax[3].plot(times[:time_idx], -u2[:time_idx,2], 'orange', lw=lw, alpha=0.7)
        ax[3].set_ylabel(r'$U_z$ [km/s]')
        ax[3].grid(True, alpha=0.3, linestyle='--')

        ax[3].set_xlabel('Times [s]')
        
        lines = [] 
        labels = [] 
        
        for ax in fig.axes: 
            Line, Label = ax.get_legend_handles_labels() 
            # print(Label) 
            lines.extend(Line) 
            labels.extend(Label)

        fig.legend(lines, labels, loc='upper center', ncols=2) 

        plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.97)

        plt.savefig(f'VDF_paper1_plots/MMS_moments/{time_idx}.png')

        plt.close()

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

if __name__ == "__main__":
    # Load in the file that we are interested in looking at. 
    file = "./input_data_files/MMS_2016-01-11_VDF_and_ERRs.cdf"

    data_xr = cdflib.cdf_to_xarray(file, to_datetime=True)
    times = func(data_xr.unix_time.values)
    times = (times - times[0]) / timedelta(seconds=1)

    data_n_arr = np.zeros((len(times), 1))
    data_u_arr = np.zeros((len(times), 3))
    data_p_arr = np.zeros((len(times), 6))
    rec_n_arr = np.zeros((len(times), 1))
    rec_u_arr = np.zeros((len(times), 3))
    rec_p_arr = np.zeros((len(times), 6))
    # n_noise_arr = np.zeros((len(times), 1))
    # u_noise_arr = np.zeros((len(times), 3))
    # p_noise_arr = np.zeros((len(times), 6))

    data_moments = read_pickle('data_moments')
    rec_moments = read_pickle('rec_moments')

    # for t_idx, time in tqdm(enumerate(times)):
    for key_idx in tqdm(data_moments.keys()):
        # n_s, u_s, p_s, n_n, u_n, p_n = get_moments(t_idx)
        data_n_s, data_u_s, data_p_s = data_moments[key_idx]
        data_n_arr[key_idx] = data_n_s / 100**3   # converting from m^-3 to cm^-3
        data_u_arr[key_idx] = data_u_s / 1e3      # converting from m/s to km/s
        data_p_arr[key_idx] = data_p_s[np.triu_indices(3)]
        # n_noise_arr[t_idx] = n_n / 1e6
        # u_noise_arr[t_idx] = u_n / 1e3
        # p_noise_arr[t_idx] = p_n[np.triu_indices(3)]

        rec_n_s, rec_u_s, rec_p_s = rec_moments[key_idx]
        rec_n_arr[key_idx] = rec_n_s / 100**3   # converting from m^-3 to cm^-3
        rec_u_arr[key_idx] = rec_u_s / 1e3      # converting from m/s to km/s
        rec_p_arr[key_idx] = rec_p_s[np.triu_indices(3)]

    
    # plotting the timeseries of the moments
    # plot_moments(n_arr, u_arr, p_arr, u_noise_arr, u_noise_arr, p_noise_arr)
    # plot_moments(data_n_arr, data_u_arr, data_p_arr, rec_n_arr, rec_u_arr, rec_p_arr)

    for time_idx in tqdm(range(len(times))):
        plot_moments(data_n_arr, data_u_arr, data_p_arr, rec_n_arr, rec_u_arr, rec_p_arr, time_idx, animate=True)

