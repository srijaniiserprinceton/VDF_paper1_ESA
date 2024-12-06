import cdflib, datetime
import numpy as np; NAX = np.newaxis
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')

if __name__=='__main__':
    filename = '../../input_data_files/2020-01-26_VDFs.cdf'
    data = cdflib.cdf_to_xarray(filename, to_datetime=True)
    tidx = 36

    # extracting the correct time stamp vdf info
    vdf_full = data.vdf.data
    vdf_full[vdf_full == 0.0] = np.nan
    vdf = vdf_full[tidx] / np.nanmin(vdf_full)
    energy = data.energy[tidx,:,0,0].data
    theta = np.radians(data.theta[tidx,0,0,:].data)
    phi = np.radians(data.phi[tidx,0,0,:].data)
    time = datetime.datetime.fromtimestamp(data.unix_time.data[tidx])

    # making the cartesian grid
    vel = 13.8 * np.sqrt(energy)
    vx = vel[:,NAX,NAX] * (np.cos(theta) * np.cos(phi))[NAX,:,:]
    vy = vel[:,NAX,NAX] * (np.cos(theta) * np.sin(phi))[NAX,:,:]
    vz = vel[:,NAX,NAX] * np.sin(theta)[NAX,:,:]

    # making the 2D slices
    vdf_theta = np.nanmean(vdf, axis=1)
    vdf_phi = np.nanmean(vdf, axis=2)

    # plotting the two cut planes
    fig, ax = plt.subplots(2, 1, figsize=(5,8), sharex=True)

    ax[0].contourf(-vx[:,0], vz[:,0], np.log10(vdf_theta), vmin=1, vmax=8, cmap='hot')
    ax[1].contourf(-vx[:,:,4], vy[:,:,4], np.log10(vdf_phi), vmin=1, vmax=8, cmap='hot')
    ax[0].set_xlim([0, 1000])
    ax[1].set_xlim([0, 1000])
    ax[0].set_ylim([-500,500])
    ax[1].set_ylim([-500,500])
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set_ylabel(r'$V_z$ [km/s]', fontsize=16)
    ax[1].set_ylabel(r'$V_y$ [km/s]', fontsize=16)
    ax[1].set_xlabel(r'$V_x$ [km/s]', fontsize=16)
    ax[0].set_title(f'SPAN-Ai: {time.year}-{time.month}-{time.day}/{time.hour}:{time.minute}:{time.second}', fontsize=16)

    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.1, right=0.98, hspace=0.1, wspace=0.1)
    plt.savefig('SPAN_data.pdf')

