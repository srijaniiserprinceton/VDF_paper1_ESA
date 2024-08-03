import spherepy as sp
import numpy as np
import mat73
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
plt.ion()

from source_scripts import sph2slep

# reading the Slepian functions
Slepian_dict = mat73.loadmat('./input_data_files/Slepian_functions/glmalphapto-20-40-200-90-0.mat')
lon, lat = Slepian_dict['lon'], Slepian_dict['lat']

# the (theta, phi) coorindate system we want our basis functions on
pp, tt = np.meshgrid(lon * 180/np.pi, lat * 180/np.pi)

# converting these to Slepian functions on a spherical polar coordinate system
make_Slep = sph2slep.sph2slep(Slepian_dict)

# plotting
fig, ax = plt.subplots(4, 4, figsize=(15,9))
for i in range(16):
    i_eff = i + 0

    z = make_Slep.G[i_eff]

    # doing something sneaky to make the first plot look correct
    if(i_eff == 0): 
        z = np.abs(z)
        z[0,0] = -np.max(z)

    norm = TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())
    ax[i//4,i%4].pcolormesh(pp, tt, z, norm=norm,
                            cmap='seismic', rasterized=True)
    leak_percent = (1 - make_Slep.V[i_eff]) * 100
    ax[i//4,i%4].set_title(f'Leak percent={leak_percent:.4f} %')
    ax[i//4,i%4].set_aspect('equal')
    ax[i//4,i%4].text(0.02, 0.95, f'Slepian number: {(i_eff + 1)}', transform=ax[i//4,i%4].transAxes,
                            va='top', ha='left', color='black', fontweight='bold')

plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02)