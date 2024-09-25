import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')
import numpy as np
NAX = np.newaxis

def plot_VDF(StepII_bundle, time_idx):
    # making the Cartesian grid
    Vmag = 13.8 * np.sqrt(StepII_bundle.ENERGY[0,:,0,0])
    THETA, PHI = np.radians(StepII_bundle.SLEP_THETA), np.radians(StepII_bundle.SLEP_PHI)
    VX = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.cos(PHI)[NAX, :, :]
    VY = Vmag[:, NAX, NAX] * np.sin(THETA)[NAX, :, :] * np.sin(PHI)[NAX, :, :]
    VZ = Vmag[:, NAX, NAX] * np.cos(THETA)[NAX, :, :]

    REC_VDF = StepII_bundle.fine_from_fine

    plt.figure(figsize=(10,10), dpi=120)
    plt.pcolormesh(VX[:,StepII_bundle.NTHETA_SLEP//2,:], VY[:,StepII_bundle.NTHETA_SLEP//2,:], REC_VDF[:,StepII_bundle.NTHETA_SLEP//2,:],
                   cmap='inferno', vmin=1, vmax=7)
    plt.gca().set_aspect('equal')
    plt.savefig(f'./VDF_paper1_plots/SolO_VDF_rec_2D/{time_idx}.png')
    plt.close()