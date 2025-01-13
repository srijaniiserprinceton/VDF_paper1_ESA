import numpy as np
import matplotlib.pyplot as plt


LmaxMMS = np.arange(4, 15, 1)
LmaxSolO = np.arange(4, 29, 1)

MMS_Grid  = 32 * 16
SolO_Grid = 11 * 9 
PSP_Grid  = 8 * 8

NbasisMMS = (LmaxMMS + 1)**2
NbasisPSP = (LmaxMMS + 1)
NbasisSolO = (LmaxSolO + 1)**2
N_gyro_basis = (LmaxSolO + 1)

fig, ax = plt.subplots(figsize=(8,4), layout='constrained')

ax.plot(LmaxMMS, MMS_Grid/NbasisMMS, marker='s', color='k', lw=3, markersize=10, label=r'$\frac{N_{MMS}}{N_{Basis}}$')
ax.plot(LmaxSolO, SolO_Grid/NbasisSolO, marker='v', color='b', lw=3, markersize=10, label=r'$\frac{N_{SolO}}{N_{Basis}}$')
ax.plot(LmaxSolO, SolO_Grid/N_gyro_basis, marker='o', color='r', lw=3, markersize=10, label=r'$\frac{N_{SolO}}{N_{Gyro}}$')
ax.plot(LmaxMMS, PSP_Grid/NbasisPSP, marker='o', markerfacecolor='w', color='tab:orange', lw=3, markersize=10, label=r'$\frac{N_{PSP}}{N_{Gyro}}$')
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.legend(ncols=4, fontsize=20, frameon=False, columnspacing=0.8)
ax.set_xlabel(r'$L_{max}$', fontsize=20)
# ax.set_ylabel(r'$\frac{N_{Data}}{N_{Basis}}$', fontsize=22, rotation='horizontal', labelpad=24)

ax.axvline(14, linestyle='dotted', color='grey', lw=1, alpha=0.7)
ax.axvline(28, linestyle='dashed', color='grey', lw=1, alpha=0.7)

ax.grid(True)
plt.savefig('NDATA_vs_Lmax.pdf')