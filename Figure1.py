import matplotlib as mpl
mpl.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib_scalebar.scalebar import ScaleBar

import cmasher as cmr

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 8

# data = '../standard/qicriteriacubic-spline50n1.0.hdf5'
data = '../IllustrisTNG_L35n2160TNG.hdf5'

sm  = None
gm  = None
sfr = None
z   = None

max_count = 1
count     = 0

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in all_subhalos.keys():
        this_subhalo = all_subhalos[key]
        
        this_sm  = np.array(this_subhalo['TotalStellarMass'])
        this_gm  = np.array(this_subhalo['TotalGasMass'])
        this_sfr = np.array(this_subhalo['TotalStarFormationRate'])
        
        if this_sm > 11 and this_sfr > 1:
        
            sm  = np.array(this_subhalo['StellarMass'])
            gm  = np.array(this_subhalo['GasMass'])
            sfr = np.array(this_subhalo['StarFormationRate'])
            z   = np.array(this_subhalo['Metallicity']) / 0.0127
            count += 1
            
            if count > max_count:
                print(this_sm, this_gm, this_sfr)
                print(key)
                break

sm  = np.log10(sm )
gm  = np.log10(gm )
sfr = np.log10(sfr+1e-16)
z   = np.log10(z  )

valid_pix = (
    (sm < 7) |
    (sfr < -4)
)

valid_pix_int = valid_pix.astype(int)

sm_og  = np.copy(sm )
gm_og  = np.copy(gm )
sfr_og = np.copy(sfr)
z_og   = np.copy(z  )

sm [valid_pix] = -16
gm [valid_pix] = -16
sfr[valid_pix] = -16
z  [valid_pix] = -16

fig, axs = plt.subplots(2, 2, figsize=(15,15), edgecolor='k')
axs = axs.flatten()

ax = axs[0]
mpb = ax.imshow(sm, vmin=7.0, vmax=10.0, cmap='inferno', rasterized=True)
ax.imshow(sm_og, vmin=7.0, vmax=10.0, cmap='inferno', rasterized=True, alpha=0.7)

cax = fig.add_axes([0.05, 0.55, 0.15, 0.015])
cbar = plt.colorbar(mpb,cax=cax,extend='both',orientation='horizontal',
                    ticks=[7.0,8.0,9.0,10.0],extendfrac=0.25)

cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(length=8, width=1, labelsize=20, labelcolor='white', color='white')
cbar.set_label('$\mathrm{log}(\Sigma_{\star} / \mathrm{M}_{\odot}\, \mathrm{kpc}^{-2})$',
               labelpad=10, fontsize=25, color='white')
cbar.ax.xaxis.set_label_position('top')

ax = axs[1]
mpb = ax.imshow(gm, vmin=6.5, vmax=8.0, cmap='cmr.emerald', rasterized=True)
ax.imshow(gm_og, vmin=6.5, vmax=8.0, cmap='cmr.emerald', rasterized=True, alpha=0.7)

cax = fig.add_axes([0.81, 0.55, 0.15, 0.015])
cbar = plt.colorbar(mpb,cax=cax,extend='both',orientation='horizontal',
                    ticks=[6.5,7.0,7.5,8.0,8.5,9.0],extendfrac=0.25)

cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(length=8, width=1, labelsize=20, labelcolor='white', color='white')
cbar.set_label('$\mathrm{log}(\Sigma_{\mathrm{gas}} / \mathrm{M}_{\odot}\, \mathrm{kpc}^{-2})$',
               labelpad=10, fontsize=25, color='white')
cbar.ax.xaxis.set_label_position('top')

ax = axs[2]
mpb = ax.imshow(sfr, vmin=-4.0, vmax=-1.0, cmap='cmr.cosmic', rasterized=True)
ax.imshow(sfr_og, vmin=-4.0, vmax=-1.0, cmap='cmr.cosmic', rasterized=True, alpha=0.7)

cax = fig.add_axes([0.075, 0.05, 0.15, 0.015])
cbar = plt.colorbar(mpb,cax=cax,extend='both',orientation='horizontal',
                    ticks=[-4.0,-3.0,-2.0,-1.0,0.0],extendfrac=0.25)

cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(length=8, width=1, labelsize=20, labelcolor='white', color='white')
cbar.set_label('$\mathrm{log}(\Sigma_{\mathrm{SFR}} / \mathrm{M}_{\odot}\, \mathrm{kpc}^{-2}\, \mathrm{yr}^{-1})$',
               labelpad=10, fontsize=25, color='white')
cbar.ax.xaxis.set_label_position('top')

ax = axs[3]

mpb = ax.imshow(z, vmin=-0.3, vmax=0.6, cmap='inferno', rasterized=True)
ax.imshow(z_og, vmin=-0.3, vmax=0.6, cmap='inferno', rasterized=True, alpha=0.7)

ax.plot([8,18],[90,90],color='white',lw=3)
ax.text(0.135,0.11,'10 kpc',transform=ax.transAxes,color='white',fontsize=20, ha='center')

for ax in axs:
    ax.contour(valid_pix_int, levels=[0.5], colors='gray', linewidths=1.5, alpha=0.5)

cax = fig.add_axes([0.81, 0.05, 0.15, 0.015])
cbar = plt.colorbar(mpb,cax=cax,extend='both',orientation='horizontal',
                    ticks=[-0.3,0.1,0.5],extendfrac=0.25)

cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(length=8, width=1, labelsize=20, labelcolor='white', color='white')
cbar.set_label('$\mathrm{log}(Z_\mathrm{gas} / Z_\odot)$',
               labelpad=10, fontsize=25, color='white')
cbar.ax.xaxis.set_label_position('top')

xstart = 0.75/2 
ystart = 0.75/2 
xlen = 0.5/2 
ylen = 0.5/2

ax_inset = fig.add_axes([xstart, ystart, xlen, ylen])

xlen = z.shape[0]
ylen = z.shape[1]

div        = 8
patch_idx  = 2
xup, xdown = int(xlen/2-xlen/div), int(xlen/2+xlen/div)
yup, ydown = int(ylen/2-ylen/div), int(ylen/2+ylen/div)

rect = mpl.patches.Rectangle((xup, yup), xdown-xup, ydown-yup,
                             linewidth=2, edgecolor='w', facecolor='none',transform=axs[patch_idx].transData)
axs[patch_idx].add_patch(rect)

x0, y0 = 40, 3.2 ## Attached to inset
x1, y1 = 3.8, 66 ## Attached to square on ax

x0, y0 = axs[patch_idx].transData.transform((x0, y0))
x1, y1 = axs[patch_idx].transData.transform((x1, y1))
line = plt.Line2D([x0, x1], [y0, y1], color='white', linestyle='-', linewidth=2)
fig.patches.append(line)

x0, y0 = 28.5, 91 ## Attached to square on ax
x1, y1 = 91.25, 54 ## Attached to inset

x0, y0 = axs[patch_idx].transData.transform((x0, y0))
x1, y1 = axs[patch_idx].transData.transform((x1, y1))
line = plt.Line2D([x0, x1], [y0, y1], color='white', linestyle='-', linewidth=2)
# fig_lines.append(line)
fig.patches.append(line)

ax_inset.imshow(sfr[xup:xdown, yup:ydown],
                vmin=-4, vmax=-1, cmap='cmr.cosmic', rasterized=True)

ax_inset.contour(valid_pix_int[xup:xdown, yup:ydown], levels=[0.5], colors='white', linewidths=1.5, alpha=1)

ax_inset.spines['bottom'].set_linewidth(3); ax_inset.spines['top']  .set_linewidth(3)
ax_inset.spines['left']  .set_linewidth(3); ax_inset.spines['right'].set_linewidth(3)

ax_inset.spines['bottom'].set_color('white'); ax_inset.spines['top']  .set_color('white')
ax_inset.spines['left']  .set_color('white'); ax_inset.spines['right'].set_color('white')

ax_inset.set_yticks((np.arange(yup,ydown+1,1) - float(ylen/2-ylen/div)))
ax_inset.set_xticks((np.arange(xup,xdown+1,1) - float(ylen/2-ylen/div)))

ax_inset.grid('true',color='w',lw=1.5,which='both',alpha=0.5)

ax_inset.tick_params(length=0, width=0, labelsize=0, labelcolor='none', color='none')

# ax_inset.text(0.5,1.02,'kpc-Scale Maps',transform=ax_inset.transAxes,ha='center',
#               color='white',va='bottom',
#               bbox=dict(facecolor='black', edgecolor='none'), fontsize=40)

ax_inset.plot([0.5,10.5],[22,22],color='white',lw=5)
ax_inset.text(0.25,0.14,'10 kpc',transform=ax_inset.transAxes,color='white',fontsize=20, ha='center',
              bbox=dict(facecolor='black', edgecolor='none', alpha=0.75))


order = ['Stellar Mass','Gas Mass','SFR','Metallicity']

for index,ax in enumerate(axs):
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    
    ax.spines['bottom'].set_color('white')
    ax.spines['top']   .set_color('white')
    ax.spines['left']  .set_color('white')
    ax.spines['right'] .set_color('white')
    
    if index % 2 == 0:
        ax.text(0.025,0.925,order[index],transform=ax.transAxes, color='white', fontsize=40)
    else:
        ax.text(0.975,0.925,order[index],transform=ax.transAxes, color='white', fontsize=40, ha='right')

plt.tight_layout()

plt.subplots_adjust(wspace=0.0,hspace=0.0)

plt.savefig('./figs/Figure1.pdf',bbox_inches='tight',facecolor='black')