import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cmasher as cmr

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size']   = 20

fs_og = 45
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5 * 3
mpl.rcParams['ytick.major.width'] = 1.5 * 3
mpl.rcParams['xtick.minor.width'] = 1.0 * 3
mpl.rcParams['ytick.minor.width'] = 1.0 * 3
mpl.rcParams['xtick.major.size'] = 7.5 * 3
mpl.rcParams['ytick.major.size'] = 7.5 * 3
mpl.rcParams['xtick.minor.size'] = 3.5 * 3
mpl.rcParams['ytick.minor.size'] = 3.5 * 3
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


data = '../IllustrisTNG_L35n2160TNG.hdf5'

all_gm  = []
all_sm  = []
all_sfr = []
all_rad = []

def median(x,y):
    xs = np.arange(np.min(x), 10, 0.25)
    ys = np.zeros(len(xs))
    up = np.zeros(len(xs))
    dn = np.zeros(len(xs))
    
    dx = xs[1] - xs[0]
    
    for index, this_x in enumerate(xs):
        within_dx = (x > this_x) & (x < this_x+dx)
        
        ys[index] = np.median(y[within_dx]) if sum(within_dx) > 50 else np.nan
        up[index] = np.percentile(y[within_dx],84) if sum(within_dx) > 50 else np.nan
        dn[index] = np.percentile(y[within_dx],16) if sum(within_dx) > 50 else np.nan
    
    nan_mask = ~np.isnan(ys)
    xs, ys = xs[nan_mask], ys[nan_mask]
    up, dn = up[nan_mask], dn[nan_mask]

    return xs, ys, dn, up

counter = 0

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in all_subhalos.keys():
        this_subhalo = all_subhalos[key]
        
        this_sm = np.array(this_subhalo['TotalStellarMass'])
        
        this_gmsd  = np.array(this_subhalo['GasMass']).flatten()
        this_smsd  = np.array(this_subhalo['StellarMass']).flatten()
        this_sfrsd = np.array(this_subhalo['StarFormationRate']).flatten()
        center = (50,50)
        
        y_indices, x_indices = np.indices((100, 100)) ## based on size=50, pixl=1.0
        
        this_rad = np.sqrt((x_indices - center[1])**2 + (y_indices - center[0])**2).flatten()
        
        if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
            all_gm.extend(this_gmsd)
            all_sm.extend(this_smsd)
            all_sfr.extend(this_sfrsd)
            all_rad.extend(this_rad)
            counter += 1
        
print('Number of Galaxies in this sample:',counter)
all_gm  = np.array(all_gm )
all_sm  = np.array(all_sm )
all_sfr = np.array(all_sfr)
all_rad = np.array(all_rad)

inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))

all_gm  = all_gm [inf_mask]
all_sm  = all_sm [inf_mask]
all_sfr = all_sfr[inf_mask]
all_rad = all_rad[inf_mask]

all_gm  = np.log10(all_gm )
all_sm  = np.log10(all_sm )
all_sfr = np.log10(all_sfr)

min_x = 7
max_x = 9.5
min_y = -4
max_y = 1

keep_mask = ((all_sm > min_x) & (all_sm < max_x) & (all_sfr > min_y) & (all_sfr < max_y))

all_gm  = all_gm [keep_mask]
all_sm  = all_sm [keep_mask]
all_sfr = all_sfr[keep_mask]
all_rad = all_rad[keep_mask]

plt.clf()
fig = plt.figure(figsize=(15.0, 15.0))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

bins = 75
sums  , xbins, ybins = np.histogram2d(all_gm, all_sfr, bins=bins, weights=all_rad)
counts,     _,     _ = np.histogram2d(all_gm, all_sfr, bins=(xbins,ybins))

sums   = np.transpose(sums)
counts = np.transpose(counts)

valid_bins = counts > 0
masked_ratio = np.full_like(sums, np.nan)  # Create an array full of NaNs
masked_ratio[valid_bins] = sums[valid_bins] / counts[valid_bins]  # Only fill valid bins

histogram = ax.pcolor(xbins, ybins, masked_ratio, vmin=0, vmax=20,
                      cmap=cmr.get_sub_cmap('magma_r', 0.1, 1), rasterized=True)

xs, ys, dn, up = median(all_gm, all_sfr)

print('SK',np.polyfit(xs, ys, 1))

plt.xlabel(r'$\log(\Sigma_{\rm gas}~[M_\odot/{\rm kpc^2}])$')
plt.ylabel(r'$\log(\Sigma_{\rm SFR}~[M_\odot/{\rm yr}/{\rm kpc^2}])$')

cbaxes = fig.add_axes([0.1, 1.01, 0.85, 0.03])
cbar = plt.colorbar(histogram, cax=cbaxes, pad=0.03, orientation='horizontal', fraction=0.08)
cbar.ax.text( 0.5,1.2, r'${\rm Radius/kpc}$', transform=cbar.ax.transAxes, fontsize=50, ha='center') 
cbar.ax.tick_params(labelsize=50)
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_label_position('top')

plt.tight_layout()
plt.savefig('./figs/Figure10.pdf', bbox_inches='tight')