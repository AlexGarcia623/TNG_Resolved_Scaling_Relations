import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cmasher as cmr

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

fs_og = 25
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5 * 1.5
mpl.rcParams['ytick.major.width'] = 1.5 * 1.5
mpl.rcParams['xtick.minor.width'] = 1.0 * 1.5
mpl.rcParams['ytick.minor.width'] = 1.0 * 1.5
mpl.rcParams['xtick.major.size'] = 7.5 * 1.5
mpl.rcParams['ytick.major.size'] = 7.5 * 1.5
mpl.rcParams['xtick.minor.size'] = 3.5 * 1.5
mpl.rcParams['ytick.minor.size'] = 3.5 * 1.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

data = '../IllustrisTNG_L35n2160TNG.hdf5'

all_sm  = []
all_sfr = []
all_rad = []
all_total_mass = []

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
        
        this_smsd  = np.array(this_subhalo['StellarMass']).flatten()
        this_sfrsd = np.array(this_subhalo['StarFormationRate']).flatten()
        center = (50,50)
        
        y_indices, x_indices = np.indices((100, 100)) ## based on size=50, pixl=1.0
        
        this_rad = np.sqrt((x_indices - center[1])**2 + (y_indices - center[0])**2).flatten()
        
        if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
            all_sm.extend(this_smsd)
            all_sfr.extend(this_sfrsd)
            all_rad.extend(this_rad)
            all_total_mass.extend(np.ones_like(this_smsd)*this_sm)
            counter += 1
        
print('Number of Galaxies in this sample:',counter)
all_sm  = np.array(all_sm )
all_sfr = np.array(all_sfr)
all_rad = np.array(all_rad)
all_total_mass = np.array(all_total_mass)

inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))

all_sm  = all_sm [inf_mask]
all_sfr = all_sfr[inf_mask]
all_rad = all_rad[inf_mask]
all_total_mass = all_total_mass[inf_mask]

all_sm  = np.log10(all_sm )
all_sfr = np.log10(all_sfr)

N = 6
cmap = cmr.get_sub_cmap('cmr.ember', 0.25, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

plt.clf()
fig = plt.figure(figsize=(15.0, 15.0))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

min_x = 7.0
max_x = 9.5
min_y = -4.0
max_y = 1

keep_mask = ((all_sm > min_x) & (all_sm < max_x) & (all_sfr > min_y) & (all_sfr < max_y))

all_sm  = all_sm [keep_mask]
all_sfr = all_sfr[keep_mask]
all_rad = all_rad[keep_mask]
all_total_mass = all_total_mass[keep_mask]

fig, axs = plt.subplots(2,2, figsize=(15.0,15.0), sharex=True, sharey=True)
axs = axs.flatten()

titles = ['$10^{9.0} \,<\,{M}_{\star}/M_\odot\,<\,10^{9.5} $', \
          '$10^{9.5} \,<\,{M}_{\star}/M_\odot\,<\,10^{10.0}$', \
          '$10^{10.0}\,<\,{M}_{\star}/M_\odot\,<\,10^{10.5}$', \
          '$10^{10.5}\,<\,{M}_{\star}/M_\odot\,<\,10^{11.0}$', \
          '$10^{11.0}\,<\,{M}_{\star}/M_\odot$']
N = 5
cmap = cmr.get_sub_cmap('cmr.torch', 0.2, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

mmin = 9.0
dm   = 0.5
mmax = mmin + dm
for k in range(len(titles)):
    if k > 3:
        continue
    within_dm = (all_total_mass > mmin) & (all_total_mass < mmax)
    
    print(len(np.unique(all_total_mass[within_dm])))
    
    bins = 50
    sums  , xbins, ybins = np.histogram2d(all_sm[within_dm], all_sfr[within_dm], bins=bins, weights=all_rad[within_dm])
    counts,     _,     _ = np.histogram2d(all_sm[within_dm], all_sfr[within_dm], bins=(xbins,ybins))

    sums   = np.transpose(sums)
    counts = np.transpose(counts)

    valid_bins = counts > 5
    masked_ratio = np.full_like(sums, np.nan)  # Create an array full of NaNs
    masked_ratio[valid_bins] = sums[valid_bins] / counts[valid_bins]  # Only fill valid bins
    
    histogram = axs[k].pcolor(xbins, ybins, masked_ratio, vmin=0, vmax=20,
                              cmap=cmr.get_sub_cmap('magma_r', 0.1, 1), rasterized=True)
    
    mass, sfr, sfr1, sfr2 = median(all_sm[within_dm], all_sfr[within_dm])

    axs[k].plot(mass, sfr , color='k', lw=8)
    
    axs[k].fill_between(mass, sfr1, sfr2, color='k', alpha=0.25)
    axs[k].plot(mass, sfr1, color='k', lw=4)
    axs[k].plot(mass, sfr2, color='k', lw=4)
    
    axs[k].text(0.05,0.95,titles[k],transform=axs[k].transAxes,va='top')
    
    mmin+=dm
    if k == len(titles) - 2:
        dm = 100
    mmax+=dm

ax_cbar = fig.add_axes([0.15, 1., 0.7, 0.03])

cb = plt.colorbar(histogram, cax=ax_cbar,orientation='horizontal',aspect=10,shrink=0.75)
# cb.set_label(r'${\rm Radius~(kpc)}$',fontsize=50,labelpad=100)

cb.ax.text( 0.5,1.25, r'${\rm Radius/kpc}$' , transform=cb.ax.transAxes, fontsize=fs_og*1.25, ha='center' )

# cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')

cb.ax.tick_params(labelsize=fs_og)
    
axs[2].set_xlabel(r'$\log(\Sigma_{\star}~[M_\odot/{\rm kpc^2}])$')
axs[3].set_xlabel(r'$\log(\Sigma_{\star}~[M_\odot/{\rm kpc^2}])$')

axs[0].set_ylabel(r'$\log(\Sigma_{\rm SFR}~[M_\odot/{\rm yr}/{\rm kpc^2}])$')
axs[2].set_ylabel(r'$\log(\Sigma_{\rm SFR}~[M_\odot/{\rm yr}/{\rm kpc^2}])$')

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin, 9.49)
ax.set_ylim(ymin, 0.99)
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig('./figs/Figure11.pdf', bbox_inches='tight')