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

data = '../../IllustrisTNG_L35n2160TNG.hdf5'

all_sm  = []
all_sfr = []

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

def ret(x):
    return list(np.array(x).flatten())

counter = 0

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in all_subhalos.keys():
        this_subhalo = all_subhalos[key]
        
        this_sm = np.array(this_subhalo['TotalStellarMass'])
        
        this_smsd  = np.array(ret(this_subhalo['StellarMass']))
        this_sfrsd = np.array(ret(this_subhalo['StarFormationRate']))
        
        if np.any(this_smsd > 10**6) and np.any(this_sfrsd > 10**-5.0):
            all_sm.extend(this_smsd)
            all_sfr.extend(this_sfrsd)
            counter += 1
        
print('Number of Galaxies in this sample:',counter)
all_sm  = np.array(all_sm )
all_sfr = np.array(all_sfr)

inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))

all_sm  = all_sm [inf_mask]
all_sfr = all_sfr[inf_mask]

all_sm  = np.log10(all_sm )
all_sfr = np.log10(all_sfr)

N = 4
cmap = cmr.get_sub_cmap('cmr.ember', 0.25, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

plt.clf()
fig = plt.figure(figsize=(15.0, 15.0))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

min_x = 5.0
max_x = 9.5
min_y = -10.0
max_y = 1

keep_mask = ((all_sm > min_x) & (all_sm < max_x) & (all_sfr > min_y) & (all_sfr < max_y))

all_sm  = all_sm [keep_mask]
all_sfr = all_sfr[keep_mask]

bins=100
counts, xbins, ybins = np.histogram2d(all_sm, all_sfr, bins=bins)

counts = np.transpose(counts)

valid_bins = counts > 10
masked_ratio = np.full_like(counts, np.nan)
masked_ratio[valid_bins] = counts[valid_bins]

ax.pcolor(xbins, ybins, masked_ratio, cmap=cmr.get_sub_cmap('Greys', 0, 0.8),
          alpha=0.5, zorder=-1, norm='log', rasterized=True)

# plt.hist2d(all_sm, all_sfr, bins=(100,100), cmap=cmr.get_sub_cmap('Greys', 0, 0.8), 
           # rasterized=True, norm='log', alpha=0.5, zorder=2)

############
mass, sfr, sfr1, sfr2 = median(all_sm, all_sfr)

poly = np.polyfit(mass, sfr, 1)

print(f'No Cutoff Fit:', poly)

ax.plot(mass, sfr , color=col[0], lw=8, ls='-',
        label=r'$\log\Sigma_{\star,\,{\rm min}}={\rm None}, \log\Sigma_{\rm SFR,\,{\rm min}}={\rm None}$')
ax.fill_between(mass, sfr1, sfr2, color=col[0], alpha=0.25)
############

############
low_mask = ((all_sm > 6.0) & (all_sfr > -5.0))
mass, sfr, sfr1, sfr2 = median(all_sm[low_mask], all_sfr[low_mask])

poly = np.polyfit(mass, sfr, 1)

print(f'Lowest Cutoff Fit:', poly)

ax.plot(mass, sfr , color=col[1], lw=8, ls='--',
        label=r'$\log\Sigma_{\star,\,{\rm min}}=6.0, \log\Sigma_{\rm SFR,\,{\rm min}}=-5.0$')
ax.fill_between(mass, sfr1, sfr2, color=col[1], alpha=0.25)
############

############
middle_mask = ((all_sm > 6.5) & (all_sfr > -4.5))
mass, sfr, sfr1, sfr2 = median(all_sm[middle_mask], all_sfr[middle_mask])

poly = np.polyfit(mass, sfr, 1)

print(f'Middle Cutoff Fit:', poly)

ax.plot(mass, sfr , color=col[2], lw=8, ls=':',
        label=r'$\log\Sigma_{\star,\,{\rm min}}=6.5, \log\Sigma_{\rm SFR,\,{\rm min}}=-4.5$')
ax.fill_between(mass, sfr1, sfr2, color=col[2], alpha=0.25)
############

############
high_mask = ((all_sm > 7.0) & (all_sfr > -4.0))
mass, sfr, sfr1, sfr2 = median(all_sm[high_mask], all_sfr[high_mask])

poly = np.polyfit(mass, sfr, 1)

print(f'Highest Cutoff fit:', poly)

ax.plot(mass, sfr , color=col[3], lw=8, ls='-.',
        label=r'$\log\Sigma_{\star,\,{\rm min}}=7.0, \log\Sigma_{\rm SFR,\,{\rm min}}=-4.0~{\rm (fiducial)}$')
ax.fill_between(mass, sfr1, sfr2, color=col[3], alpha=0.25)
############

plt.xlabel(r'$\log(\Sigma_{\star}~[M_\odot/{\rm kpc^2}])$')
plt.ylabel(r'$\log(\Sigma_{\rm SFR}~[M_\odot/{\rm yr}/{\rm kpc^2}])$')

leg = ax.legend( frameon=False,handletextpad=0.25, labelspacing=0.05,
                 loc='upper left', fontsize=fs_og*0.8 )

for index, text in enumerate(leg.get_texts()):
    text.set_color(col[index])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin, 9.49)
ax.set_ylim(ymin, 0.99)

ax.axvline(5.5, color='gray', ls='--', alpha=0.5)
ax.text(5.525,-8.75,r'$\sim3~{\rm Star~Particles}/{\rm pixel}$', rotation=90, fontsize=30, color='gray')
ax.axvline(6.0, color='gray', ls='--', alpha=0.5)
ax.text(6.025,-8.75,r'$\sim10~{\rm Star~Particles}/{\rm pixel}$', rotation=90, fontsize=30, color='gray')
ax.axvline(6.5, color='gray', ls='--', alpha=0.5)
ax.text(6.525,-8.75,r'$\sim32~{\rm Star~Particles}/{\rm pixel}$', rotation=90, fontsize=30, color='gray')
ax.axvline(7.0, color='gray', ls='--', alpha=0.5)
ax.text(7.025,-8.75,r'$\sim100~{\rm Star~Particles}/{\rm pixel}$', rotation=90, fontsize=30, color='gray')

ax.axhline(-3, color='k', ls='--', alpha=0.5)
ax.axhline(-4, color='k', ls='--', alpha=0.5)

ax.text(5.55,-2.9,r'${\rm CALIFA~Sensitivity~Limit}$', fontsize=24, color='k')
ax.text(5.55,-3.9,r'${\rm MaNGA~Sensitivity~Limit}$' , fontsize=24, color='k')

plt.tight_layout()
plt.savefig('./figs/AppendixD.pdf', bbox_inches='tight')