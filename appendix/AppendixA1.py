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

N = 2
cmap = cmr.get_sub_cmap('cmr.ember', 0.25, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

plt.clf()
fig = plt.figure(figsize=(15.0, 15.0))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

for index, which in enumerate(['../../IllustrisTNG_L35n2160TNG.hdf5',
                               '../../standard/updated.hdf5']):
    counter = 0
    
    all_sm  = []
    all_sfr = []
    
    with h5py.File(which, 'r') as f:
        if index == 1:
            all_subhalos = f['IllustrisTNG']['L75n1820TNG']['snap_99']
        else:
            all_subhalos = f['snap_99']
        
        for key in all_subhalos.keys():
            this_subhalo = all_subhalos[key]

            this_sm = np.array(this_subhalo['TotalStellarMass'])

            this_smsd  = np.array(this_subhalo['StellarMass']).flatten()
            this_sfrsd = np.array(this_subhalo['StarFormationRate']).flatten()

            if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
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

    min_x = 7.0
    max_x = 9.5
    min_y = -4.0
    max_y = 1

    keep_mask = ((all_sm > min_x) & (all_sm < max_x) & (all_sfr > min_y) & (all_sfr < max_y))

    all_sm  = all_sm [keep_mask]
    all_sfr = all_sfr[keep_mask]
    
    mass, sfr, sfr1, sfr2 = median(all_sm, all_sfr)

    poly = np.polyfit(mass, sfr, 1)

    if index == 0:
        print(f'TNG50 fit:', poly)
        label = r'${\rm TNG50}$'
        ls = '-'
    elif index == 1:
        print(f'TNG100 fit:', poly)
        label = r'${\rm TNG100}$'
        ls = '--'
        
    ax.plot(mass, sfr, color=col[index], lw=8, label=label, ls=ls)
    ax.fill_between(mass, sfr1, sfr2, color=col[index], alpha=0.25)
    
plt.xlabel(r'$\log(\Sigma_{\star}~[M_\odot/{\rm kpc^2}])$')
plt.ylabel(r'$\log(\Sigma_{\rm SFR}~[M_\odot/{\rm yr}/{\rm kpc^2}])$')

leg = ax.legend( frameon=False,handletextpad=0.25, labelspacing=0.05,
                 loc='upper left', fontsize=fs_og )

for index, text in enumerate(leg.get_texts()):
    text.set_color(col[index])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin, 9.49)
ax.set_ylim(ymin, -0.5)

plt.tight_layout()
plt.savefig('./figs/AppendixA1.pdf', bbox_inches='tight')