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

def median(x,y,max_n=50):
    xs = np.arange(np.min(x), 10, 0.25)
    ys = np.zeros(len(xs))
    up = np.zeros(len(xs))
    dn = np.zeros(len(xs))
    
    dx = xs[1] - xs[0]
        
    for index, this_x in enumerate(xs):
        within_dx = (x > this_x) & (x < this_x+dx)
        
        ys[index] = np.median(y[within_dx]) if sum(within_dx) > max_n else np.nan
        up[index] = np.percentile(y[within_dx],84) if sum(within_dx) > max_n else np.nan
        dn[index] = np.percentile(y[within_dx],16) if sum(within_dx) > max_n else np.nan
    
    nan_mask = ~np.isnan(ys)
    xs, ys = xs[nan_mask], ys[nan_mask]
    up, dn = up[nan_mask], dn[nan_mask]

    return xs, ys, dn, up

N = 2
cmap = cmr.get_sub_cmap('cmr.ember', 0.25, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

plt.clf()
fig = plt.figure(figsize=(15.0, 14.25))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

for index, which in enumerate(['../../IllustrisTNG_L35n2160TNG.hdf5',
                               '../../standard/updated.hdf5']):
    
    counter = 0
    
    all_z   = []
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

            this_zsd   = np.array(this_subhalo['Metallicity']).flatten()
            this_smsd  = np.array(this_subhalo['StellarMass']).flatten()
            this_sfrsd = np.array(this_subhalo['StarFormationRate']).flatten()

            if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
                all_z.extend(this_zsd)
                all_sm.extend(this_smsd)
                all_sfr.extend(this_sfrsd)
                counter += 1
        
    print('Number of Galaxies in this sample:',counter)
    all_z   = np.array(all_z  )
    all_sm  = np.array(all_sm )
    all_sfr = np.array(all_sfr)

    inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))
    invalid_pix = (
        (all_sm < 10**7) |
        (all_sfr < 10**-4)
    )

    x = np.log10(all_sm [~invalid_pix&inf_mask])
    y = np.log10(all_z  [~invalid_pix&inf_mask] * 0.35/0.76 * 1/16) + 12

    # if index == 1:
    #     bins=100
    #     ax.hist2d(x, y, range=[[7.0, 9.5], [8.0, 9.25]], bins=100, norm='log',
    #                cmap='Greys',alpha=0.5, rasterized=True, zorder=-1)
    
    mass, metal, metal1, metal2 = median(x, y, max_n=50 if index == 0 else 200)

    if index == 0:
        label = r'${\rm Full~Sample}$'
        ls = '-'
    elif index == 1:
        label = r'${\rm Centrals}$'
        ls = '--'

    ax.plot(mass, metal, color=col[index], lw=8, label=label, ls=ls)
    ax.fill_between(mass, metal1, metal2, color=col[index], alpha=0.25)

ax.set_xlabel(r'$\mathrm{log}(\Sigma_{\star} ~[\mathrm{M}_{\odot}\, \mathrm{kpc}^{-2}])$', fontsize=fs_og)
ax.set_ylabel(r'$12 + \mathrm{log}(\mathrm{O}/\mathrm{H})~[{\rm dex}]$', fontsize=fs_og)
ax.tick_params(labelsize=fs_og)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(7.01, 9.74)

plt.tight_layout()
plt.savefig('./figs/AppendixA2.pdf',bbox_inches='tight')