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

### Observations ###
tf18_mass, tf18_z = np.loadtxt('tf18_z.txt',
                               usecols=(0, 1), unpack=True)
califa_mass, califa_z = np.loadtxt('califa_z.txt',
                                   usecols=(0, 1), unpack=True)
manga_mass, manga_z = np.loadtxt('manga_z.txt',
                                 usecols=(0, 1), unpack=True)


data = '../IllustrisTNG_L35n2160TNG.hdf5'

all_sm  = []
all_sfr = []
all_zs  = []

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
        this_metal = np.array(this_subhalo['Metallicity']).flatten()
        
        if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
            all_sm.extend(this_smsd)
            all_sfr.extend(this_sfrsd)
            all_zs.extend(this_metal)
            counter += 1
        
print('Number of Galaxies in this sample:',counter)
all_sm  = np.array(all_sm )
all_sfr = np.array(all_sfr)
all_zs  = np.array(all_zs)

inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))
alex_mask = (all_sm > 10**7.0) & (all_sfr > 10**-4.0)

all_sm  = all_sm [alex_mask & inf_mask]
all_sfr = all_sfr[alex_mask & inf_mask]
all_zs  = all_zs [alex_mask & inf_mask]

all_sm  = np.log10(all_sm )
all_sfr = np.log10(all_sfr)
all_zs  = np.log10(all_zs * 0.35/0.76 * 1.00/16.00) + 12 - 0.2

N = 5
cmap = cmr.get_sub_cmap('cmr.ember', 0.25, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

plt.clf()
fig = plt.figure(figsize=(15.0, 15.0))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])
plt.hist2d(all_sm  , all_zs, bins=(75,75), cmap=cmr.get_sub_cmap('Greys', 0, 0.8), 
           rasterized=True, norm='log', alpha=0.5, zorder=2)

mass, z, z1, z2 = median(all_sm, all_zs)

poly = np.polyfit(mass, z, 1)

print(f'Full fit:', poly)

ax.plot(mass, z, color=col[0], label=r'${\rm IllustrisTNG}$', lw=8)

ax.plot(tf18_mass, tf18_z, color=col[1], lw=5, label=r'${\rm EAGLE}~{\rm (Trayford\, \&\, Schaye\, 2019)}$',ls='--')
ax.plot(manga_mass, manga_z, color=col[2], lw=5, label='MaNGA (Barrera-Ballesteros+2016)',ls=':')
ax.plot(califa_mass, califa_z, color=col[3], lw=5, label='CALIFA (Sánchez+2013)',ls='-.')

ax.plot(mass, z , color='k', lw=11)
ax.plot(mass, z , color=col[0], lw=8)
ax.plot(mass, z1, color='k', lw=2)
ax.plot(mass, z2, color='k', lw=2)
ax.fill_between(mass, z1, z2, color=col[0], alpha=0.5)

plt.xlabel(r'$\log(\Sigma_{\star}~[M_\odot/{\rm kpc^2}])$')
plt.ylabel(r'$\log({\rm O/H}+12)~[{\rm dex}]$')

leg = ax.legend( frameon=False,handletextpad=0.25, labelspacing=0.05,
                 loc='upper left', fontsize=fs_og*0.8 )

for index, text in enumerate(leg.get_texts()):
    text.set_color(col[index])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin, 9.74)
    
plt.tight_layout()
plt.savefig('./figs/Figure5.pdf', bbox_inches='tight')