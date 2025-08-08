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

### OBSERVATIONS ###
PHANGS_mass = [6.201298701298701, 10.506493506493506]
PHANGS_sfr  = [-4.10410094637224, 0.3775861362612133]

ALMA_QUEST_mass = [6.8012684989429175, 9.997885835095138]
ALMA_QUEST_sfr  = [-2.775362318840579, -0.6116064589269845]

califa_mass = [7.0, 9.0]
califa_sfr  = [7.0 * 0.72 - 7.95, 9.0 * 0.72 - 7.95]

tf18_sfr = np.array([ -3.086206896551724, -2.994252873563218, -2.913793103448276 , -2.8333333333333335,
                     -2.7413793103448274, -2.67816091954023 , -2.53448275862069  , -2.413793103448276,
                     -2.2758620689655173, -2.109195402298851, -1.8735632183908046, -1.5919540229885059,
                     -1.1896551724137931])
tf18_mass = np.array([ 7.139759036144578, 7.351807228915662, 7.53012048192771 , 7.693975903614458,
                      7.877108433734939 , 7.992771084337349, 8.132530120481928, 8.27710843373494 ,
                      8.412048192771085 , 8.57590361445783 , 8.807228915662652, 9.4              ,
                      9.679518072289156])
tf18_sfr_l = np.array([ -3.6149425287356323, -3.574712643678161, -3.557471264367816, -3.5114942528735633,
                       -3.4827586206896552, -3.4827586206896552, -3.350574712643678, -3.235632183908046,
                       -3.0747126436781613, -2.913793103448276, -2.672413793103448, -2.505747126436782,
                       -2.0919540229885056])
tf18_sfr_h = np.array([ -2.5977011494252875, -2.5, -2.3850574712643677, -2.2701149425287355, 
                       -2.149425287356322, -2.068965517241379, -1.9080459770114944, -1.706896551724138,
                       -1.5977011494252875, -1.4482758620689655, -1.2586206896551726, -0.8275862068965517,
                       -0.28160919540229884])

manga_mass, manga_sfr = np.loadtxt('manga.txt', usecols=(0, 1), unpack=True)
manga_sfr  += 6.0
manga_mass += 6.0
####################


# data = '../standard/qicriteriacubic-spline50n1.0.hdf5'
# data = '../cubic-spline50n1.0maps.hdf5'
data = '../IllustrisTNG_L35n2160TNG.hdf5'

all_sm  = []
all_sfr = []

def median(x,y):
    xs = np.arange(np.min(x), 10, 0.25)
    ys = np.zeros(len(xs))
    up = np.zeros(len(xs))
    dn = np.zeros(len(xs))
    
    dx = xs[1] - xs[0]
    
    print(dx)
    
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

plt.hist2d(all_sm  , all_sfr, bins=(100,100), cmap=cmr.get_sub_cmap('Greys', 0, 0.8), 
           rasterized=True, norm='log', alpha=0.5, zorder=2)

mass, sfr, sfr1, sfr2 = median(all_sm, all_sfr)

poly = np.polyfit(mass, sfr, 1)

print(f'Full fit:', poly)

ax.plot(mass, sfr, color=col[0], label=r'${\rm IllustrisTNG}$', lw=8)

ax.plot(califa_mass, califa_sfr, color=col[1], lw=5, label=r'CALIFA (Cano-Díaz+2016)' ,ls='--', alpha=0.5)
ax.plot(manga_mass, manga_sfr, color=col[2], lw=5, label='MaNGA (Hsieh+2017)',ls=':', alpha=0.5)
ax.plot(ALMA_QUEST_mass, ALMA_QUEST_sfr, color=col[3], lw=5, label='ALMaQUEST (Ellison+2021)', ls='-.', alpha=0.5)
ax.plot(PHANGS_mass, PHANGS_sfr, color=col[4], lw=5, label='PHANGS (Pessa+2021)',ls='-', alpha=0.5)
    
ax.plot(tf18_mass,   tf18_sfr,   color=col[5], lw=5, label='EAGLE (Trayford \& Schaye 2019)',ls='--', alpha=0.5)
ax.fill_between(tf18_mass, tf18_sfr_l, tf18_sfr_h, color=col[5], alpha=0.1)

ax.plot(mass, sfr , color='k', lw=11)
ax.plot(mass, sfr , color=col[0], lw=8)
ax.plot(mass, sfr1, color='k', lw=2)
ax.plot(mass, sfr2, color='k', lw=2)
ax.fill_between(mass, sfr1, sfr2, color=col[0], alpha=0.5)

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
    
plt.tight_layout()
plt.savefig('./figs/Figure2.pdf', bbox_inches='tight')