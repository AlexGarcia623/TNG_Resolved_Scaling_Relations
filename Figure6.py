import sys
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

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
all_total_mass = []

def median_relation(x, y, dx=0.25, up=84, down=16, req=30):    
    xs = np.arange(np.min(x), np.max(x), dx)
    
    median = np.zeros(len(xs))
    lower  = np.zeros(len(xs))
    upper  = np.zeros(len(xs))
    
    for index, xval in enumerate(xs):
        within_dx = (x > xval) & (x < xval + dx) 
        y_within = y[within_dx]
        
        if len(y_within) > req:
            median[index] = np.median(y_within)
            lower[index] = np.percentile(y_within,down)
            upper[index] = np.percentile(y_within,up)
            
        else:
            median[index] = np.nan
            
    return_mask = ~np.isnan(median)
            
    return xs[return_mask], median[return_mask], lower[return_mask], upper[return_mask]

def get_fixed_SFR_bin(all_m, all_z, all_SFR, current_SFR, dSFR, req=200):
    
    mask = (all_SFR > current_SFR) & (all_SFR < current_SFR + dSFR)
    
    current_m = all_m[mask]
    current_z = all_z[mask]
    
    worm_x = []
    worm_y = []
    
    if sum(mask) > 30:
        mbins = np.linspace(np.min(current_m), np.max(current_m), 10)
        dm = mbins[1] - mbins[0]
        
        for index, mbin in enumerate(mbins):
            mask = (current_m > mbin) & (current_m < mbin + dm)
            worm_x.append( mbin + dm/2 )
            if sum(mask) > req:
                worm_y.append( np.median(current_z[mask]) )
            else:
                worm_y.append( np.nan )
            
    return worm_x, worm_y

all_z   = []
all_sm  = []
all_sfr = []
all_rad = []

counter = 0

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in all_subhalos.keys():
        this_subhalo = all_subhalos[key]
        
        this_sm = np.array(this_subhalo['TotalStellarMass'])
        
        this_zsd   = np.array(this_subhalo['Metallicity'])
        this_smsd  = np.array(this_subhalo['StellarMass'])
        this_sfrsd = np.array(this_subhalo['StarFormationRate'])
        
        center = (50,50)
        
        y_indices, x_indices = np.indices((100, 100)) ## based on size=50, pixl=1.0
        
        this_rad = np.sqrt((x_indices - center[1])**2 + (y_indices - center[0])**2)
        
        if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
            all_z.extend(this_zsd)
            all_sm.extend(this_smsd)
            all_sfr.extend(this_sfrsd)
            all_rad.extend(this_rad)
            all_total_mass.extend(np.ones_like(this_smsd)*this_sm)
            counter += 1
        
print('Number of Galaxies in this sample:',counter)
all_z   = np.array(all_z  )
all_sm  = np.array(all_sm )
all_sfr = np.array(all_sfr)
all_rad = np.array(all_rad)

inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))
invalid_pix = (
    (all_sm < 10**7) |
    (all_sfr < 10**-4)
)

x = np.log10(all_sm [~invalid_pix&inf_mask])
y = np.log10(all_z  [~invalid_pix&inf_mask] * 0.35/0.76 * 1/16) + 12 - 0.2
z = np.log10(all_sfr[~invalid_pix&inf_mask])

r = all_rad[~invalid_pix&inf_mask]

fig = plt.figure(figsize=(15.0, 15.0))
ax  = plt.gca()

dsSFR = 0.75
sSFR_bins = np.arange(-4,-1,dsSFR)[::-1]

N = len(sSFR_bins)
cmap = cmr.get_sub_cmap('copper_r', 0.2, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

mass, metal, metal1, metal2 = median_relation(x, y, dx=0.25, req=200)

ax.plot(mass*100, metal*100 , color='gray', lw=5, label='Median rMZR', alpha=0.9)
ax.plot(mass, metal , color='gray', lw=5, alpha=0.75)
ax.plot(mass, metal1, color='gray', ls='--', lw=5, alpha=0.75)
ax.plot(mass, metal2, color='gray', ls='--', lw=5, alpha=0.75)

for idx, sSFR_bin in enumerate(sSFR_bins):
    worm_x, worm_y = get_fixed_SFR_bin(x,y,z,sSFR_bin,dsSFR)
    ax.plot(worm_x, worm_y, lw=7, color=col[idx],
             label=f"${sSFR_bin:0.2f}$" + r"$\,< \log(\Sigma_{\mathrm{SFR}}~[M_\odot/{\rm kpc}^2/{\rm yr}]) <\,$" + f"${(sSFR_bin+dsSFR):0.2f}$")
    

ax.set_xlabel(r'$\mathrm{log}(\Sigma_{\star} ~[\mathrm{M}_{\odot}\, \mathrm{kpc}^{-2}])$', fontsize=fs_og)
ax.set_ylabel(r'$12 + \mathrm{log}(\mathrm{O}/\mathrm{H})~[{\rm dex}]$', fontsize=fs_og)
ax.tick_params(labelsize=fs_og)

leg = ax.legend( frameon=False,handletextpad=0.25, labelspacing=0.05, fontsize=32, loc='upper left' )

for index, text in enumerate(leg.get_texts()):
    if index == 0:
        text.set_color('gray')
    else:
        text.set_color(col[index-1])

bins=100
ax.hist2d(x, y, range=[[7.0, 9.5], [8.0, 9.25]], bins=100, norm='log',
           cmap='Greys',alpha=0.5, rasterized=True)

ax.set_ylim([8.00, 9.39])
ax.set_xlim([7.01, 9.49])

plt.tight_layout()
plt.savefig('./figs/Figure6.pdf',bbox_inches='tight')