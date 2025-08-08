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


def offsets_from_relation(all_m, all_y, dm=0.5):
    mbins = np.arange(7,10,dm)
    ybins = np.ones(len(mbins))*np.nan
    
    for idx, m in enumerate(mbins):
        
        within_dm = (all_m > m) & (all_m < m + dm)
        
        if np.sum(within_dm) > 30:
            ybins[idx] = np.median(all_y[within_dm])
            
    relation = interp1d(mbins,ybins,fill_value='extrapolate')
    offsets = all_y - relation(all_m)
    
    return offsets

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

fig, axs = plt.subplots(1, 2, figsize=(28,15), sharex=True, sharey=True, 
                       gridspec_kw=dict(width_ratios=[0.8,1], height_ratios=[1]))

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

axs[0].hist2d(x, y, range=[[7.0, 9.5], [8.0, 9.25]], bins=100, norm='log',
           cmap='Greys',alpha=0.5, rasterized=True)

dsSFR = 0.67
sSFR_bins = np.arange(-4,-1,dsSFR)[::-1]

N = len(sSFR_bins)
cmap = cmr.get_sub_cmap('cmr.pride', 0.2, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

mass, metal, metal1, metal2 = median_relation(x, y, dx=0.25, req=200)

axs[0].plot(mass, metal , color='k', lw=5, label='Median rMZR',alpha=0.5)
axs[0].plot(mass, metal1, color='k', ls='--', lw=5, alpha=0.5)
axs[0].plot(mass, metal2, color='k', ls='--', lw=5, alpha=0.5)

for idx, sSFR_bin in enumerate(sSFR_bins):
    worm_x, worm_y = get_fixed_SFR_bin(x,y,z,sSFR_bin,dsSFR)
    axs[0].plot(worm_x, worm_y, lw=7, color=col[idx],
             label=f"${sSFR_bin:0.2f}$" + "$\,< \Sigma_{\mathrm{SFR}} <\,$" + f"${(sSFR_bin+dsSFR):0.2f}$")
    
ax = axs[0]

ax.set_xlabel('$\mathrm{log}(\Sigma_{\star} / \mathrm{M}_{\odot}\, \mathrm{kpc}^{-2})$', fontsize=fs_og)
ax.set_ylabel('$12 + \mathrm{log}(\mathrm{O}/\mathrm{H})$', fontsize=fs_og)
ax.tick_params(labelsize=fs_og)

ax.set_ylim([8.00, 9.60])
ax.set_xlim([7.01, 9.50])

leg = ax.legend( frameon=False,handletextpad=0.25, labelspacing=0.05, fontsize=fs_og, loc='lower right' )

for index, text in enumerate(leg.get_texts()):
    if index == 0:
        text.set_color('gray')
    else:
        text.set_color(col[index-1])

offsets_rSFMS = offsets_from_relation(x, z)
offsets_rMZR  = offsets_from_relation(x, y)

dsSFR = 1
sSFR_bins = np.arange(-4,0,dsSFR)[::-1]

N = len(sSFR_bins)
cmap = cmr.get_sub_cmap('copper', 0.2, 0.65, N=N)
newcolors = np.linspace(0, 1, N)
col = [ cmap(x) for x in newcolors[::-1] ]

mass, metal, metal1, metal2 = median_relation(x, y, dx=0.25, req=200)

axs[1].plot(mass, metal , color='k', lw=5)
axs[1].plot(mass, metal1, color='k', ls='--', lw=5)
axs[1].plot(mass, metal2, color='k', ls='--', lw=5)

# for idx, sSFR_bin in enumerate(sSFR_bins):
#     worm_x, worm_y = get_fixed_SFR_bin(x,y,z,sSFR_bin,dsSFR)
#     plt.plot(worm_x, worm_y, lw=7, color=col[idx],
#              label=f"{sSFR_bin:0.1f}" + "$\,< \Sigma_{\mathrm{SFR}} <\,$" + f"{(sSFR_bin+dsSFR):0.1f}")
     
bins=100
sums  , xbins, ybins = np.histogram2d(x, y, bins=bins, weights=offsets_rSFMS)
counts,     _,     _ = np.histogram2d(x, y, bins=(xbins,ybins))

sums   = np.transpose(sums)
counts = np.transpose(counts)

valid_bins = counts > 10
masked_ratio = np.full_like(sums, np.nan)  # Create an array full of NaNs
masked_ratio[valid_bins] = sums[valid_bins] / counts[valid_bins]  # Only fill valid bins

cmap = cmr.get_sub_cmap('cmr.pride', 0.1, 0.9)
img = axs[1].pcolor(xbins, ybins, masked_ratio, vmin=-0.5,vmax=0.5, alpha=0.75, cmap=cmap, rasterized=True)

cbar = plt.colorbar(img, ax=axs[1])
cbar.set_label(r'$\Delta \mathrm{rSFMS}$', rotation=-90, fontsize=fs_og)
cbar.ax.tick_params(labelsize=40)

ax = axs[1]

ax.set_xlabel('$\mathrm{log}(\Sigma_{\star} / \mathrm{M}_{\odot}\, \mathrm{kpc}^{-2})$', fontsize=fs_og)
# ax.set_ylabel('$12 + \mathrm{log}(\mathrm{O}/\mathrm{H})$', fontsize=fs_og)
ax.tick_params(labelsize=fs_og)

ax.set_ylim([8.00-0.2, 9.39-0.2])
ax.set_xlim([7.01, 9.49])

leg = ax.legend( frameon=False,handletextpad=0.25, labelspacing=0.05, fontsize=fs_og, loc='upper left' )

for index, text in enumerate(leg.get_texts()):
    if index == 0:
        text.set_color('k')
    else:
        text.set_color(col[index-1])

xstart = 0.7
ystart = 0.225
xlen = 0.15
ylen = 0.15

ax_inset = fig.add_axes([xstart, ystart, xlen, ylen])

### Plotting purposes only!! Removing this doesn't change anything
plot_mask = ((offsets_rSFMS > -2.01) & (offsets_rSFMS < 1.501) &
             (offsets_rMZR > -0.6) & (offsets_rMZR < 0.6))

bins = 50
sums  , xbins, ybins = np.histogram2d(offsets_rSFMS[plot_mask], offsets_rMZR[plot_mask], bins=bins, weights=r[plot_mask])
counts,     _,     _ = np.histogram2d(offsets_rSFMS[plot_mask], offsets_rMZR[plot_mask], bins=(xbins,ybins))

sums   = np.transpose(sums)
counts = np.transpose(counts)

valid_bins = counts > 10
masked_ratio = np.full_like(sums, np.nan)  # Create an array full of NaNs
masked_ratio[valid_bins] = sums[valid_bins] / counts[valid_bins]  # Only fill valid bins

cmap = cmr.get_sub_cmap('magma_r', 0, 0.8)
# cmap = cmr.get_sub_cmap('cmr.pride', 0.1, 0.9)
img = plt.pcolor(xbins, ybins, masked_ratio, vmin=0, vmax=15, cmap=cmap, rasterized=True)

cbar = fig.colorbar(img, ax=ax_inset, orientation='vertical')  
# cbar.set_label('Radius (kpc)', rotation=-90, fontsize=fs_og/1.5)
ax_inset.text(1.2,0.5,'Radius (kpc)', va='center',transform=ax_inset.transAxes, rotation=-90, fontsize=fs_og/2)

cbar.outline.set_linewidth(1.2)
cbar.ax.tick_params(axis='y', which='both', width=1, length=5, direction='in', labelsize=fs_og/2)  
cbar.ax.tick_params(axis='y', which='minor', width=0.5, length=3, direction='in')
# ax_inset.hist2d(offsets_rSFMS, offsets_rMZR, bins=bins, norm=LogNorm(), cmap='Greys', rasterized=True)

# corr_coeff, p_value = pearsonr(offsets_rSFMS, offsets_rMZR)

# ax_inset.text(0.025,0.85,'$R^2 = %s$' %round(corr_coeff**2,3), transform=ax_inset.transAxes, fontsize=fs_og/1.75)

ax_inset.set_xticks([-2,-1,0,1])

# slope, intercept = np.polyfit(offsets_rSFMS, offsets_rSFMS, 1)
# _x_ = np.linspace(np.min(offsets_rSFMS),np.max(offsets_rSFMS),100)
# _y_ = slope * _x_ + intercept

# ax_inset.plot(_x_, _y_, color='r', lw=3)

ax_inset.axhline(0, color='k', linestyle=':')
ax_inset.axvline(0, color='k', linestyle=':')

ax_inset.set_xlabel('$\Delta \mathrm{rSFMS}$',fontsize=fs_og/2)
ax_inset.set_ylabel('$\Delta \mathrm{rMZR}$' ,fontsize=fs_og/2)
    
ax_inset.spines['bottom'].set_linewidth(1.2); ax_inset.spines['top']  .set_linewidth(1.2)
ax_inset.spines['left']  .set_linewidth(1.2); ax_inset.spines['right'].set_linewidth(1.2)
    
ax_inset.tick_params(axis='both', width=1, length=6, which='major', labelsize=fs_og/2)
ax_inset.tick_params(axis='both', width=1, length=3, which='minor', labelsize=fs_og/2)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0)
plt.savefig('./figs/Figure6.pdf',bbox_inches='tight')