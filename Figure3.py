import sys
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cmasher as cmr
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

fs_og = 15
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 2.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5 
mpl.rcParams['ytick.major.width'] = 1.5 
mpl.rcParams['xtick.minor.width'] = 1.0 
mpl.rcParams['ytick.minor.width'] = 1.0 
mpl.rcParams['xtick.major.size'] = 7.5 
mpl.rcParams['ytick.major.size'] = 7.5 
mpl.rcParams['xtick.minor.size'] = 3.5 
mpl.rcParams['ytick.minor.size'] = 3.5 
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


data = '../IllustrisTNG_L35n2160TNG.hdf5'

def median(x,y):
    xs = np.arange(7, 10, 0.01)
    ys = np.zeros(len(xs))
    up = np.zeros(len(xs))
    dn = np.zeros(len(xs))
    
    dx = xs[1] - xs[0]
    
    for index, this_x in enumerate(xs):
        within_dx = (x > this_x) & (x < this_x+dx)
        
        ys[index] = np.median(y[within_dx]) if sum(within_dx) > 1 else np.nan
        up[index] = np.percentile(y[within_dx],84) if sum(within_dx) > 1 else np.nan
        dn[index] = np.percentile(y[within_dx],16) if sum(within_dx) > 1 else np.nan
    
    nan_mask = ~np.isnan(ys)
    xs, ys = xs[nan_mask], ys[nan_mask]
    up, dn = up[nan_mask], dn[nan_mask]

    return xs, ys, dn, up

counter = 0
skip_count1 = 0
skip_count2 = 0

norm_9   = []
norm_95  = []
norm_10  = []
norm_105 = []
norm_11  = []
all_norm = []

slope_9   = []
slope_95  = []
slope_10  = []
slope_105 = []
slope_11  = []
all_slope = []

scatter_9   = []
scatter_95  = []
scatter_10  = []
scatter_105 = []
scatter_11  = []
all_scatter = []

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in tqdm(list(all_subhalos.keys())):
        this_subhalo = all_subhalos[key]
        
        this_sm = np.array(this_subhalo['TotalStellarMass'])
        
        this_smsd  = np.log10(np.array(this_subhalo['StellarMass']).flatten())
        this_sfrsd = np.array(this_subhalo['StarFormationRate']).flatten()
        
        this_sfrsd[this_sfrsd <= 0] = np.nan
        
        this_sfrsd = np.log10(this_sfrsd)
        
        if np.any(this_smsd > 7) and np.any(this_sfrsd > -4.0):
            mask = (this_smsd > 7) & (this_sfrsd > -4.0)
            
            if sum(mask) <= 20:
                skip_count1 += 1
                continue
            
            mass, sfr, sfr1, sfr2 = median(this_smsd[mask], this_sfrsd[mask])
            
            if len(mass) < 2:
                skip_count2 += 1
                continue
            slope, inter = np.polyfit(this_smsd[mask], this_sfrsd[mask], 1)
            sfr_fit   = slope * mass + inter
            residuals = sfr - sfr_fit
            
            val_at_8 = slope * 8 + inter
            
            # finite_mask = np.isfinite(this_smsd) & np.isfinite(this_sfrsd)
            # plt.hist2d(this_smsd[finite_mask], this_sfrsd[finite_mask], bins=100, cmap=plt.cm.Greys)
            # plt.plot(mass, sfr)
            # plt.plot(mass, sfr_fit, color='r')
            # plt.savefig(f'./figs/idv_galaxies/{key}.pdf', bbox_inches='tight')
            # plt.close()
            
            if this_sm > 9 and this_sm < 9.5:
                norm_9.append( val_at_8 )
                slope_9.append( slope )
                scatter_9.append( np.std(residuals) )
                
            elif this_sm > 9.5 and this_sm < 10.0:
                norm_95.append( val_at_8 )
                slope_95.append( slope )
                scatter_95.append( np.std(residuals) )
                
            elif this_sm > 10.0 and this_sm < 10.5:
                norm_10.append( val_at_8 )
                slope_10.append( slope )
                scatter_10.append( np.std(residuals) )
                
            elif this_sm > 10.5 and this_sm < 11:
                norm_105.append( val_at_8 )
                slope_105.append( slope )
                scatter_105.append( np.std(residuals) )
                
            elif this_sm > 11:
                norm_11.append( val_at_8 )
                slope_11.append( slope )
                scatter_11.append( np.std(residuals) )
            
            all_norm.append( val_at_8 )
            all_slope.append( slope )
            all_scatter.append( np.std(residuals) )

            counter += 1
            
norm_9   = np.array(norm_9  )
norm_95  = np.array(norm_95 )
norm_10  = np.array(norm_10 )
norm_105 = np.array(norm_105)
norm_11  = np.array(norm_11 )
all_norm = np.array(all_norm)

slope_9   = np.array(slope_9  )
slope_95  = np.array(slope_95 )
slope_10  = np.array(slope_10 )
slope_105 = np.array(slope_105)
slope_11  = np.array(slope_11 )
all_slope = np.array(all_slope)

scatter_9   = np.array(scatter_9  )
scatter_95  = np.array(scatter_95 )
scatter_10  = np.array(scatter_10 )
scatter_105 = np.array(scatter_105)
scatter_11  = np.array(scatter_11 )
all_scatter = np.array(all_scatter)

print('Number of Galaxies in this sample:',counter)
print('Number of Galaxies we skipped for too few pixels:', skip_count1)
print('Number of Galaxies we skipped for other:', skip_count2)

N = 5
cmap = cmr.get_sub_cmap('cmr.torch', 0.2, 0.8, N=N)
newcolors = np.linspace(0, 1, N)
colors = [ cmap(x) for x in newcolors[::-1] ]

titles = ['$10^{9.0} \,<\,{M}_{\star}/M_\odot\,<\,10^{9.5} $', \
          '$10^{9.5} \,<\,{M}_{\star}/M_\odot\,<\,10^{10.0}$', \
          '$10^{10.0}\,<\,{M}_{\star}/M_\odot\,<\,10^{10.5}$', \
          '$10^{10.5}\,<\,{M}_{\star}/M_\odot\,<\,10^{11.0}$', \
          '$10^{11.0}\,<\,{M}_{\star}/M_\odot$']

ls = [
    'dotted','dashed','dashdot','dotted',(0, (1, 1))
]

fig, axs = plt.subplots(1,3,figsize=(11,4.1),sharey=True)

def plot_hist(ax, hist_vals, color, nbins=10, label='', fill=True, ls='-'):    
    counts, bins = np.histogram(hist_vals, bins=nbins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    f = UnivariateSpline(bin_centers, counts, s=30)
    y = f(bin_centers) / float(len(hist_vals))
    
    ax.plot(bin_centers, y, lw=2.5, color=color,label=label, ls=ls)
    
    if fill:
        ax.fill_between(bin_centers, 0, counts / len(hist_vals), color=color, alpha=0.25)
    
plot_hist(axs[0],all_norm,'k',fill=False,label=r'${\rm All~Galaxies}$')
plot_hist(axs[0],norm_9  ,colors[0], ls=ls[0], label=titles[0])
plot_hist(axs[0],norm_95 ,colors[1], ls=ls[1], label=titles[1])
plot_hist(axs[0],norm_10 ,colors[2], ls=ls[2], label=titles[2])
plot_hist(axs[0],norm_105,colors[3], ls=ls[3], label=titles[3])
plot_hist(axs[0],norm_11 ,colors[4], ls=ls[4], label=titles[4])

plot_hist(axs[1],all_slope,'k',fill=False)
plot_hist(axs[1],slope_9  ,colors[0], ls=ls[0])
plot_hist(axs[1],slope_95 ,colors[1], ls=ls[1])
plot_hist(axs[1],slope_10 ,colors[2], ls=ls[2])
plot_hist(axs[1],slope_105,colors[3], ls=ls[3])
plot_hist(axs[1],slope_11 ,colors[4], ls=ls[4])

plot_hist(axs[2],all_scatter,'k',fill=False)
plot_hist(axs[2],scatter_9  ,colors[0], ls=ls[0])
plot_hist(axs[2],scatter_95 ,colors[1], ls=ls[1])
plot_hist(axs[2],scatter_10 ,colors[2], ls=ls[2])
plot_hist(axs[2],scatter_105,colors[3], ls=ls[3])
plot_hist(axs[2],scatter_11 ,colors[4], ls=ls[4])

ymin,ymax = axs[0].get_ylim()
axs[0].set_ylim([0.00, ymax*1.2])

leg = axs[0].legend( frameon=False, handletextpad=0.25, labelspacing=0.05,
                     loc='upper left', fontsize=11 )
colors = ['k', colors[0], colors[1], colors[2], colors[3], colors[4]]
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])

axs[0].set_ylabel(r'${\rm PDF}$')
axs[0].set_xlabel(r'$\Sigma_{\mathrm{SFR},\, \Sigma_{\star}=8.0}$/[$M_\odot~\mathrm{kpc}^{-2}~\mathrm{yr}^{-1}$]')
axs[1].set_xlabel(r'$\mathrm{Power\!-\!law~Index~}(\alpha)$')
axs[2].set_xlabel(r'Scatter/[dex]')
    
plt.tight_layout()
plt.savefig('./figs/Figure3.pdf', bbox_inches='tight')