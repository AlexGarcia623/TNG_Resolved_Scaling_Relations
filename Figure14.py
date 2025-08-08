import matplotlib as mpl
mpl.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import time
import h5py
from matplotlib.colors import LogNorm

import cmasher as cmr

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

fs_og = 30
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

####################################################################################
# Initial Condition
sig_gas0 = 1.0 * 10**9.0
sig_star0 = 0.0

t_end = 8.0
nsteps = 10000
#dt = t_end / nsteps
t0 = 0.0

####################################################################################
#def gas_inflow(inflow_rate, dt):
#return inflow_rate * dt

def gas_inflow(sfr, r, dt):
    return r * sfr * dt

def calc_sfr(sig_gas, efficiency=1.0):
    # a0 = 4.78 * 10**(-16.0)
    # alpha = 1.77
    # return a0 * sig_gas**alpha
    a0 = 10**(-13.22995892)
    alpha = 1.52051897
    return a0*sig_gas**alpha


def one_step(sig_gas, sig_star, dt, mu = 0.5, inflow_ratio=-0.3):
    sfr = calc_sfr(sig_gas)
    sig_star += sfr * dt * mu
    sig_gas -= sfr * dt * mu
    inflow = gas_inflow(sfr, inflow_ratio, dt)
    #inflow = gas_inflow(inflow_rate, dt)
    sig_gas += inflow

    #print sig_gas, sig_star
    return sig_gas, sig_star

###################################################################################
def evolve(sig_gas0, sig_star0, t_end, nsteps, inflow_ratio=-0.2):
    t = 0.0
    t_end *= 10**9.0
    dt = ( t_end - t0 ) / nsteps
    sig_gas = sig_gas0
    sig_star = sig_star0
    gas_list = []
    star_list = []

    cnt = 0
    while t < t_end:
        sig_gas, sig_star = one_step(sig_gas, sig_star, dt, inflow_ratio=inflow_ratio)
        t += dt
        gas_list.append(sig_gas)
        star_list.append(sig_star)
        cnt += 1
    gas_list = np.array(gas_list)
    star_list = np.array(star_list)
    
    fgas_list = gas_list / (gas_list + star_list)
    # fgas_list = gas_list / star_list
    
    return gas_list, star_list

data = '../IllustrisTNG_L35n2160TNG.hdf5'

all_sm  = []
all_sfr = []
all_gm  = []
all_total_mass = []
counter = 0

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in all_subhalos.keys():
        this_subhalo = all_subhalos[key]
        
        this_sm = np.array(this_subhalo['TotalStellarMass'])
        
        this_smsd  = np.array(this_subhalo['StellarMass']).flatten()
        this_sfrsd = np.array(this_subhalo['StarFormationRate']).flatten()
        this_gmsd  = np.array(this_subhalo['GasMass']).flatten()
        
        if np.any(this_smsd > 10**7) and np.any(this_sfrsd > 10**-4.0):
            all_sm.extend(this_smsd)
            all_sfr.extend(this_sfrsd)
            all_gm.extend(this_gmsd)
            all_total_mass.extend(np.ones_like(this_smsd)*this_sm)
            counter += 1
        
print('Number of Galaxies in this sample:',counter)
all_sm  = np.array(all_sm )
all_sfr = np.array(all_sfr)
all_gm  = np.array(all_gm)
all_total_mass = np.array(all_total_mass)

inf_mask  = (np.isfinite(all_sm)) & (np.isfinite(all_sfr))
alex_mask = (all_sm > 10**7.0) & (all_sfr > 10**-4.0)

all_sm  = all_sm [alex_mask & inf_mask]
all_sfr = all_sfr[alex_mask & inf_mask]
all_gm  = all_gm [alex_mask & inf_mask]
all_total_mass = all_total_mass[alex_mask & inf_mask]

all_sm  = np.log10(all_sm )
all_sfr = np.log10(all_sfr)
all_gm  = np.log10(all_gm )

###################################################################################
fig = plt.figure(figsize=(15.0, 15.0))
ax = fig.add_axes([0.13, 0.10, 0.85, 0.85])

plt.hist2d(all_sm, all_gm, bins=100, norm=LogNorm(), cmap='Greys',
           rasterized=True, alpha=0.5)

ics = [[1.0*10**7.5, 0.0], [1.0*10**8.0, 0.0], [1.0*10**8.5, 0.0], \
       [1.0*10**9.0, 0.0], [1.0*10**9.5, 0.0], [1.0*10**10.0, 0.0]]
#ics = [[1.0*10**9.5, 1.0*10**7.5], [1.0*10**9.0, 1.0*10**8.0], [1.0*10**9.5, 8.5], \
#        [1.0*10**9.0, 1.0*10**9.0], [1.0*10**9.0, 1.0*10**7.0], [1.0*10**9.0, 1.0*10**6.5]]

'''
inflow_ratio = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
for ratio in inflow_ratio:
    sig_gas0, sig_star0 = 1.0 * 10**9.0, 0.0
    fgas_list, star_list = evolve(sig_gas0, sig_star0, 8.0, 16000, inflow_ratio=ratio)
    label = 'ratio = ' + str(ratio)
    #plt.plot(np.log10(star_list), np.log10(fgas_list), lw=5, label=label)


'''

#etas = [[0.0, 0.0], [0.0, 0.3], [0.0, 0.7], [0.3, 0.0], [0.3, 0.3], [0.3, 0.7], [0.7, 0.0], [0.7, 0.3], [0.7, 0.7]]
etas = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.5, 1.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
N = 3
cmap = cmr.get_sub_cmap('cmr.guppy', 0, 1, N=N)
newcolors = np.linspace(0, 1, N)
colors = [ cmap(x) for x in newcolors ]
styles = ['-', ':', '--']

col_i = -1
xloc = 0.175 - 0.33
legs = []
locs = ['lower left', 'lower center', 'lower right']

for index, flows in enumerate(etas):
    # print(flows[1], flows[0])
    flows[0] *= 0.5
    flows[1] *= 0.5
    inflow_ratio = flows[1] - flows[0]
    multi_time = []
    for ic in ics:
        sig_gas0, sig_star0 = ic[0], ic[1]
        fgas_list, star_list = evolve(sig_gas0, sig_star0, 8.0, 16000, inflow_ratio=inflow_ratio)
        multi_time.append(fgas_list[1999::2000])
        multi_time.append(star_list[1999::2000])
        #plt.plot(np.log10(star_list), np.log10(fgas_list), '--', lw=5)

    cur_fgas, cur_star = [], []
    for j in range(len(multi_time) // 2):
        cur_fgas.append(multi_time[2*j][-4])
        cur_star.append(multi_time[2*j+1][-4])
    if index in [0, 3, 6]:
        legs = []
        col_i += 1
        xloc += 0.33
    #     label = '$\eta_{\mathrm{in}} = \,$' + f'{flows[1]:0.2f}' + '; $\eta_{\mathrm{out}}$ = \,' + f'{flows[0]:0.2f}'
    # else:
    label = '$\eta_{\mathrm{in}} = \,$' + f'{flows[1]:0.2f}'
    # label = '$\eta_{\mathrm{in}}$ = ' + str(flows[1]) + '; $\eta_{\mathrm{out}}$ = ' + str(flows[0]) #str(i+1) + ' Gyr'
    if flows[0] == 0.0:
        color = colors[0]
    if flows[0] == 0.25:
        color = colors[1]
    if flows[0] == 0.5:
        color = colors[2]
    if flows[1] == 0.0:
        style = styles[0]
    if flows[1] == 0.25:
        style = styles[1]
    if flows[1] == 0.5:
        style = styles[2]

    line, = plt.plot(np.log10(np.array(cur_star)), np.log10(np.array(cur_fgas)),
                     lw=5, color=color, ls = style, label=label)
    
    legs.append(line)
    if index in [2, 5, 8]:
        legend = ax.legend(handles=legs, frameon=False, handletextpad=0.25, loc=locs[col_i], fontsize=40)
        ax.add_artist(legend)  # This is important to add the first legend to the plot

        c = [color, color, color]
        for index, text in enumerate(legend.get_texts()):
            text.set_color(c[index])
        
        plt.text(xloc,0.25,'$\eta_{\mathrm{out}} = \,$' + f'{flows[0]:0.2f}',
                 transform=plt.gca().transAxes, ha='center', fontsize=50, color=color)
    
ax.set_xlabel('$\mathrm{log}(\Sigma_{\star} / \mathrm{M}_{\odot}\, \mathrm{kpc}^{-2})$', fontsize=40)
# ax.set_ylabel('$\mathrm{log}(f_{\mathrm{gas}})$', fontsize=40)
ax.set_ylabel('$\mathrm{log}(\mathrm{\Sigma}_{\mathrm{gas}} / M_{\odot}\, \mathrm{kpc}^{-2})$', fontsize=40)
ax.tick_params(labelsize=40)
# cbar = plt.colorbar(pad=0.03)
# cbar.set_label('Count', fontsize=50, rotation=90, labelpad=1.1)
# cbar.ax.tick_params(labelsize=40)
# ax.legend(fancybox=True, framealpha=0.5, fontsize=30)
ax.set_xlim([7.1, 9.49])
# ax.set_ylim([-3.25, -0.1])
    
plt.tight_layout()

plt.savefig('./figs/Figure14.pdf', bbox_inches='tight')
