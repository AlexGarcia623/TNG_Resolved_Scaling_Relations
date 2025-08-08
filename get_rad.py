import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size']   = 20

data = '../IllustrisTNG_L35n2160TNG.hdf5'

with h5py.File(data, 'r') as f:
    all_subhalos = f['snap_99']
    
    for key in all_subhalos.keys():
        this_subhalo = all_subhalos[key]    
        this_smsd = np.array(this_subhalo['StellarMass'])
        
        center = (50,50)
        y_indices, x_indices = np.indices((100, 100)) ## based on size=50, pixl=1.0
        radii = np.sqrt((x_indices - center[1])**2 + (y_indices - center[0])**2)
        
        plt.clf()
        mp = plt.imshow(radii)
        
        cbar = plt.colorbar(mp,label=r'${\rm Radius}$')
        plt.savefig('./figs/test_rad.pdf',bbox_inches='tight')
        
        break