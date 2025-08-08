import os, sys
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI

import illustris_python as il # https://github.com/illustristng/illustris_python
import scripts as scr

comm     = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

size = 50
pixl = 1.0

SIM  = 'IllustrisTNG'
run  = 'L35n2160TNG'
snap = 99

smoothing_method = 'cubic-spline'
output_directory = './data/'

f_name = f'{SIM}_{run}_{mpi_rank}.hdf5'
hdf5_path = output_directory + f_name

with h5py.File(hdf5_path, "w") as f:
   ## write file first
   snap_group = f.create_group(f'snap_{snap}')

out_dir = '/standard/torrey-group/' + SIM + '/Runs/' + run + '/output/'

### Constants
xh = 7.600E-01
zo = 3.500E-01
mh = 1.6726219E-24
kb = 1.3806485E-16
mc = 1.270E-02
###

print(f'Loading {SIM} Header...')
hdr      = il.groupcat.loadHeader(out_dir, snap)
box_size = hdr['BoxSize']
scf      = hdr['Time']
z        = hdr['Redshift']
h = hdr['HubbleParam']

print(f'Loading {SIM} Subhalos...')
fields  = ['SubhaloMassType','SubhaloPos','SubhaloVel','SubhaloSFR','SubhaloLenType']
prt     = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)

prt['SubhaloMassType'] *= 1.00E+10/h
prt['SubhaloMassType'] = np.log10(prt['SubhaloMassType'])

print(f'Loading {SIM} Halos...')
subs = il.groupcat.loadHalos(out_dir,snap,fields=['GroupFirstSub'])
print('')
###

subs = np.arange(len(prt['SubhaloSFR']), dtype=int)

desired = subs[~(
    (prt['SubhaloMassType'][:,4] < 9.0) |
    (prt['SubhaloSFR'] < 1e-2) | 
    (
        (prt['SubhaloLenType'][:,4] < 64) | (prt['SubhaloLenType'][:,0] < 64) 
    )
)]

print(f'Number of galaxies: {len(desired)}')

def process_single(index, sub):
    print(f'Processing galaxy #{index}/{len(desired)} (subhalo ID: {sub}) on rank {mpi_rank}')

    this_stellar_mass = prt['SubhaloMassType'][sub, 4] 
    this_gas_mass     = prt['SubhaloMassType'][sub, 0] 
    this_SFR          = prt['SubhaloSFR'][sub] 
    this_pos          = prt['SubhaloPos'][sub]
    this_vel          = prt['SubhaloVel'][sub]

    ### Load in TNG particle catalogs for single galaxy
    star_pos  = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Coordinates'      ])
    star_vel  = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Velocities'       ])
    star_mass = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Masses'           ])
    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
    gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
    gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
    GFM_Metal = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])
    
    # Gas helpers
    gas_pos    = scr.center(gas_pos, this_pos, box_size)
    gas_pos   *= (scf / h)
    gas_vel   *= np.sqrt(scf)
    gas_vel   -= this_vel
    gas_mass  *= (1.000E+10 / h)
    gas_rho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
    gas_rho   *= (1.989E+33    ) / (3.086E+21**3.00E+00)
    gas_rho   *= XH / mh

    ri, ro = scr.calc_rsfr_io(gas_pos, gas_sfr)

    sf_idx = gas_rho > 1.300E-01

    incl   = scr.calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

    gas_pos  = scr.trans(gas_pos, incl)
    gas_vel  = scr.trans(gas_vel, incl)

    # Star Helpers
    star_pos    = scr.center(star_pos, this_pos, box_size)
    star_pos   *= (scf / h)
    star_vel   *= np.sqrt(scf)
    star_vel   -= this_vel
    star_mass  *= (1.000E+10 / h)

    star_pos  = scr.trans(star_pos, incl)
    star_vel  = scr.trans(star_vel, incl)
    
    OH = ZO/XH * 1.00/16.00

    Zgas = np.log10(OH) + 12

    Npix = int(np.ceil(2 * size / pixl))
    x = np.linspace(-size, size, Npix)
    y = np.linspace(-size, size, Npix)
    z = np.linspace(-size, size, Npix)

    pixlims = np.linspace(-size, size, Npix + 1)
    mask = (np.abs(gas_pos[:,0]) <= size) & (np.abs(gas_pos[:,1]) <= size) & (np.abs(gas_pos[:,2]) <= size)

    Npix = len(x)
    grid_shape = (Npix, Npix, Npix)
    grid_points = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T

    density = scr.get_density_from_neighbors(smoothing_method,
        len(gas_mass), len(grid_points),
        gas_pos[:,0], gas_pos[:,1], gas_pos[:,2], gas_mass,
        grid_points[:,0], grid_points[:,1], grid_points[:,2], DesNgb=32
    )
    metal_dens = scr.get_density_from_neighbors(smoothing_method,
        len(gas_mass), len(grid_points),
        gas_pos[:,0], gas_pos[:,1], gas_pos[:,2], gas_mass*gas_met,
        grid_points[:,0], grid_points[:,1], grid_points[:,2], DesNgb=32
    )
    sfr_density = scr.get_density_from_neighbors(smoothing_method,
        len(gas_sfr), len(grid_points),
        gas_pos[:,0], gas_pos[:,1], gas_pos[:,2], gas_sfr,
        grid_points[:,0], grid_points[:,1], grid_points[:,2], DesNgb=32
    )
    stellar_density = scr.get_density_from_neighbors(smoothing_method,
        len(star_mass), len(grid_points),
        star_pos[:,0], star_pos[:,1], star_pos[:,2], star_mass,
        grid_points[:,0], grid_points[:,1], grid_points[:,2], DesNgb=32
    )

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    rhomet_3d  = metal_dens.reshape(Npix, Npix, Npix).copy()
    rhogas_3d  = density.reshape(Npix, Npix, Npix).copy()
    rhostar_3d = stellar_density.reshape(Npix, Npix, Npix).copy()
    rhosfr_3d  = sfr_density.reshape(Npix, Npix, Npix).copy()

    gas_dens  = np.sum(rhogas_3d , axis=2) * dz
    star_dens = np.sum(rhostar_3d, axis=2) * dz
    sfr_dens  = np.sum(rhosfr_3d , axis=2) * dz
    met_dens  = np.sum(rhomet_3d , axis=2) * dz

    xymmet = met_dens / gas_dens

    with h5py.File(hdf5_path, "r+") as f:
        snap_group = f[f"snap_{snap}"]
        
        subhalo_group = snap_group.create_group(f'subhalo_{sub}')

        subhalo_group.create_dataset("TotalStarFormationRate", data=this_SFR)
        subhalo_group.create_dataset("TotalStellarMass", data=this_stellar_mass)
        subhalo_group.create_dataset("TotalGasMass", data=this_gas_mass)
        subhalo_group.create_dataset("GasMass", data=gas_dens)
        subhalo_group.create_dataset("StarFormationRate", data=sfr_dens)
        subhalo_group.create_dataset("StellarMass", data=star_dens)
        subhalo_group.create_dataset("Metallicity", data=xymmet)
        
galaxies_per_rank = np.array_split(desired, mpi_size)
indexes_per_rank  = np.array_split(np.arange(len(desired)), mpi_size)
        
for jjj, sub in enumerate(galaxies_per_rank[mpi_rank]):
    idx = indexes_per_rank[mpi_rank][jjj]
    try:
        process_single(idx, sub)
    except:
        print(f'Subhalo {sub} did not work')
