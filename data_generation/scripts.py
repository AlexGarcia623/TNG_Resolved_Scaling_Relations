import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

def gaussian_kernel(q, h, truncate_at=3):
    W             = np.zeros_like(q)
    mask          = q <= truncate_at
    W[mask]       = np.exp(-q[mask]**2)
    normalization = 1 / ((np.pi) ** 1.5 * h ** 3)
    W            *= normalization
    return W
    
def tophat_kernel(q, h):
    W         = np.zeros_like(q)
    W[q <= 1] = 1
    norm      = 3 / (4 * np.pi * h**3)
    W         *= norm
    return W
    
def cubic_spline_kernel(u, h):
    u        = np.asarray(u)
    norm     = 8 / (np.pi * h**3)
    W        = np.zeros_like(u)
    mask1    = u < 0.5
    mask2    = (u >= 0.5) & (u < 1.0)
    W[mask1] = norm * (1 - 6 * u[mask1]**2 + 6 * u[mask1]**3)
    W[mask2] = norm * (2 * (1 - u[mask2])**3)
    return W

def get_neighbors(tree, grid_positions, DesNgb):
    dist_matrix, idx_matrix = tree.query(grid_positions, k=DesNgb)
    return dist_matrix, idx_matrix

def get_density_from_neighbors(
        kernel, N_mass, N_grid,
        particle_x, particle_y, particle_z, mass_particle,
        box_x, box_y, box_z,
        DesNgb, Hmax=33.0
    ):
    particle_pos = np.vstack((particle_x, particle_y, particle_z)).T
    particle_masses = np.array(mass_particle)
    tree = cKDTree(particle_pos)

    dens = np.zeros(N_grid)
    
    grid_positions = np.vstack((box_x, box_y, box_z)).T

    kernel_map = {
        'gaussian': gaussian_kernel,
        'cubic-spline': cubic_spline_kernel,
        'tophat': tophat_kernel
    }

    if kernel not in kernel_map:
        raise ValueError("Choose either 'gaussian', 'cubic-spline', or 'tophat'")

    kernel_fn = kernel_map[kernel]
    
    dists, idxs = get_neighbors(tree, grid_positions, DesNgb)

    h = 1.04 * dists[:, -1]
    
    for i in (range(N_grid)):
        grid_pos = grid_positions[i]
        h_guess  = h[i]
        h2       = h_guess ** 2
        hinv     = 1.0 / h_guess

        neighbor_idxs = idxs[i]
        particle_pos_neighbors  = particle_pos[neighbor_idxs]
        particle_mass_neighbors = particle_masses[neighbor_idxs]

        displacements = grid_pos - particle_pos_neighbors
        r2 = np.sum(displacements**2, axis=1)

        mask = r2 < h2
        if not np.any(mask):
            dens[i] = 0.0
            continue

        r2 = r2[mask]
        particle_mass_neighbors = particle_mass_neighbors[mask]

        r = np.sqrt(r2)
        u = r * hinv
        W = kernel_fn(u, h_guess)

        dens[i] = np.sum(particle_mass_neighbors * W)

    return dens

def trans(arr0, incl0):
    arr      = np.copy( arr0)
    incl     = np.copy(incl0)
    deg2rad  = np.pi / 1.800E+02
    incl    *= deg2rad
    arr[:,0] = -arr0[:,2] * np.sin(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.cos(incl[0])
    arr[:,1] = -arr0[:,0] * np.sin(incl[1]) + (arr0[:,1] * np.cos(incl[1])                                                )
    arr[:,2] =  arr0[:,2] * np.cos(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.sin(incl[0])
    del incl
    return arr

def calc_incl(pos0, vel0, m0, ri, ro):
    rpos = np.sqrt(pos0[:,0]**2.000E+00 +
                   pos0[:,1]**2.000E+00 +
                   pos0[:,2]**2.000E+00 )
    rpos = rpos[~np.isnan(rpos)]
    idx  = (rpos > ri) & (rpos < ro)
    pos  = pos0[idx]
    vel  = vel0[idx]
    m    =   m0[idx]
        
    hl = np.cross(pos, vel)
    L  = np.array([np.multiply(m, hl[:,0]),
                   np.multiply(m, hl[:,1]),
                   np.multiply(m, hl[:,2])])
    L  = np.transpose(L)
    L  = np.array([np.sum(L[:,0]),
                   np.sum(L[:,1]),
                   np.sum(L[:,2])])
    Lmag  = np.sqrt(L[0]**2.000E+00 +
                    L[1]**2.000E+00 +
                    L[2]**2.000E+00 )
    Lhat  = L / Lmag
    incl  = np.array([np.arccos(Lhat[2]), np.arctan2(Lhat[1], Lhat[0])])
    incl *= 1.800E+02 / np.pi
    if   incl[1]  < 0.000E+00:
         incl[1] += 3.600E+02
    elif incl[1]  > 3.600E+02:
         incl[1] -= 3.600E+02
    return incl

def center(pos0, centpos, boxsize = None):
    pos       = np.copy(pos0)
    pos[:,0] -= centpos[0]
    pos[:,1] -= centpos[1]
    pos[:,2] -= centpos[2]
    if (boxsize != None):
        pos[:,0][pos[:,0] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,0][pos[:,0] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,1][pos[:,1] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,1][pos[:,1] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,2][pos[:,2] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,2][pos[:,2] > ( boxsize / 2.000E+00)] -= boxsize
    return pos

def calc_rsfr_io(pos0, sfr0):
    fraci = 5.000E-02
    fraco = 9.000E-01
    r0    = 1.000E+01
    rpos  = np.sqrt(pos0[:,0]**2.000E+00 +
                    pos0[:,1]**2.000E+00 +
                    pos0[:,2]**2.000E+00 )
    sfr0  = sfr0[~np.isnan(rpos)]
    rpos  = rpos[~np.isnan(rpos)]
    sfr   = sfr0[np.argsort(rpos)]
    rpos  = rpos[np.argsort(rpos)]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr)/sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxi   = idx0[(sfrf > fraci)]
    idxi   = idxi[0]
    rsfri  = rpos[idxi]
    dskidx = rpos < (rsfri + r0)
    sfr    =  sfr[dskidx]
    rpos   = rpos[dskidx]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxo   = idx0[(sfrf > fraco)]
    idxo   = idxo[0]
    rsfro  = rpos[idxo]
    return rsfri, rsfro

if __name__ == '__main__':
    print('Hello World!')