#######################################################
# All things averaging
#######################################################

import numpy as np
from utils import z_to_xyz

# Vertically average the given field over all depths.

# Arguments:
# data: 3D (depth x lat x lon) or 4D (time x depth x lat x lon, needs time_dependent=True) array of data to average
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_lon_lat
# time_dependent: as in function apply_mask

# Output: array of dimension lat x lon (if time_dependent=False) or time x lat x lon (if time_dependent=True)

def vertical_average (data, grid, gtype='t', time_dependent=False):

    # Choose the correct integrand of depth
    if gtype == 'w':
        dz = grid.dz_t
    else:
        dz = grid.dz
    # Make it 3D
    dz = z_to_xyz(dz, grid)
    # Get the correct hFac
    hfac = grid.get_hfac(gtype=gtype)
    if time_dependent:
        # There's also a time dimension
        num_time = data.shape[0]
        dz = np.tile(dz, (num_time, 1, 1, 1))
        hfac = np.tile(hfac, (num_time, 1, 1, 1))
    # Vertically average    
    return np.sum(data*dz*hfac, axis=-3)/np.sum(dz*hfac, axis=-3)

    
        
