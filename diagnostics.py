#######################################################
# Offline calculation of diagnostic variables
#######################################################

import numpy as np
import sys

import constants as const


# Calculate the in-situ freezing point (helper function for t_minus_tf)

# Arguments:
# temp, salt, z: arrays of any dimension (but all the same dimension) containing temperature (C), salinity (psu), and depth (m, sign doesn't matter).

# Output: array of the same dimension as temp, salt, and z, containing the in-situ freezing point.

def tfreeze (temp, salt, z):

    a0 = -0.0575
    b = -7.61e-4
    c0 = 0.0901

    return a0*salt + b*abs(z) + c0


# Calculate the difference from the in-situ freezing point.

# Arguments:
# temp, salt: arrays of temperature and salinity. They can be 3D (depth x lat x lon) or 4D (time x depth x lat x lon), in which case you need time_dependent=True.
# grid = Grid object

# Optional keyword arguments:
# time_dependent: boolean indicating that temp and salt are 4D, with a time dimension. Default False.

# Output: array of the same dimensions as temp and salt, containing the difference from the in-situ freezing point.

def t_minus_tf (temp, salt, grid, time_dependent=False):

    # Tile the z coordinates to be the same size as temp and salt
    if time_dependent:
        # 4D arrays
        num_time = temp.shape[0]
        z = np.tile(np.expand_dims(np.expand_dims(grid.z,1),2), (num_time, 1, grid.ny, grid.nx))
    else:
        # 3D arrays
        z = np.tile(np.expand_dims(np.expand_dims(grid.z,1),2), (1, grid.ny, grid.nx))

    return temp - tfreeze(temp, salt, z)


# Calculate the total mass loss or area-averaged melt rate.

# Arguments:
# ismr: 2D (lat x lon) array of ice shelf melt rate in m/y
# mask: boolean array which is True in the points to be included in the calculation (such as grid.fris_mask or grid.zice_mask)
# grid: Grid object

# Optional keyword argument:
# result: 'massloss' (default) calculates the total mass loss in Gt/y. 'meltrate' calculates the area-averaged melt rate in m/y.

# Output: float containing mass loss or average melt rate

def total_melt (ismr, mask, grid, result='massloss'):

    if result == 'meltrate':
        # Area-averaged melt rate
        return np.sum(ismr*grid.dA*mask)/np.sum(grid.dA*mask)
    elif result == 'massloss':
        # Total mass loss
        return np.sum(ismr*grid.dA*mask)*const.rho_ice*1e-12



