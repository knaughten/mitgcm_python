#######################################################
# Offline calculation of diagnostic variables
#######################################################

import numpy as np
import sys

from constants import rho_ice
from utils import z_to_xyz, add_time_dim, xy_to_xyz
from averaging import area_integral


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
    # First assume 3D arrays
    z = z_to_xyz(grid.z, grid)
    if time_dependent:
        # 4D arrays
        z = add_time_dim(z, temp.shape[0])

    return temp - tfreeze(temp, salt, z)


# Calculate the total mass loss or area-averaged melt rate.

# Arguments:
# ismr: 2D (lat x lon) array of ice shelf melt rate in m/y
# mask: boolean array which is True in the points to be included in the calculation (such as grid.fris_mask or grid.ice_mask)
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
        return np.sum(ismr*grid.dA*mask)*rho_ice*1e-12


# Find the time indices of minimum and maximum sea ice area.

# Arguments:
# aice: 3D (time x lat x lon) array of sea ice area at each time index
# grid: Grid object

# Output: two integers containing the time indices (0-indexed) of minimum and maximum sea ice area, respectively

def find_aice_min_max (aice, grid):

    total_aice = area_integral(aice, grid, time_dependent=True)
    return np.argmin(total_aice), np.argmax(total_aice)


# Calculate the barotropic transport streamfunction.
def barotropic_streamfunction (u, grid):

    # Get integrands and partial cell fractions
    # hfac and dz should be 3D
    hfac = grid.hfac_w
    dz = z_to_xyz(grid.dz, grid)
    # dy should be 2D
    dy = grid.dy_w
    if len(u.shape)==4:
        # Time-dependent. Add a time dimension to everything.
        num_time = u.shape[0]
        hfac = add_time_dim(hfac, num_time)
        dz = add_time_dim(dz, num_time)
        dy = add_time_dim(dy, num_time)
    # Vertically integrate
    udz_int = np.sum(u*hfac*dz, axis=-3)
    # Indefinite integral from south to north, and convert to Sv
    return np.cumsum(udz_int*dy, axis=-2)*1e-6
    
    





