#######################################################
# Offline calculation of diagnostic variables
#######################################################

import numpy as np
import sys

from constants import rho_ice, wed_gyre_bounds
from utils import z_to_xyz, add_time_dim, xy_to_xyz, var_min_max
from averaging import area_integral, vertical_integral, indefinite_ns_integral


# Calculate the adiabatic temperature gradient exactly like MITgcm does. This originates from section 7 of "Algorithms for computation of fundamental properties of seawater", UNESCO technical papers in marine science 44, 1983.
# Helper function for in_situ_temp.

# Arguments:
# temp: temperature (degC)
# salt: salinity (psu)
# press: pressure (dbar ~= absolute value of depth in m)
# Note that temp, salt, and press can be of any dimension, as long as they all match.

# Output: adiabatic temperature gradient in degC/dbar, same dimension as input arguments.

def ad_temp_grad (temp, salt, press):

    s_ref = 35.
    a0 = 3.5803e-5
    a1 = 8.5258e-6
    a2 = -6.836e-8
    a3 = 6.6228e-10
    b0 = 1.8932e-6
    b1 = -4.2393e-8
    c0 = 1.8741e-8
    c1 = -6.7795e-10
    c2 = 8.733e-12
    c3 = -5.4481e-14
    d0 = -1.1351e-10
    d1 = 2.7759e-12
    e0 = -4.6206e-13
    e1 = 1.8676e-14
    e2 = -2.1687e-16

    return a0 + a1*temp + a2*temp**2 + a3*temp**3 + (b0 + b1*temp)*(salt - s_ref) + (c0 + c1*temp + c2*temp**2 + c3*temp**3 + (d0 + d1*temp)*(salt - s_ref))*press + (e0 + e1*temp + e2*temp**2)*press**2


# Calculate in-situ temperature from potential temperature, exactly like MITgcm does.

# Arguments:
# temp: potential temperature (degC)
# salt: salinity (psu)
# z: depth (m, sign doesn't matter)

# Output: in-situ temperature (degC), same dimension as input arguments

def in_situ_temp (temp, salt, z):

    press_ref = 0.
    sqrt_2 = np.sqrt(2)

    # Step 1
    dpress = np.abs(z) - press_ref
    dtemp = dpress*ad_temp_grad(temp, salt, press_ref)
    temp_new = temp + 0.5*dtemp
    q = dtemp

    # Step 2
    dtemp = dpress*ad_temp_grad(temp_new, salt, press_ref + 0.5*dpress)
    temp_new += (1 - 1/sqrt_2)*(dtemp - q)
    q = (2 - sqrt_2)*dtemp + (-2 + 3/sqrt_2)*q

    # Step 3
    dtemp = dpress*ad_temp_grad(temp_new, salt, press_ref + 0.5*dpress)
    temp_new += (1 + 1/sqrt_2)*(dtemp - q)
    q = (2 + sqrt_2)*dtemp + (-2 - 3/sqrt_2)*q

    # Step 4
    dtemp = dpress*ad_temp_grad(temp_new, salt, press_ref + dpress)
    temp_new += (dtemp - 2*q)/6

    return temp_new


# Calculate the in-situ freezing point (helper function for t_minus_tf)

# Arguments:
# salt, z: arrays of any dimension (but both the same dimension, or else one is a scalar) containing salinity (psu) and depth (m, sign doesn't matter).

# Output: array of the same dimension as salt and z, containing the in-situ freezing point in degC.

def tfreeze (salt, z):

    a0 = -0.0575
    b = -7.61e-4
    c0 = 0.0901

    return a0*salt + b*abs(z) + c0


# Calculate the temperature difference from the in-situ freezing point.

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

    return in_situ_temp(temp, salt, z) - tfreeze(salt, z)


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


# Calculate the barotropic transport streamfunction. u is assumed not to be time-dependent.
def barotropic_streamfunction (u, grid):

    if len(u.shape) == 4:
        print 'Error (barotropic_streamfunction): u cannot be time-dependent.'
        sys.exit()
        
    # Vertically integrate
    udz_int = vertical_integral(u, grid, gtype='u')
    # Indefinite integral from south to north
    strf = indefinite_ns_integral(udz_int, grid, gtype='u')
    # Convert to Sv
    return strf*1e-6


# Calculate the Weddell Gyre transport: absolute value of the most negative streamfunction within the Weddell Gyre bounds.
def wed_gyre_trans (u, grid):

    strf = barotropic_streamfunction(u, grid)
    vmin, vmax = var_min_max(strf, grid, xmin=wed_gyre_bounds[0], xmax=wed_gyre_bounds[1], ymin=wed_gyre_bounds[2], ymax=wed_gyre_bounds[3], gtype='u')
    return -1*vmin


# Calculate seawater density for a linear equation of state.
def dens_linear (salt, temp, rhoConst, Tref, Sref, tAlpha, sBeta):

    return rhoConst*(1 - tAlpha*(temp-Tref) + sBeta*(salt-Sref))

    
    





