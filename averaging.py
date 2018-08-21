#######################################################
# All things averaging and integrating
#######################################################

import numpy as np
from utils import z_to_xyz, xy_to_xyz, add_time_dim


# Helper functions to set up integrands and masks, tiled to be the same dimension as the "data" array

# Returns area integrand, and whichever mask is already applied to the MaskedArray "data"
def prepare_area_mask (data, grid, gtype='t', time_dependent=False):
    
    if gtype != 't':
        print 'Error (prepare_area_mask): non-tracer grids not yet supported'
        sys.exit()

    # Get the mask as 1s and 0s
    if isinstance(data, np.ma.MaskedArray):
        mask = np.invert(data.mask).astype(float)
    else:
        # No mask, just use 1s everywhere
        mask = np.ones(data.shape)
    # Get the integrand of area
    dA = grid.dA
    if (time_dependent and len(data.shape)==4) or (not time_dependent and len(data.shape)==3):
        # There's also a depth dimension; tile in z
        dA = xy_to_xyz(dA, grid)
    if time_dependent:
        # Tile in time
        dA = add_time_dim(dA, data.shape[0])
    return dA, mask


# Returns depth integrand and hfac
def prepare_dz_mask (data, grid, gtype='t', time_dependent=False):

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
        dz = add_time_dim(dz, data.shape[0])
        hfac = add_time_dim(hfac, data.shape[0])
    return dz, hfac


# Helper functions to average/integrate over area or volume (option='average' or 'integrate')

def over_area (option, data, grid, gtype='t', time_dependent=False):

    dA, mask = prepare_area_mask(data, grid, gtype=gtype, time_dependent=time_dependent)
    if option == 'average':
        return np.sum(data*dA*mask, axis=(-2,-1))/np.sum(dA*mask, axis=(-2,-1))
    elif option == 'integrate':
        return np.sum(data*dA*mask, axis=(-2,-1))
    else:
        print 'Error (over_area): invalid option ' + option


def over_volume (option, data, grid, gtype='t', time_dependent=False):

    # Get dz and hfac
    dz, hfac = prepare_dz_mask(data, grid, gtype=gtype, time_dependent=time_dependent)
    # Get dA and mask
    dA, mask = prepare_area_mask(data, grid, gtype=gtype, time_dependent=time_dependent)
    # Now get the volume integrand
    dV = dA*dz
    if option == 'average':
        return np.sum(data*dV*hfac*mask, axis=(-3,-2,-1))/np.sum(dV*hfac*mask, axis=(-3,-2,-1))
    elif option == 'integrate':
        return np.sum(data*dV*hfac*mask, axis=(-3,-2,-1))
    else:
        print 'Error (over_volume): invalid option ' + option


# Now here are the APIs.


# Vertically average the given field over all depths.

# Arguments:
# data: 3D (depth x lat x lon) or 4D (time x depth x lat x lon, needs time_dependent=True) array of data to average
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_lon_lat
# time_dependent: as in function apply_mask

# Output: array of dimension lat x lon (if time_dependent=False) or time x lat x lon (if time_dependent=True)

def vertical_average (data, grid, gtype='t', time_dependent=False):

    dz, hfac = prepare_dz_mask(data, grid, gtype=gtype, time_dependent=time_dependent)
    return np.sum(data*dz*hfac, axis=-3)/np.sum(dz*hfac, axis=-3)


# Vertically average a specific water column with fixed latitude and longitude. So "data" is a depth-dependent array, either 1D or 2D (if time_dependent=True). You also need to supply hfac at the same water column (1D, depth-dependent).
def vertical_average_column (data, hfac, grid, time_dependent=False):

    if gtype == 'w':
        dz = grid.dz_t
    else:
        dz = grid.dz
    if time_dependent:
        # Add time dimension to dz and hfac
        dz = add_time_dim(dz, data.shape[0])
        hfac = add_time_dim(hfac, data.shape[0])
    return np.sum(data*dz*hfac, axis=0)/np.sum(dz*hfac, axis=0)


# Area-average the given field over the unmasked region, using whatever mask is already applied as a MaskedArray.

# Arguments:
# data: 2D (lat x lon) or 3D (time x lat x lon, needs time_dependent=True) array of data to average. It could also be depth-dependent but then the output array will have a depth dimension which is a bit weird.
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_lon_lat; only 't' is supported right now
# time_dependent: as in function apply_mask

# Output: array of dimension time (if time_dependent=True) or a single value (if time_dependent=False)

def area_average (data, grid, gtype='t', time_dependent=False):

    return over_area('average', data, grid, gtype=gtype, time_dependent=time_dependent)


# Like area_average, but for area integrals.
def area_integral (data, grid, gtype='t', time_dependent=False):

    return over_area('integrate', data, grid, gtype=gtype, time_dependent=time_dependent)


# Volume-average the given field, taking hfac into account, plus any mask which is on data as as MaskedArray.

# Arguments:
# data: 3D (depth x lat x lon) or 4D (time x depth x lat x lon, needs time_dependent=True) array of data to average
# grid: Grid object:

# Optional keyword arguments and output as in function area_average.

def volume_average (data, grid, gtype='t', time_dependent=False):

    return over_volume('average', data, grid, gtype=gtype, time_dependent=time_dependent)


# Like volume_average, but for volume integrals.
def volume_integral (data, grid, gtype='t', time_dependent=False):

    return over_volume('integrate', data, grid, gtype=gtype, time_dependent=time_dependent)        
