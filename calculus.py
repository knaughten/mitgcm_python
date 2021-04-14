###############################################################
# All things averaging, integrating, and differentiating
###############################################################

import numpy as np
import datetime
from utils import z_to_xyz, xy_to_xyz, add_time_dim, is_depth_dependent


# Helper functions to set up integrands and masks, tiled to be the same dimension as the "data" array

# Returns area, volume, or distance integrand (option='dA', 'dV', 'dx', or 'dy'), and whichever mask is already applied to the MaskedArray "data".
def prepare_integrand_mask (option, data, grid, gtype='t', time_dependent=False):
    
    if option in ['dA','dV'] and gtype != 't':
        print 'Error (prepare_integrand_mask): non-tracer grids not yet supported'
        sys.exit()
    elif option == 'dx' and gtype == 'u':
        print 'Error (prepare_integrand_mask): u-grid not yet supported for dx'
        sys.exit()
    elif option == 'dy' and gtype == 'v':
        print 'Error (prepare_integrand_mask): v-grid not yet supported for dy'
        sys.exit()

    # Get the mask as 1s and 0s
    if isinstance(data, np.ma.MaskedArray):
        mask = np.invert(data.mask).astype(float)
    else:
        # No mask, just use 1s everywhere
        mask = np.ones(data.shape)
    # Get the integrand
    if option == 'dA':
        integrand = grid.dA
    elif option == 'dV':
        integrand = grid.dV
    elif option == 'dx':
        integrand = grid.dx_s
    elif option == 'dy':
        integrand = grid.dy_w
    else:
        print 'Error (prepare_integrand_mask): invalid option ' + option
    if (len(integrand.shape)==2) and is_depth_dependent(data, time_dependent=time_dependent):
        # There's also a depth dimension; tile in z
        integrand = xy_to_xyz(integrand, grid)
    if time_dependent:
        # Tile in time
        integrand = add_time_dim(integrand, data.shape[0])
    return integrand, mask


# Returns depth integrand and hfac
def prepare_dz_hfac (data, grid, gtype='t', time_dependent=False):

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


# Helper functions to average/integrate over depth, area, or volume (option='average' or 'integrate')

def over_depth (option, data, grid, gtype='t', time_dependent=False):

    dz, hfac = prepare_dz_hfac(data, grid, gtype=gtype, time_dependent=time_dependent)
    if isinstance(data, np.ma.MaskedArray):
        mask = np.invert(data.mask).astype(float)
    else:
        mask = np.ones(data.shape)
    if option == 'average':
        return np.sum(data*dz*hfac*mask, axis=-3)/np.sum(dz*hfac*mask, axis=-3)
    elif option == 'integrate':
        return np.sum(data*dz*hfac*mask, axis=-3)
    else:
        print 'Error (over_depth): invalid option ' + option
        sys.exit()


def over_area (option, data, grid, gtype='t', time_dependent=False):

    dA, mask = prepare_integrand_mask('dA', data, grid, gtype=gtype, time_dependent=time_dependent)
    if option == 'average':
        return np.sum(data*dA*mask, axis=(-2,-1))/np.sum(dA*mask, axis=(-2,-1))
    elif option == 'integrate':
        return np.sum(data*dA*mask, axis=(-2,-1))
    else:
        print 'Error (over_area): invalid option ' + option
        sys.exit()


def over_volume (option, data, grid, gtype='t', time_dependent=False):

    dV, mask = prepare_integrand_mask('dV', data, grid, gtype=gtype, time_dependent=time_dependent)
    if option == 'average':
        return np.sum(data*dV*mask, axis=(-3,-2,-1))/np.sum(dV*mask, axis=(-3,-2,-1))
    elif option == 'integrate':
        return np.sum(data*dV*mask, axis=(-3,-2,-1))
    else:
        print 'Error (over_volume): invalid option ' + option
        sys.exit()


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
    return over_depth('average', data, grid, gtype=gtype, time_dependent=time_dependent)


# Vertically integrate.
def vertical_integral (data, grid, gtype='t', time_dependent=False):
    return over_depth('integrate', data, grid, gtype=gtype, time_dependent=time_dependent)


# Vertically average a specific water column with fixed latitude and longitude. So "data" is a depth-dependent array, either 1D or 2D (if time_dependent=True). You also need to supply hfac at the same water column (1D, depth-dependent).
def vertical_average_column (data, hfac, grid, gtype='t', time_dependent=False):

    if gtype == 'w':
        dz = grid.dz_t
    else:
        dz = grid.dz
    if time_dependent:
        # Add time dimension to dz and hfac
        dz = add_time_dim(dz, data.shape[0])
        hfac = add_time_dim(hfac, data.shape[0])
    return np.sum(data*dz*hfac, axis=-1)/np.sum(dz*hfac, axis=-1)


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


# Indefinite integral from south to north.
def indefinite_ns_integral (data, grid, gtype='t', time_dependent=False):

    dy, mask = prepare_integrand_mask('dy', data, grid, gtype=gtype, time_dependent=time_dependent)
    return np.cumsum(data*dy*mask, axis=-2)


# First-order derivatives (just forward difference with the last row/column copied over)


# Helper function: assumes coordinates have same dimension as data
def derivative (data, coordinates, axis=0):

    # Forward difference
    result = np.diff(data, axis=axis)/np.diff(coordinates, axis=axis)
    # Just copy the last row/column/whatever
    pad_width = [(0,0)]*len(data.shape)
    pad_width[axis] = (0,1)
    return np.pad(result, pad_width, 'edge')


# Helper function to prepare spatial coordinates to match the shape of the data
def prepare_coord (shape, grid, option, gtype='t', time_dependent=False):

    if option == 'lon':
        # Get 2D lon
        coordinates = grid.get_lon_lat(gtype=gtype)[0]
    elif option == 'lat':
        # Get 2D lat
        coordinates = grid.get_lon_lat(gtype=gtype)[1]
    elif option == 'depth':
        # Get 3D z
        if gtype == 'w':
            print 'Error (prepare_coord): w-grid not yet supported for depth derivatives'
            sys.exit()
        coordinates = z_to_xyz(grid.z, z)
    if option in ['lon', 'lat'] and ((len(shape)==3 and not time_dependent) or (len(shape)==4 and time_dependent)):
        # Add depth dimension
        coordinates = xy_to_xyz(coordinates, grid)
    if time_dependent:
        # Add time dimension
        coordinates = add_time_dim(coordinates, shape[0])
    return coordinates


# APIs for each dimension now. Assumes data has lat and lon dimensions, plus possibly depth and time dimensions.

def lon_derivative (data, grid, gtype='t', time_dependent=False):

    lon = prepare_coord(data.shape, grid, 'lon', gtype=gtype, time_dependent=time_dependent)
    return derivative(data, lon, axis=-1)


def lat_derivative (data, grid, gtype='t', time_dependent=False):

    lat = prepare_coord(data.shape, grid, 'lat', gtype=gtype, time_dependent=time_dependent)
    return derivative(data, lat, axis=-2)


def depth_derivative (data, grid, gtype='t', time_dependent=False):

    z = prepare_coord(data.shape, grid, 'depth', gtype=gtype, time_dependent=time_dependent)
    return derivative(data, z, axis=-3)


def time_derivative (data, time):

    if isinstance(time[0], datetime.datetime):
        # Get time intervals in seconds
        dt = np.array([(time[n]-time[n-1]).total_seconds() for n in range(1,time.size)])
    ddata = data[1:,...] - data[:-1,...]
    # Expand the dimensions of dt to match ddata
    for n in range(len(ddata.shape)-1):
        dt = np.expand_dims(dt,-1)
    ddata_dt = ddata/dt
    # Now pad with zeros at the first time index
    return np.concatenate((np.expand_dims(np.zeros(ddata_dt[0,...].shape),0), ddata_dt), axis=0)


def time_integral (data, time):

    if isinstance(time[0], datetime.datetime):
        # Get time in seconds
        time_sec = np.array([(t-time[0]).total_seconds() for t in time])
        # Get midpoints
        time_sec_mid = 0.5*(time_sec[:-1] + time_sec[1:])
        # Extrapolate to edges
        time_edges = np.concatenate(([2*time_sec_mid[0]-time_sec_mid[1]], time_sec_mid, [2*time_sec_mid[-1]-time_sec_mid[-2]]))
        # Get difference
        dt = time_edges[1:] - time_edges[:-1]
    # Expand the dimensions of dt to match data
    for n  in range(len(data.shape)-1):
        dt = np.expand_dims(dt,-1)
    # Integrate
    return np.cumsum(data*dt, axis=0)
        
    
        

    
    
