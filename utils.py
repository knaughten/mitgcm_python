#######################################################
# Miscellaneous useful tools
#######################################################

import numpy as np
import sys

from constants import rho_fw, sec_per_year, fris_bounds, deg2rad
from diagnostics import total_aice


# Given an array containing longitude, make sure it's either in the range (-180, 180) (if max_lon=180) or (0, 360) (if max_lon=360).
def fix_lon_range (lon, max_lon=180):

    if max_lon == 180:
        # Range (-180, 180)
        index = lon > 180
        lon[index] = lon[index] - 360
        index = lon < -180
        lon[index] = lon[index] + 360
    elif max_lon == 360:
        # Range (0, 360)
        index = lon < 0
        lon[index] = lon[index] + 360
        index = lon > 360
        lon[index] = lon[index] - 360
    else:
        print 'Error (fix_lon_range): max_lon must be either 180 or 360'
        sys.exit()

    return lon


# Given an array containing longitude, make sure it's in the range (0, 360) as opposed to (-180, 180).
def revert_lon_range (lon):

    index = lon < 0
    lon[index] = lon[index] + 360
    
    return lon


# Convert freshwater flux into the ice shelf (diagnostic SHIfwFlx) (kg/m^2/s, positive means freezing) to ice shelf melt rate (m/y, positive means melting).
def convert_ismr (shifwflx):

    return -shifwflx/rho_fw*sec_per_year


# Select the top layer from the given array of data. This is useful to see conditions immediately beneath ice shelves.
# The only assumptions about the input array are that 1) it is masked with hfac (see mask_3d below), 2) the third last dimension is the vertical dimension. So it can be depth x lat x lon, or time x depth x lat x lon, or even something like experiment x time x depth x lat x lon.
def select_top (data):

    # Figure out the dimensions of the data when the vertical dimension is removed
    collapsed_shape = data.shape[:-3] + data.shape[-2:]
    # Array which will hold values at the top level, initialised to NaN
    data_top = np.zeros(collapsed_shape)
    data_top[:] = np.nan
    # Loop from surface to bottom
    for k in range(data.shape[-3]):
        curr_data = data[...,k,:,:]
        # Find points which are unmasked at this vertical level, and which
        # haven't already been assigned a top level
        index = np.nonzero(np.invert(curr_data.mask)*np.isnan(data_top))
        data_top[index] = curr_data[index]
    # Anything still NaN is land; mask it out
    data_top = np.ma.masked_where(np.isnan(data_top), data_top)

    return data_top


# Select the bottom layer from the given array of data. See select_top for more documentation.
def select_bottom (data):

    # Same as select_top, but loop from bottom to top.
    collapsed_shape = data.shape[:-3] + data.shape[-2:]
    data_bottom = np.zeros(collapsed_shape)
    data_bottom[:] = np.nan
    for k in range(data.shape[-3]-1, -1, -1):
        curr_data = data[...,k,:,:]
        index = np.nonzero(np.invert(curr_data.mask)*np.isnan(data_bottom))
        data_bottom[index] = curr_data[index]
    data_bottom = np.ma.masked_where(np.isnan(data_bottom), data_bottom)

    return data_bottom


# Helper function for masking functions below
def apply_mask (data, mask, time_dependent=False):

    if time_dependent:
        # Tile the mask in the time dimension
        num_time = data.shape[0]
        if len(mask.shape) == 2:
            # Starting with a 2D mask
            mask = np.tile(mask, (num_time, 1, 1))
        elif len(mask.shape) == 3:
            # Starting with a 3D mask
            mask = np.tile(mask, (num_time, 1, 1, 1))
        else:
            print 'Error (apply_mask): invalid dimensions of mask'
            sys.exit()

    if len(mask.shape) != len(data.shape):
        print 'Error (apply_mask): invalid dimensions of data'
        sys.exit()

    data = np.ma.masked_where(mask, data)
    return data


# Mask land out of a 2D field.

# Arguments:
# data: array of data to mask, assumed to be 2D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_land (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, grid.get_land_mask(gtype=gtype), time_dependent=time_dependent)


# Mask land and ice shelves out of a 2D field, just leaving the open ocean.

# Arguments:
# data: array of data to mask, assumed to be 2D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_land_zice (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, grid.get_land_mask(gtype=gtype)+grid.get_zice_mask(gtype=gtype), time_dependent=time_dependent)


# Mask land and open ocean out of a 2D field, just leaving the ice shelves.

# Arguments:
# data: array of data to mask, assumed to be 2D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_except_zice (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, np.invert(grid.get_zice_mask(gtype=gtype)), time_dependent=time_dependent)


# Mask everything except FRIS out of a 2D field.

# Arguments:
# data: array of data to mask, assumed to be 2D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_except_fris (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, np.invert(grid.get_fris_mask(gtype=gtype)), time_dependent=time_dependent)


# Apply the 3D hfac mask. Dry cells are masked out; partial cells are untouched.

# Arguments:
# data: array of data to mask, assumed to be 3D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_3d (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, grid.get_hfac(gtype=gtype)==0, time_dependent=time_dependent)


# Find the indices bounding the given year in the given time array. This script doesn't check that the entire year is within the array! Partial years are supported.

# Arguments:
# time: array of Datetime objects (can be created by the function netcdf_time)
# year: integer containing the year we care about

# Output: two integers containing the first index of year in time, and the first index of the next year (i.e. the last index of the year plus one, following python convention).

def select_year (time, year):

    t_start = -1
    for t in range(time.size):
        if time[t].year == year:
            t_start = t
            break
    if t_start == -1:
        print 'Error (trim_year): this array contains no instances of the year ' + str(year)
        sys.exit()
    t_end = time.size
    for t in range(t_start+1, time.size):
        if time[t].year == year+1:
            t_end = t
            break
    return t_start, t_end


# Find the time indices of minimum and maximum sea ice area.

# Arguments:
# aice: 3D (time x lat x lon) array of sea ice area at each time index
# grid: Grid object

# Output: two integers containing the time indices (0-indexed) of minimum and maximum sea ice area, respectively

def find_aice_min_max (aice, grid):

    num_time = aice.shape[0]
    aice_int = np.zeros(num_time)
    for t in range(num_time):
        aice_int[t] = total_aice(aice[t,:], grid)
    return np.argmin(aice_int), np.argmax(aice_int)


# Find the minimum and maximum values of a 2D (lat x lon) array in the given region.
def var_min_max (data, grid, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, gtype='t'):

    # Choose the correct longitude and latitude arrays
    lon, lat = grid.get_lon_lat(gtype=gtype)

    # Set limits on axes
    if zoom_fris:
        xmin = fris_bounds[0]
        xmax = fris_bounds[1]
        ymin = fris_bounds[2]
        ymax = fris_bounds[3]
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lon)
    if ymax is None:
        ymax = np.amax(lon)

    # Select the correct indices
    loc = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)
    # Find the min and max values
    return np.amin(data[loc]), np.amax(data[loc])


# Find all the factors of the integer n.
def factors (n):

    factors = []
    for i in range(1, n+1):
        if n % i == 0:
            factors.append(i)
    return factors


# Convert longitude and latitude to polar stereographic projection used by BEDMAP2. Adapted from polarstereo_fwd.m in the MITgcm Matlab toolbox.
def polar_stereo (lon, lat, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    # Deep copies of arrays in case they are reused
    lon = np.copy(lon)
    lat = np.copy(lat)

    if lat_c < 0:
        # Southern hemisphere
        pm = -1
    else:
        # Northern hemisphere
        pm = 1

    # Prepare input
    lon = lon*pm*deg2rad
    lat = lat*pm*deg2rad
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*pm*deg2rad

    # Calculations
    t = np.tan(np.pi/4 - lat/2)/((1 - e*np.sin(lat))/(1 + e*np.sin(lat)))**(e/2)
    t_c = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    m_c = np.cos(lat_c)/np.sqrt(1 - (e*np.sin(lat_c))**2)
    rho = a*m_c*t/t_c
    x = pm*rho*np.sin(lon - lon0)
    y = -pm*rho*np.cos(lon - lon0)

    return x, y


# Given a path to a directory, make sure it ends with /
def real_dir (dir_path):

    if not dir_path.endswith('/'):
        dir_path += '/'
    return dir_path

    


