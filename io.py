import netCDF4 as nc4
import numpy as np
import sys
from utils import fix_lon_range

# FUNCTION read_netcdf
# Read a single variable from a NetCDF file. The default behaviour is to read and return the entire record (all time indices), but you can also select a subset of time indices, and/or time-average - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read
# var_name: name of variable in NetCDF file

# Optional keyword arguments:
# time_index: integer (0-based) containing a time index. If set, the variable will be read for this specific time index, rather than for the entire record.
# t_start: integer (0-based) containing the time index to start reading at. Default is 0 (beginning of the record).
# t_end: integer (0-based) containing the time index to stop reading before (i.e. the first index not read, following python conventions). Default is the length of the record.
# time_average: boolean indicating to time-average the record before returning (will honour t_start and t_end if set, otherwise will average over the entire record). Default False.
# mask: boolean indicating to mask values that are identically zero. Default True.

# Output: numpy array containing the variable

# Examples:
# Read the entire record:
# temp = read_netcdf('temp.nc', 'temp')
# Read just the first time index:
# temp = read_netcdf('temp.nc', 'temp', time_index=0)
# Read the first 12 time indices:
# temp = read_netcdf('temp.nc', 'temp', t_end=12)
# Read the entire record and time-average:
# temp = read_netcdf('temp.nc', 'temp', time_average=True)
# Read the last 12 time indices and time-average:
# temp = read_netcdf('temp.nc', 'temp', t_start=-12, time_average=True)

def read_netcdf (file_path, var_name, time_index=None, t_start=None, t_end=None, time_average=False, mask=True):

    # Check for conflicting arguments
    if time_index is not None and time_average==True:
        print 'Error (function read_netcdf): you selected a specific time index (time_index=' + str(time_index) + '), and also want time averaging (time_average=True). Choose one or the other.'
        sys.exit()

    # Open the file
    id = nc4.Dataset(file_path, 'r')

    # Figure out if this variable is time-dependent. We consider this to be the case if the name of its first dimension clearly looks like a time variable (not case sensitive) or if its first dimension is unlimited.
    first_dim = id.variables[var_name].dimensions[0]

    if first_dim.upper() in ['T', 'TIME', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'TIME_INDEX', 'DELTAT'] or id.dimensions[first_dim].isunlimited():
        # Time-dependent

        num_time = id.dimensions[first_dim].size
        # Make sure the variable itself isn't time
        if len(id.variables[var_name].shape) == 1:
            print 'Error (function read_netcdf): you are trying to read the time variable. Use netcdf_time instead.'
            sys.exit()

        # Choose range of time values to consider
        # If t_start and/or t_end are already set, use those bounds
        # Otherwise, start at the first time_index and/or end at the last time_index in the file
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = num_time

        # Now read the variable
        if time_index is not None:
            data = id.variables[var_name][time_index,:]
        else:
            data = id.variables[var_name][t_start:t_end,:]
        id.close()

        # Time-average if necessary
        if time_average:
            data = np.mean(data, axis=0)    

    else:
        # Not time-dependent

        if time_index is not None or time_average==True or t_start is not None or t_end is not None:
            print 'Error (function read_netcdf): you want to do something fancy with the time dimension of variable ' + var_name + ' in file ' + file_path + ', but this does not appear to be a time-dependent variable.'
            sys.exit()

        # Read the variable
        data = id.variables[var_name][:]

    # Remove any one-dimensional entries
    data = np.squeeze(data)

    # Mask
    if mask:
        data = np.ma.masked_where(data==0, data)        

    return data


# FUNCTION netcdf_time
# Read the time axis from a NetCDF file. The default behaviour is to read and return the entire axis as Date objects, but you can also select a subset of time indices, and/or return as scalars - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read

# Optional keyword arguments
# var_name: name of time axis. Default 'T'.
# t_start: integer (0-based) containing the time index to start reading at. Default is 0 (beginning of the record).
# t_end: integer (0-based) containing the time index to stop reading before (i.e. the first index not read, following python conventions). Default is the length of the record.
# return_date: boolean indicating to return the time axis as Date objects (so you can easily get year, month, day as attributes). Default True. If False, will just return the axis as scalars.

# Output: 1D numpy array containing the time values (either scalars or Date objects)

# Examples:
# Read the entire time axis as Date objects:
# time = netcdf_time('sample.nc')
# Read a time axis which is named something other than 'T':
# time = netcdf_time('sample.nc', var_name='time')
# Read only the first 12 time indices:
# time = netcdf_time('sample.nc', t_end=12)
# Read only the last 12 time indices:
# time = netcdf_time('sample.nc', t_start=-12)
# Return as scalars rather than Date objects:
# time = netcdf_time('sample.nc', return_date=False)

def netcdf_time (file_path, var_name='T', t_start=None, t_end=None, return_date=True):

    # Open the file and get the length of the record
    id = nc4.Dataset(file_path, 'r')
    time_id = id.variables[var_name]    
    num_time = time_id.size

    # Choose range of time values to consider
    # If t_start and/or t_end are already set, use those bounds
    # Otherwise, start at the first time_index and/or end at the last time_index in the file
    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = num_time

    # Read the variable
    if return_date:
        # Return as handy Date objects
        time = nc4.num2date(time_id[t_start:t_end], units=time_id.units)
    else:
        # Return just as scalar values
        time = time_id[t_start:t_end]
    id.close()

    return time


# CLASS Grid
# Load all the useful grid variables and store them in a Grid object.

# (Initialisation) Arguments:
# path: path to NetCDF grid file (if nc=True) or run directory containing all the binary grid files (if nc=False)
# nc: Default True. If False, reads the grid from binary files instead of from a NetCDF file.

# Output: Grid object containing lots of grid variables - read comments in code to find them all.

# Examples:
# Read grid from NetCDF file:
# grid = Grid('grid.nc')
# Read grid from binary files in run directory
# grid = Grid('../run/', nc=False)
# A few of the grid variables:
# grid.lon_2d
# grid.lat_2d
# grid.z
# grid.dA
# grid.hfac

class Grid:
    
    def __init__ (self, path, nc=True):

        if nc:
            # Read grid from a NetCDF file

            # 1D lon and lat axes on regular grids
            # Make sure longitude is between -180 and 180
            # Cell centres
            self.lon_1d = fix_lon_range(read_netcdf(path, 'X'))
            self.lat_1d = read_netcdf(path, 'Y')
            # Cell corners
            self.lon_psi_1d = fix_lon_range(read_netcdf(path, 'Xp1'))
            self.lat_psi_1d = read_netcdf(path, 'Yp1')

            # 2D lon and lat fields on any grid
            # Cell centres
            self.lon_2d = fix_lon_range(read_netcdf(path, 'XC'))
            self.lat_2d = read_netcdf(path, 'YC')
            # Cell corners
            self.lon_psi_2d = fix_lon_range(read_netcdf(path, 'XG'))
            self.lat_psi_2d = read_netcdf(path, 'YG')

            # 2D integrands of distance
            # Across faces
            self.dx = read_netcdf(path, 'dxF')
            self.dy = read_netcdf(path, 'dyF')
            # Between centres
            self.dx_t = read_netcdf(path, 'dxC')
            self.dy_t = read_netcdf(path, 'dyC')
            # Between u-points
            self.dx_u = self.dx  # Equivalent to distance across face
            self.dy_u = read_netcdf(path, 'dyU')
            # Between v-points
            self.dx_v = read_netcdf(path, 'dxV')
            self.dy_v = self.dy  # Equivalent to distance across face
            # Between corners
            self.dx_psi = read_netcdf(path, 'dxG')
            self.dy_psi = read_netcdf(path, 'dyG')

            # 2D integrands of area
            # Area of faces
            self.dA = read_netcdf(path, 'rA')
            # Centered on u-points
            self.dA_u = read_netcdf(path, 'rAw')
            # Centered on v-points
            self.dA_v = read_netcdf(path, 'rAs')
            # Centered on corners
            self.dA_psi = read_netcdf(path, 'rAz')

            # Vertical grid
            # Assumes we're in the ocean so using z-levels - not sure how this
            # would handle atmospheric pressure levels.
            # Depth axis at centres of z-levels
            self.z = read_netcdf(path, 'Z')
            # Depth axis at edges of z-levels
            self.z_edges = read_netcdf(path, 'Zp1')

            # Vertical integrands of distance
            # Across cells
            self.dz = read_netcdf(path, 'drF')
            # Between centres
            self.dz_t = read_netcdf(path, 'drC')

            # Partial cell fractions
            # At centres
            self.hfac = read_netcdf(path, 'HFacC')
            # At u-points
            self.hfac_u = read_netcdf(path, 'HFacW')
            # At v-points
            self.hfac_v = read_netcdf(path, 'HFacS')

            # Topography
            # Bathymetry (bottom depth)
            self.bathy = read_netcdf(path, 'R_low')
            # Ice shelf draft (surface depth)
            self.zice = read_netcdf(path, 'Ro_surf')
            # Water column thickness
            self.wct = read_netcdf(path, 'Depth')

            # Create a couple of masks
            self.land_mask = self.bathy == 0
            self.zice_mask = self.zice == 0

            # Apply land mask to the topography
            self.bathy = np.ma.masked_where(self.land_mask, self.bathy)
            self.zice = np.ma.masked_where(self.land_mask, self.zice)
            self.wct = np.ma.masked_where(self.land_mask, self.wct)

        else:

            print 'Error (class Grid): the code has only been written for NetCDF grids so far.'
            sys.exit()
