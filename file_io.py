#####################################################################
# Utilities for file reading and writing, both NetCDF and binary
#####################################################################

import numpy as np
import sys
import os
import datetime

from utils import days_per_month


# Read a single variable from a NetCDF file. The default behaviour is to read and return the entire record (all time indices), but you can also select a subset of time indices, and/or time-average - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read
# var_name: name of variable in NetCDF file

# Optional keyword arguments:
# time_index: integer (0-based) containing a time index. If set, the variable will be read for this specific time index, rather than for the entire record.
# t_start: integer (0-based) containing the time index to start reading at. Default is 0 (beginning of the record).
# t_end: integer (0-based) containing the time index to stop reading before (i.e. the first index not read, following python conventions). Default is the length of the record.
# time_average: boolean indicating to time-average the record before returning (will honour t_start and t_end if set, otherwise will average over the entire record). Default False.

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

def read_netcdf (file_path, var_name, time_index=None, t_start=None, t_end=None, time_average=False):

    import netCDF4 as nc

    # Check for conflicting arguments
    if time_index is not None and time_average==True:
        print 'Error (function read_netcdf): you selected a specific time index (time_index=' + str(time_index) + '), and also want time averaging (time_average=True). Choose one or the other.'
        sys.exit()

    # Open the file
    id = nc.Dataset(file_path, 'r')

    # Figure out if this variable is time-dependent. We consider this to be the case if the name of its first dimension clearly looks like a time variable (not case sensitive) or if its first dimension is unlimited.
    first_dim = id.variables[var_name].dimensions[0]

    if first_dim.upper() in ['T', 'TIME', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'TIME_INDEX', 'DELTAT'] or id.dimensions[first_dim].isunlimited():
        # Time-dependent

        num_time = id.dimensions[first_dim].size
        # Check for 1D timeseries variables
        timeseries = len(id.variables[var_name].shape) == 1

        # Choose range of time values to consider
        # If t_start and/or t_end are already set, use those bounds
        # Otherwise, start at the first time_index and/or end at the last time_index in the file
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = num_time

        # Now read the variable
        if time_index is not None:
            if timeseries:
                data = id.variables[var_name][time_index]
            else:
                data = id.variables[var_name][time_index,:]
        else:
            if timeseries:
                data = id.variables[var_name][t_start:t_end]
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

    return data


# Read the time axis from a NetCDF file. The default behaviour is to read and return the entire axis as Date objects, but you can also select a subset of time indices, and/or return as scalars - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read

# Optional keyword arguments
# var_name: name of time axis. Default 'time'.
# t_start, t_end: as in function read_netcdf
# return_date: boolean indicating to return the time axis as Date objects (so you can easily get year, month, day as attributes). Default True. If False, will just return the axis as scalars.
# monthly: indicates that the output is monthly averaged, so everything will be stamped with the first day of the next month. If True, the function will subtract one month from each timestamp, so it's at the beginning of the correct month.

# Output: 1D numpy array containing the time values (either scalars or Date objects)

def netcdf_time (file_path, var_name='time', t_start=None, t_end=None, return_date=True, monthly=True, return_units=False):

    import netCDF4 as nc

    if return_units and not return_date:
        print 'Error (netcdf_time): need return_date=True if return_units=True'
        sys.exit()

    # Open the file and get the length of the record
    id = nc.Dataset(file_path, 'r')
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
        units = time_id.units
        time = nc.num2date(time_id[t_start:t_end], units=units)
    else:
        # Return just as scalar values
        time = time_id[t_start:t_end]
    id.close()

    if monthly and return_date:
        # Back up to previous month
        for t in range(time.size):
            # First back up one day so the year and month are correct
            time[t] = time[t] - datetime.timedelta(days=1)
            # Now use the timestamp from the beginning of the month
            time[t] = datetime.datetime(time[t].year, time[t].month, 1)

    if return_units:
        return time, units
    else:
        return time


# Given two NetCDF files, figure out which one the given variable is in.
def find_variable (file_path_1, file_path_2, var_name):

    import netCDF4 as nc

    if var_name in nc.Dataset(file_path_1).variables:
        return file_path_1
    elif var_name in nc.Dataset(file_path_2).variables:
        return file_path_2
    else:
        print 'Error (find_variable): variable ' + var_name + ' not in ' + file_path_1 + ' or ' + file_path_2
        sys.exit()


# Given time parameters, make sure we will end up with a single record in time.
def check_single_time (time_index, time_average):
    if time_index is None and not time_average:
        print 'Error (check_single_time): either specify time_index or set time_average=True.'
        sys.exit()


# Helper function for read_binary and write_binary. Given a precision (32 or 64) and endian-ness ('big' or 'little'), construct the python data type string.
def set_dtype (prec, endian):

    if endian == 'big':
        dtype = '>'
    elif endian == 'little':
        dtype = '<'
    else:
        print 'Error (set_dtype): invalid endianness'
        sys.exit()
    if prec == 32:
        dtype += 'f4'
    elif prec == 64:
        dtype += 'f8'
    else:
        print 'Error (set_dtype): invalid precision'
        sys.exit()
    return dtype


# Read an array from a binary file and reshape to the correct dimensions. If it's an MITgcm array, use rdmds (built into MITgcmutils) instead.

# Arguments:
# filename: path to binary file
# grid_sizes: list of length 2 or 3 containing [nx, ny] or [nx, ny, nz] grid sizes
# dimensions: string containing dimension characters ('x', 'y', 'z', 't') in any order, e.g. 'xyt'

# Optional keyword arguments:
# prec: precision of data: 32 (default) or 64
# endian: endian-ness of data: 'big' (default) or 'little'

def read_binary (filename, grid_sizes, dimensions, prec=32, endian='big'):

    print 'Reading ' + filename

    dtype = set_dtype(prec, endian)

    # Extract grid sizes
    nx = grid_sizes[0]
    ny = grid_sizes[1]
    if len(grid_sizes) == 3:
        # It's a 3D grid
        nz = grid_sizes[2]
    elif 'z' in dimensions:
        print 'Error (read_binary): ' + dimensions + ' is depth-dependent, but your grid sizes are 2D.'
        sys.exit()

    # Read data
    data = np.fromfile(filename, dtype=dtype)

    # Expected shape of data
    shape = []
    # Expected size of data, not counting time
    size0 = 1
    # Now update these initial values
    if 'x' in dimensions:
        shape = [nx] + shape
        size0 *= nx
    if 'y' in dimensions:
        shape = [ny] + shape
        size0 *= ny
    if 'z' in dimensions:
        shape = [nz] + shape
        size0 *= nz

    if 't' in dimensions:
        # Time-dependent field; figure out how many timesteps
        if np.mod(data.size, size0) != 0:
            print 'Error (read_binary): incorrect dimensions or precision'
            sys.exit()
        num_time = data.size/size0
        shape = [num_time] + shape
    else:
        # Time-independent field; just do error checking
        if data.size != size0:
            print 'Error (read_binary): incorrect dimensions or precision'
            sys.exit()

    # Reshape the data and return
    return np.reshape(data, shape)            

        

# Write an array ("data"), of any dimension, to a binary file ("file_path"). Optional keyword arguments ("prec" and "endian") are as in function read_binary.
def write_binary (data, file_path, prec=32, endian='big'):

    print 'Writing ' + file_path

    dtype = set_dtype(prec, endian)    
    # Make sure data is in the right precision
    data = data.astype(dtype)

    # Write to file
    id = open(file_path, 'w')
    data.tofile(id)
    id.close()


# NCfile object to simplify writing of NetCDF files.
class NCfile:

    # Initialisation arguments:
    # filename: name for desired NetCDF file
    # grid: Grid object
    # dimensions: string containing dimension characters in any order, eg 'xyz' or 'xyt'. Include all the dimensions (from x, y, z, t) that any of the variables in the file will need.
    def __init__ (self, filename, grid, dimensions):

        import netCDF4 as nc

        # Open the file
        self.id = nc.Dataset(filename, 'w')

        # Set up the grid
        if 't' in dimensions:
            self.id.createDimension('time', None)
        if 'z' in dimensions:
            self.id.createDimension('Z', grid.nz)
            self.id.createVariable('Z', 'f8', ('Z'))
            self.id.variables['Z'].long_name = 'vertical coordinate of cell center'
            self.id.variables['Z'].units = 'm'
            self.id.variables['Z'][:] = grid.z
            self.id.createDimension('Zl', grid.nz)
            self.id.createVariable('Zl', 'f8', ('Zl'))
            self.id.variables['Zl'].long_name = 'vertical coordinate of upper cell interface'
            self.id.variables['Zl'].units = 'm'
        if 'y' in dimensions:
            self.id.createDimension('Y', grid.ny)
            self.id.createVariable('Y', 'f8', ('Y'))
            self.id.variables['Y'].long_name = 'latitude at cell center'
            self.id.variables['Y'].units = 'degrees_north'
            self.id.variables['Y'][:] = grid.lat_1d
            self.id.createDimension('Yp1', grid.ny)
            self.id.createVariable('Yp1', 'f8', ('Yp1'))
            self.id.variables['Yp1'].long_name = 'latitude at SW corner'
            self.id.variables['Yp1'].units = 'degrees_north'
            self.id.variables['Yp1'][:] = grid.lat_corners_1d
        if 'x' in dimensions:
            self.id.createDimension('X', grid.nx)
            self.id.createVariable('X', 'f8', ('X'))
            self.id.variables['X'].long_name = 'longitude at cell center'
            self.id.variables['X'].units = 'degrees_east'
            self.id.variables['X'][:] = grid.lon_1d
            self.id.createDimension('Xp1', grid.nx)
            self.id.createVariable('Xp1', 'f8', ('Xp1'))
            self.id.variables['Xp1'].long_name = 'longitude at SW corner'
            self.id.variables['Xp1'].units = 'degrees_east'
            self.id.variables['Xp1'][:] = grid.lon_corners_1d


    # Create and write a variable.

    # Arguments:
    # var_name: desired name for variable
    # data: array of data for that variable
    # dimensions: as in initialisation

    # Optional keyword arguments:
    # gtype: as in function cell_boundaries (plus 'w' for w-grid)
    # long_name: descriptor for this variable
    # units: units for this variable
    # dtype: data type of variable (default 'f8' which is float)

    def add_variable (self, var_name, data, dimensions, gtype='t', long_name=None, units=None, dtype='f8'):

        # Sort out dimensions
        shape = []
        if 't' in dimensions:
            shape.append('time')
        if 'z' in dimensions:
            if gtype == 'w':
                shape.append('Zl')
            else:
                shape.append('Z')
        if 'y' in dimensions:
            if gtype in ['v', 'psi']:
                shape.append('Yp1')
            else:
                shape.append('Y')
        if 'x' in dimensions:
            if gtype in ['u', 'psi']:
                shape.append('Xp1')
            else:
                shape.append('X')
        shape = tuple(shape)

        # Initialise the variable
        self.id.createVariable(var_name, dtype, shape)
        if long_name is not None:
            self.id.variables[var_name].long_name = long_name
        if units is not None:
            self.id.variables[var_name].units = units

        # Fill data
        self.id.variables[var_name][:] = data


    # Special case to simplify writing the time variable.

    # Argument:
    # time: time values (either numeric values or DateTime objects)

    # Optional keyword argument:
    # units: units of time (eg 'seconds since 1979-01-01 00:00:00')
    def add_time (self, time, units=None):

        import netCDF4 as nc

        if isinstance(time[0], datetime.datetime):
            # These are DateTime objects
            if units is None:
                # Set some default units
                units = 'seconds since 1979-01-01 00:00:00'
            # Convert to numeric values
            time = nc.date2num(time, units)            

        self.add_variable('time', time, 't', units=units)


    # Call this function when you're ready to close the file.
    def close (self):

        self.id.close()



# Basic version of NCfile for a simple lat-lon file on any regular grid (eg intermediate domain generation steps, see make_domain.py).
class NCfile_basiclatlon:

    # Initialisation arguments:
    # filename: name for desired NetCDF file
    # lon, lat: 1D longitude and latitude arrays
    def __init__ (self, filename, lon, lat):

        import netCDF4 as nc

        self.id = nc.Dataset(filename, 'w')
        self.id.createDimension('lat', lat.size)
        self.id.createVariable('lat', 'f8', ('lat'))
        self.id.variables['lat'].long_name = 'latitude at cell centre'
        self.id.variables['lat'].units = 'degrees_north'
        self.id.variables['lat'][:] = lat
        self.id.createDimension('lon', lon.size)
        self.id.createVariable('lon', 'f8', ('lon'))
        self.id.variables['lon'].long_name = 'longitude at cell centre'
        self.id.variables['lon'].units = 'degrees_east'
        self.id.variables['lon'][:] = lon

        
    # Create and write a lat-lon variable.
    def add_variable (self, var_name, data, long_name=None, units=None, dtype='f8'):
        
        self.id.createVariable(var_name, dtype, ('lat', 'lon'))
        if long_name is not None:
            self.id.variables[var_name].long_name = long_name
        if units is not None:
            self.id.variables[var_name].units = units
        self.id.variables[var_name][:] = data

        
    # Call this function when you're ready to close the file.
    def close (self):

        self.id.close()


# Given a list of output files (chronological, could concatenate to make the entire simulation) and a time index we want relative to the beginning of the simulation (0-indexed), find the individual file that time index falls within, and what that time index is relative to the beginning of that file.
def find_time_index (file_list, time_index):

    for file_path in file_list:
        num_time = netcdf_time(file_path).size
        if num_time > time_index:
            return file_path, time_index
        else:
            time_index -= num_time
    # If we're still here, we didn't find it
    print "Error (find_time_index): this simulation isn't long enough to contain time_index=" + str(time_index)
    sys.exit()

    




    



            

            
            

