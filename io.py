#####################################################################
# Utilities for file reading and writing, both NetCDF and binary
#####################################################################

import netCDF4 as nc
import numpy as np
import sys
import os
import datetime


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

    return data


# Read the time axis from a NetCDF file. The default behaviour is to read and return the entire axis as Date objects, but you can also select a subset of time indices, and/or return as scalars - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read

# Optional keyword arguments
# var_name: name of time axis. Default 'time'.
# t_start, t_end: as in function read_netcdf
# return_date: boolean indicating to return the time axis as Date objects (so you can easily get year, month, day as attributes). Default True. If False, will just return the axis as scalars.
# monthly: indicates that the output is monthly averaged, so everything will be stamped with the first day of the next month. If True, the function will subtract one day from each timestamp, so at least the month and year are correct.

# Output: 1D numpy array containing the time values (either scalars or Date objects)

def netcdf_time (file_path, var_name='time', t_start=None, t_end=None, return_date=True, monthly=True):

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
        time = nc.num2date(time_id[t_start:t_end], units=time_id.units)
    else:
        # Return just as scalar values
        time = time_id[t_start:t_end]
    id.close()

    if monthly:
        # Back up one day so at least the year and month are correct
        for t in range(time.size):
            time[t] = time[t] - datetime.timedelta(days=1)

    return time


# Given two NetCDF files, figure out which one the given variable is in.
def find_variable (file_path_1, file_path_2, var_name):

    if var_name in nc.Dataset(file_path_1).variables:
        return file_path_1
    elif var_name in nc.Dataset(file_path_2).variables:
        return file_path_2
    else:
        print 'Error (find_variable): variable ' + var_name + ' not in ' + file_path_1 + ' or ' + file_path_2
        sys.exit()


# Read an array from a binary file. This is useful for input files (eg bathymetry) and output files which don't get converted to NetCDF (eg crashes).

# Arguments:
# filename: path to binary file
# grid: Grid object
# dimensions: string containing dimension characters in any order, eg 'xyz' or 'xyt'. For a 1D array of a special shape, use '1'.

# Optional keyword arguments:
# prec: 32 or 64, corresponding to the precision of the file. Default 32. 
#       Here is how you work out the expected precision of MITgcm files:
#       Input OBC files: exf_iprec_obcs (data.exf, default equal to exf_iprec)
#       Input forcing files: exf_iprec (data.exf, default 32)
#       All other input files: readBinaryPrec (data, default 32)
#       Restarts/dumps: writeStatePrec (data, default 64)
#       All other output files: writeBinaryPrec (data, default 32)

# Output: array of specified dimension

def read_binary (filename, grid, dimensions, prec=32):

    # Set dtype
    if prec == 32:
        dtype = '>f4'
    elif prec == 64:
        dtype = '>f8'
    else:
        print 'Error (read_binary): invalid precision'
        sys.exit()

    # Read data
    data = np.fromfile(filename, dtype=dtype)

    if dimensions == '1':
        # No need to reshape
        return data

    # Work out dimensions
    if 'z' in dimensions:
        if 't' in dimensions:
            if mod(data.size, grid.nx*grid.ny*grid.nz) != 0:
                print 'Error (read_binary): incorrect dimensions or precision'
                sys.exit()
            num_time = data.size/(grid.nx*grid.ny*grid.nz)
            data_shape = [num_time, grid.nz, grid.ny, grid.nx]
        else:
            if data.size != grid.nx*grid.ny*grid.nz:
                print 'Error (read_binary): incorrect dimensions or precision'
                sys.exit()
            data_shape = [grid.nz, grid.ny, grid.nx]
    else:
        if 't' in dimensions:
            if mod(data.size, grid.nx*grid.ny) != 0:
                print 'Error (read_binary): incorrect dimensions or precision'
                sys.exit()
            num_time = data.size/(grid.nx*grid.ny)
            data_shape = [num_time, grid.ny, grid.nx]
        else:
            if data.size != grid.nx*grid.ny:
                print 'Error (read_binary): incorrect dimensions or precision'
                sys.exit()
            data_shape = [grid.ny, grid.nx]

    # Reshape the data and return
    return np.reshape(data, data_shape)


# Write an array ("data"), of any dimension, to a binary file ("file_path"). Default is 32-bit (prec=32) but can also do 64-bit (prec=64).
def write_binary (data, file_path, prec=32):

    # Set dtype
    if prec == 32:
        dtype = '>f4'
    elif prec == 64:
        dtype = '>f8'
    else:
        print 'Error (write_binary): invalid precision'
        sys.exit()

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
    # dimensions: as in function read_binary
    def __init__ (self, filename, grid, dimensions):

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
    # dimensions: as in function read_binary

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
    # time: time values

    # Optional keyword argument:
    # units: units of time (eg 'seconds since 1979-01-01 00:00:00')
    def add_time (self, time, units=None):

        self.add_variable('time', time, 't', units=units)


    # Call this function when you're ready to close the file.
    def finished (self):

        self.id.close()



    



            

            
            

