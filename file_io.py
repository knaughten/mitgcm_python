#####################################################################
# Utilities for file reading and writing, both NetCDF and binary
#####################################################################

import numpy as np
import sys
import os
import datetime

from utils import days_per_month, real_dir, is_depth_dependent, average_12_months


# Read a single variable from a NetCDF file. The default behaviour is to read and return the entire record (all time indices), but you can also select a subset of time indices, and/or time-average - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read
# var_name: name of variable in NetCDF file

# Optional keyword arguments:
# time_index: integer (0-based) containing a time index. If set, the variable will be read for this specific time index, rather than for the entire record.
# t_start: integer (0-based) containing the time index to start reading at. Default is 0 (beginning of the record).
# t_end: integer (0-based) containing the time index to stop reading before (i.e. the first index not read, following python conventions). Default is the length of the record.
# time_average: boolean indicating to time-average the record before returning (will honour t_start and t_end if set, otherwise will average over the entire record). Default False.
# return_info: boolean indicating to return the 'description'/'long_name' and 'units' variables. Default False.
# return_minmax: boolean indicating to return the 'vmin' and 'vmax' attributes. Default False.

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

def read_netcdf (file_path, var_name, time_index=None, t_start=None, t_end=None, time_average=False, return_info=False, return_minmax=False):

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

    if return_info:
        try:
            description = id.variables[var_name].description
        except(AttributeError):
            description = id.variables[var_name].long_name
        units = id.variables[var_name].units
    if return_minmax:
        vmin = id.variables[var_name].vmin
        vmax = id.variables[var_name].vmax
    id.close()

    if return_info and return_minmax:
        return data, description, units, vmin, vmax
    elif return_info:
        return data, description, units
    elif return_minmax:
        return data, vmin, vmax
    else:
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

    # Open the file and get the length of the record
    id = nc.Dataset(file_path, 'r')
    time_id = id.variables[var_name]
    units = time_id.units
    try:
        calendar = time_id.calendar
    except(AttributeError):
        calendar = 'standard'
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
        time = nc.num2date(time_id[t_start:t_end], units=units, calendar=calendar)
    else:
        # Return just as scalar values
        time = time_id[t_start:t_end]
    id.close()

    if return_date:
        # Want to convert to a datetime object
        if monthly:
            # Back up to previous month
            for t in range(time.size):
                month = time[t].month-1
                year = time[t].year
                if month < 1:
                    month += 12
                    year -= 1
                time[t] = datetime.datetime(year, month, 1)
        else:
            for t in range(time.size):
                time[t] = datetime.datetime(time[t].year, time[t].month, time[t].day)             

    if return_units:
        return time, units, calendar
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

    if isinstance(data, np.ma.MaskedArray):
        # Need to remove the mask
        data = data.data

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
    # vmin, vmax: optional attributes
    # dtype: data type of variable (default 'f8' which is float)

    def add_variable (self, var_name, data, dimensions, gtype='t', long_name=None, units=None, calendar=None, vmin=None, vmax=None, dtype='f8'):

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
        if calendar is not None:
            self.id.variables[var_name].calendar = calendar
        if vmin is not None:
            self.id.variables[var_name].vmin = vmin
        if vmax is not None:
            self.id.variables[var_name].vmax = vmax

        # Fill data
        self.id.variables[var_name][:] = data


    # Special case to simplify writing the time variable.

    # Argument:
    # time: time values (either numeric values or DateTime objects)

    # Optional keyword argument:
    # units: units of time (eg 'seconds since 1979-01-01 00:00:00')
    def add_time (self, time, units=None, calendar=None):

        import netCDF4 as nc

        if isinstance(time[0], datetime.datetime):
            # These are DateTime objects
            if units is None:
                # Set some default units
                units = 'seconds since 1979-01-01 00:00:00'
            # Convert to numeric values
            time = nc.date2num(time, units, calendar=calendar)            

        self.add_variable('time', time, 't', units=units, calendar=calendar)


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


# Save a super-basic NetCDF file with one variable and no information about the axes. Must be xy with possible time and depth dimensions.
def write_netcdf_basic (data, var_name, filename, time_dependent=True, units=None):
    import netCDF4 as nc
    id = nc.Dataset(filename, 'w')
    depth_dependent = is_depth_dependent(data, time_dependent=time_dependent)
    if time_dependent:
        id.createDimension('time', None)
    if depth_dependent:
        id.createDimension('Z', data.shape[-3])
    id.createDimension('Y', data.shape[-2])
    id.createDimension('X', data.shape[-1])
    id.createVariable('time', 'f8', ['time'])
    id.variables['time'][:] = np.arange(data.shape[0])+1
    if depth_dependent:
        id.createVariable(var_name, 'f8', ['time', 'Z', 'Y', 'X'])
    else:
        id.createVariable(var_name, 'f8', ['time', 'Y', 'X'])
    if units is not None:
        id.variables[var_name].units = units
    id.variables[var_name][:] = data
    id.close()

# Save a very basic NetCDF file with one variable as an error dump from the discard_and_fill function
def write_netcdf_very_basic (data, var_name, filename, use_3d=False):
    import netCDF4 as nc
    id = nc.Dataset(filename, 'w')
    if use_3d:
        id.createDimension('Z', data.shape[-3])
        id.createDimension('Y', data.shape[-2])
        id.createDimension('X', data.shape[-1])
        id.createVariable(var_name, 'f8', ['Z', 'Y', 'X'])
    else:
        id.createDimension('Y', data.shape[-2])
        id.createDimension('X', data.shape[-1])
        id.createVariable(var_name, 'f8', ['Y', 'X'])
    id.variables[var_name][:] = data
    id.close()

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


# Given information about a CMIP6 dataset (path to model directory, ensemble member, experiment, variable, and time code eg 'day' or 'Omon'), return a list of the files containing this data, and the years covered by each file.
# This assumes there are 30-day months in the simulation (true for UKESM1).

# Input arguments:
# model_path: path to directory containing output from that model and the given MIP, eg '/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/'
# expt: experiment name, eg 'piControl'
# ensemble_member: ensemble member name, eg 'r1i1p1f2'
# var: variable name, eg 'tas' or 'thetao'
# time_code: name of subdirectory that mentions the frequency and sometimes the domain. eg 'day' for daily atmospheric, 'Omon' for monthly ocean.

# Output:
# in_files: list of files containing this data, in chronological order
# start_years, end_years: lists of integers containing the starting and ending years (inclusive) of each file. The code will check to make sure there are no gaps.
def find_cmip6_files (model_path, expt, ensemble_member, var, time_code):

    # Construct the path to the directory containing all the data files, and make sure it exists
    in_dir = real_dir(model_path)+expt+'/'+ensemble_member+'/'+time_code+'/'+var+'/gn/latest/'
    if not os.path.isdir(in_dir):
        print 'Error (find_cmip6_files): no such directory ' + in_dir
        sys.exit()

    # Get the names of all the data files in this directory, in chronological order
    in_files = []
    for fname in os.listdir(in_dir):
        if fname.endswith('.nc'):
            in_files.append(in_dir+fname)
    in_files.sort()

    # Work out the start and end years for each file
    start_years = []
    end_years = []
    for file_path in in_files:
        # Dates encoded in file names
        if time_code.endswith('day'):
            start_date = file_path[-20:-12]
            end_date = file_path[-11:-3]
        elif time_code.endswith('mon'):
            start_date = file_path[-16:-10]
            end_date = file_path[-9:-3]
        start_year = start_date[:4]
        end_year = end_date[:4]
        # Make sure they are 30-day months and complete years        
        if (time_code.endswith('day') and start_date[4:] != '0101') or (time_code.endswith('mon') and start_date[4:] != '01'):
            print 'Error (find_cmip6_files): '+file_path+' does not start at the beginning of January'
            sys.exit()
        if (time_code.endswith('day') and end_date[4:] != '1230') or (time_code.endswith('mon') and end_date[4:] != '12'):
            print 'Error (find_cmip6_files): '+file_path+' does not end at the end of December'
            sys.exit()
        # Save the start and end years
        start_years.append(int(start_year))
        end_years.append(int(end_year))
    # Now make sure there are no missing years
    for t in range(1, len(in_files)):
        if start_years[t] != end_years[t-1]+1:
            print 'Error (find_cmip6_files): there are missing years in '+in_dir
            sys.exit()

    return in_files, start_years, end_years


# Read a list of variables from the same NetCDF file. They all must have the same time index / averaging / etc.
def read_netcdf_list (file_path, var_list, time_index=None, t_start=None, t_end=None, time_average=False):

    data = []
    for var in var_list:
        data.append(read_netcdf(file_path, var, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))
    return data


# Read the long name and units of the given variable in the NetCDF file.
def read_title_units (file_path, var_name):

    import netCDF4 as nc

    id = nc.Dataset(file_path)
    return id.variables[var_name].long_name, id.variables[var_name].units


# Read sea ice production, which is the sum of 4 variables. Interface to read_netcdf and adding the four.
def read_iceprod (file_path, time_index=None, t_start=None, t_end=None, time_average=False):

    data = None
    for var_name in ['SIdHbOCN', 'SIdHbATC', 'SIdHbATO', 'SIdHbFLO']:
        data_tmp = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        if data is None:
            data = data_tmp
        else:
            data += data_tmp
    return data


# Calculate annual averages of the given variable in the given (chronological) list of files.
def read_annual_average (var_name, file_paths, return_years=False):

    # Inner function to calculate the average and save to data_annual
    def update_data_annual (data_read, t0, year, data_annual):
        data_avg = average_12_months(data_read, t0, calendar=calendar, year=year)
        # Add a dummy time dimension
        data_avg = np.expand_dims(data_avg, axis=0)
        if data_annual is None:
            data_annual = data_avg
        else:
            data_annual = np.concatenate((data_annual, data_avg), axis=0)
        return data_annual

    # Now read all the data
    data_tmp = None
    data_annual = None
    years = []
    for f in file_paths:
        time, units, calendar = netcdf_time(f, return_units=True)
        data = read_netcdf(f, var_name)
        if time.size == 1:
            # Single time record in this file: add a time dimension
            data = np.expand_dims(data, axis=0)
        if data_tmp is not None:
            # There is a partial year from last time - complete it
            num_months = 12-data_tmp.shape[0]
            if data.shape[0] < num_months:
                print 'Error (read_annual_average): '+f+' has only '+str(data.shape[0])+' time indices. This is too short. Concatenate it with the next one and re-run.'
                sys.exit()
            data_tmp2 = data[:num_months,...]
            data_year = np.concatenate((data_tmp, data_tmp2), axis=0)
            data_annual = update_data_annual(data_year, 0, time[0].year, data_annual)
            t_start = num_months
        else:
            # This file starts at the beginning of a year
            t_start = 0
        # Loop over complete years
        for t in range(t_start, (time.size-t_start)/12*12, 12):
            print time[t].year
            years.append(time[t].year)
            data_annual = update_data_annual(data, t, time[t].year, data_annual)
        if t+12 < time.size:
            # Read partial year from end
            data_tmp = data[t+12:,...]
            print time[t+12].year
            years.append(time[t+12].year)
        else:
            # Reset
            data_tmp = None

    if return_years:
        return data_annual, years
    else:
        return data_annual


            

            
            

