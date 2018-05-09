import netCDF4 as nc4
import numpy as np

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
    time
    if return_date:
        # Return as handy Date objects
        time = nc4.num2date(time_id[:], units=time_id.units, calendar=time_id.calendar.lower())
    else:
        # Return just as scalar values
        time = time_id[:]
    id.close()

    return time
