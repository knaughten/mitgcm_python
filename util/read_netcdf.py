import netCDF4 as nc4
import numpy as np
import sys

# Read a single variable from a NetCDF file. The default behaviour is to read and return the entire record (all time indices), but you can also select a subset of time indices, and/or time-average - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file to read
# var_name: name of variable in NetCDF file

# Optional keyword arguments:
# time_index: integer (0-based) containing a time index. If set, the variable will be read for this specific time index, rather than for the entire record.
# t_start: integer (0-based) containing the time index to start reading at. Default is 0 (beginning of the record).
# t_end: integer (0-based) containing the time index to stop reading before (i.e. the first index not read, following python conventions). Default is the length of the record.
# time_average: boolean indicating to time-average the record before returning (will honour t_start and t_end if set, otherwise will average over the entire record). Default False.

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
        print 'Error (read_netcdf.py): you selected a specific time index (time_index=' + str(time_index) + '), and also want time averaging (time_average=True). Choose one or the other.'
        sys.exit()

    # Open the file
    id = nc4.Dataset(file_path, 'r')

    # Figure out if this variable is time-dependent. We consider this to be the case if the name of its first dimension clearly looks like a time variable (not case sensitive) or if its first dimension is unlimited.
    first_dim = id.variables[var_name].dimensions[0]
    if first_dim.upper() in ['T', 'TIME', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'TIME_INDEX', 'DELTAT'] or id.dimensions[first_dim].isunlimited():
        time_dependent = True
        num_time = id.dimensions[first_dim].size
    else:
        time_dependent = False
        if time_index is not None or time_average==True or t_start is not None or t_end is not None:
            print 'Error (read_netcdf.py): you want to do something fancy with the time dimension of variable ' + var_name + ' in file ' + file_path + ', but this does not appear to be a time-dependent variable.'
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

    # Remove any one-dimensional entries
    data = np.squeeze(data)

    return data
