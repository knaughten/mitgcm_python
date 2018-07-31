#######################################################
# Calculation of integral timeseries
#######################################################

import numpy as np

from grid import choose_grid
from file_io import read_netcdf, netcdf_time
from utils import convert_ismr, var_min_max, mask_land_ice, mask_except_fris
from diagnostics import total_melt
from averaging import over_area, volume_average


# Calculate total mass loss or area-averaged melt rate from FRIS in the given NetCDF file. The default behaviour is to calculate the melt at each time index in the file, but you can also select a subset of time indices, and/or time-average - see optional keyword arguments. You can also split into positive (melting) and negative (freezing) components.

# Arguments:
# file_path: path to NetCDF file containing 'SHIfwFlx' variable
# grid = Grid object

# Optional keyword arguments:
# result: 'massloss' (default) calculates the total mass loss in Gt/y. 'meltrate' calculates the area-averaged melt rate in m/y.
# time_index, t_start, t_end, time_average: as in function read_netcdf
# mass_balance: if True, split into positive (melting) and negative (freezing) terms. Default False.

# Output:
# If time_index is set, or time_average=True: single value containing mass loss or average melt rate
# Otherwise: 1D array containing timeseries of mass loss or average melt rate
# If mass_balance=True: two values/arrays will be returned, with the positive and negative components.

def timeseries_fris_melt (file_path, grid, result='massloss', time_index=None, t_start=None, t_end=None, time_average=False, mass_balance=False):

    # Read ice shelf melt rate and convert to m/y
    ismr = convert_ismr(read_netcdf(file_path, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))

    if mass_balance:
        # Split into melting and freezing
        ismr_positive = np.maximum(ismr, 0)
        ismr_negative = np.minimum(ismr, 0)
    
    if time_index is not None or time_average:
        # Just one timestep
        if mass_balance:
            melt = total_melt(ismr_positive, grid.fris_mask, result=result)
            freeze = total_melt(ismr_negative, grid.fris_mask, result=result)
            return melt, freeze
        else:
            return total_melt(ismr, grid.fris_mask, grid, result=result)
    else:
        # Loop over timesteps
        num_time = ismr.shape[0]
        if mass_balance:
            melt = np.zeros(num_time)
            freeze = np.zeros(num_time)
            for t in range(num_time):
                melt[t] = total_melt(ismr_positive[t,:], grid.fris_mask, grid, result=result)
                freeze[t] = total_melt(ismr_negative[t,:], grid.fris_mask, grid, result=result)
            return melt, freeze
        else:
            melt = np.zeros(num_time)
            for t in range(num_time):
                melt[t] = total_melt(ismr[t,:], grid.fris_mask, grid, result=result)
            return melt


# Read the given lat x lon variable from the given NetCDF file, and calculate timeseries of its maximum value in the given region.
def timeseries_max (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, xmin=None, xmax=None, ymin=None, ymax=None):

    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)

    if time_index is not None or time_average:
        return var_min_max(data, grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)[1]
    else:
        num_time = data.shape[0]
        max_data = np.zeros(num_time)
        for t in range(num_time):
            max_data[t] = var_min_max(data[t,:], grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)[1]
        return max_data


# Helper function for timeseries_avg_sfc and timeseries_int_sfc.
def timeseries_area_sfc (option, file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False):
    
    # Read the data
    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data.shape[0]):
        # Mask
        data_tmp = mask_land_ice(data[t,:], grid, gtype=gtype)
        # Area-average or integrate
        timeseries.append(over_area(option, data_tmp, grid, gtype=gtype))
    return np.array(timeseries)


# Read the given lat x lon variable from the given NetCDF file, and calculate timeseries of its area-averaged value over the sea surface.
def timeseries_avg_sfc (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False):
    return timeseries_area_sfc('average', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)


# Like timeseries_avg_sfc, but for area-integrals over the sea surface.
def timeseries_int_sfc (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False):
    return timeseries_area_sfc('integrate', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)


# Read the given 3D variable from the given NetCDF file, and calculate timeseries of its volume-averaged value. This can be restricted to the FRIS cavity if you set fris=True.
def timeseries_avg_3d (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, fris=False):

    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data.shape[0]):
        # Mask everything except FRIS out of the array
        data_tmp = mask_except_fris(data[t,:], grid, gtype=gtype, depth_dependent=True)
        # Volume average
        timeseries.append(volume_average(data_tmp, grid, gtype=gtype))
    return np.array(timeseries)


# Calculate timeseries from one or more files.

# Arguments:
# file_path: either a single filename or a list of filenames

# Optional keyword arguments:
# option: 'fris_melt': calculates total melting and freezing beneath FRIS
#          'max': calculates maximum value of variable in region; must specify var_name and possibly xmin etc.
#          'avg_sfc': calculates area-averaged value over the sea surface, i.e. not counting cavities
#          'int_sfc': calculates area-integrated value over the sea surface
#          'avg_fris': calculates volume-averaged value in the FRIS cavity
# grid: as in function read_plot_latlon
# gtype: as in function read_plot_latlon
# var_name: variable name to process. Only matters for 'max', 'avg_sfc', 'int_sfc', and 'avg_fris'.
# xmin, xmax, ymin, ymax: as in function var_min_max
# monthly: as in function netcdf_time

# Output:
# if option='fris_melt', returns three 1D arrays of time, melting, and freezing.
# if option='max', 'avg_sfc', or 'avg_fris', returns two 1D arrays of time and the relevant timeseries.
# if option='time', just returns the time array.

def read_timeseries (file_path, option=None, grid=None, gtype='t', var_name=None, xmin=None, xmax=None, ymin=None, ymax=None, monthly=True):

    if isinstance(file_path, str):
        # Just one file
        first_file = file_path
    elif isinstance(file_path, list):
        # More than one
        first_file = file_path[0]
    else:
        print 'Error (read_timeseries): file_path must be a string or a list'
        sys.exit()

    if option in ['max', 'avg_sfc', 'int_sfc', 'avg_fris'] and var_name is None:
        print 'Error (read_timeseries): must specify var_name'
        sys.exit()

    # Build the grid if needed
    if option != 'time':
        grid = choose_grid(grid, first_file)

    # Calculate timeseries on the first file
    if option == 'fris_melt':
        melt, freeze = timeseries_fris_melt(first_file, grid, mass_balance=True)
    elif option == 'max':
        values = timeseries_max(first_file, var_name, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    elif option == 'avg_sfc':
        values = timeseries_avg_sfc(first_file, var_name, grid, gtype=gtype)
    elif option == 'int_sfc':
        values = timeseries_int_sfc(first_file, var_name, grid, gtype=gtype)
    elif option == 'avg_fris':
        values = timeseries_avg_3d(first_file, var_name, grid, gtype=gtype, fris=True)
    elif option != 'time':
        print 'Error (read_timeseries): invalid option ' + option
        sys.exit()
    # Read time axis
    time = netcdf_time(first_file, monthly=monthly)
    if isinstance(file_path, list):
        # More files to read
        for file in file_path[1:]:
            if option == 'fris_melt':
                melt_tmp, freeze_tmp = timeseries_fris_melt(file, grid, mass_balance=True)
            elif option == 'max':
                values_tmp = timeseries_max(file, var_name, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            elif option == 'avg_sfc':
                values_tmp = timeseries_avg_sfc(file, var_name, grid, gtype=gtype)
            elif option == 'int_sfc':
                values_tmp = timeseries_int_sfc(file, var_name, grid, gtype=gtype)
            elif option == 'avg_fris':
                values_tmp = timeseries_avg_3d(file, var_name, grid, gtype=gtype, fris=True)
            time_tmp = netcdf_time(file, monthly=monthly)
            # Concatenate the arrays
            if option == 'fris_melt':
                melt = np.concatenate((melt, melt_tmp))
                freeze = np.concatenate((freeze, freeze_tmp))
            elif option in ['max', 'avg_sfc', 'int_sfc', 'avg_fris']:
                values = np.concatenate((values, values_tmp))
            time = np.concatenate((time, time_tmp))

    if option == 'fris_melt':
        return time, melt, freeze
    elif option in ['max', 'avg_sfc', 'int_sfc', 'avg_fris']:
        return time, values
    elif option == 'time':
        return time


# Helper function to calculate difference timeseries, trimming if needed.

# Arguments:
# time_1, time_2: 1D arrays containing time values for the two simulations (assumed to start at the same time, but might not be the same length)
# data_1, data_2: 1D arrays containing timeseries for the two simulations

# Output:
# time: 1D array containing time values for the overlapping period of simulation
# data_diff: 1D array containing differences (data_2 - data_1) at these times
def trim_and_diff (time_1, time_2, data_1, data_2):

    num_time = min(time_1.size, time_2.size)
    time = time_1[:num_time]
    data_diff = data_2[:num_time] - data_1[:num_time]
    return time, data_diff


# Call read_timeseries twice, for two simulations, and calculate the difference in the timeseries. Doesn't work for the complicated case of timeseries_fris_melt.
def read_timeseries_diff (file_path_1, file_path_2, option=None, var_name=None, grid=None, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, monthly=True):

    if option == 'fris_melt':
        print "Error (read_timeseries_diff): this function can't be used for option="+option
        sys.exit()

    # Calculate timeseries for each
    time_1, values_1 = read_timeseries(file_path_1, option=option, var_name=var_name, grid=grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, monthly=monthly)
    time_2, values_2 = read_timeseries(file_path_2, option=option, var_name=var_name, grid=grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, monthly=monthly)
    # Find the difference, trimming if needed
    time, values_diff = trim_and_diff(time_1, time_2, values_1, values_2)
    return time, values_diff


# Set a bunch of parameters corresponding to a given timeseries variable.
def set_parameters (var):

    xmin = None
    xmax = None
    ymin = None
    ymax = None

    if var == 'fris_melt':
        option = 'fris_melt'
        var_name = 'SHIfwFlx'
        title = 'Basal mass balance of FRIS'
        units = 'Gt/y'
    elif var in ['hice_corner', 'mld_ewed']:
        # Maximum between spatial bounds
        option = 'max'
        if var == 'hice_corner':
            var_name = 'SIheff'
            xmin = -62
            xmax = -59.5
            ymin = -75.5
            ymax = -74
            title = 'Maximum sea ice thickness in problematic corner'
            units = 'm'
        elif var == 'mld_ewed':
            var_name = 'MXLDEPTH'
            xmin = -30
            xmax = 30
            ymin = -69
            ymax = -60
            title = 'Maximum mixed layer depth in Eastern Weddell'
            units = 'm'
    elif var == 'eta_avg':
        option = 'avg_sfc'
        var_name = 'ETAN'
        title = 'Area-averaged sea surface height'
        units = 'm'
    elif var == 'seaice_area':
        option = 'int_sfc'
        var_name = 'SIarea'
        title = 'Total sea ice area'
        units = r'million km$^2$'
    elif var in ['fris_temp', 'fris_salt']:
        option = 'avg_fris'
        if var == 'fris_temp':
            var_name = 'THETA'
            title = 'Volume-averaged temperature in FRIS cavity'
            units = r'$^{\circ}$C'
        elif var == 'fris_salt':
            var_name = 'SALT'
            title = 'Volume-averaged salinity in FRIS cavity'
            units = 'psu'
    else:
        print 'Error (set_parameters): invalid variable ' + var
        sys.exit()

    return option, var_name, title, units, xmin, xmax, ymin, ymax
