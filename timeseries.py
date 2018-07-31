#######################################################
# Calculation of integral timeseries
#######################################################

import numpy as np

from file_io import read_netcdf
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

def fris_melt (file_path, grid, result='massloss', time_index=None, t_start=None, t_end=None, time_average=False, mass_balance=False):

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

    
    # Figure out if there's a time dimension
    #time_dependent = time_index is None and not time_average
    # Mask
    #data = mask_land_ice(data, grid, gtype=gtype, time_dependent=time_dependent)
    # Area-average or integrate
    #return over_area(option, data, grid, gtype=gtype, time_dependent=time_dependent)


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
    return timeseries
        
    
    #time_dependent = time_index is None and not time_average
    #if fris:
    #    # Mask everything except FRIS out of the array
    #    data = mask_except_fris(data, grid, gtype=gtype, time_dependent=time_dependent, depth_dependent=True)
    #return volume_average(data, grid, gtype=gtype, time_dependent=time_dependent)

    
