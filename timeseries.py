from io import read_netcdf
from utils import convert_ismr
from diagnostics import total_melt

# FUNCTION fris_melt

# Calculate total mass loss or area-averaged melt rate from FRIS in the given NetCDF file. The default behaviour is to calculate the melt at each time index in the file, but you can also select a subset of time indices, and/or time-average - see optional keyword arguments.

# Arguments:
# file_path: path to NetCDF file containing 'SHIfwFlx' variable
# grid = Grid object

# Optional keyword arguments:
# result: 'massloss' (default) calculates the total mass loss in Gt/y. 'meltrate' calculates the area-averaged melt rate in m/y.
# time_index, t_start, t_end, time_average: as in the read_netcdf function

# Output:
# If time_index is set, or time_average=True: single value containing melt rate
# Otherwise: 1D array containing timeseries of melt rate

def fris_melt (file_path, grid, result='massloss', time_index=None, t_start=None, t_end=None, time_average=False):

    # Read ice shelf melt rate and convert to m/y
    ismr = convert_ismr(read_netcdf(file_path, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))
    
    if time_index is not None or time_average:
        # Calculate total melt at each timestep
        num_time = ismr.shape[0]
        melt = np.zeros(num_time)
        for t in range(num_time):
            melt[t] = calc_melt(ismr[t,:], dA, grid.fris_mask, result=result)
    else:
        # Just one timestep
        melt = calc_melt(ismr, dA, grid.fris_mask, result=result)

    return melt
