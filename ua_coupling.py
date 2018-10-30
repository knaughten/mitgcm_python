#######################################################
# File transfer between MITgcm and Ua
#######################################################

import numpy as np
from scipy.io import savemat

from file_io import read_netcdf, check_single_time
from utils import convert_ismr


# Pass data from MITgcm to Ua (melt rates).
# All we need to do is extract melt rates from MITgcm's output, convert the units, and write to a .mat file. We don't even need to interpolate them to the Ua nodes, as Ua does this at runtime.

# Arguments:
# mit_file: path to MITgcm NetCDF file containing melt rates (SHIfwFlx).
# ua_out_file: desired path to .mat file for Ua to read melt rates from.

# Optional keyword arguments:
# time_average, time_index, t_start, t_end: Time index options to read the melt rates from mit_file, as in function read_netcdf.

def couple_ocn2ice (mit_file, ua_out_file, time_average=False, time_index=None, t_start=None, t_end=None):

    # Make sure we'll end up with a single record in time
    check_single_time(time_index, time_average)

    # Read MITgcm grid
    lon = read_netcdf(mit_file, 'LONGITUDE')
    lat = read_netcdf(mit_file, 'LATITUDE')
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    
    # Read MITgcm melt rates and convert to m/y
    ismr = convert_ismr(read_netcdf(mit_file, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))

    # Put everything in exactly the format that Ua wants: long 1D arrays with an empty second dimension, and double precision
    lon_points = np.ravel(lon_2d)[:,None].astype('float64')
    lat_points = np.ravel(lat_2d)[:,None].astype('float64')
    ismr_points = np.ravel(ismr)[:,None].astype('float64')

    # Write to Matlab file for Ua, as long 1D arrays
    savemat(ua_out_file, {'meltrate':ismr_points, 'x':lon_points, 'y':lat_points})
    

    
    
    
