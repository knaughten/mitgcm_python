#######################################################
# File transfer between MITgcm and Ua
#######################################################

from scipy.io import loadmat

from grid import Grid
from file_io import read_netcdf, check_single_time
from utils import convert_ismr

def couple_ocn2ice (mit_file, ua_geom_file, ua_melt_file, time_average=True, time_index=None, t_start=None, t_end=None):

    # Make sure we'll end up with a single record in time
    check_single_time(time_index, time_average)

    # Build MITgcm grid
    grid = Grid(mit_file)
    # Read MITgcm melt rates and convert to m/y
    ismr = convert_ismr(read_netcdf(mit_file, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))

    
    
    
