##################################################################
# UaMITgcm / UaPICO intercomparison
##################################################################

import numpy as np
import netCDF4 as nc
import shutil

from ..file_io import read_netcdf
from ..interpolation import discard_and_fill
from ..plot_ua import read_ua_mesh


# Given the Moholdt basal melt rate data, extend into the mask until the entire Ua domain is covered.
def extend_moholdt_data (old_file, new_file, ua_mesh_file):

    var_name = 'basalMassBalance'
    # Make a copy of the original data
    shutil.copy(old_file, new_file)

    # Read melt data
    x = read_netcdf(new_file, 'x')
    y = read_netcdf(new_file, 'y')
    x, y = np.meshgrid(x,y)
    melt = read_netcdf(new_file, var_name)

    # Read Ua points
    x_ua, y_ua = read_ua_mesh(ua_mesh_file)[:2]
    # Find bounds
    xmin = np.amin(x_ua)
    xmax = np.amax(x_ua)
    ymin = np.amin(y_ua)
    ymax = np.amax(y_ua)

    # Extend melt data into mask until entire Ua region is covered
    discard = melt.mask
    fill = (x >= xmin)*(x <= xmax)*(y >= ymin)*(y <= ymax)
    melt = discard_and_fill(melt, discard, fill, use_3d=False)

    # Overwrite data
    id = nc.Dataset(new_file, 'a')
    id.variables[var_name][:] = melt
    id.close()
    
