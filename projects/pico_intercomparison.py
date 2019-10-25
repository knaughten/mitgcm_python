##################################################################
# UaMITgcm / UaPICO intercomparison
##################################################################

import numpy as np
import netCDF4 as nc
import shutil

from ..file_io import read_netcdf
from ..interpolation import discard_and_fill
from ..plot_ua import read_ua_mesh
from ..grid import SOSEGrid
from ..utils import real_dir, apply_mask, select_bottom
from ..calculus import area_average
from ..plot_utils.labels import round_to_decimals

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


# Calculate the monthly climatology of B-SOSE's bottom temperature and salinity on the continental shelf in front of FRIS, as inputs to PICO. The results (24 values) will be printed on screen and also written to an ASCII file.
def bsose_pico_input (bsose_dir, out_file):

    var_names = ['THETA', 'SALT']
    file_tail = '_climatology.nc'
    bsose_dir = real_dir(bsose_dir)

    print 'Building B-SOSE grid'
    grid = SOSEGrid(bsose_dir+'grid.nc')

    f = open(out_file, 'w')
    for i in range(2):
        print 'Processing ' + var_names[i]
        f.write(var_names[i]+'\n')
        # Read the data for all months
        data = grid.read_field(bsose_dir+var_names[i]+file_tail, 'xyzt', var_name=var_names[i])        
        # Select the bottom layer
        data = select_bottom(data, masked=False, grid=grid, time_dependent=True)
        # Mask out everything except the SWS continental shelf
        data = apply_mask(data, np.invert(grid.sws_shelf_mask), time_dependent=True)
        # Area-average
        data = area_average(data, grid, time_dependent=True)
        for t in range(12):
            data_val = round_to_decimals(data[t], 2)
            print data_val
            f.write(data_val+'\n')
    f.close()
    print 'Data saved in ' + out_file
