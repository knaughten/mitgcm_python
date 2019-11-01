##################################################################
# UaMITgcm / UaPICO intercomparison
##################################################################

import numpy as np
import netCDF4 as nc
import shutil

from ..file_io import read_netcdf
from ..interpolation import discard_and_fill
from ..plot_ua import read_ua_mesh
from ..utils import real_dir, apply_mask, select_bottom, days_per_month
from ..calculus import area_average
from ..plot_utils.labels import round_to_decimals
from ..grid import WOAGrid

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


# Calculate the annual mean and monthly climatology of WOA 2018's bottom temperature and salinity on the continental shelf in front of FRIS, as inputs to PICO. The results (24 values) will be printed on screen and also written to an ASCII file.
def woa18_pico_input (woa_dir, out_file):

    file_head = 'woa18_decav_'
    file_tail = '_04.nc'
    var_names = ['t', 's']
    var_name_tail = '_an'
    var_names_long = ['Temperature', 'Salinity']
    woa_dir = real_dir(woa_dir)

    print 'Building WOA grid'
    # Use the January temperature file
    grid = WOAGrid(woa_dir + file_head + var_names[0] + '01' + file_tail)

    # Build array of days per month, with February as 28.25
    ndays = np.array([days_per_month(t+1, 1) for t in range(12)])
    ndays[1] += 0.25

    monthly_data = np.zeros([2, 12])
    annual_data = np.zeros(2)
    # Loop over variables
    for n in range(2):
        print 'Processing ' + var_names[n]
        # Loop over months
        for t in range(12):
            print 'Month ' + str(t+1)
            # Construct the filename
            file_path = woa_dir + file_head + var_names[n] + str(t+1).zfill(2) + file_tail
            # Read the data
            data = read_netcdf(file_path, var_names[n]+var_name_tail)
            # Select the bottom layer
            data = select_bottom(data)
            # Mask out everything except the SWS continental shelf
            data = apply_mask(data, np.invert(grid.sws_shelf_mask))
            # Area-average and save
            data = area_average(data, grid)
            # Save monthly data
            monthly_data[n,t] = data
            # Accumulate annual data
            annual_data[n] += data*ndays[t]/np.sum(ndays)

    # Save results to file
    f = open(out_file, 'w')
    for n in range(2):
        f.write(var_names_long[n]+'\n')
        f.write('Monthly values:\n')
        for t in range(12):
            f.write(round_to_decimals(monthly_data[n,t], 2)+'\n')
        f.write('Annual value: ')
        f.write(round_to_decimals(annual_data[n], 2)+'\n')
    f.close()

    # Print the contents of that file
    f = open(out_file, 'r')
    for line in f:
        print line
    f.close()    
    print 'Data saved in ' + out_file
