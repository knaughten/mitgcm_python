##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys

from ..grid import Grid, UKESMGrid
from ..file_io import read_binary, find_cmip6_files, NCfile
from ..interpolation import interp_reg_xy

# Functions to build a katabatic wind correction file between UKESM and ERA5, following the method of Mathiot et al 2010.

# Read the daily wind output from UKESM's historical simulation, over the period 1979-2014, and time_average. Interpolate to the MITgcm grid and save the output to a NetCDF file.
# Run this function on JASMIN and then copy the output file to ARCHER.
def process_ukesm_wind (mit_grid_dir, out_file, model_path='/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/', ensemble_member='r1i1p1f2'):

    expt = 'historical'
    start_year = 1979
    end_year = 2014
    var_names_in = ['uas', 'vas']
    var_names_out = ['uwind', 'vwind']
    gtype = ['u', 'v']
    days_per_year = 12*30

    # Build grids
    ukesm_grid = UKESMGrid(start_year=start_year)
    # Make sure MITgcm longitude is in the range 0-360 to match UKESM
    model_grid = Grid(mit_grid_dir, max_lon=360)

    # Open NetCDF file
    ncfile = NCfile(out_file, model_grid, 'xy')

    for n in range(2):
        print 'Processing variable ' + var_names_out[n]

        # Read the data, time-integrating as we go
        in_files, start_years, end_years = find_cmip6_files(model_path, expt, ensemble_member, var_names_in[n], 'day')
        data = None
        num_time = 0
        # Loop over each file
        for t in range(len(in_files)):
            file_path = in_files[t]
            print 'Processing ' + file_path
            print 'Covers years ' + str(start_years[t]) + ' to ' + str(end_years[t])
            # Loop over years
            t_start = 0  # Time index in file
            t_end = t_start+days_per_year
            for year in range(start_years[t], end_years[t]+1):
                if years >= start_year and year <= end_year:
                    print 'Processing ' + str(year)
                    # Read data
                    print 'Reading ' + str(year) + ' from indicies ' + str(t_start) + '0' + str(t_end)
                    data_tmp = read_netcdf(file_path, var, t_start=t_start, t_end=t_end)
                    if data is None:
                        data = np.sum(data_tmp, axis=0)
                    else:
                        data += np.sum(data_tmp, axis=0)
                    num_time += days_per_year
                # Update time range for next time
                t_start = t_end
                t_end = t_start + days_per_year
        # Now convert from time-integral to time-average
        data /= num_time

        # Interpolate to MITgcm tracer grid
        ukesm_lon, ukesm_lat = ukesm_grid.get_lon_lat(gtype=gtype[n], dim=1)
        mit_lon, mit_lat = mit_grid.get_lon_lat(gtype='t', dim=1)
        print 'Interpolating'
        data_interp = interp_reg_xy(ukesm_lon, ukesm_lat, data, mit_lon, mit_lat)
        print 'Saving to ' + out_file
        ncfile.add_variable(var_names_out[n], data_interp, 'xy', units='m/s')

    ncfile.close()
        
