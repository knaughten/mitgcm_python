##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys

from ..grid import Grid, UKESMGrid, ERA5Grid
from ..file_io import read_binary, find_cmip6_files, NCfile, read_netcdf
from ..interpolation import interp_reg_xy
from ..utils import fix_lon_range, split_longitude, real_dir

# Functions to build a katabatic wind correction file between UKESM and ERA5, following the method of Mathiot et al 2010.

# Read the daily wind output from either UKESM's historical simulation (option='UKESM') or ERA5 (option='ERA5') over the period 1979-2014, and time-average. Interpolate to the MITgcm grid and save the output to a NetCDF file.
def process_wind_forcing (option, mit_grid_dir, out_file, source_dir=None):

    start_year = 1979
    end_year = 2014
    var_names = ['uwind', 'vwind']
    if option == 'UKESM':
        if source_dir is None:
            source_dir = '/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/'
        ensemble_member = 'r1i1p1f2'
        var_names_in = ['uas', 'vas']
        gtype = ['u', 'v']
        days_per_year = 12*30
    elif option == 'ERA5':
        if source_dir is None:
            source_dir = '/work/n02/n02/shared/baspog/MITgcm/reanalysis/ERA5/'
        file_head = 'ERA5_'
    else:
        print 'Error (process_wind_forcing); invalid option ' + option
        sys.exit()

    mit_grid_dir = real_dir(mit_grid_dir)
    source_dir = real_dir(source_dir)

    print 'Building grids'
    if option == 'UKESM':
        forcing_grid = UKESMGrid()
    elif option == 'ERA5':
        forcing_grid = ERA5Grid()
    mit_grid = Grid(mit_grid_dir)

    # Open NetCDF file
    ncfile = NCfile(out_file, mit_grid, 'xy')

    # Loop over variables
    for n in range(2):
        print 'Processing variable ' + var_names[n]
        # Read the data, time-integrating as we go
        data = None
        num_time = None
        
        if option == 'UKESM':
            in_files, start_years, end_years = find_cmip6_files(source_dir, expt, ensemble_member, var_names_in[n], 'day')
            # Loop over each file
            for t in range(len(in_files)):
                file_path = in_files[t]
                print 'Processing ' + file_path
                print 'Covers years ' + str(start_years[t]) + ' to ' + str(end_years[t])
                # Loop over years
                t_start = 0  # Time index in file
                t_end = t_start+days_per_year
                for year in range(start_years[t], end_years[t]+1):
                    if year >= start_year and year <= end_year:
                        print 'Processing ' + str(year)
                        # Read data
                        print 'Reading ' + str(year) + ' from indices ' + str(t_start) + '-' + str(t_end)
                        data_tmp = read_netcdf(file_path, var_names_in[n], t_start=t_start, t_end=t_end)
                        if data is None:
                            data = np.sum(data_tmp, axis=0)
                        else:
                            data += np.sum(data_tmp, axis=0)
                        num_time += days_per_year
                    # Update time range for next time
                    t_start = t_end
                    t_end = t_start + days_per_year        

        elif option == 'ERA5':
            # Loop over years
            for year in range(start_year, end_year+1):
                file_path = source_dir + file_head + var_names[n] + '_' + str(year)
                data_tmp = read_binary(file_path, [era5_grid.nx, era5_grid.ny], 'xyt')
                if data is None:
                    data = np.sum(data_tmp, axis=0)
                else:
                    data += np.sum(data_tmp, axis=0)
                num_time += data_tmp.shape[0]

        # Now convert from time-integral to time-average
        data /= num_time
        
        # Get longitude in the range -180 to 180, then split and rearrange so it's monotonically increasing
        forcing_lon, forcing_lat = forcing_grid.get_lon_lat(dim=1)
        forcing_lon = fix_lon_range(forcing_lon)
        i_split = np.nonzero(forcing_lon < 0)[0][0]
        forcing_lon = split_longitude(forcing_lon, i_split)
        data = split_longitude(data, i_split)
        # Now interpolate to MITgcm tracer grid        
        mit_lon, mit_lat = mit_grid.get_lon_lat(gtype='t', dim=1)
        print 'Interpolating'
        data_interp = interp_reg_xy(forcing_lon, forcing_lat, data, mit_lon, mit_lat)
        print 'Saving to ' + out_file
        ncfile.add_variable(var_names[n], data_interp, 'xy', units='m/s')

    ncfile.close()
