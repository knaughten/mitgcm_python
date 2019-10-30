##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid, UKESMGrid, ERA5Grid
from ..file_io import read_binary, find_cmip6_files, NCfile, read_netcdf
from ..interpolation import interp_reg_xy
from ..utils import fix_lon_range, split_longitude, real_dir
from ..plot_utils.windows import finished_plot

# Functions to build a katabatic wind correction file between UKESM and ERA5, following the method of Mathiot et al 2010.

# Read the daily wind output from either UKESM's historical simulation (option='UKESM') or ERA5 (option='ERA5') over the period 1979-2014, and time-average. Interpolate to the MITgcm grid and save the output to a NetCDF file.
def process_wind_forcing (option, mit_grid_dir, out_file, source_dir=None):

    start_year = 1979
    end_year = 2014
    var_names = ['uwind', 'vwind']
    if option == 'UKESM':
        if source_dir is None:
            source_dir = '/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/'
        expt = 'historical'
        ensemble_member = 'r1i1p1f2'
        var_names_in = ['uas', 'vas']
        gtype = ['u', 'v']
        days_per_year = 12*30
    elif option == 'ERA5':
        if source_dir is None:
            source_dir = '/work/n02/n02/shared/baspog/MITgcm/reanalysis/ERA5/'
        file_head = 'ERA5_'
        gtype = ['t', 't']
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
        num_time = 0
        
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
                data_tmp = read_binary(file_path, [forcing_grid.nx, forcing_grid.ny], 'xyt')
                if data is None:
                    data = np.sum(data_tmp, axis=0)
                else:
                    data += np.sum(data_tmp, axis=0)
                num_time += data_tmp.shape[0]

        # Now convert from time-integral to time-average
        data /= num_time
        
        # Get longitude in the range -180 to 180, then split and rearrange so it's monotonically increasing
        forcing_lon, forcing_lat = forcing_grid.get_lon_lat(gtype=gtype[n], dim=1)
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


# Analyse the coastal winds in UKESM vs ERA5:
#   1. Suggest possible caps on the ERA5/UKESM ratio
#   2. Make scatterplots of both components
#   3. Plot the wind vectors and their differences along the coast
def analyse_coastal_winds (grid_dir, ukesm_file, era5_file, save_fig=False, fig_dir='./'):

    fig_name = None
    fig_dir = real_dir(fig_dir)

    print 'Selecting coastal points'
    grid = Grid(grid_dir)
    coast_mask = grid.get_coast_mask(ignore_iceberg=True)
    var_names = ['uwind', 'vwind']

    ukesm_wind_vectors = []
    era5_wind_vectors = []
    for n in range(2):
        print 'Processing ' + var_names[n]
        # Read the data and select coastal points only
        ukesm_wind = (read_netcdf(ukesm_file, var_names[n])[coast_mask]).ravel()
        era5_wind = (read_netcdf(era5_file, var_names[n])[coast_mask]).ravel()
        ratio = np.abs(era5_wind/ukesm_wind)
        # Save this component
        ukesm_wind_vectors.append(ukesm_wind)
        era5_wind_vectors.append(era5_wind)

        print 'Analysing ratios'
        print 'Minimum ratio of ' + str(np.amin(ratio))
        print 'Maximum ratio of ' + str(np.amax(ratio))
        print 'Mean ratio of ' + str(np.mean(ratio))
        percent_exceed = np.empty(20)
        for i in range(20):
            percent_exceed[i] = float(np.count_nonzero(ratio > i+1))/ratio.size*100
        # Find first value of ratio which includes >90% of points
        i_cap = np.nonzero(percent_exceed < 10)[0][0] + 1
        print 'A ratio cap of ' + str(i_cap) + ' will cover ' + str(100-percent_exceed[i_cap]) + '%  of points'
        # Plot the percentage of points that exceed each threshold ratio
        fig, ax = plt.subplots()
        ax.plot(np.arange(20)+1, percent_exceed, color='blue')
        ax.axhline(y=10, color='red')
        plt.xlabel('Ratio', fontsize=16)
        plt.ylabel('%', fontsize=16)
        plt.title('Percentage of points exceeding given ratios', fontsize=18)
        if save_fig:
            fig_name = fig_dir + 'ratio_caps.png'
        finished_plot(fig, fig_name=fig_name)

        print 'Making scatterplot'
        fig, ax = plt.subplots()
        ax.scatter(era5_wind, ukesm_wind, color='blue')
        # Plot the y=x diagonal line in red
        xlim = np.array(ax.get_xlim())
        ax.plot(xlim, xlim, color='red')
        # Plot the ratio cap in green
        ax.plot(xlim, i_cap*xlim, color='green')
        plt.xlabel('ERA5', fontsize=16)
        plt.ylabel('UKESM', fontsize=16)
        plt.title(var_names[n] + ' (m/s) at coastal points, 1979-2014 mean', fontsize=18)
        # Construct figure name, if needed
        if save_fig:
            fig_name = fig_dir + 'scatterplot_' + var_names[n] + '.png'
        finished_plot(fig, fig_name=fig_name)

        
            
        
