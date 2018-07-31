#######################################################
# Files to create when the simulation is done
#######################################################

import os
import sys

from grid import Grid
from file_io import NCfile, netcdf_time, find_time_index
from plot_1d import read_plot_timeseries, read_plot_timeseries_diff
from plot_latlon import read_plot_latlon, plot_aice_minmax, read_plot_latlon_diff
from plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff
from utils import real_dir
from plot_utils.labels import parse_date

from MITgcmutils import rdmds


# Helper function to build lists of output files in a directory.
# If unravelled=True, it will look for filenames of the form 1979.nc, etc. If unravelled=False, it will look for filenames of the form output_001.nc, etc.
def build_file_list (output_dir, unravelled=False):

    output_files = []
    for file in os.listdir(output_dir):
        if (unravelled and file[0] in ['1', '2'] and file.endswith('.nc')) or (not unravelled and file.startswith('output_') and file.endswith('.nc')):
            output_files.append(output_dir+file)
    # Make sure in chronological order
    output_files.sort()
    return output_files


# Make a bunch of plots when the simulation is done.
# This will keep evolving over time!

# Optional keyword arguments:
# output_dir: path to directory containing output NetCDF files (assumed to be in one file per segment a la scripts/convert_netcdf.py)
# grid_path: path to binary grid directory, or NetCDF file containing grid variables
# fig_dir: path to directory to save figures in
# file_path: specific output file to analyse for non-time-dependent plots (default the most recent segment)
# monthly: as in function netcdf_time
# unravelled: set to True if the simulation is done and you've run netcdf_finalise.sh, so the files are 1979.nc, 1980.nc, etc. instead of output_001.nc, output_002., etc.

def plot_everything (output_dir='.', grid_path='../grid/', fig_dir='.', file_path=None, monthly=True, date_string=None, time_index=-1, time_average=False, unravelled=False):

    if time_average:
        time_index = None

    # Make sure proper directories
    output_dir = real_dir(output_dir)
    fig_dir = real_dir(fig_dir)
    
    # Build the list of output files in this directory (use them all for timeseries)
    output_files = build_file_list(output_dir, unravelled=unravelled)
    if file_path is None:
        # Select the last file for single-timestep analysis
        file_path = output_files[-1]        

    # Build the grid
    grid = Grid(grid_path)

    # Timeseries
    var_names = ['fris_melt', 'hice_corner', 'mld_ewed', 'eta_avg', 'seaice_area']
    for var in var_names:
        read_plot_timeseries(var, file_path, grid=grid, fig_name=fig_dir+'timeseries_'+var+'.png', monthly=monthly)

    # Lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'hsnow', 'mld', 'eta', 'saltflx', 'tminustf', 'vel', 'velice']
    for var in var_names:
        # Customise bounds and zooming
        vmin = None
        vmax = None
        zoom_fris = False
        fig_name = fig_dir + var + '.png'
        if var == 'bwtemp':
            vmax = 1
        if var == 'bwsalt':
            vmin = 34.3
        if var == 'eta':
            vmin = -2.5
        if var == 'hice':
            vmax = 4
        if var == 'saltflx':
            vmin = -0.001
            vmax = 0.001
        if var == 'tminustf':
            vmax = 1.5
            zoom_fris = True
            fig_name = fig_dir + var + '_min.png'
        if zoom_fris:
            figsize = (8,6)
        else:
            figsize = (10,6)
        # Plot
        read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_name, date_string=date_string, figsize=figsize)
        # Make additional plots if needed
        if var in ['ismr', 'vel', 'bwtemp', 'bwsalt']:
            # Make another plot zoomed into FRIS
            figsize = (8,6)
            # First adjust bounds
            if var == 'bwtemp':
                vmax = -1.5
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vmin=vmin, vmax=vmax, zoom_fris=True, fig_name=fig_dir+var+'_zoom.png', date_string=date_string, figsize=figsize)
        if var == 'tminustf':
            # Call the other options for vertical transformations
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, tf_option='max', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_max.png', date_string=date_string, figsize=figsize)
        if var == 'vel':
            # Call the other options for vertical transformations
            figsize = (10,6)
            for vel_option in ['sfc', 'bottom']:
                read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vel_option=vel_option, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_'+vel_option+'.png', date_string=date_string, figsize=figsize)
        if var in ['eta', 'hice']:
            # Make another plot with unbounded colour bar
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, zoom_fris=zoom_fris, fig_name=fig_dir + var + '_unbound.png', date_string=date_string, figsize=figsize)

    # Slice plots
    read_plot_ts_slice(file_path, grid=grid, lon0=-40, hmax=-75, zmin=-1450, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_filchner.png', date_string=date_string)
    read_plot_ts_slice(file_path, grid=grid, lon0=-55, hmax=-72, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_ronne.png', date_string=date_string)
    read_plot_ts_slice(file_path, grid=grid, lon0=-25, zmin=-2000, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_eweddell.png', date_string=date_string)


# Compare one simulation to another. Assumes the simulations have monthly averaged output. They don't need to be the same length.

# Keyword arguments:
# output_dir: as in function plot_everything
# baseline_dir: like output_dir, but for the simulation you want to use as a baseline. All the plots will show results from output_dir minus baseline_dir. This is the only non-optional keyword argument; it is a named keyword argument with no meaningful default so there's no chance of the user mixing up which simulation is which.
# grid_path: as in function plot_everything
# fig_dir: as in function plot_everything
# option: either 'last_year' (averages over the last 12 months of the overlapping period of the simulations) or 'last_month' (just considers the last month of the overlapping period).
# unravelled: as in function plot_everything

def plot_everything_diff (output_dir='./', baseline_dir=None, grid_path='../grid/', fig_dir='.', option='last_year', unravelled=False):

    # Check that baseline_dir is set
    # It's a keyword argument on purpose so that the user can't mix up which simulation is which.
    if baseline_dir is None:
        print 'Error (plot_everything_diff): must set baseline_dir'
        sys.exit()

    # Make sure proper directories, and rename so 1=baseline, 2=current
    output_dir_1 = real_dir(baseline_dir)
    output_dir_2 = real_dir(output_dir)    
    fig_dir = real_dir(fig_dir)

    # Build lists of output files in each directory
    output_files_1 = build_file_list(baseline_dir, unravelled=unravelled)
    output_files_2 = build_file_list(output_dir, unravelled=unravelled)

    # Build the grid
    grid = Grid(grid_path)

    # Timeseries through the entire simulation
    var_names = ['fris_melt', 'hice_corner', 'mld_ewed', 'eta_avg', 'seaice_area']
    for var in var_names:
        read_plot_timeseries_diff(var, output_files_1, output_files_2, grid=grid, fig_name=fig_dir+'timeseries_'+var+'.png', monthly=monthly)

    # Now figure out which time indices to use for plots with no time dependence
    # Concatenate the time arrays from all files
    time_1 = read_timeseries(output_files_1, option='time')
    time_2 = read_timeseries(output_files_2, option='time')
    # Find the last time index in the shortest simulation
    time_index = min(time_1.size, time_2.size) - 1
    file_path_1, time_index_1 = find_time_index(output_files_1, time_index)
    file_path_2, time_index_2 = find_time_index(output_files_2, time_index)
    # Make sure we got this right
    if netcdf_time(file_path_1)[time_index_1] != netcdf_time(file_path_2)[time_index_2]:
        print 'Error (plot_everything_diff): something went wrong when matching time indices between the two files.'
        sys.exit()
    if option == 'last_year':
        # Add 1 to get the upper bound on the time range we care about
        t_end_1 = time_index_1 + 1
        t_end_2 = time_index_2 + 1
        # Find the index 12 before that
        t_start_1 = t_end_1 - 12
        t_start_2 = t_end_2 - 12
        # Make sure it's still contained within one file
        if t_start_1 < 0 or t_start_2 < 0:
            print "Error (plot_everything_diff): option last_year doesn't work if that year isn't contained within one file, for both simulations."
            sys.exit()
        # Set the other options
        time_index_1 = None
        time_index_2 = None
        time_average = True
        # Set date string
        date_string = 'year beginning ' + parse_date(file_path=file_path_1, time_index=t_start_1)
    elif option == 'last_month':
        # Set the other options
        t_start_1 = None
        t_start_2 = None
        t_end_1 = None
        t_end_2 = None
        time_average = False
        # Set date string
        date_string = parse_date(file_path=file_path_1, time_index=time_index_1)
    else:
        print 'Error (plot_everything_diff): invalid option ' + option
        sys.exit()

    # Now make lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'hsnow', 'mld', 'eta', 'vel', 'velice']
    for var in var_names:
        read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+var+'_diff.png', figsize=(10,6))
        # Zoom into some variables
        if var in ['ismr', 'bwtemp', 'bwsalt', 'vel']:
            read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, zoom_fris=True, date_string=date_string, fig_name=fig_dir+var+'_zoom_diff.png')
        if var == 'vel':
            # Call the other options for vertical transformations
            for vel_option in ['sfc', 'bottom']:
                read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, vel_option=vel_option, date_string=date_string, fig_name=fig_dir+var+'_'+vel_option+'_diff.png')

    # Slice plots
    read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-40, hmax=-75, zmin=-1450, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_filchner_diff.png')
    read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-55, hmax=-72, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_ronne_diff.png')
    read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-25, zmin=-2000, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_eweddell_diff.png')    
    


# Plot the sea ice annual min and max for each year of the simulation. First you have to concatenate the sea ice area into a single file, such as:
# ncrcat -v SIarea output_*.nc aice_tot.nc

# Arguments:
# file_path: path to concatenated NetCDF file with sea ice area for the entire simulation

# Optional keyword arguments:
# grid_path: as in function plot_everything
# fig_dir: path to directory to save figures in
# monthly: as in function netcdf_time

def plot_seaice_annual (file_path, grid_path='../grid/', fig_dir='.', monthly=True):

    fig_dir = real_dir(fig_dir)

    grid = Grid(grid_path)

    time = netcdf_time(file_path, monthly=monthly)
    first_year = time[0].year
    if time[0].month > 2:
        first_year += 1
    last_year = time[-1].year
    if time[-1].month < 8:
        last_year -= 1
    for year in range(first_year, last_year+1):
        plot_aice_minmax(file_path, year, grid=grid, fig_name=fig_dir+'aice_minmax_'+str(year)+'.png')
    


# When the model crashes, convert its crash-dump to a NetCDF file.

# Arguments:
# crash_dir: directory including all the state*crash.* files. The NetCDF file will be saved here too, with the name crash.nc.
# grid_path: as in function plot_everything

def crash_to_netcdf (crash_dir, grid_path):

    # Make sure crash_dir is a proper directory
    crash_dir = real_dir(crash_dir)

    # Read the grid
    grid = Grid(grid_path)
    # Initialise the NetCDF file
    ncfile = NCfile(crash_dir+'crash.nc', grid, 'xyz')

    # Find all the crash files
    for file in os.listdir(crash_dir):
        if file.startswith('stateThetacrash') and file.endswith('.data'):
            # Found temperature
            # Read it from binary
            temp = rdmds(crash_dir + file.replace('.data', ''))
            # Write it to NetCDF
            ncfile.add_variable('THETA', temp, 'xyz', units='C')
        if file.startswith('stateSaltcrash') and file.endswith('.data'):
            salt = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('SALT', salt, 'xyz', units='psu')
        if file.startswith('stateUvelcrash') and file.endswith('.data'):
            u = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('UVEL', u, 'xyz', gtype='u', units='m/s')
        if file.startswith('stateVvelcrash') and file.endswith('.data'):
            v = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('VVEL', v, 'xyz', gtype='v', units='m/s')
        if file.startswith('stateWvelcrash') and file.endswith('.data'):
            w = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('WVEL', w, 'xyz', gtype='w', units='m/s')
        if file.startswith('stateEtacrash') and file.endswith('.data'):
            eta = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('ETAN', eta, 'xy', units='m')
        if file.startswith('stateAreacrash') and file.endswith('.data'):
            area = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('SIarea', area, 'xy', units='fraction')
        if file.startswith('stateHeffcrash') and file.endswith('.data'):
            heff = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('SIheff', heff, 'xy', units='m')
        if file.startswith('stateUicecrash') and file.endswith('.data'):
            uice = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('SIuice', uice, 'xy', gtype='u', units='m/s')
        if file.startswith('stateVicecrash') and file.endswith('.data'):
            vice = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('SIvice', vice, 'xy', gtype='v', units='m/s')
        if file.startswith('stateQnetcrash') and file.endswith('.data'):
            qnet = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('Qnet', qnet, 'xy', units='W/m^2')
        if file.startswith('stateMxlcrash') and file.endswith('.data'):
            mld = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('MXLDEPTH', mld, 'xy', units='m')
        if file.startswith('stateEmpmrcrash') and file.endswith('.data'):
            empmr = rdmds(crash_dir + file.replace('.data', ''))
            ncfile.add_variable('Empmr', empmr, 'xy', units='kg/m^2/s')

    ncfile.close()
