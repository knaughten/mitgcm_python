#######################################################
# Files to create when the simulation is done
#######################################################

import os
import sys
import numpy as np
import shutil
import netCDF4 as nc

from grid import Grid
from file_io import NCfile, netcdf_time, find_time_index, read_netcdf
from timeseries import calc_timeseries, calc_special_timeseries, set_parameters
from plot_1d import read_plot_timeseries, read_plot_timeseries_diff
from plot_latlon import read_plot_latlon, plot_aice_minmax, read_plot_latlon_diff
from plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff
from utils import real_dir, days_per_month
from plot_utils.labels import parse_date


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
# timeseries_file: filename created by precompute_timeseries, within output_dir
# grid_path: path to binary grid directory, or NetCDF file containing grid variables
# fig_dir: path to directory to save figures in
# file_path: specific output file to analyse for non-time-dependent plots (default the most recent segment)
# monthly: as in function netcdf_time
# unravelled: set to True if the simulation is done and you've run netcdf_finalise.sh, so the files are 1979.nc, 1980.nc, etc. instead of output_001.nc, output_002., etc.

def plot_everything (output_dir='.', timeseries_file='timeseries.nc', grid_path='../grid/', fig_dir='.', file_path=None, monthly=True, date_string=None, time_index=-1, time_average=False, unravelled=False, key='WSS'):

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
    var_names = ['fris_mass_balance', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt', 'fris_age']
    for var in var_names:
        read_plot_timeseries(var, output_dir+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_'+var+'.png', monthly=monthly)

    # Lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'hsnow', 'mld', 'eta', 'saltflx', 'vel', 'velice', 'psi', 'bwage', 'iceprod']
    for var in var_names:
        # Customise bounds and zooming
        vmin = None
        vmax = None
        zoom_fris = False
        chunk = None
        fig_name = fig_dir + var + '.png'
        if var == 'bwtemp':
            if key == 'WSS':
                vmin = -2.5
                vmax = -1.5
            elif key == 'WSK':
                vmax = 1
        if var == 'bwsalt':
            vmin = 34.3
        if var == 'bwage':
            vmin = 0
            if key == 'WSS':
                vmax = 12
        if var == 'eta':
            vmin = -2.5
        if var == 'hice':
            vmax = 4
        if var == 'saltflx':
            vmin = -0.001
            vmax = 0.001
        if var == 'iceprod':
            vmin = 0
            vmax = 5
        if var == 'psi' and key=='WSS':
            vmin = -0.5
            vmax = 0.5
        if var in ['vel', 'velice'] and key=='WSS':
            chunk = 6
        if not zoom_fris and key=='WSK':
            figsize = (10,6)
        else:
            figsize = (8,6)
        # Plot
        read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_name, date_string=date_string, figsize=figsize, chunk=chunk)
        # Make additional plots if needed
        if key=='WSK' and var in ['ismr', 'vel', 'bwtemp', 'bwsalt', 'psi', 'bwage']:
            # Make another plot zoomed into FRIS
            figsize = (8,6)
            # First adjust bounds
            if var == 'bwtemp':
                vmax = -1.5
            if var == 'bwage':
                vmax = 10
            if var == 'psi':
                vmax = 0.5
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vmin=vmin, vmax=vmax, zoom_fris=True, fig_name=fig_dir+var+'_zoom.png', date_string=date_string, figsize=figsize)
        if var == 'vel':
            # Call the other options for vertical transformations
            if key=='WSK':
                figsize = (10,6)
            for vel_option in ['sfc', 'bottom']:
                read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vel_option=vel_option, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_'+vel_option+'.png', date_string=date_string, figsize=figsize, chunk=chunk)
        if var in ['eta', 'hice']:
            # Make another plot with unbounded colour bar
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, zoom_fris=zoom_fris, fig_name=fig_dir + var + '_unbound.png', date_string=date_string, figsize=figsize)

    # Slice plots
    read_plot_ts_slice(file_path, grid=grid, lon0=-40, hmax=-75, zmin=-1450, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_filchner.png', date_string=date_string)
    read_plot_ts_slice(file_path, grid=grid, lon0=-55, hmax=-72, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_ronne.png', date_string=date_string)
    if key == 'WSK':
        read_plot_ts_slice(file_path, grid=grid, lon0=0, hmax=-71, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_eweddell.png', date_string=date_string)


# Given lists of files from two simulations, find the file and time indices corresponding to the last year (if option='last_year') or last month/timestep (if option='last_month') in the shortest simulation.
def select_common_time (output_files_1, output_files_2, option='last_year', monthly=True):

    if not monthly and option == 'last_year':
        print 'Error (select_common_time): need monthly output to correctly select the last year.'
        sys.exit()

    # Concatenate the time arrays from all files
    time_1 = calc_timeseries(output_files_1, option='time')
    time_2 = calc_timeseries(output_files_2, option='time')
    # Find the last time index in the shortest simulation
    time_index = min(time_1.size, time_2.size) - 1
    file_path_1, time_index_1 = find_time_index(output_files_1, time_index)
    file_path_2, time_index_2 = find_time_index(output_files_2, time_index)
    # Make sure we got this right
    if netcdf_time(file_path_1, monthly=monthly)[time_index_1] != netcdf_time(file_path_2, monthly=monthly)[time_index_2]:
        print 'Error (select_common_time): something went wrong when matching time indices between the two files.'
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
            print "Error (select_common_time): option last_year doesn't work if that year isn't contained within one file, for both simulations."
            sys.exit()
        # Set the other options
        time_index_1 = None
        time_index_2 = None
        time_average = True
    elif option == 'last_month':
        # Set the other options
        t_start_1 = None
        t_start_2 = None
        t_end_1 = None
        t_end_2 = None
        time_average = False
    else:
        print 'Error (select_common_time): invalid option ' + option
        sys.exit()
    return file_path_1, file_path_2, time_index_1, time_index_2, t_start_1, t_start_2, t_end_1, t_end_2, time_average


# Compare one simulation to another. Assumes the simulations have monthly averaged output. They don't need to be the same length.

# Keyword arguments:
# output_dir: as in function plot_everything
# baseline_dir: like output_dir, but for the simulation you want to use as a baseline. All the plots will show results from output_dir minus baseline_dir. This is the only non-optional keyword argument; it is a named keyword argument with no meaningful default so there's no chance of the user mixing up which simulation is which.
# timeseries_file: filename created by precompute_timeseries, within output_dir and baseline_dir
# grid_path: as in function plot_everything
# fig_dir: as in function plot_everything
# option: either 'last_year' (averages over the last 12 months of the overlapping period of the simulations) or 'last_month' (just considers the last month of the overlapping period).
# unravelled: as in function plot_everything

def plot_everything_diff (output_dir='./', baseline_dir=None, timeseries_file='timeseries.nc', grid_path='../grid/', fig_dir='.', option='last_year', unravelled=False, monthly=True, key='WSS'):

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
    output_files_1 = build_file_list(output_dir_1, unravelled=unravelled)
    output_files_2 = build_file_list(output_dir_2, unravelled=unravelled)

    # Build the grid
    grid = Grid(grid_path)

    # Timeseries through the entire simulation
    var_names = ['fris_mass_balance', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt', 'fris_age']
    for var in var_names:
        read_plot_timeseries_diff(var, output_dir_1+timeseries_file, output_dir_2+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_'+var+'_diff.png', monthly=monthly)

    # Now figure out which time indices to use for plots with no time dependence
    file_path_1, file_path_2, time_index_1, time_index_2, t_start_1, t_start_2, t_end_1, t_end_2, time_average = select_common_time(output_files_1, output_files_2, option=option, monthly=monthly)
    # Set date string
    if option == 'last_year':
        date_string = 'year beginning ' + parse_date(file_path=file_path_1, time_index=t_start_1)
    elif option == 'last_month':
        date_string = parse_date(file_path=file_path_1, time_index=time_index_1)

    # Now make lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'hsnow', 'mld', 'eta', 'vel', 'velice', 'bwage', 'iceprod']
    if key == 'WSK':
        figsize = (10,6)
    else:
        figsize = (8,6)
    for var in var_names:
        if var == 'iceprod':
            vmin = -2
            vmax = 2            
        else:
            vmin = None
            vmax = None
        read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+var+'_diff.png', figsize=figsize, vmin=vmin, vmax=vmax)
        # Zoom into some variables
        if key=='WSK' and var in ['ismr', 'bwtemp', 'bwsalt', 'vel', 'bwage']:
            if var == 'bwage':
                vmin = -5
                vmax = None
            else:
                vmin = None
                vmax = None
            read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, zoom_fris=True, date_string=date_string, fig_name=fig_dir+var+'_zoom_diff.png', vmin=vmin, vmax=vmax)
        if var == 'vel':
            # Call the other options for vertical transformations
            for vel_option in ['sfc', 'bottom']:
                read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, vel_option=vel_option, date_string=date_string, fig_name=fig_dir+var+'_'+vel_option+'_diff.png')

    # Slice plots
    read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-40, hmax=-75, zmin=-1450, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_filchner_diff.png')
    read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-55, hmax=-72, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_ronne_diff.png')
    if key == 'WSS':
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


# Pre-compute timeseries and save them in a NetCDF file which concatenates after each simulation segment.

# Arguments:
# mit_file: path to a single NetCDF file output by MITgcm
# timeseries_file: path to a NetCDF file for saving timeseries. If it exists, it will be appended to; if it doesn't exist, it will be created.

# Optional keyword arguments:
# timeseries_types: list of timeseries types to compute (subset of the options from set_parameters). If None, a default set will be used.
# lon0, lat0: if timeseries_types includes 'temp_polynya' and/or 'salt_polynya', use these points as the centre.

def precompute_timeseries (mit_file, timeseries_file, timeseries_types=None, monthly=True, lon0=None, lat0=None):

    # Timeseries to compute
    if timeseries_types is None:
        timeseries_types = ['fris_mass_balance', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt', 'fris_age'] #['fris_mass_balance', 'hice_corner', 'mld_ewed', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt']

    # Build the grid
    grid = Grid(mit_file)

    # Check if the timeseries file already exists
    file_exists = os.path.isfile(timeseries_file)
    if file_exists:
        # Open it
        id = nc.Dataset(timeseries_file, 'a')
    else:
        # Create it
        ncfile = NCfile(timeseries_file, grid, 't')

    # Define/update time
    # Read the time array from the MITgcm file, and its units
    time, time_units = netcdf_time(mit_file, return_units=True)
    if file_exists:
        # Update the units to match the old time array
        time_units = id.variables['time'].units
        # Also figure out how many time indices are in the file so far
        num_time = id.variables['time'].size
        # Convert to numeric values
        time = nc.date2num(time, time_units)
        # Append to file
        id.variables['time'][num_time:] = time
    else:
        # Add the time variable to the file
        ncfile.add_time(time, units=time_units)

    # Inner function to define/update non-time variables
    def write_var (data, var_name, title, units):
        if file_exists:
            # Append to file
            id.variables[var_name][num_time:] = data
        else:
            # Add the variable to the file
            ncfile.add_variable(var_name, data, 't', long_name=title, units=units)

    # Now process all the timeseries
    for ts_name in timeseries_types:
        print 'Processing ' + ts_name
        # Get information about the variable; only care about title and units
        title, units = set_parameters(ts_name)[2:4]
        if ts_name == 'fris_mass_balance':
            melt, freeze = calc_special_timeseries(ts_name, mit_file, grid=grid, monthly=monthly)[1:]
            # We need two titles now
            title_melt = 'Total melting beneath FRIS'
            title_freeze = 'Total refreezing beneath FRIS'
            # Update two variables
            write_var(melt, 'fris_total_melt', title_melt, units)
            write_var(freeze, 'fris_total_freeze', title_freeze, units)
        else:
            data = calc_special_timeseries(ts_name, mit_file, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)[1]
            write_var(data, ts_name, title, units)

    # Finished
    if file_exists:
        id.close()
    else:
        ncfile.close()



# When the model crashes, convert its crash-dump to a NetCDF file.

# Arguments:
# crash_dir: directory including all the state*crash.* files. The NetCDF file will be saved here too, with the name crash.nc.
# grid_path: as in function plot_everything

def crash_to_netcdf (crash_dir, grid_path):

    from MITgcmutils import rdmds

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


# Helper function for average_monthly_files and make_climatology
# Find all the time-dependent variables in a NetCDF file (not counting 1D time-dependent variables such as 'time' and 'iters') and return a list of their names.
def time_dependent_variables (file_name):

    var_names = []
    id = nc.Dataset(file_name, 'r')
    for var in id.variables:
        if 'time' in id.variables[var].dimensions and len(id.variables[var].shape) > 1:
            var_names.append(var)
    id.close()
    return var_names


# Do a proper time-average of files with monthly output, where each month is weighted with the number of days it represents. Make sure you load NCO before calling this function.

# Arguments:
# input_files: list of paths to filenames to time-average over. They will all be collapsed into a single record.
# output_file: path to filename to save the time-average.

# Optional keyword arguments:
# t_start: index (0-based) of the time record to start the average in input_files[0].
# t_end: index (0-based) of the time record to end the average in input_files[-1]. In python convention, this is the first index to ignore.
# For example, to average from month 7 (index 6) of the first file to month 10 (index 9)  of the last file, do
# average_monthly_files(input_files, output_file, t_start=6, t_end=10)

def average_monthly_files (input_files, output_file, t_start=0, t_end=None):

    from nco import Nco
    from nco.custom import Limit

    if isinstance(input_files, str):
        # Only one file
        input_files = [input_files]

    # Extract the first time record from the first file
    # This will make a skeleton file with 1 time record and all the right metadata; later we will overwrite the values of all the time-dependent variables with the weighted time-averages.
    print 'Initialising ' + output_file
    nco = Nco()
    nco.ncks(input=input_files[0], output=output_file, options=[Limit('time', t_start, t_start)])

    # Get the starting date
    time0 = netcdf_time(output_file)
    year0 = time0[0].year
    month0 = time0[0].month    

    # Find all the time-dependent variables
    var_names = time_dependent_variables(output_file)

    # Time-average each variable
    id_out = nc.Dataset(output_file, 'a')
    for var in var_names:
        print 'Processing ' + var

        # Reset time
        year = year0
        month = month0
        # Set up accumulation array of the right dimension for this variable
        shape = id_out.variables[var].shape[1:]
        data = np.zeros(shape)
        # Also accumulate total number of days
        total_days = 0

        # Loop over input files
        for i in range(len(input_files)):

            file_name = input_files[i]
            print '...' + file_name
            
            # Figure out how many time indices there are
            id_in = nc.Dataset(file_name, 'r')
            num_time = id_in.variables[var].shape[0]

            # Choose which indices we will loop over: special cases for first and last files, if t_start or t_end are set
            if i == 0:
                t_start_curr = t_start
            else:
                t_start_curr = 0
            if i == len(input_files)-1 and t_end is not None:
                t_end_curr = t_end
            else:
                t_end_curr = num_time

            # Now loop over time indices
            for t in range(t_start_curr, t_end_curr):
                # Integrate
                ndays = days_per_month(month, year)
                data += id_in.variables[var][t,:]*ndays
                total_days += ndays
                # Increment month (and year if needed)
                month += 1
                if month == 13:
                    month = 1
                    year += 1

            id_in.close()

        # Now convert from integral to average
        data /= total_days
        # Overwrite this variable in the output file
        id_out.variables[var][0,:] = data

    id_out.close()

    

# Call average_monthly_files for each year in the simulation. Make sure you load NCO before calling this function.

# Optional keyword arguments:
# in_dir: path to directory containing output_*.nc files
# out_dir: path to directory to save the annually averaged files
def make_annual_averages (in_dir='./', out_dir='./'):

    in_dir = real_dir(in_dir)
    out_dir = real_dir(out_dir)

    # Find all the files of the form output_*.nc
    file_names = build_file_list(in_dir)
    num_files = len(file_names)
    # Make sure their names go from 1 to n where  n is the number of files
    if '001' not in file_names[0] or '{0:03d}'.format(num_files) not in file_names[-1]:
        print 'Error (make_annual_average): based on filenames, you seem to be missing some files.'
        sys.exit()

    # Get the starting date
    time0 = netcdf_time(file_names[0])[0]
    if time0.month != 1:
        print "Error (make_annual_average): this simulation doesn't start in January."
        sys.exit()
    year0 = time0.year

    # Save the number of months in each file
    num_months = []
    for file in file_names:
        id = nc.Dataset(file)
        num_months.append(id.variables['time'].size)
        id.close()

    # Now the work starts
    year = year0
    i = 0  # file number
    t = 0  # the next time index that needs to be dealt with
    files_to_average = [] # list of files containing timesteps from the current year
    t_start = None  # time index of files_to_average[0] to start the averaging from

    # New iteration of loop each time we process a chunk of time from a file.
    while True:

        if len(files_to_average)==0 and t+12 <= num_months[i]:
            # Option 1: Average a full year
            files_to_average.append(file_names[i])
            t_start = t
            t_end = t+12
            print 'Processing all of ' + str(year) + ' from ' + file_names[i] + ', indices ' + str(t_start) + ' to ' + str(t_end-1)
            average_monthly_files(files_to_average, out_dir+str(year)+'_avg.nc', t_start=t_start, t_end=t_end)
            files_to_average = []
            t_start = None
            t += 12
            year += 1

        elif len(files_to_average)==0 and t+12 > num_months[i]:
            # Option 2: Start a new year
            files_to_average.append(file_names[i])
            t_start = t
            print 'Processing beginning of ' + str(year) + ' from ' + file_names[i] + ', indices ' + str(t_start) + ' to ' + str(num_months[i]-1)
            tmp_months = num_months[i] - t_start
            print '(have processed ' + str(tmp_months) + ' months of ' + str(year) + ')'
            t = num_months[i]

        elif len(files_to_average)>0 and t+12-tmp_months > num_months[i]:
            # Option 3: Add onto an existing year, but can't complete it
            files_to_average.append(file_names[i])
            if t != 0:
                print 'Error (make_annual_averages): something weird happened with Option 3'
                sys.exit()
            print 'Processing middle of ' + str(year) + ' from ' + file_names[i] + ', indices ' + str(t) + ' to ' + str(num_months[i]-1)
            tmp_months += num_months[i] - t
            print '(have processed ' + str(tmp_months) + ' months of ' + str(year) + ')'
            t = num_months[i]

        elif len(files_to_average)>0 and t+12-tmp_months <= num_months[i]:
            # Option 4: Add onto an existing year and complete it
            files_to_average.append(file_names[i])
            if t != 0:
                print 'Error (make_annual_averages): something weird happened with Option 4'
                sys.exit()
            t_end = t+12-tmp_months
            print 'Processing end of ' + str(year) + ' from ' + file_names[i] + ', indices ' + str(t) + ' to ' + str(t_end-1)
            average_monthly_files(files_to_average, out_dir+str(year)+'_avg.nc', t_start=t_start, t_end=t_end)
            files_to_average = []
            t_start = None
            t += 12-tmp_months
            year += 1

        if t == num_months[i]:
            print 'Reached the end of ' + file_names[i]
            # Prepare for the next file
            i += 1
            t = 0
            if i == num_files:
                # No more files
                if len(files_to_average)>0:
                    print 'Warning: ' + str(year) + ' is incomplete. Ignoring it.'
                break


# Make a monthly climatology from unravelled files (in the form 1979.nc, etc. using netcdf_finalise.sh) stored in the given directory. Restrict the climatology to the years start_year to end_year inclusive, and save the result in output_file.
def make_climatology (start_year, end_year, output_file, directory='./'):
    
    directory = real_dir(directory)

    # Copy the first file
    # This will make a skeleton file with 12 time records and all the right metadata; later we will overwrite the values of all the time-dependent variables.
    print 'Setting up ' + output_file
    shutil.copyfile(directory+str(start_year)+'.nc', output_file)

    # Find all the time-dependent variables
    var_names = time_dependent_variables(output_file)

    # Calculate the monthly climatology for each variable
    id_out = nc.Dataset(output_file, 'a')
    for var in var_names:
        print 'Processing ' + var

        # Start with the first year
        print '...' + str(start_year)
        data = id_out.variables[var][:]
        
        # Add subsequent years
        for year in range(start_year+1, end_year+1):
            print '...' + str(year)
            data += read_netcdf(directory+str(year)+'.nc', var)

        # Divide by number of years to get average
        data /= (end_year-start_year+1)
        # Overwrite in output_file
        id_out.variables[var][:] = data

    id_out.close()


# Calculate sea ice production and save the result in a new file. This selects all the positive values and sets all the negative values to zero. So in practice it is gross sea ice production calculated monthly from net sea ice production.
def calc_ice_prod (file_path, out_file, monthly=True):

    # Build the grid from the file
    grid = Grid(file_path)

    # Add up all the terms to get sea ice production at each time index
    ice_prod = read_netcdf(file_path, 'SIdHbOCN') + read_netcdf(file_path, 'SIdHbATC') + read_netcdf(file_path, 'SIdHbATO') + read_netcdf(file_path, 'SIdHbFLO')
    # Also need time
    time = netcdf_time(file_path, monthly=monthly)

    # Set negative values to 0
    ice_prod = np.maximum(ice_prod, 0)

    # Write a new file
    ncfile = NCfile(out_file, grid, 'xyt')
    ncfile.add_time(time)
    ncfile.add_variable('ice_prod', ice_prod, 'xyt', long_name='Net sea ice production', units='m/s')
    ncfile.close()

    

    

    
    

    

