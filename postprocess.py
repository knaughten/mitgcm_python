#######################################################
# Files to create when the simulation is done
#######################################################

import os
import sys
import numpy as np
import shutil
import netCDF4 as nc

from grid import Grid
from file_io import NCfile, netcdf_time, find_time_index, read_netcdf, read_iceprod
from timeseries import calc_timeseries, calc_special_timeseries, set_parameters
from utils import real_dir, days_per_month, str_is_int, mask_3d, mask_except_ice, mask_land, mask_land_ice, select_top, select_bottom, mask_outside_box, var_min_max, add_time_dim, apply_mask
from constants import deg_string, region_names
from calculus import area_average
from diagnostics import density


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


# Helper function to get all the output directories from a coupled model, one per segment, in order.
def get_segment_dir (output_dir):

    segment_dir = []
    for name in os.listdir(output_dir):
        # Look for directories composed of numbers (date codes)
        if os.path.isdir(output_dir+name) and str_is_int(name):
            segment_dir.append(name)
    # Make sure in chronological order
    segment_dir.sort()
    return segment_dir


# Either call get_segment_dir (if segment_dir is None), or make sure segment_dir is an array (and not a string, i.e. just one entry).
def check_segment_dir (output_dir, segment_dir):

    if segment_dir is None:
        segment_dir = get_segment_dir(output_dir)
    else:
        if isinstance(segment_dir, str):
            segment_dir = [segment_dir]
    return segment_dir


# Get the path to the MITgcm output file for each segment.
def segment_file_paths (output_dir, segment_dir=None, file_name='output.nc'):

    if segment_dir is None:
        segment_dir = get_segment_dir(output_dir)
    file_paths = []
    for sdir in segment_dir:
        file_paths.append(output_dir + sdir + '/MITgcm/' + file_name)
    return file_paths


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

def plot_everything (output_dir='./', timeseries_file='timeseries.nc', grid_path=None, fig_dir='.', file_path=None, monthly=True, date_string=None, time_index=-1, time_average=True, unravelled=False, key='WSFRIS', hovmoller_file='hovmoller.nc', ctd_file='../../ctddatabase.mat'):

    from plot_1d import read_plot_timeseries, read_plot_timeseries_multi
    from plot_latlon import read_plot_latlon
    from plot_slices import read_plot_ts_slice
    from plot_misc import read_plot_hovmoller_ts
    from plot_misc import ctd_cast_compare, amundsen_rignot_comparison

    if time_average:
        time_index = None

    # Make sure proper directories
    output_dir = real_dir(output_dir)
    fig_dir = real_dir(fig_dir)
    
    # Build the list of output files in this directory (use them all for timeseries)
    if key in ['WSFRIS', 'FRIS']:
        # Coupled
        output_files = segment_file_paths(output_dir)
    else:
        # Uncoupled
        output_files = build_file_list(output_dir, unravelled=unravelled)
    if file_path is None:
        # Select the last file for single-timestep analysis
        file_path = output_files[-1]        

    # Build the grid
    if grid_path is None:
        grid_path = file_path
    grid = Grid(grid_path)

    # Timeseries
    if key == 'WSS':
        var_names = ['fris_mass_balance', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt', 'fris_age']
    elif key == 'WSK':
        var_names = ['fris_mass_balance', 'hice_corner', 'mld_ewed', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt']
    elif key == 'WSFRIS':
        var_names = ['fris_mass_balance', 'fris_massloss', 'fris_temp', 'fris_salt', 'fris_density', 'sws_shelf_temp', 'sws_shelf_salt', 'sws_shelf_density', 'filchner_trough_temp', 'filchner_trough_salt', 'filchner_trough_density', 'wdw_core_temp', 'wdw_core_salt', 'wdw_core_density', 'seaice_area', 'wed_gyre_trans', 'filchner_trans', 'sws_shelf_iceprod'] #['fris_mass_balance', 'hice_corner', 'mld_ewed', 'fris_temp', 'fris_salt', 'ocean_vol', 'eta_avg', 'seaice_area']
    elif key == 'FRIS':
        var_names = ['fris_mass_balance', 'fris_temp', 'fris_salt', 'ocean_vol', 'eta_avg', 'seaice_area']
    elif key == 'PAS':
        melt_names = ['getz_melting', 'dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting']
        read_plot_timeseries_multi(melt_names, output_dir+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_multi_melt.png', monthly=monthly)
        var_names = ['eta_avg', 'seaice_area']
    for var in var_names:
        read_plot_timeseries(var, output_dir+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_'+var+'.png', monthly=monthly)

    # Hovmoller plots, CTD casts, and melt rate comparisons
    if key == 'PAS':
        for loc in ['pine_island_bay', 'dotson_bay']:
            read_plot_hovmoller_ts(hovmoller_file, loc, grid, tmax=1.5, smin=34, t_contours=[0,1], s_contours=[34.5, 34.7], fig_name=fig_dir+'hovmoller_ts_'+loc+'.png', monthly=monthly, smooth=12)
            ctd_cast_compare(loc, hovmoller_file, ctd_file, grid, fig_name=fig_dir+'casts_'+loc+'.png')
        amundsen_rignot_comparison(output_dir+timeseries_file, precomputed=True, fig_name=fig_dir+'rignot.png') 

    # Lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'eta', 'vel', 'velice']
    if key in ['WSS', 'WSK', 'FRIS', 'WSFRIS']:
        var_names += ['hsnow', 'mld', 'saltflx', 'psi', 'iceprod']
        if key in ['WSS', 'WSK']:
            var_names += ['bwage']
    for var in var_names:
        # Customise bounds and zooming
        vmin = None
        vmax = None
        zoom_fris = False
        ymax = None
        chunk = None
        fig_name = fig_dir + var + '.png'
        if key == 'PAS' and var in ['bwsalt', 'bwtemp', 'hice', 'ismr', 'vel', 'velice']:
            ymax = -70
        if var == 'bwtemp':
            if key == 'WSS':
                vmin = -2.5
                vmax = -1.5
            elif key in ['WSK', 'WSFRIS']:
                vmax = 1
        if var == 'bwsalt':
            if key == 'PAS':
                vmin = 34.1
            else:
                vmin = 34.3
        if var == 'bwage':
            vmin = 0
            if key == 'WSS':
                vmax = 12
        if var == 'eta':
            vmin = -2.5
        if var == 'hice':
            if key == 'PAS':
                vmax = 2
            else:
                vmax = 4
        if var == 'saltflx':
            vmin = -0.001
            vmax = 0.001
        if var == 'iceprod':
            vmin = 0
            vmax = 5
        if var == 'psi' and key in ['WSS', 'FRIS']:
            vmin = -0.5
            vmax = 0.5
        if var in ['vel', 'velice'] and key=='WSS':
            chunk = 6
        if not zoom_fris and key in ['WSK', 'WSFRIS']:
            figsize = (10,6)
        elif key == 'PAS':
            if ymax == -70:
                figsize = (14,5)
            else:
                figsize = (12,6)
        else:
            figsize = (8,6)
        # Plot
        read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, ymax=ymax, fig_name=fig_name, date_string=date_string, figsize=figsize, chunk=chunk)
        # Make additional plots if needed
        if key in ['WSK', 'WSFRIS'] and var in ['ismr', 'vel', 'bwtemp', 'bwsalt', 'psi', 'bwage']:
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
            if key in ['WSK', 'WSFRIS']:
                figsize = (10,6)
            for vel_option in ['sfc', 'bottom']:
                read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, vel_option=vel_option, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, ymax=ymax, fig_name=fig_dir+var+'_'+vel_option+'.png', date_string=date_string, figsize=figsize, chunk=chunk)
        if var in ['eta', 'hice']:
            # Make another plot with unbounded colour bar
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, time_average=time_average, zoom_fris=zoom_fris, ymax=ymax, fig_name=fig_dir + var + '_unbound.png', date_string=date_string, figsize=figsize)

    # Slice plots
    if key in ['WSK', 'WSS', 'WSFRIS', 'FRIS']:
        read_plot_ts_slice(file_path, grid=grid, lon0=-40, hmax=-75, zmin=-1450, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_filchner.png', date_string=date_string)
        read_plot_ts_slice(file_path, grid=grid, lon0=-55, hmax=-72, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_ronne.png', date_string=date_string)
    if key in ['WSK', 'WSFRIS']:
        read_plot_ts_slice(file_path, grid=grid, lon0=0, time_index=time_index, time_average=time_average, fig_name=fig_dir+'ts_slice_eweddell.png', date_string=date_string)


# Given lists of files from two simulations, find the file and time indices corresponding to the last year (if option='last_year') or last month/timestep (if option='last_month') in the shortest simulation.
def select_common_time (output_files_1, output_files_2, option='last_year', monthly=True, check_match=True):

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
    if check_match:
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
# file_name: name of file containing 1 time index, which is present in both directories.

def plot_everything_diff (output_dir='./', baseline_dir=None, timeseries_file='timeseries.nc', grid_path=None, fig_dir='.', option='last_year', unravelled=False, monthly=True, key='WSFRIS', hovmoller_file='hovmoller.nc', file_name=None):

    from plot_1d import read_plot_timeseries, read_plot_timeseries_multi
    from plot_latlon import read_plot_latlon_diff
    from plot_slices import read_plot_ts_slice_diff
    from plot_misc import read_plot_hovmoller_ts_diff
    from plot_utils.labels import parse_date

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
    coupled = key in ['WSFRIS', 'FRIS']
    if coupled:
        output_files_1 = segment_file_paths(output_dir_1)
        output_files_2 = segment_file_paths(output_dir_2)
    else:
        output_files_1 = build_file_list(output_dir_1, unravelled=unravelled)
        output_files_2 = build_file_list(output_dir_2, unravelled=unravelled)

    # Now figure out which time indices to use for plots with no time dependence
    if file_name is None:
        file_path_1, file_path_2, time_index_1, time_index_2, t_start_1, t_start_2, t_end_1, t_end_2, time_average = select_common_time(output_files_1, output_files_2, option=option, monthly=monthly, check_match=False)
        # Set date string
        if option == 'last_year':
            date_string = 'year beginning ' + parse_date(file_path=file_path_1, time_index=t_start_1)
        elif option == 'last_month':
            date_string = parse_date(file_path=file_path_1, time_index=time_index_1)
    else:
        file_path_1 = output_dir_1 + file_name
        file_path_2 = output_dir_2 + file_name
        time_index_1 = 0
        time_index_2 = 0
        t_start_1 = None
        t_start_2 = None
        t_end_1 = None
        t_end_2 = None
        time_average = False
        date_string = ''

    # Build the grid
    if grid_path is None:
        grid_path = file_path_1
    grid = Grid(grid_path)

    # Timeseries through the entire simulation
    if key == 'WSS':
        var_names = ['fris_mass_balance', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt', 'fris_age']
    elif key == 'WSK':
        var_names = ['fris_mass_balance', 'hice_corner', 'mld_ewed', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt']
    elif key == 'WSFRIS':
        var_names = ['fris_mass_balance', 'fris_massloss', 'fris_temp', 'fris_salt', 'fris_density', 'sws_shelf_temp', 'sws_shelf_salt', 'sws_shelf_density', 'filchner_trough_temp', 'filchner_trough_salt', 'filchner_trough_density', 'wdw_core_temp', 'wdw_core_salt', 'wdw_core_density', 'seaice_area', 'wed_gyre_trans', 'filchner_trans', 'sws_shelf_iceprod'] #['fris_mass_balance', 'hice_corner', 'mld_ewed', 'fris_temp', 'fris_salt', 'ocean_vol', 'eta_avg', 'seaice_area']
    elif key == 'FRIS':
        var_names = ['fris_mass_balance', 'fris_temp', 'fris_salt', 'ocean_vol', 'eta_avg', 'seaice_area']
    elif key == 'PAS':
        melt_names = ['getz_melting', 'dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting']
        read_plot_timeseries_multi(melt_names, [output_dir_1+timeseries_file, output_dir_2+timeseries_file], diff=True, precomputed=True, fig_name=fig_dir+'timeseries_multi_melt_diff.png', monthly=monthly)
        var_names = ['eta_avg', 'seaice_area']
    for var in var_names:
        read_plot_timeseries(var, [output_dir_1+timeseries_file, output_dir_2+timeseries_file], diff=True, precomputed=True, fig_name=fig_dir+'timeseries_'+var+'_diff.png', monthly=monthly)

    # Hovmoller plots
    if key == 'PAS':
        for loc in ['pine_island_bay', 'dotson_bay']:
            read_plot_hovmoller_ts_diff(output_dir_1+hovmoller_file, output_dir_2+hovmoller_file, loc, grid, fig_name=fig_dir+'hovmoller_ts_'+loc+'_diff.png', monthly=monthly, smooth=12)

    # Now make lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'hsnow', 'eta', 'vel', 'velice']
    if key in ['WSK', 'WSS', 'WSFRIS', 'FRIS']:
        var_names += ['iceprod', 'mld']
        if key in ['WSK', 'WSS']:
            var_names += ['bwage']
    for var in var_names:        
        if var == 'iceprod':
            vmin = -2
            vmax = 2            
        else:
            vmin = None
            vmax = None
        ymax = None
        if key == 'PAS' and var in ['bwsalt', 'bwtemp', 'hice', 'ismr', 'vel', 'velice']:
            ymax = -70
        if key in ['WSK', 'WSFRIS']:
            figsize = (10, 6)
        elif key == 'PAS':
            if ymax == -70:
                figsize = (14, 5)
            else:
                figsize = (12, 6)
        else:
            figsize = (8, 6)
        read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, ymax=ymax, fig_name=fig_dir+var+'_diff.png', figsize=figsize, vmin=vmin, vmax=vmax, coupled=coupled)
        # Zoom into some variables
        if key in['WSK', 'WSFRIS'] and var in ['ismr', 'bwtemp', 'bwsalt', 'vel', 'bwage']:
            if var == 'bwage':
                vmin = -5
                vmax = None
            else:
                vmin = None
                vmax = None
            read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, zoom_fris=True, date_string=date_string, fig_name=fig_dir+var+'_zoom_diff.png', vmin=vmin, vmax=vmax, coupled=coupled)
        if var == 'vel':
            # Call the other options for vertical transformations
            for vel_option in ['sfc', 'bottom']:
                read_plot_latlon_diff(var, file_path_1, file_path_2, grid=grid, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, vel_option=vel_option, date_string=date_string, fig_name=fig_dir+var+'_'+vel_option+'_diff.png', coupled=coupled)

    # Slice plots
    if key in ['WSK', 'WSS', 'WSFRIS', 'FRIS']:
        read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-40, hmax=-75, zmin=-1450, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_filchner_diff.png', coupled=coupled)
        read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=-55, hmax=-72, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_ronne_diff.png', coupled=coupled)
    if key in ['WSK', 'WSFRIS']:
        read_plot_ts_slice_diff(file_path_1, file_path_2, grid=grid, lon0=0, zmin=-2000, time_index=time_index_1, t_start=t_start_1, t_end=t_end_1, time_average=time_average, time_index_2=time_index_2, t_start_2=t_start_2, t_end_2=t_end_2, date_string=date_string, fig_name=fig_dir+'ts_slice_eweddell_diff.png', coupled=coupled)    
    


# Plot the sea ice annual min and max for each year of the simulation. First you have to concatenate the sea ice area into a single file, such as:
# ncrcat -v SIarea output_*.nc aice_tot.nc

# Arguments:
# file_path: path to concatenated NetCDF file with sea ice area for the entire simulation

# Optional keyword arguments:
# grid_path: as in function plot_everything
# fig_dir: path to directory to save figures in
# monthly: as in function netcdf_time

def plot_seaice_annual (file_path, grid_path='../grid/', fig_dir='.', monthly=True):

    from plot_latlon import plot_aice_minmax

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


# Helper functions for precompute_timeseries and precompute_hovmoller:

# Check if the precomputed file already exists, and either open it or create it.
def set_update_file (precomputed_file, grid, dimensions):
    if os.path.isfile(precomputed_file):
        # Open it
        return nc.Dataset(precomputed_file, 'a')
    else:
        # Create it
        return NCfile(precomputed_file, grid, dimensions)

# Define or update the time axis.
def set_update_time (id, mit_file, monthly=True, time_average=False):
    # Read the time array from the MITgcm file, and its units
    time, time_units, calendar = netcdf_time(mit_file, return_units=True, monthly=monthly)
    if time_average:
        # Only save the first one
        time = np.array([time[0]])
    if isinstance(id, nc.Dataset):
        # File is being updated
        # Update the units to match the old time array
        time_units = id.variables['time'].units
        # Also figure out how many time indices are in the file so far
        num_time = id.variables['time'].size
        # Convert to numeric values
        time = nc.date2num(time, time_units, calendar=calendar)
        # Append to file
        id.variables['time'][num_time:] = time
        return num_time
    elif isinstance(id, NCfile):
        # File is new
        # Add the time variable to the file
        id.add_time(time, units=time_units, calendar=calendar)
        return 0
    else:
        print 'Error (set_update_time): unknown id type'
        sys.exit()

# Define or update non-time variables.
def set_update_var (id, num_time, data, dimensions, var_name, title, units):
    if isinstance(id, nc.Dataset):
        # File is being updated
        # Append to file
        id.variables[var_name][num_time:] = data
    elif isinstance(id, NCfile):
        # File is new
        # Add the variable to the file
        id.add_variable(var_name, data, dimensions, long_name=title, units=units)
    else:
        print 'Error (set_update_var): unknown id type'
        sys.exit()        


# Pre-compute timeseries and save them in a NetCDF file which concatenates after each simulation segment.

# Arguments:
# mit_file: path to a single NetCDF file output by MITgcm
# timeseries_file: path to a NetCDF file for saving timeseries. If it exists, it will be appended to; if it doesn't exist, it will be created.

# Optional keyword arguments:
# timeseries_types: list of timeseries types to compute (subset of the options from set_parameters). If None, a default set will be used.
# lon0, lat0: if timeseries_types includes 'temp_polynya' and/or 'salt_polynya', use these points as the centre.

def precompute_timeseries (mit_file, timeseries_file, timeseries_types=None, monthly=True, lon0=None, lat0=None, key='PAS', eosType='MDJWF', rhoConst=None, Tref=None, Sref=None, tAlpha=None, sBeta=None, time_average=False, grid=None):

    # Timeseries to compute
    if timeseries_types is None:
        if key == 'WSS':
            timeseries_types = ['fris_mass_balance', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt', 'fris_age']
        elif key == 'WSK':
            timeseries_types = ['fris_mass_balance', 'hice_corner', 'mld_ewed', 'eta_avg', 'seaice_area', 'fris_temp', 'fris_salt']
        elif key == 'PAS':
            timeseries_types = ['dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'getz_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting', 'eta_avg', 'hice_max', 'pine_island_bay_temp_btw_200_700m', 'pine_island_bay_salt_btw_200_700m', 'dotson_bay_temp_btw_200_700m', 'dotson_bay_salt_btw_200_700m', 'inner_amundsen_shelf_temp_btw_200_700m', 'inner_amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_break_uwind_avg', 'dotson_massloss', 'pig_massloss', 'getz_massloss', 'inner_amundsen_shelf_sss_avg']

    # Build the grid
    if grid is None:
        grid = Grid(mit_file)
    if any (['density' in s for s in timeseries_types]):
        # Precompute density so we don't have to re-calculate it for each density variable. If there's only one density variable, this won't make a difference.
        temp = read_netcdf(mit_file, 'THETA', time_average=time_average)
        salt = read_netcdf(mit_file, 'SALT', time_average=time_average)
        rho = density(eosType, salt, temp, 0, rhoConst=rhoConst, Tref=Tref, Sref=Sref, tAlpha=tAlpha, sBeta=sBeta)
    else:
        rho = None

    # Set up or update the file and time axis
    id = set_update_file(timeseries_file, grid, 't')
    num_time = set_update_time(id, mit_file, monthly=monthly, time_average=time_average)

    # Now process all the timeseries
    for ts_name in timeseries_types:
        print 'Processing ' + ts_name
        # Get information about the variable; only care about title and units
        title, units = set_parameters(ts_name)[2:4]
        if ts_name == 'fris_mass_balance':
            melt, freeze = calc_special_timeseries(ts_name, mit_file, grid=grid, monthly=monthly, time_average=time_average)[1:]
            # We need two titles now
            title_melt = 'Total melting beneath FRIS'
            title_freeze = 'Total refreezing beneath FRIS'
            # Update two variables
            set_update_var(id, num_time, melt, 't', 'fris_total_melt', title_melt, units)
            set_update_var(id, num_time, freeze, 't', 'fris_total_freeze', title_freeze, units)
        else:
            data = calc_special_timeseries(ts_name, mit_file, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly, rho=rho, time_average=time_average)[1]
            set_update_var(id, num_time, data, 't', ts_name, title, units)

    id.close()


# Precompute ocean timeseries from a coupled UaMITgcm simulation.
# Optional keyword arguments:
# output_dir: path to master output directory for experiment. Default the current directory.
# timeseries_file: as in precompute_timeseries. Default 'timeseries.nc'.
# file_name: name of the output NetCDF file within the output/XXXXXX/MITgcm/ directories. Default 'output.nc'.
# segment_dir: list of date codes, in chronological order, corresponding to the subdirectories within output_dir. This must be specified if timeseries_file already exists. If it is not specified, all available subdirectories of output_dir will be used.
# timeseries_types: as in precompute_timeseries
# time_average: Average each year to create an annually averaged timeseries
def precompute_timeseries_coupled (output_dir='./', timeseries_file='timeseries.nc', hovmoller_file='hovmoller.nc', file_name='output.nc', segment_dir=None, timeseries_types=None, hovmoller_loc=None, key='PAS', time_average=False):

    if timeseries_types is None:
        if key == 'WSFRIS':
            timeseries_types = ['fris_mass_balance', 'fris_massloss', 'fris_temp', 'fris_salt', 'fris_density', 'sws_shelf_temp', 'sws_shelf_salt', 'sws_shelf_density', 'filchner_trough_temp', 'filchner_trough_salt', 'filchner_trough_density', 'wdw_core_temp', 'wdw_core_salt', 'wdw_core_density', 'seaice_area', 'wed_gyre_trans', 'filchner_trans', 'sws_shelf_iceprod']
            #timeseries_types = ['fris_mass_balance', 'hice_corner', 'mld_ewed', 'fris_temp', 'fris_salt', 'ocean_vol', 'eta_avg', 'seaice_area']
        elif key == 'WSFRIS_pt2':
            timeseries_types = ['fris_massloss', 'fris_temp', 'filchner_trough_temp', 'filchner_trough_salt', 'ronne_depression_temp', 'ronne_depression_salt', 'ronne_depression_iceprod', 'ronne_depression_atemp_avg', 'ronne_depression_wind_avg', 'ronne_depression_sst_avg', 'ronne_depression_sss_avg', 'sws_shelf_temp', 'sws_shelf_salt', 'sws_shelf_iceprod', 'sws_shelf_atemp_avg', 'sws_shelf_wind_avg', 'sws_shelf_sst_avg', 'sws_shelf_sss_avg', 'sws_shelf_pminuse']
        elif key == 'WSFRIS_diag':
            timeseries_types = ['sws_shelf_salt_adv', 'sws_shelf_salt_dif', 'sws_shelf_salt_sfc', 'sws_shelf_salt_sfc_corr', 'sws_shelf_salt_tend', 'sws_shelf_seaice_melt', 'sws_shelf_seaice_freeze', 'sws_shelf_pmepr', 'fris_age']
        elif key == 'FRIS':
            timeseries_types = ['fris_mass_balance', 'fris_temp', 'fris_salt', 'ocean_vol', 'eta_avg', 'seaice_area']
        elif key == 'PAS':
            timeseries_types = ['dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'getz_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting', 'eta_avg', 'hice_max', 'pine_island_bay_temp_below_500m', 'pine_island_bay_salt_below_500m', 'dotson_bay_temp_below_500m', 'dotson_bay_salt_below_500m', 'inner_amundsen_shelf_temp_below_500m', 'inner_amundsen_shelf_salt_below_500m', 'amundsen_shelf_break_uwind_avg', 'dotson_massloss', 'pig_massloss', 'getz_massloss', 'inner_amundsen_shelf_sss_avg', 'amundsen_shelf_break_adv_heat_ns_300_1500m']
    if hovmoller_loc is None:
        if key == 'PAS':
            hovmoller_loc = ['pine_island_bay', 'dotson_bay']
        else:
            hovmoller_loc = []

    output_dir = real_dir(output_dir)

    if segment_dir is None and os.path.isfile(output_dir+timeseries_file):
        print 'Error (precompute_timeseries_coupled): since ' + timeseries_file + ' exists, you must specify segment_dir'
        sys.exit()
    segment_dir = check_segment_dir(output_dir, segment_dir)
    file_paths = segment_file_paths(output_dir, segment_dir, file_name)

    # Call precompute_timeseries for each segment
    for file_path in file_paths:
        print 'Processing ' + file_path
        precompute_timeseries(file_path, output_dir+timeseries_file, timeseries_types=timeseries_types, monthly=True, time_average=time_average)
        if len(hovmoller_loc) > 0 and hovmoller_loc is not None:
            precompute_hovmoller(file_path, output_dir+hovmoller_file, loc=hovmoller_loc)


# Make animations of lat-lon variables throughout a coupled UaMITgcm simulation, and also images of the first and last frames.
# Currently supported: ismr, bwtemp, bwsalt, draft, aice, hice, mld, eta, psi.
def animate_latlon_coupled (var, output_dir='./', file_name='output.nc', segment_dir=None, vmin=None, vmax=None, change_points=None, mov_name=None, fig_name_beg=None, fig_name_end=None, figsize=(8,6), zoom_fris=False):

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from plot_latlon import latlon_plot
    from plot_utils.labels import parse_date
    from plot_utils.colours import get_extend

    output_dir = real_dir(output_dir)
    segment_dir = check_segment_dir(output_dir, segment_dir)
    file_paths = segment_file_paths(output_dir, segment_dir, file_name)

    # Inner function to read and process data from a single file
    def read_process_data (file_path, var_name, grid, mask_option='3d', gtype='t', lev_option=None, ismr=False, psi=False):
        data = read_netcdf(file_path, var_name)
        if mask_option == '3d':
            data = mask_3d(data, grid, gtype=gtype, time_dependent=True)
        elif mask_option == 'except_ice':
            data = mask_except_ice(data, grid, gtype=gtype, time_dependent=True)
        elif mask_option == 'land':
            data = mask_land(data, grid, gtype=gtype, time_dependent=True)
        elif mask_option == 'land_ice':
            data = mask_land_ice(data, grid, gtype=gtype, time_dependent=True)
        else:
            print 'Error (read_process_data): invalid mask_option ' + mask_option
            sys.exit()
        if lev_option is not None:
            if lev_option == 'top':
                data = select_top(data)
            elif lev_option == 'bottom':
                data = select_bottom(data)
            else:
                print 'Error (read_process_data): invalid lev_option ' + lev_option
                sys.exit()
        if ismr:
            data = convert_ismr(data)
        if psi:
            data = np.sum(data, axis=-3)*1e-6
        return data

    all_data = []
    all_grids = []
    all_dates = []
    # Loop over segments
    for file_path in file_paths:
        print 'Processing ' + file_path
        # Build the grid
        grid = Grid(file_path)
        # Read and process the variable we need
        ctype = 'basic'
        gtype = 't'
        include_shelf = var not in ['aice', 'hice', 'mld']
        if var == 'ismr':
            data = read_process_data(file_path, 'SHIfwFlx', grid, mask_option='except_ice', ismr=True)
            title = 'Ice shelf melt rate (m/y)'
            ctype = 'ismr'
        elif var == 'bwtemp':
            data = read_process_data(file_path, 'THETA', grid, lev_option='bottom')
            title = 'Bottom water temperature ('+deg_string+'C)'
        elif var == 'bwsalt':
            data = read_process_data(file_path, 'SALT', grid, lev_option='bottom')
            title = 'Bottom water salinity (psu)'
        elif var == 'draft':
            data = mask_except_ice(grid.draft, grid)
            title = 'Ice shelf draft (m)'
        elif var == 'aice':
            data = read_process_data(file_path, 'SIarea', grid, mask_option='land_ice')
            title = 'Sea ice concentration'
        elif var == 'hice':
            data = read_process_data(file_path, 'SIheff', grid, mask_option='land_ice')
            title = 'Sea ice thickness (m)'
        elif var == 'mld':
            data = read_process_data(file_path, 'MXLDEPTH', grid, mask_option='land_ice')
            title = 'Mixed layer depth (m)'
        elif var == 'eta':
            data = read_process_data(file_path, 'ETAN', grid, mask_option='land')
            title = 'Free surface (m)'
        elif var == 'psi':
            data = read_process_data(file_path, 'PsiVEL', grid, psi=True)
            title = 'Vertically integrated streamfunction (Sv)'
            ctype = 'plusminus'
        else:
            print 'Error (animate_latlon): invalid var ' + var
            sys.exit()
        # Loop over timesteps
        if var == 'draft':
            # Just one timestep
            all_data.append(data)
            all_grids.append(grid)
            all_dates.append(parse_date(file_path=file_path, time_index=0))
        else:
            for t in range(data.shape[0]):
                # Extract the data from this timestep
                # Save it and the grid to the long lists
                all_data.append(data[t,:])
                all_grids.append(grid)
                all_dates.append(parse_date(file_path=file_path, time_index=t))

    extend = get_extend(vmin=vmin, vmax=vmax)
    vmin_tmp = np.amax(data)
    vmax_tmp = np.amin(data)
    for elm in all_data:
        vmin_2, vmax_2 = var_min_max(elm, grid, zoom_fris=zoom_fris)
        vmin_tmp = min(vmin_tmp, vmin_2)
        vmax_tmp = max(vmax_tmp, vmax_2)
    if vmin is None:
        vmin = vmin_tmp
    if vmax is None:
        vmax = vmax_tmp

    num_frames = len(all_data)

    # Make the first and last frames as stills
    tsteps = [0, -1]
    fig_names = [fig_name_beg, fig_name_end]
    for t in range(2):
        latlon_plot(all_data[tsteps[t]], all_grids[tsteps[t]], gtype=gtype, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points, title=title, date_string=all_dates[tsteps[t]], figsize=figsize, fig_name=fig_names[t], zoom_fris=zoom_fris)

    # Now make the animation

    fig, ax = plt.subplots(figsize=figsize)

    # Inner function to plot a frame
    def plot_one_frame (t):
        img = latlon_plot(all_data[t], all_grids[t], ax=ax, gtype=gtype, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points, title=title+'\n'+all_dates[t], make_cbar=False, zoom_fris=zoom_fris)
        if t==0:
            return img

    # First frame
    img = plot_one_frame(0)        
    plt.colorbar(img, extend=extend)

    # Function to update figure with the given frame
    def animate(t):
        print 'Frame ' + str(t) + ' of ' + str(num_frames)
        ax.cla()
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=range(num_frames))
    writer = animation.FFMpegWriter(bitrate=500, fps=12)
    anim.save(mov_name, writer=writer)
    if mov_name is not None:
        print 'Saving ' + mov_name
        anim.save(mov_name, writer=writer)
    else:
        plt.show()
    


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
# leap_years: boolean (default true) for whether to consider leap years
# For example, to average from month 7 (index 6) of the first file to month 10 (index 9)  of the last file, do
# average_monthly_files(input_files, output_file, t_start=6, t_end=10)

def average_monthly_files (input_files, output_file, t_start=0, t_end=None, leap_years=True):

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
                if month==2 and not leap_years:
                    # Remove any leap day in February
                    ndays = 28
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


# Do a basic average with no weighting of months.
def simple_average_files (input_files, output_file, t_start=0, t_end=None):

    from nco import Nco
    from nco.custom import Limit
    nco = Nco()

    if isinstance(input_files, str):
        input_files = [input_files]

    # Extract partial first and last files, if needed
    num_time_start = netcdf_time(input_files[0]).size
    num_time_end = netcdf_time(input_files[-1]).size
    if t_end is None:
        t_end = num_time_end
    tmp_file_start = None
    tmp_file_end = None
    if t_start > 0:
        tmp_file_start = input_files[0][:-3] + '_tmp.nc'
        nco.ncks(input=input_files[0], output=tmp_file_start, options=[Limit('time', t_start, num_time_start-1)])
        input_files[0] = tmp_file_start
    if t_end < num_time_end:
        tmp_file_end = input_files[-1][:-3] + '_tmp.nc'
        nco.ncks(input=input_files[-1], output=tmp_file_end, options=[Limit('time', 0, t_end-1)])

    # Now average
    nco.ncra(input=input_files, output=output_file)
    # Remove temporary files
    if tmp_file_start is not None:
        os.remove(tmp_file_start)
    if tmp_file_end is not None:
        os.remove(tmp_file_end)

    
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
    ice_prod = read_iceprod(file_path)
    # Also need time
    time = netcdf_time(file_path, monthly=monthly)

    # Set negative values to 0
    ice_prod = np.maximum(ice_prod, 0)

    # Write a new file
    ncfile = NCfile(out_file, grid, 'xyt')
    ncfile.add_time(time)
    ncfile.add_variable('ice_prod', ice_prod, 'xyt', long_name='Net sea ice production', units='m/s')
    ncfile.close()


# Precompute Hovmoller plots (time x depth) for each of the given variables (default temperature and salinity), area-averaged over each of the given regions (default boxes in Pine Island Bay and in front of Dotson).
def precompute_hovmoller (mit_file, hovmoller_file, loc=['pine_island_bay', 'dotson_bay', 'amundsen_west_shelf_break'], var=['temp', 'salt'], monthly=True):

    if isinstance(loc, str):
        # Make it a list
        loc = [loc]

    # Build the grid
    grid = Grid(mit_file)

    # Set up or update the file and time axis
    id = set_update_file(hovmoller_file, grid, 'zt')
    num_time = set_update_time(id, mit_file, monthly=monthly)

    for v in var:
        print 'Processing ' + v
        if v == 'temp':
            var_name = 'THETA'
            title = 'Temperature'
            units = 'degC'
        elif v == 'salt':
            var_name = 'SALT'
            title = 'Salinity'
            units = 'psu'
        # Read data
        data_full = read_netcdf(mit_file, var_name)
        if netcdf_time(mit_file).size == 1:
            # Need a dummy time dimension
            data_full = add_time_dim(data_full, 1)
        # Mask land/ice shelves
        data_full = mask_3d(data_full, grid, time_dependent=True)
        for l in loc:
            print '...at ' + l
            loc_name = region_names[l]
            # Average over the correct region
            if l == 'filchner_front':
                mask = grid.get_icefront_mask(shelf='filchner')
            else:
                mask = grid.get_region_mask(l)
            data = apply_mask(data_full, np.invert(mask), time_dependent=True, depth_dependent=True)
            data = area_average(data, grid, time_dependent=True)
            set_update_var(id, num_time, data, 'zt', l+'_'+v, loc_name+' '+title, units)

    # Finished
    if isinstance(id, nc.Dataset):
        id.close()
    elif isinstance(id, NCfile):
        id.close()


# Call precompute_hovmoller for every segment in a coupled simulation.
def precompute_hovmoller_all_coupled (output_dir='./', hovmoller_file='hovmoller.nc', file_name='output.nc', segment_dir=None, loc=['filchner_trough'], var=['temp', 'salt'], monthly=True):

    output_dir = real_dir(output_dir)
    if segment_dir is None and os.path.isfile(hovmoller_file):
        print 'Error (precompute_hovmoller_all_coupled): since ' + hovmoller_file + ' exists, you must specify segment_dir'
        sys.exit()
    segment_dir = check_segment_dir(output_dir, segment_dir)
    file_paths = segment_file_paths(output_dir, segment_dir, file_name)

    # Call precompute_hovmoller for each segment
    for file_path in file_paths:
        print 'Processing ' + file_path
        precompute_hovmoller(file_path, hovmoller_file, loc=loc, var=var, monthly=monthly)
        

# Make figures to compare two simulations (generally 3-panel figures with 1, 2, and 2-1).
# Input arguments:
# name_1, name_2: simulation names to add to plot titles. No spaces allowed so they can be used in file names.
# dir_1, dir_2: paths to directories containing fname
# fname: name of NetCDF output file (assumes one time index, previously time-averaged with nco)
# fig_dir: directory to save figures in
# Optional keyword arguments:
# hovmoller_file, timeseries_file: 
# key: simulation type key which will set variable types and other settings
def plot_everything_compare (name_1, name_2, dir_1, dir_2, fname, fig_dir, hovmoller_file='hovmoller.nc', timeseries_file='timeseries.nc', key='PAS', ctd_file='../../ctddatabase.mat'):

    from plot_1d import read_plot_timeseries_multi
    from plot_latlon import read_plot_latlon_comparison
    from plot_misc import read_plot_hovmoller_ts, read_plot_hovmoller_ts_diff, amundsen_rignot_comparison, ctd_cast_compare

    if key == 'PAS':
        latlon_names_forcing = ['atemp', 'aqh', 'uwind', 'vwind', 'wind', 'windangle', 'precip', 'swdown', 'lwdown']
        latlon_names = ['bwsalt', 'bwtemp', 'ismr', 'aice', 'hice']
        vmin = [34, None, None, None, None]
        vmax = [None, 1.5, None, None, 4]
        vmin_diff = [-0.3, None, -10, None, None]
        vmax_diff = [0.3, None, None, None, 4]
        change_points = [None, None, [5,10,30], None, None]
        ymax = -70
        melt_types = ['dotson_crosson_melting', 'thwaites_melting', 'pig_melting']
        hovmoller_loc = ['pine_island_bay', 'dotson_bay']
        hovmoller_bounds = [-1.8, 1.2, 34, 34.725]
        hovmoller_t_contours = [0, 1]
        hovmoller_s_contours = [34.5, 34.6, 34.7]
    else:
        print 'Error (plot_everything_compare): need to write the code for simulation key ' + key
        sys.exit()

    dir_1 = real_dir(dir_1)
    dir_2 = real_dir(dir_2)
    fig_dir = real_dir(fig_dir)
    if ' ' in name_1 or ' ' in name_2:
        print 'Error (plot_everything_compare): no spaces allowed in simulation names'
        sys.exit()
    dirs = [dir_1, dir_2]
    names = [name_1, name_2]

    grid = Grid(dir_1+fname)
    # Plot lat-lon forcing variables
    for var_name in latlon_names_forcing:
        read_plot_latlon_comparison(var_name, name_1, name_2, dir_1, dir_2, fname, grid=grid, time_index=0, fig_name=fig_dir+var_name+'.png')
    # Plot lat-lon diagnostic variables
    for n in range(len(latlon_names)):
        read_plot_latlon_comparison(latlon_names[n], name_1, name_2, dir_1, dir_2, fname, grid=grid, time_index=0, fig_name=fig_dir+latlon_names[n]+'.png', vmin=vmin[n], vmax=vmax[n], vmin_diff=vmin_diff[n], vmax_diff=vmax_diff[n], change_points=change_points[n], ymax=ymax)
    # Plot multi timeseries: 1, 2, and difference
    for n in range(2):
        read_plot_timeseries_multi(melt_types, dirs[n]+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_melt_multi_'+names[n]+'.png')
    read_plot_timeseries_multi(melt_types, [dir_1+timeseries_file, dir_2+timeseries_file], diff=True, precomputed=True, fig_name=fig_dir+'timeseries_melt_multi_diff.png')
    # Plot CTD casts and Hovmoller plots: 1, 2, and difference
    for loc in hovmoller_loc:
        for n in range(2):
            ctd_cast_compare(loc, dirs[n]+hovmoller_file, ctd_file, grid, fig_name=fig_dir+'casts_'+loc+'_'+names[n]+'.png')
            read_plot_hovmoller_ts(dirs[n]+hovmoller_file, loc, grid, tmin=hovmoller_bounds[0], tmax=hovmoller_bounds[1], smin=hovmoller_bounds[2], smax=hovmoller_bounds[3], t_contours=hovmoller_t_contours, s_contours=hovmoller_s_contours, fig_name=fig_dir+'hovmoller_ts_'+loc+'_'+names[n]+'.png', smooth=12)            
        read_plot_hovmoller_ts_diff(dir_1+hovmoller_file, dir_2+hovmoller_file, loc, grid, fig_name=fig_dir+'hovmoller_ts_'+loc+'_diff.png', smooth=12)
    if key == 'PAS':
        amundsen_rignot_comparison(dir_1+timeseries_file, file_path_2=dir_2+timeseries_file, precomputed=True, sim_names=[name_1, name_2], fig_name=fig_dir+'rignot.png')


# Read all the output files, and sort them by number
def get_output_files (output_dir):

    output_dir = real_dir(output_dir)
    fnames = os.listdir(output_dir)
    fnames = [f for f in fnames if f.startswith('output_') and f.endswith('.nc')]
    fnames.sort()
    return fnames


# Calculate the long-term mean of a simulation between the given years (inclusive). Return the name of the generated file. Load NCO before you run this.
def long_term_mean (output_dir, year_start, year_end, proper_weighting=True, leap_years=True):

    # Read all the output files, and sort them by number
    output_dir = real_dir(output_dir)
    fnames = get_output_files(output_dir)

    # Loop through to find filenames and indices to start and end
    start_file = None
    end_file = None
    files_to_avg = []
    for f in fnames:
        file_path = output_dir + f
        time = netcdf_time(file_path)
        if start_file is None:
            # Look for start file
            if ((time[0].year < year_start) or (time[0].year == year_start and time[0].month == 1)) and (time[-1].year >= year_start):
                # This is the right file
                start_file = file_path
                files_to_avg.append(file_path)
                # Now find the index of Jan year_start
                for t in range(time.size):
                    if time[t].year == year_start and time[t].month == 1:
                        t_start = t
                        break
        else:
            # Have already found start_file, so we know we need this file
            files_to_avg.append(file_path)
            # Now look for end file
            if (time[-1].year > year_end) or (time[-1].year == year_end and time[-1].month == 12):
                # This is the right file
                end_file = file_path
                # Now find the index of Dec year_end and add one (as per python convention for first index to ignore)
                for t in range(time.size):
                    if time[t].year == year_end and time[t].month == 12:
                        t_end = t+1
                        break
                break

    # Make sure we found them
    if start_file is None:
        print 'Error (long_term_mean): simulation finishes before ' + str(year_start)
        sys.exit()
    if end_file is None:
        print 'Error (long_term_mean): simulation ends before the end of ' + str(year_end)
        sys.exit()

    # Now average them
    print 'Averaging from index ' + str(t_start) + ' of ' + start_file + ' to index ' + str(t_end) + ' of ' + end_file
    out_file = output_dir + str(year_start) + '_' + str(year_end) + '_avg.nc'
    if os.path.isfile(out_file):
        print 'Already exists'
    else:
        if proper_weighting:
            average_monthly_files(files_to_avg, out_file, t_start=t_start, t_end=t_end, leap_years=leap_years)
        else:
            simple_average_files(files_to_avg, out_file, t_start=t_start, t_end=t_end)        
    return out_file


# Precompute both timeseries and Hovmollers for all output files in the (standalone) simulation, or the files in fnames (if set).
def precompute_all (output_dir='./', fnames=None, timeseries_file='timeseries.nc', hovmoller_file='hovmoller.nc', timeseries_types=None, hovmoller_loc=None, obs_file=None, key='PAS', grid=None, time_average=False):

    if key == 'PAS':
        if timeseries_types is None:
            timeseries_types = ['dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'getz_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting', 'eta_avg', 'hice_max', 'pine_island_bay_temp_below_500m', 'pine_island_bay_salt_below_500m', 'dotson_bay_temp_below_500m', 'dotson_bay_salt_below_500m', 'inner_amundsen_shelf_temp_below_500m', 'inner_amundsen_shelf_salt_below_500m', 'amundsen_shelf_break_uwind_avg', 'dotson_massloss', 'pig_massloss', 'getz_massloss', 'inner_amundsen_shelf_sss_avg', 'amundsen_shelf_break_adv_heat_ns_300_1500m']
        if hovmoller_loc is None:
            hovmoller_loc = ['pine_island_bay', 'dotson_bay', 'amundsen_west_shelf_break']
    else:
        print 'Error (precompute_all): invalid key ' + key
        sys.exit()
    if obs_file is None:
        obs_file = '/data/oceans_output/shelf/kaight/ctddatabase.mat'
    output_dir = real_dir(output_dir)

    if fnames is None:
        fnames = get_output_files(output_dir)
    for f in fnames:
        file_path = output_dir + f
        if grid is None:
            grid = Grid(file_path)
        print 'Processing ' + file_path
        precompute_timeseries(file_path, output_dir+timeseries_file, timeseries_types=timeseries_types, grid=grid, time_average=time_average)
        if len(hovmoller_loc) > 0:
            precompute_hovmoller(file_path, output_dir+hovmoller_file, loc=hovmoller_loc)            
    

# All the steps to analyse a newly finished ERA5 run and matching PACE ensemble!
def analyse_pace_ensemble (era5_dir, pace_dir, fig_dir='./', year_start=1979, year_end=2013):

    # Could do:
    # Fix grid lines being on top of timeseries ensembles
    # Fix isotherm NaNs
    # Split into seasonal averages
    # Plot standard deviation as well as mean
    # Compare ismr estimates to what Paul uses in obs/

    from plot_1d import read_plot_timeseries_ensemble
    from plot_latlon import read_plot_latlon_comparison
    from plot_misc import amundsen_rignot_comparison, ctd_cast_compare
    
    latlon_types = ['bwtemp', 'bwsalt', 'sst', 'sss', 'ismr', 'aice', 'hice', 'fwflx']
    ymax = -70
    change_points = [5, 10, 30]

    if isinstance(pace_dir, str):
        # Case for a single ensemble member
        pace_dir = [pace_dir]
    num_ens = len(pace_dir)
    print 'PACE ensemble has ' + str(num_ens) + ' members'
    era5_dir = real_dir(era5_dir) + 'output/'
    for n in range(num_ens):
        pace_dir[n] = real_dir(pace_dir[n]) + 'output/'
    fig_dir = real_dir(fig_dir)
    directories = [era5_dir] + pace_dir
    sim_names = ['ERA5'] + ['PACE '+str(n+1) for n in range(num_ens)]
    
    # Calculate long-term means
    for d in directories:
        print 'Calculating long term mean of ' + d
        file_path = long_term_mean(d, year_start, year_end, leap_years=(d==era5_dir))
        avg_file = file_path[file_path.rfind('/')+1:]

    # Calculate timeseries and Hovmollers
    for d in directories:
        print 'Calculating timeseries and Hovmollers for ' + d
        precompute_all(output_dir=d, key='PAS')

    # Plot ensemble for all timeseries
    for var_name in timeseries_types:
        read_plot_timeseries_ensemble(var_name, timeseries_paths, sim_names=sim_names, precomputed=True, time_use=None, vline=year_start, fig_name=fig_dir+'timeseries_'+var_name+'.png')

    # Plot lat-lon comparison with ERA5
    for var_name in latlon_types:
        vmin = None
        vmax = None
        vmin_diff = None
        vmax_diff = None
        if var_name == 'bwsalt':
            vmin = 34
        if var_name == 'hice':
            vmax = 3
            vmin_diff = -1.5
        if var_name == 'ismr':
            vmin_diff = -6
        if var_name == 'fwflx':
            vmax = 600
            vmin_diff = -50
            vmax_diff = 50
        read_plot_latlon_comparison(var_name, 'ERA5', 'PACE ensemble', era5_dir, pace_dir, avg_file, time_index=0, grid=grid, ymax=ymax, change_points=change_points, vmin=vmin, vmax=vmax, vmin_diff=vmin_diff, vmax_diff=vmax_diff, fig_name=fig_dir+'latlon_'+var_name+'.png')
        
    # Make ismr plots vs Rignot to show range of ensemble
    amundsen_rignot_comparison(timeseries_paths[0], file_path_2=timeseries_paths[1:], precomputed=True, sim_names=['ERA5', 'PACE ensemble'], fig_name=fig_dir+'mean_ismr_rignot.png')

    # Make casts plot showing full ensemble, ERA5, and obs
    for loc in hovmoller_loc:
        for std in [True, False]:
            fig_name = fig_dir + 'casts_' + loc
            if std:
                fig_name += '_std'
            fig_name += '.png'
        ctd_cast_compare(loc, hovmoller_paths[0], obs_file, grid, ens_hovmoller_files=hovmoller_paths[1:], std=std, fig_name=fig_name)
    
    

