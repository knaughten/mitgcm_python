#######################################################
# Files to create when the simulation is done
#######################################################

import os

from grid import Grid
from io import NCfile, read_binary
from plots_1d import plot_fris_massbalance
from plots_latlon import read_plot_latlon


# Make a bunch of plots when the simulation is done.
# This will keep evolving over time! For now it is all the 2D lat-lon plots at the last time index, and a timeseries of FRIS mass loss.

# Arguments:
# file_path: path to output NetCDF file for this simulation chunk (assumed to be all in one file a la scripts/convert_netcdf.py)
# grid_path: path to NetCDF grid file
# fig_dir: path to directory to save figures in

def plot_everything (file_path, grid_path, fig_dir):

    # Make sure fig_dir is a proper directory
    if not fig_dir.endswith('/'):
        fig_dir += '/'

    # Build the grid
    grid = Grid(grid_path)

    # Timeseries
    plot_fris_massbalance(file_path, grid, fig_name=fig_dir+'fris_massloss.png')

    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'mld', 'eta', 'saltflx', 'tminustf', 'vel', 'velice']
    for var in var_names:
        # Customise bounds and zooming
        vmin = None
        vmax = None
        zoom_fris = False
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
            zoom_fris=True
        # Plot
        read_plot_latlon(var, file_path, grid, time_index=-1, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'.png')
        # Make additional plots if needed
        if var in ['ismr', 'vel']:
            # Make another plot zoomed into FRIS
            read_plot_latlon(var, file_path, grid, time_index=-1, vmin=vmin, vmax=vmax, zoom_fris=True, fig_name=fig_dir+var+'_zoom.png')
        if var == 'tminustf':
            # Call the other options for vertical transformations
            read_plot_latlon(var, file_path, grid, time_index=-1, tf_option='max', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_max.png')
            read_plot_latlon(var, file_path, grid, time_index=-1, tf_option='min', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_min.png')
        if var == 'vel':
            # Call the other options for vertical transformations            
            read_plot_latlon(var, file_path, grid, time_index=-1, vel_option='sfc', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_sfc.png')
            read_plot_latlon(var, file_path, grid, time_index=-1, vel_option='bottom', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_bottom.png')    


# When the model crashes, convert its crash-dump to a NetCDF file.

# Arguments:
# crash_dir: directory including all the state*crash.*.data files. The NetCDF file will be saved here too, with the name crash.nc.
# grid_path: path to NetCDF grid file.

def crash_to_netcdf (crash_dir, grid_path):

    # Make sure crash_dir is a proper directory
    if not crash_dir.endswith('/'):
        crash_dir += '/'

    # Read the grid
    grid = Grid(grid_path)
    # Initialise the NetCDF file
    ncfile = NCfile(crash_dir+'crash.nc', grid, 'xyz')

    # Find all the crash files
    for file in os.listdir(crash_dir):
        if file.startswith('stateThetacrash') and file.endswith('.data'):
            # Found temperature
            # Read it from binary
            temp = read_binary(crash_dir + file, grid, 'xyz')
            # Write it to NetCDF
            ncfile.add_variable('THETA', temp, 'xyz', units='C')
        if file.startswith('stateSaltcrash') and file.endswith('.data'):
            salt = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('SALT', salt, 'xyz', units='psu')
        if file.startswith('stateUvelcrash') and file.endswith('.data'):
            u = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('UVEL', u, 'xyz', gtype='u', units='m/s')
        if file.startswith('stateVvelcrash') and file.endswith('.data'):
            v = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('VVEL', v, 'xyz', gtype='v', units='m/s')
        if file.startswith('stateWvelcrash') and file.endswith('.data'):
            w = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('WVEL', w, 'xyz', gtype='w', units='m/s')
        if file.startswith('stateEtacrash') and file.endswith('.data'):
            eta = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('ETAN', eta, 'xy', units='m')
        if file.startswith('stateAreacrash') and file.endswith('.data'):
            area = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIarea', area, 'xy', units='fraction')
        if file.startswith('stateHeffcrash') and file.endswith('.data'):
            heff = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIheff', heff, 'xy', units='m')
        if file.startswith('stateUicecrash') and file.endswith('.data'):
            uice = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIuice', uice, 'xy', gtype='u', units='m/s')
        if file.startswith('stateVicecrash') and file.endswith('.data'):
            vice = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIvice', vice, 'xy', gtype='v', units='m/s')

    ncfile.finished()
