#######################################################
# Files to create when the simulation is done
#######################################################

import os

from grid import Grid
from io import NCfile, read_binary, netcdf_time
from plots_1d import plot_fris_massbalance
from plots_latlon import read_plot_latlon, plot_aice_minmax


# Make a bunch of plots when the simulation is done.
# This will keep evolving over time! For now it is all the 2D lat-lon plots at the last time index, a timeseries of FRIS mass loss throughout the entire simulation, and sea ice min and max at each year within the last segment.

# Arguments:
# output_dir: path to directory containing output NetCDF files (assumed to be in one file per segment a la scripts/convert_netcdf.py)
# grid_path: path to NetCDF grid file

# Optional keyword arguments:
# fig_dir: path to directory to save figures in
# file_path: specific output file to analyse for non-time-dependent plots (default the most recent segment)

def plot_everything (output_dir, grid_path, fig_dir='.', file_path=None):

    # Make sure proper directories
    if not output_dir.endswith('/'):
        output_dir += '/'
    if not fig_dir.endswith('/'):
        fig_dir += '/'
    # Build the list of output files in this directory (use them all for timeseries)
    output_files = []
    for file in os.listdir(output_dir):
        if file.startswith('output') and file.endswith('.nc'):
            output_files.append(output_dir+file)
    # Make sure in chronological order
    output_files.sort()
    if file_path is None:
        # Select the last file for single-timestep analysis
        file_path = output_files[-1]        

    # Build the grid
    grid = Grid(grid_path)

    # Timeseries
    plot_fris_massbalance(output_files, grid, fig_name=fig_dir+'fris_massloss.png')

    # Lat-lon plots
    var_names = ['ismr', 'bwtemp', 'bwsalt', 'sst', 'sss', 'aice', 'hice', 'mld', 'eta', 'saltflx', 'tminustf', 'vel', 'velice']
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
        # Plot
        read_plot_latlon(var, file_path, grid, time_index=-1, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_name)
        # Make additional plots if needed
        if var in ['ismr', 'vel', 'bwtemp', 'bwsalt']:
            # Make another plot zoomed into FRIS
            # First adjust bounds
            if var == 'bwtemp':
                vmax = -1.5
            read_plot_latlon(var, file_path, grid, time_index=-1, vmin=vmin, vmax=vmax, zoom_fris=True, fig_name=fig_dir+var+'_zoom.png')
        if var == 'tminustf':
            # Call the other options for vertical transformations
            read_plot_latlon(var, file_path, grid, time_index=-1, tf_option='max', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_max.png')
        if var == 'vel':
            # Call the other options for vertical transformations            
            read_plot_latlon(var, file_path, grid, time_index=-1, vel_option='sfc', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_sfc.png')
            read_plot_latlon(var, file_path, grid, time_index=-1, vel_option='bottom', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, fig_name=fig_dir+var+'_bottom.png')
        if var in ['eta', 'hice']:
            # Make another plot with unbounded colour bar
            read_plot_latlon(var, file_path, grid, time_index=-1, zoom_fris=zoom_fris, fig_name=fig_dir + var + '_unbound.png')

    # Sea ice min and max
    time = netcdf_time(file_path)
    first_year = time[0].year
    last_year = time[-1].year
    for year in range(first_year, last_year+1):
        plot_aice_minmax(file_path, grid, year, fig_name=fig_dir+'aice_minmax_'+str(year)+'.png')


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
