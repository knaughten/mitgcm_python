#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from grid import Grid
from io import netcdf_time
from timeseries import fris_melt
from plot_utils.labels import monthly_ticks, yearly_ticks
from plot_utils.windows import finished_plot


# Plot timeseries of FRIS' basal mass balance components (melting, freezing, total) at every time index in the given file.

# Arguments:
# file_path: path to NetCDF file containing variable "SHIfwFlx", or a list of such files to plot sequentially
# grid: either a Grid object, or the path to NetCDF grid file

# Optional keyword argument:
# fig_name: as in function finished_plot

def plot_fris_massbalance (file_path, grid, fig_name=None):

    if not isinstance(grid, Grid):
        # This is the path to the NetCDF grid file, not a Grid object
        # Make a grid object from it
        grid = Grid(grid)

    if isinstance(file_path, str):
        # Just one file
        first_file = file_path
    elif isinstance(file_path, list):
        # More than one
        first_file = file_path[0]
    else:
        print 'Error (plot_fris_massbalance): file_path must be a string or a list'
        sys.exit()
    # Calculate timeseries on the first file
    melt, freeze = fris_melt(first_file, grid, mass_balance=True)
    # Read time axis
    time = netcdf_time(first_file)
    if isinstance(file_path, list):
        # More files to read
        for file in file_paths[1:]:
            melt_tmp, freeze_tmp = fris_melt(file, grid, mass_balance=True)
            time_tmp = netcdf_time(file)
            # Concatenate the arrays
            melt = np.concatenate(melt, melt_tmp)
            freeze = np.concatenate(freeze, freeze_tmp)
            time = np.concatenate(time, time_tmp)    

    # Plot
    fig, ax = plt.subplots()
    ax.plot_date(time, melt, '-', color='red', linewidth=1.5, label='Melting')
    ax.plot_date(time, freeze, '-', color='blue', linewidth=1.5, label='Freezing')
    ax.plot_date(time, melt+freeze, '-', color='black', linewidth=1.5, label='Total')
    ax.axhline(color='black')
    ax.grid(True)
    yearly_ticks(ax)
    plt.title('Basal mass balance of FRIS', fontsize=18)
    plt.ylabel('Gt/y', fontsize=16)
    ax.legend()
    finished_plot(fig, fig_name=fig_name)


    
