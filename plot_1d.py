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
from timeseries import fris_melt, timeseries_max
from plot_utils.labels import monthly_ticks, yearly_ticks
from plot_utils.windows import finished_plot


# Plot timeseries of FRIS' basal mass balance components (melting, freezing, total) at every time index in the given files.

# Arguments:
# file_path: path to NetCDF file containing variable "SHIfwFlx", or a list of such files to concatenate
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
        for file in file_path[1:]:
            melt_tmp, freeze_tmp = fris_melt(file, grid, mass_balance=True)
            time_tmp = netcdf_time(file)
            # Concatenate the arrays
            melt = np.concatenate((melt, melt_tmp))
            freeze = np.concatenate((freeze, freeze_tmp))
            time = np.concatenate((time, time_tmp))

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


# Plot timeseries of the maximum value of the given variable in the given region, at every time index in the given files.

# Arguments:
# file_path: path to NetCDF file containing the variable, or a list of such files to concatenate
# grid: either a Grid object, or the path to a NetCDF grid file

# Optional keyword arguments:
# xmin, xmax, ymin, ymax: as in function var_min_max
# title: title to add to the plot
# units: units for the y-axis
# fig_name: as in function finished_plot

def plot_timeseries_max (file_path, var_name, grid, xmin=None, xmax=None, ymin=None, ymax=None, title='', units='', fig_name=None):

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
        print 'Error (plot_timeseries_max): file_path must be a string or a list'
        sys.exit()
    # Calculate timeseries on the first file
    values = timeseries_max(first_file, var_name, grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # Read time axis
    time = netcdf_time(first_file)
    if isinstance(file_path, list):
        # More files to read
        for file in file_path[1:]:
            values_tmp = timeseries_max(file, var_name, grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            time_tmp = netcdf_time(file)
            # Concatenate the arrays
            values = np.concatenate((values, values_tmp))
            time = np.concatenate((time, time_tmp))

    # Plot
    fig, ax = plt.subplots()
    ax.plot_date(time, values, '-', linewidth=1.5)
    ax.grid(True)
    yearly_ticks(ax)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Maximum sea ice thickness in the southwest corner of the Weddell Sea, between the Ronne and the peninsula.
def plot_hice_corner (file_path, grid, fig_name=None):

    plot_timeseries_max(file_path, 'SIheff', grid, xmin=-62, xmax=-59.5, ymin=-75.5, ymax=-74, title='Maximum sea ice thickness in problematic corner', units='m', fig_name=fig_name)


# Maximum mixed layer depth in the open Eastern Weddell
def plot_mld_ewed (file_path, grid, fig_name=None):

    plot_timeseries_max(file_path, 'MXLDEPTH', grid, xmin=-30, ymin=-69, title='Maximum mixed layer depth in Eastern Weddell', units='m', fig_name=fig_name)
