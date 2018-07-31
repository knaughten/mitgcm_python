#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from grid import choose_grid
from file_io import netcdf_time
from timeseries import fris_melt, timeseries_max, timeseries_avg_ss
from plot_utils.labels import monthly_ticks, yearly_ticks
from plot_utils.windows import finished_plot


# Helper function to calculate timeseries from one or more files.

# Arguments:
# file_path: either a single filename or a list of filenames

# Optional keyword arguments:
# option: either 'fris_melt' (calculates total melting and freezing beneath FRIS), 'max' (calculates maximum value of variable in region; must specify var_name and possibly xmin etc.), or 'avg_ss' (calculates area-averaged value over the sea surface, i.e. not counting cavities)
# grid: as in function read_plot_latlon
# gtype: as in function read_plot_latlon
# var_name: if option='max' or 'avg_ss', variable name to calculate the maximum or area-average of
# xmin, xmax, ymin, ymax: as in function var_min_max
# monthly: as in function netcdf_time

# Output:
# if option='fris_melt', returns three 1D arrays of time, melting, and freezing.
# if option='max', returns two 1D arrays of time and maximum values.
# if option='avg_ss', returns two 1D arrays of time and area-averaged values.
# if option='time', just returns the time array.

def read_timeseries (file_path, option=None, grid=None, gtype='t', var_name=None, xmin=None, xmax=None, ymin=None, ymax=None, monthly=True):

    if isinstance(file_path, str):
        # Just one file
        first_file = file_path
    elif isinstance(file_path, list):
        # More than one
        first_file = file_path[0]
    else:
        print 'Error (read_timeseries): file_path must be a string or a list'
        sys.exit()

    if option in ['max', 'avg_ss'] and var_name is None:
        print 'Error (read_timeseries): must specify var_name'
        sys.exit()

    # Build the grid if needed
    if option != 'time':
        grid = choose_grid(grid, first_file)

    # Calculate timeseries on the first file
    if option == 'fris_melt':
        melt, freeze = fris_melt(first_file, grid, mass_balance=True)
    elif option == 'max':
        values = timeseries_max(first_file, var_name, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    elif option == 'avg_ss':
        values = timeseries_avg_ss(first_file, var_name, grid, gtype=gtype)
    elif option != 'time':
        print 'Error (read_timeseries): invalid option ' + option
        sys.exit()
    # Read time axis
    time = netcdf_time(first_file, monthly=monthly)
    if isinstance(file_path, list):
        # More files to read
        for file in file_path[1:]:
            if option == 'fris_melt':
                melt_tmp, freeze_tmp = fris_melt(file, grid, mass_balance=True)
            elif option == 'max':
                values_tmp = timeseries_max(file, var_name, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            elif option == 'avg_ss':
                values_tmp = timeseries_avg_ss(file, var_name, grid, gtype=gtype)
            time_tmp = netcdf_time(file, monthly=monthly)
            # Concatenate the arrays
            if option == 'fris_melt':
                melt = np.concatenate((melt, melt_tmp))
                freeze = np.concatenate((freeze, freeze_tmp))
            elif option in ['max', 'avg_ss']:
                values = np.concatenate((values, values_tmp))
            time = np.concatenate((time, time_tmp))

    if option == 'fris_melt':
        return time, melt, freeze
    elif option in ['max', 'avg_ss']:
        return time, values
    elif option == 'time':
        return time


# Helper function to plot timeseries.

# Arguments:
# time: 1D array of Date objects corresponding to time of each record
# data: 1D array of timeseries to plot

# Optional keyword arguments:
# melt_freeze: boolean (default False) indicating to plot melting, freezing, and total. Assumes melting is given by "data" and freezing by "data_2".
# data_2: if melt_freeze=True, array of freezing timeseries
# diff: boolean (default False) indicating this is an anomaly timeseries. Only matters for melt_freeze as it will change the legend labels.
# title: title for plot
# units: units of timeseries
# monthly: as in function netcdf_time
# fig_name: as in function finished_plot

def plot_timeseries (time, data, data_2=None, melt_freeze=False, diff=False, title='', units='', monthly=True, fig_name=None):

    fig, ax = plt.subplots()
    if melt_freeze:
        if diff:
            melt_label = 'Change in melting (>0)'
            freeze_label = 'Change in freezing (<0)'
            total_label = 'Change in net'
        else:
            melt_label = 'Melting'
            freeze_label = 'Freezing'
            total_label = 'Net'
        ax.plot_date(time, data, '-', color='red', linewidth=1.5, label=melt_label)
        ax.plot_date(time, data_2, '-', color='blue', linewidth=1.5, label=freeze_label)
        ax.plot_date(time, data+data_2, '-', color='black', linewidth=1.5, label=total_label)
        ax.legend()
    else:
        ax.plot_date(time, data, '-', linewidth=1.5)
    ax.axhline(color='black')
    ax.grid(True)
    if not monthly:
        monthly_ticks(ax)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Helper function to calculate difference timeseries, trimming if needed.

# Arguments:
# time_1, time_2: 1D arrays containing time values for the two simulations (assumed to start at the same time, but might not be the same length)
# data_1, data_2: 1D arrays containing timeseries for the two simulations

# Output:
# time: 1D array containing time values for the overlapping period of simulation
# data_diff: 1D array containing differences (data_2 - data_1) at these times
def trim_and_diff (time_1, time_2, data_1, data_2):

    num_time = min(time_1.size, time_2.size)
    time = time_1[:num_time]
    data_diff = data_2[:num_time] - data_1[:num_time]
    return time, data_diff
    

# Plot timeseries of FRIS' basal mass balance components (melting, freezing, total) at every time index in the given files.

# Arguments:
# file_path: path to NetCDF file containing variable "SHIfwFlx", or a list of such files to concatenate

# Optional keyword arguments:
# grid: as in function read_plot_latlon
# fig_name: as in function finished_plot
# monthly: as in function netcdf_time

def plot_fris_massbalance (file_path, grid=None, fig_name=None, monthly=True):

    # Calculate timeseries
    time, melt, freeze = read_timeseries(file_path, option='fris_melt', grid=grid, monthly=monthly)
    # Plot
    plot_timeseries(time, melt, data_2=freeze, melt_freeze=True, title='Basal mass balance of FRIS', units='Gt/y', monthly=monthly, fig_name=fig_name)


# Plot the difference in FRIS melting and freezing for two simulations (2 minus 1). It is assumed the two simulations start at the same time, but it's okay if one is longer - it will get trimmed.

def plot_fris_massbalance_diff (file_path_1, file_path_2, grid=None, fig_name=None, monthly=True):

    # Calculate timeseries for each
    time_1, melt_1, freeze_1 = read_timeseries(file_path_1, option='fris_melt', grid=grid, monthly=monthly)
    time_2, melt_2, freeze_2 = read_timeseries(file_path_2, option='fris_melt', grid=grid, monthly=monthly)
    # Find the difference, trimming if needed
    time, melt_diff = trim_and_diff(time_1, time_2, melt_1, melt_2)
    freeze_diff = trim_and_diff(time_1, time_2, freeze_1, freeze_2)[1]
    # Plot
    plot_timeseries(time, melt_diff, data_2=freeze_diff, melt_freeze=True, diff=True, title='Change in basal mass balance of FRIS', units='Gt/y', monthly=monthly, fig_name=fig_name)    


# Plot timeseries of the maximum value of the given variable in the given region, at every time index in the given files.

# Arguments:
# file_path: path to NetCDF file containing the variable, or a list of such files to concatenate

# Optional keyword arguments:
# grid: as in function read_plot_latlon
# xmin, xmax, ymin, ymax: as in function var_min_max
# title: title to add to the plot
# units: units for the y-axis
# fig_name: as in function finished_plot
# monthly: as in function netcdf_time

def plot_timeseries_max (file_path, var_name, grid=None, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, title='', units='', fig_name=None, monthly=True):

    time, values = read_timeseries(file_path, option='max', var_name=var_name, grid=grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, monthly=monthly)

    plot_timeseries(time, values, title=title, units=units, monthly=monthly, fig_name=fig_name)


# Plot the difference in the maximum value of the given variable in the given region, between two simulations (2 minus 1). It is assumed the two simulations start at the same time, but it's okay if one is longer - it will get trimmed.

def plot_timeseries_max_diff (file_path_1, file_path_2, var_name, grid=None, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, title='', units='', fig_name=None, monthly=True):

    time_1, values_1 = read_timeseries(file_path_1, option='max', var_name=var_name, grid=grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, monthly=monthly)
    time_2, values_2 = read_timeseries(file_path_2, option='max', var_name=var_name, grid=grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, monthly=monthly)
    time, values_diff = trim_and_diff(time_1, time_2, values_1, values_2)
    plot_timeseries(time, values_diff, title=title, units=units, monthly=monthly, fig_name=fig_name)


# Maximum sea ice thickness in the southwest corner of the Weddell Sea, between the Ronne and the peninsula.
def plot_hice_corner (file_path, grid=None, fig_name=None, monthly=True):

    plot_timeseries_max(file_path, 'SIheff', grid=grid, xmin=-62, xmax=-59.5, ymin=-75.5, ymax=-74, title='Maximum sea ice thickness in problematic corner', units='m', fig_name=fig_name, monthly=monthly)


# Difference in this maximum sea ice between two simulations (2 minus 1).
def plot_hice_corner_diff (file_path_1, file_path_2, grid=None, fig_name=None, monthly=True):

    plot_timeseries_max_diff(file_path_1, file_path_2, 'SIheff', grid=grid, xmin=-62, xmax=-59.5, ymin=-75.5, ymax=-74, title='Change in maximum sea ice thickness in problematic corner', units='m', fig_name=fig_name, monthly=monthly)


# Maximum mixed layer depth in the open Eastern Weddell
def plot_mld_ewed (file_path, grid=None, fig_name=None, monthly=True):

    plot_timeseries_max(file_path, 'MXLDEPTH', grid=grid, xmin=-30, ymin=-69, title='Maximum mixed layer depth in Eastern Weddell', units='m', fig_name=fig_name, monthly=monthly)


# Difference in this maximum mixed layer depth between two simulations (2 minus 1).
def plot_mld_ewed_diff (file_path_1, file_path_2, grid=None, fig_name=None, monthly=True):

    plot_timeseries_max_diff(file_path_1, file_path_2, 'MXLDEPTH', grid=grid, xmin=-30, ymin=-69, title='Change in maximum mixed layer depth in Eastern Weddell', units='m', fig_name=fig_name, monthly=monthly)


# Plot timeseries of the area-averaged sea surface height, at every time index in the given file(s).
def plot_eta_avg (file_path, grid=None, fig_name=None, monthly=True):

    time, eta = read_timeseries(file_path, option='avg_ss', var_name='ETAN', grid=grid, monthly=monthly)
    plot_timeseries(time, eta, title='Area-averaged sea surface height', units='m', monthly=monthly, fig_name=fig_name)


# Difference in the area-averaged sea surface height between two simulations (2 minus 1).
def plot_eta_avg_diff (file_path_1, file_path_2, grid=None, fig_name=None, monthly=True):

    time_1, eta_1 = read_timeseries(file_path_1, option='avg_ss', var_name='ETAN', grid=grid, monthly=monthly)
    time_2, eta_2 = read_timeseries(file_path_2, option='avg_ss', var_name='ETAN', grid=grid, monthly=monthly)
    time, eta_diff = trim_and_diff(time_1, time_2, values_1, values_2)
    plot_timeseries(time, eta_diff, title='Change in area-averaged sea surface height', units='m', monthly=monthly, fig_name=fig_name)
