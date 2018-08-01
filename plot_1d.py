#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from timeseries import calc_special_timeseries, calc_special_timeseries_diff, set_parameters
from plot_utils.labels import monthly_ticks, yearly_ticks
from plot_utils.windows import finished_plot
from file_io import netcdf_time, read_netcdf


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

def make_timeseries_plot (time, data, data_2=None, melt_freeze=False, diff=False, title='', units='', monthly=True, fig_name=None):

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
    if melt_freeze or (np.amin(data) < 0 and np.amax(data) > 0):
        # Add a line at 0
        ax.axhline(color='black')
    ax.grid(True)
    if not monthly:
        monthly_ticks(ax)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# User interface for timeseries plots. Call this function with a specific variable key and a list of NetCDF files to get a nice timeseries plot.

# Arguments:
# var: keyword indicating which timeseries to plot. The options are defined in function set_parameters.
# file_path: if precomputed=False, either a single filename or a list of filenames, to NetCDF files containing the necessary var_name as defined in set_parameters. If precomputed=True, path to the timeseries file created by function precompute_timeseries.

# Optional keyword arguments:
# precomputed: indicates that the timeseries have been precomputed (by function precompute_timeseries in postprocess.py) and saved in file_path
# grid: as in function read_plot_latlon
# fig_name: as in function finished_plot
# monthly: indicates the model output is monthly-averaged

def read_plot_timeseries (var, file_path, precomputed=False, grid=None, fig_name=None, monthly=True):

    # Set parameters (only care about title and units)
    title, units = set_parameters(var)[2:4]

    if precomputed:
        # Read the time array
        time = netcdf_time(file_path)

    if var == 'fris_melt':
        if precomputed:
            # Read the fields from the timeseries file
            data = read_netcdf(file_path, 'fris_total_melt')
            data_2 = read_netcdf(file_path, 'fris_total_freeze')
        else:
            # Calculate the timeseries from the MITgcm file(s)
            time, data, data_2 = calc_special_timeseries(var, file_path, grid=grid, monthly=monthly)
    else:
        if precomputed:
            data = read_netcdf(file_path, var)
        else:
            time, data = calc_special_timeseries(var, file_path, grid=grid, monthly=monthly)
        data_2 = None
        
    # Plot
    make_timeseries_plot(time, data, data_2=data_2, melt_freeze=(var=='fris_melt'), title=title, units=units, monthly=monthly, fig_name=fig_name)



# User interface for difference timeseries. Given simulations 1 and 2, plot the difference (2 minus 1) for the given variable.
# Arguments are the same as in read_plot_timeseries, but two file paths/lists are supplied. It is assumed the two simulations start at the same time, but it's okay if one is longer than the other.
def read_plot_timeseries_diff (var, file_path_1, file_path_2, grid=None, fig_name=None, monthly=True):

    # Set parameters (only care about title and units)
    title, units = set_parameters(var)[2:4]
    # Edit the title to show it's a difference plot
    title = 'Change in ' + title[0].lower() + title[1:]

    # Calculate difference timeseries
    if var == 'fris_melt':
        time, data_diff, data_diff_2 = calc_special_timeseries_diff(var, file_path_1, file_path_2, grid=grid, monthly=monthly)
    else:
        time, data_diff = calc_special_timeseries_diff(var, file_path_1, file_path_2, grid=grid, monthly=monthly)
        data_diff_2 = None

    # Plot
    make_timeseries_plot(time, data_diff, data_2=data_diff_2, melt_freeze=(var=='fris_melt'), diff=True, title=title, units=units, monthly=monthly, fig_name=fig_name)
