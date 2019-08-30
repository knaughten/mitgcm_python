#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from timeseries import calc_special_timeseries, calc_special_timeseries_diff, set_parameters, trim_and_diff
from plot_utils.labels import monthly_ticks, yearly_ticks
from plot_utils.windows import finished_plot
from file_io import netcdf_time, read_netcdf


# Helper function to plot timeseries.

# Arguments:
# time: 1D array of Date objects corresponding to time of each record
# data: 1D array of timeseries to plot

# Optional keyword arguments:
# title: title for plot
# units: units of timeseries
# monthly: as in function netcdf_time
# fig_name: as in function finished_plot

def make_timeseries_plot (time, data, title='', units='', monthly=True, fig_name=None, dpi=None):

    fig, ax = plt.subplots()
    ax.plot_date(time, data, '-', linewidth=1.5)
    if np.amin(data) < 0 and np.amax(data) > 0:
        # Add a line at 0
        ax.axhline(color='black')
    ax.grid(True)
    if not monthly:
        monthly_ticks(ax)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    finished_plot(fig, fig_name=fig_name, dpi=dpi)


# Plot two different variables on the same axes, with different scales, in blue and red respectively.
def make_timeseries_plot_2sided (time, data1, data2, title, units1, units2, monthly=True, fig_name=None, dpi=None):

    fig, ax1 = plt.subplots(figsize=(9,6))
    ax1.plot_date(time, data1, '-', linewidth=1.5, color='blue')
    ax1.grid(True)
    if not monthly:
        monthly_ticks(ax1)
    ax1.set_ylabel(units1, color='blue', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot_date(time, data2, '-', linewidth=1.5, color='red')
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.set_ylabel(units2, color='red', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title(title, fontsize=18)
    finished_plot(fig, fig_name=fig_name, dpi=dpi)        


# Plot multiple timeseries on the same axes.

# Arguments:
# time: either a 1D array of time values for all timeseries, or a list of 1D arrays of time values for each timeseries
# datas: list of 1D arrays of timeseries
# labels: list of legend labels (strings) to use for each timeseries
# colours: list of colours (strings or RGB tuples) to use for each timeseries

# Optional keyword arguments: as in make_timeseries_plot

def timeseries_multi_plot (times, datas, labels, colours, title='', units='', monthly=True, fig_name=None, dpi=None, legend_in_centre=False):

    # Figure out if time is a list or a single array that applies to all timeseries
    multi_time = isinstance(times, list)
    # Boolean which will tell us whether we need a line at 0
    crosses_zero = np.amin(datas)<0 and np.amax(datas)>0

    if legend_in_centre:
        figsize=(8,6)
    else:
        figsize=(11,6)
    fig, ax = plt.subplots(figsize=figsize)
    # Plot each line
    for i in range(len(datas)):
        if multi_time:
            time = times[i]
        else:
            time = times
        ax.plot_date(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=1.5)

    ax.grid(True)
    if crosses_zero:
        # Add a line at 0
        ax.axhline(color='black')
    if not monthly:
        monthly_ticks(ax)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    if legend_in_centre:
        # Make legend
        ax.legend(loc='center')
    else:
        # Move plot over to make room for legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        # Make legend
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    finished_plot(fig, fig_name=fig_name, dpi=dpi)    
    

# User interface for timeseries plots. Call this function with a specific variable key and a list of NetCDF files to get a nice timeseries plot.

# Arguments:
# var: keyword indicating which timeseries to plot. The options are defined in function set_parameters.
# file_path: if precomputed=False, either a single filename or a list of filenames, to NetCDF files containing the necessary var_name as defined in set_parameters. If precomputed=True, path to the timeseries file created by function precompute_timeseries.

# Optional keyword arguments:
# precomputed: indicates that the timeseries have been precomputed (by function precompute_timeseries in postprocess.py) and saved in file_path
# grid: as in function read_plot_latlon
# lon0, lat0: point to interpolate to for var='temp_polynya' or 'salt_polynya'
# wm_name: name of water mass for var='watermass'
# fig_name: as in function finished_plot
# monthly: indicates the model output is monthly-averaged

def read_plot_timeseries (var, file_path, precomputed=False, grid=None, lon0=None, lat0=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None):

    # Set parameters (only care about title and units)
    title, units = set_parameters(var)[2:4]

    if precomputed:
        # Read the time array; don't need to back up one month
        time = netcdf_time(file_path, monthly=False)

    if var == 'fris_mass_balance':
        if precomputed:
            # Read the fields from the timeseries file
            melt = read_netcdf(file_path, 'fris_total_melt')
            freeze = read_netcdf(file_path, 'fris_total_freeze')
        else:
            # Calculate the timeseries from the MITgcm file(s)
            time, melt, freeze = calc_special_timeseries(var, file_path, grid=grid, monthly=monthly)
        timeseries_multi_plot(time, [melt, freeze, melt+freeze], ['Melting', 'Freezing', 'Net'], ['red', 'blue', 'black'], title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi, legend_in_centre=legend_in_centre)
    else:
        if precomputed:
            data = read_netcdf(file_path, var)
        else:
            time, data = calc_special_timeseries(var, file_path, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
        make_timeseries_plot(time, data, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi)



# User interface for difference timeseries. Given simulations 1 and 2, plot the difference (2 minus 1) for the given variable.
# Arguments are the same as in read_plot_timeseries, but two file paths/lists are supplied. It is assumed the two simulations start at the same time, but it's okay if one is longer than the other.
def read_plot_timeseries_diff (var, file_path_1, file_path_2, precomputed=False, grid=None, lon0=None, lat0=None, fig_name=None, monthly=True):

    # Set parameters (only care about title and units)
    title, units = set_parameters(var)[2:4]
    # Edit the title to show it's a difference plot
    title = 'Change in ' + title[0].lower() + title[1:]

    if precomputed:
        # Read the time arrays
        time_1 = netcdf_time(file_path_1, monthly=False)
        time_2 = netcdf_time(file_path_2, monthly=False)

    # Inner function to read a timeseries from both files and calculate the differences, trimming if needed. Only useful if precomputed=True.
    def read_and_trim (var_name):
        data_1 = read_netcdf(file_path_1, var_name)
        data_2 = read_netcdf(file_path_2, var_name)
        time, data_diff = trim_and_diff(time_1, time_2, data_1, data_2)
        return time, data_diff

    if var == 'fris_mass_balance':
        if precomputed:
            time, melt_diff = read_and_trim('fris_total_melt')
            freeze_diff = read_and_trim('fris_total_freeze')[1]
        else:
            # Calculate the difference timeseries
            time, melt_diff, freeze_diff = calc_special_timeseries_diff(var, file_path_1, file_path_2, grid=grid, monthly=monthly)
        timeseries_multi_plot(time, [melt_diff, freeze_diff, melt_diff+freeze_diff], ['Change in melting\n(>0)', 'Change in freezing\n(<0)', 'Change in net'], ['red', 'blue', 'black'], title=title, units=units, monthly=monthly, fig_name=fig_name)
    else:
        if precomputed:
            time, data_diff = read_and_trim(var)
        else:
            time, data_diff = calc_special_timeseries_diff(var, file_path_1, file_path_2, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
        make_timeseries_plot(time, data_diff, title=title, units=units, monthly=monthly, fig_name=fig_name)
