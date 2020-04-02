#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from timeseries import calc_special_timeseries, calc_special_timeseries_diff, set_parameters, trim_and_diff, calc_annual_averages
from plot_utils.labels import monthly_ticks, yearly_ticks
from plot_utils.windows import finished_plot
from file_io import netcdf_time, read_netcdf
from utils import trim_titles


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
# dates: indicates "time" is not an array of Dates, but just the values for years
# thick_last: indicates to plot the last line in a thicker weight

# Optional keyword arguments: as in make_timeseries_plot

def timeseries_multi_plot (times, datas, labels, colours, linestyles=None, title='', units='', monthly=True, fig_name=None, dpi=None, legend_in_centre=False, dates=True, thick_last=False):

    # Figure out if time is a list or a single array that applies to all timeseries
    multi_time = isinstance(times, list)
    # Boolean which will tell us whether we need a line at 0
    negative = False
    positive = False
    for data in datas:
        if np.amin(data) < 0:
            negative = True
        if np.amax(data) > 0:
            positive = True
    crosses_zero = negative and positive
    if not dates:
        if multi_time:
            start_time = times[0][0]
            end_time = start_time
            for time in times:
                end_time = max(end_time, time[-1])
        else:
            start_time = times[0]
            end_time = times[-1]

    plot_legend = labels is not None
    if labels is None:
        labels = [None for i in range(len(datas))]        
            
    if linestyles is None:
        linestyles = ['solid' for n in range(len(labels))]

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
        if thick_last and i==len(datas)-1:
            linewidth=3
        else:
            linewidth=1.5
        if dates:
            ax.plot_date(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=linewidth, linestyle=linestyles[i])
        else:
            ax.plot(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=linewidth, linestyle=linestyles[i])
            ax.set_xlim(start_time, end_time)

    ax.grid(True)
    if crosses_zero:
        # Add a line at 0
        ax.axhline(color='black')
    if not monthly:
        monthly_ticks(ax)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    if plot_legend:
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
    

# User interface for timeseries plots. Call this function with a specific variable key and a list of NetCDF files to get a nice timeseries plot. Can also do difference plots (2 minus 1).

# Arguments:
# var: keyword indicating which timeseries to plot. The options are defined in function set_parameters.
# file_path: if precomputed=False, either a single filename or a list of filenames, to NetCDF files containing the necessary var_name as defined in set_parameters. If precomputed=True, path to the timeseries file created by function precompute_timeseries. If diff=True, this must be a list of length 2 (containing either 2 lists or 2 filenames), corresponding to simulations 1 and 2.

# Optional keyword arguments:
# diff: indicates this is a difference plot between two simulations.
# precomputed: indicates that the timeseries have been precomputed (by function precompute_timeseries in postprocess.py) and saved in file_path
# grid: as in function read_plot_latlon
# lon0, lat0: point to interpolate to for var='temp_polynya' or 'salt_polynya'
# wm_name: name of water mass for var='watermass'
# fig_name: as in function finished_plot
# monthly: indicates the model output is monthly-averaged

def read_plot_timeseries (var, file_path, diff=False, precomputed=False, grid=None, lon0=None, lat0=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None):

    if diff and (not isinstance(file_path, list) or len(file_path) != 2):
        print 'Error (read_plot_timeseries): must pass a list of 2 file paths when diff=True.'
        sys.exit()

    if precomputed:
        # Read time arrays
        if diff:
            time_1 = netcdf_time(file_path[0], monthly=(monthly and not precomputed))
            time_2 = netcdf_time(file_path[1], monthly=(monthly and not precomputed))
            time = trim_and_diff(time_1, time_2, time_1, time_2)[0]
        else:
            time = netcdf_time(file_path, monthly=(monthly and not precomputed))

    # Set parameters (only care about title and units)
    title, units = set_parameters(var)[2:4]
    if diff:
        title = 'Change in ' + title[0].lower() + title[1:]

    # Inner function to read a timeseries from both files and calculate the differences, trimming if needed. Only useful if precomputed=True.
    def read_and_trim (var_name):
        data_1 = read_netcdf(file_path[0], var_name)
        data_2 = read_netcdf(file_path[1], var_name)
        data_diff = trim_and_diff(time_1, time_2, data_1, data_2)[1]
        return data_diff

    if var.endswith('mass_balance'):
        if precomputed:
            # Read the fields from the timeseries file
            shelf = var[:var.index('_mass_balance')]
            if diff:
                melt = read_and_trim(shelf+'_total_melt')
                freeze = read_and_trim(shelf+'_total_freeze')
            else:
                melt = read_netcdf(file_path, shelf+'_total_melt')
                freeze = read_netcdf(file_path, shelf+'_total_freeze')
        else:
            # Calculate the timeseries from the MITgcm file(s)
            if diff:
                time, melt, freeze = calc_special_timeseries_diff(var, file_path[0], file_path[1], grid=grid, monthly=monthly)
            else:
                time, melt, freeze = calc_special_timeseries(var, file_path, grid=grid, monthly=monthly)
        timeseries_multi_plot(time, [melt, freeze, melt+freeze], ['Melting', 'Freezing', 'Net'], ['red', 'blue', 'black'], title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi, legend_in_centre=legend_in_centre)
    else:
        if precomputed:
            if diff:
                data = read_and_trim(var)
            else:
                data = read_netcdf(file_path, var)
        else:
            if diff:
                time, data = calc_special_timeseries_diff(var, file_path[0], file_path[1], grid=grid, monthly=monthly)
            else:
                time, data = calc_special_timeseries(var, file_path, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
        make_timeseries_plot(time, data, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi)


# Helper function to set up to 7 colours automatically.
def default_colours (n):

    colours = ['blue', 'red', 'black', 'green', 'cyan', 'magenta', 'yellow']
    if n > len(colours):
        print 'Error (default_colours): need to specify colours if you need more than ' + str(len(colours))
        sys.exit()
    return colours[:n]


# NetCDF interface to timeseries_multi_plot, for multiple variables in the same simulation (that have the same units). Can set diff=True and file_path as a list of two file paths if you want a difference plot.
def read_plot_timeseries_multi (var_names, file_path, diff=False, precomputed=False, grid=None, lon0=None, lat0=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None, colours=None):

    if diff and (not isinstance(file_path, list) or len(file_path) != 2):
        print 'Error (read_plot_timeseries_multi): must pass a list of 2 file paths when diff=True.'
        sys.exit()

    if precomputed:
        # Read time arrays
        if diff:
            time_1 = netcdf_time(file_path[0], monthly=False)
            time_2 = netcdf_time(file_path[1], monthly=False)
            time = trim_and_diff(time_1, time_2, time_1, time_2)[0]
        else:
            time = netcdf_time(file_path, monthly=False)

    # Set up the colours
    if colours is None:
        colours = default_colours(len(var_names))
    
    data = []
    labels = []
    units = None        
    # Loop over variables
    for var in var_names:
        if var.endswith('mass_balance'):
            print 'Error (read_plot_timeseries_multi): ' + var + ' is already a multi-plot by itself.'
            sys.exit()
        title, units_tmp = set_parameters(var)[2:4]
        labels.append(title)
        if units is None:
            units = units_tmp
        elif units != units_tmp:
            print 'Error (read_plot_timeseries_multi): units do not match for all timeseries variables'
            sys.exit()
        if precomputed:
            if diff:
                data_1 = read_netcdf(file_path[0], var)
                data_2 = read_netcdf(file_path[1], var)
                data.append(trim_and_diff(time_1, time_2, data_1, data_2)[1])
            else:
                data.append(read_netcdf(file_path, var))
        else:
            if diff:
                time, data_tmp = calc_special_timeseries_diff(var, file_path[0], file_path[1], grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
            else:
                time, data_tmp = calc_special_timeseries(var, file_path, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
            data.append(data_tmp)
    title, labels = trim_titles(labels)
    if diff:
        title = 'Change in ' + title[0].lower() + title[1:]
    timeseries_multi_plot(time, data, labels, colours, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi, legend_in_centre=legend_in_centre)


# NetCDF interface to timeseries_multi_plot, for the same variable in multiple simulations.

# Arguments:
# var_name: name of timeseries variable to plot (anything from function set_parameters in timeseries.py)
# file_paths: list of length N, of file paths to MITgcm output files or precomputed timeseries files from different simulations.

# Optional keyword arguments:
# sim_names: list of length N, of names for each simulation to show on legend. If not set, there will be no legend.
# precomputed: indicates timeseries is precomputed (otherwise will calculate from model output file)
# grid, lon0, lat0: as in calc_special_timeseries
# plot_mean: boolean indicating to also plot the ensemble mean in thicker black
# first_in_mean: boolean indicating to include the first simulation in the mean (default True; set to False if you want to exclude it, e.g. ERA5 compared to PACe ensemble mean)
# annual_average: boolean indicating to annually average the data before plotting
# time_use: index of simulation to use the time axis for all variables (<= N); set to None if you want the different simulations to have different time axes (eg if they're not the same length and you're not plotting the mean).
# colours: list of length N of colours to use for plot - if not set, will choose colours automatically
# linestyles: list of length N of linestyles to use for the plot (default solid for all)
# fig_name, monthly, legend_in_centre, dpi: as in timeseries_multi_plot

def read_plot_timeseries_ensemble (var_name, file_paths, sim_names=None, precomputed=False, grid=None, lon0=None, lat0=None, plot_mean=False, first_in_mean=True, annual_average=False, time_use=0, colours=None, linestyles=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None):

    if var_name.endswith('mass_balance'):
        print 'Error (read_plot_timeseries_ensemble): This function does not work for mass balance terms.'
        sys.exit()

    # Read data
    all_times = []
    all_datas = []
    for f in file_paths:
        if precomputed:
            time = netcdf_time(f, monthly=False)
            data = read_netcdf(f, var_name)
        else:
            time, data = calc_special_timeseries(var_name, f, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
        all_times.append(time)
        all_datas.append(data)
    if time_use is None:
        time = all_times
    else:
        # Make sure all simulations are the same length, and then choose one time axis to use
        if any([t.size != all_times[0].size for t in all_times]):
            print 'Error (read_plot_timeseries_ensemble): not all the simulations are the same length.'
            sys.exit()
        time = all_times[time_use]

    if annual_average:
        # Make sure it's an integer number of 30-day months
        calendar = netcdf_time(file_paths[0], return_units=True)[2]
        if calendar != '360_day' or not monthly or time.size%12 != 0:
            print 'Error (read_plot_timeseries_ensemble): can only do true annual averages if there are an integer number of 30-day months.'
            sys.exit()
        time, all_datas = calc_annual_averages(time, all_datas)

    # Set other things for plot
    title, units = set_parameters(var_name)[2:4]
    if colours is None:
        colours = default_colours(len(file_paths))

    if plot_mean:
        if first_in_mean:
            n0 = 0
        else:
            n0 = 1
        if time_use is None and any([t.size != all_times[n0].size for t in all_times[n0:]]):
            print 'Error (read_plot_timeseries_ensemble): can only calculate mean if simulations are same length.'
            sys.exit()            
        # Calculate the mean
        all_datas.append(np.mean(all_datas[n0:], axis=0))
        # Plot in thicker black
        # First replace any black in the colours array
        if 'black' in colours:
            colours[colours.index('black')] = (0.6, 0.6, 0.6)
        colours.append('black')
        if sim_names is not None:
            sim_names.append('Mean')

    timeseries_multi_plot(time, all_datas, sim_names, colours, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi, legend_in_centre=legend_in_centre, thick_last=plot_mean, linestyles=linestyles)
