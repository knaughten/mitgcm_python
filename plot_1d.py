#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime

from .timeseries import calc_special_timeseries, calc_special_timeseries_diff, set_parameters, trim_and_diff, calc_annual_averages
from .plot_utils.labels import monthly_ticks, yearly_ticks
from .plot_utils.windows import finished_plot
from .file_io import netcdf_time, read_netcdf
from .utils import trim_titles, moving_average, index_period, index_year_start


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
    if np.amin(data1) < 0 and np.amax(data1) > 0 and np.amin(data2) < 0 and np.amax(data2) > 0:
        # Both timeseries cross 0. Line them up there.
        val1 = max(-np.amin(data1), np.amax(data1))
        val2 = max(-np.amin(data2), np.amax(data2))
        ax1.set_ylim([-val1, val1])
        ax2.set_ylim([-val2, val2])
        ax1.axhline(color='black')
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

def timeseries_multi_plot (times, datas, labels, colours, linestyles=None, alphas=None, title='', units='', monthly=True, fig_name=None, dpi=None, legend_in_centre=False, legend_outside=True, dates=True, thick_last=False, thick_first=False, first_on_top=False, vline=None, return_fig=False, year_ticks=None):

    # Figure out if time is a list or a single array that applies to all timeseries
    multi_time = isinstance(times, list)
    # Booleans which will tell us whether we need a line at 0 or 100
    negative = False
    positive = False
    lt_100 = False
    gt_100 = False
    for data in datas:
        if np.amin(data) < 0:
            negative = True
        if np.amax(data) > 0:
            positive = True
        if np.amin(data) < 100:
            lt_100 = True
        if np.amax(data) > 100:
            gt_100 = True
    crosses_zero = negative and positive
    crosses_100 = lt_100 and gt_100 and units[0] == '%'
    if multi_time:
        start_time = times[0][0]
        end_time = start_time
        for time in times:
            if time[-1] > end_time:
                end_time = time[-1]
    else:
        start_time = times[0]
        end_time = times[-1]

    plot_legend = labels is not None
    if labels is None:
        labels = [None for i in range(len(datas))]        
            
    if linestyles is None:
        linestyles = ['solid' for n in range(len(labels))]
    if alphas is None:
        alphas = [1 for n in range(len(labels))]

    if legend_outside:
        figsize=(11,6)
    else:
        figsize=(8,6)
    fig, ax = plt.subplots(figsize=figsize)
    # Plot each line
    for i in range(len(datas)):
        if multi_time:
            time = times[i]
        else:
            time = times
        if (thick_first and i==0) or (thick_last and i==len(datas)-1):
            linewidth=2
        else:
            linewidth=1
        if first_on_top and i==0:
            if dates:
                ax.plot_date(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=linewidth, linestyle=linestyles[i], alpha=alphas[i], zorder=len(datas))
            else:
                ax.plot(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=linewidth, linestyle=linestyles[i], alpha=alphas[i], zorder=len(datas))
        else:            
            if dates:
                ax.plot_date(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=linewidth, linestyle=linestyles[i], alpha=alphas[i])
            else:
                ax.plot(time, datas[i], '-', color=colours[i], label=labels[i], linewidth=linewidth, linestyle=linestyles[i], alpha=alphas[i])
        ax.set_xlim(start_time, end_time)

    ax.grid(linestyle='dotted')
    if crosses_zero:
        # Add a line at 0
        ax.axhline(color='black', linestyle='dashed')
    if crosses_100:
        # Add a line at 100%
        ax.axhline(100, color='black', linestyle='dashed')
    if vline is not None:
        if dates:
            vline = datetime.date(vline, 1, 1)
        ax.axvline(vline, color='black', linestyle='dashed')
    if not monthly:
        monthly_ticks(ax)
    if year_ticks is not None:
        if dates:
            year_ticks = [datetime.date(y,1,1) for y in year_ticks]
        ax.set_xticks(year_ticks)
    plt.title(title, fontsize=18)
    plt.ylabel(units, fontsize=16)
    if not dates:
        plt.xlabel('Years', fontsize=16)
    if plot_legend:
        if legend_outside:
            # Move plot over to make room for legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            # Make legend
            ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        elif legend_in_centre:
            ax.legend(loc='center')
        else:
            ax.legend(loc='best')
    if return_fig:
        return fig, ax
    else:
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

def read_plot_timeseries (var, file_path, diff=False, precomputed=False, grid=None, lon0=None, lat0=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None, annual_average=False, smooth=0):

    if diff and (not isinstance(file_path, list) or len(file_path) != 2):
        print('Error (read_plot_timeseries): must pass a list of 2 file paths when diff=True.')
        sys.exit()

    if precomputed:
        # Read time arrays
        if diff:
            time_1 = netcdf_time(file_path[0], monthly=(monthly and not precomputed))
            time_2 = netcdf_time(file_path[1], monthly=(monthly and not precomputed))
            calendar = netcdf_time(file_path[0], return_units=True)[2]
            time = trim_and_diff(time_1, time_2, time_1, time_2)[0]
        else:
            time = netcdf_time(file_path, monthly=(monthly and not precomputed))
            calendar = netcdf_time(file_path, return_units=True)[2]

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
        if annual_average:
            time, [melt, freeze] = calc_annual_averages(time, [melt, freeze])
        melt = moving_average(melt, smooth, time=time)[0]
        freeze, time = moving_average(freeze, smooth, time=time)
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
        if annual_average:
            time, data = calc_annual_averages(time, data)
        data, time = moving_average(data, smooth, time=time)
        make_timeseries_plot(time, data, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi)


# Helper function to set colours automatically.
def default_colours (n):

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if n > len(colours):
        print(('Error (default_colours): must specify colours if there are more than ' + len(colours)))
        sys.exit()
    return colours[:n]


# NetCDF interface to timeseries_multi_plot, for multiple variables in the same simulation (that have the same units). Can set diff=True and file_path as a list of two file paths if you want a difference plot.
def read_plot_timeseries_multi (var_names, file_path, diff=False, precomputed=False, grid=None, lon0=None, lat0=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None, colours=None, smooth=0, annual_average=False):

    if diff and (not isinstance(file_path, list) or len(file_path) != 2):
        print('Error (read_plot_timeseries_multi): must pass a list of 2 file paths when diff=True.')
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
    if annual_average:
        time_orig = np.copy(time)
    # Loop over variables
    for var in var_names:
        if var.endswith('mass_balance'):
            print(('Error (read_plot_timeseries_multi): ' + var + ' is already a multi-plot by itself.'))
            sys.exit()
        title, units_tmp = set_parameters(var)[2:4]
        labels.append(title)
        if units is None:
            units = units_tmp
        elif units != units_tmp:
            print('Error (read_plot_timeseries_multi): units do not match for all timeseries variables')
            sys.exit()
        if precomputed:
            if diff:
                data_1 = read_netcdf(file_path[0], var)
                data_2 = read_netcdf(file_path[1], var)
                data_tmp = trim_and_diff(time_1, time_2, data_1, data_2)[1]
            else:
                data_tmp = read_netcdf(file_path, var)
        else:
            if diff:
                time, data_tmp = calc_special_timeseries_diff(var, file_path[0], file_path[1], grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
            else:
                time, data_tmp = calc_special_timeseries(var, file_path, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
        if annual_average:
            time, data_tmp = calc_annual_averages(time_orig, data_tmp)
        data.append(data_tmp)
    for n in range(len(data)):
        data_tmp, time_tmp = moving_average(data[n], smooth, time=time)        
        data[n] = data_tmp        
    time = time_tmp
    title, labels = trim_titles(labels)
    if diff:
        title = 'Change in ' + title[0].lower() + title[1:]
    timeseries_multi_plot(time, data, labels, colours, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi, legend_in_centre=legend_in_centre)


# NetCDF interface to timeseries_multi_plot, for the same variable in multiple simulations.

# Arguments:
# var_name: name of timeseries variable to plot (anything from function set_parameters in timeseries.py) - can also be several such variable names, which will be added together (must set title and units)
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
# title, units: set these strings if var_name is multiple variables to sum.
# print_mean: set to True if you want to print the mean value for each ensemble member
# operator: 'add' or 'subtract' each additional variable name after the first one (if var_name is a list); default add
# plot_anomaly, base_year_start, base_year_end: will plot as an anomaly from the average over the given years
# trim_before: if base_year_start is set, don't show any timeseries before that year

def read_plot_timeseries_ensemble (var_name, file_paths, sim_names=None, precomputed=False, grid=None, lon0=None, lat0=None, plot_mean=False, first_in_mean=True, annual_average=False, time_use=0, colours=None, linestyles=None, fig_name=None, monthly=True, legend_in_centre=False, dpi=None, smooth=0, title=None, units=None, print_mean=False, operator='add', vline=None, alpha=False, plot_anomaly=False, base_year_start=None, base_year_end=None, trim_before=False, base_year_start_first=None, percent=False, year_ticks=None):

    if isinstance(var_name, str):
        var_name = [var_name]
    if (plot_anomaly or percent) and (base_year_start is None or base_year_end is None):
        print('Error (read_plot_timeseries_ensemble): must set base_year_start and base_year_end')
        sys.exit()

    # Read data
    all_times = []
    all_datas = []
    for f in file_paths:
        data = None
        for var in var_name:
            if var.endswith('mass_balance'):
                print('Error (read_plot_timeseries_ensemble): This function does not work for mass balance terms.')
                sys.exit()
            if precomputed:
                time = netcdf_time(f, monthly=False)
                data_tmp = read_netcdf(f, var)
            else:
                time, data_tmp = calc_special_timeseries(var, f, grid=grid, lon0=lon0, lat0=lat0, monthly=monthly)
            if data is None:
                data = data_tmp
            else:
                if operator == 'add':
                    data += data_tmp
                elif operator == 'subtract':
                    data -= data_tmp
                else:
                    print(('Error (read_plot_timeseries_ensemble): invalid operator ' + operator))
                    sys.exit()
        if plot_anomaly or percent or trim_before:
            # Find the time indices that define the baseline period
            if time[0].year > base_year_start:
                if (not plot_anomaly) and (not percent):
                    # This is ok
                    t_start = 0
                    if base_year_start_first is not None and f == file_paths[0]:
                        # A tighter constraint on start year
                        t_start = index_year_start(time, base_year_start_first)
                else:
                    print('Error (read_plot_timeseries_ensemble): this simulation does not cover the baseline period')
                    sys.exit()
            else:
                t_start, t_end = index_period(time, base_year_start, base_year_end)
                # Calculate the mean over that period
                data_mean = np.mean(data[t_start:t_end])
            if plot_anomaly:
                # Subtract the mean
                data -= data_mean
            if percent:
                # Express as percentage of mean
                data = data/data_mean*100
            if trim_before:
                # Trim everything before the baseline period
                data = data[t_start:]
                time = time[t_start:]                
        all_times.append(time)
        all_datas.append(data)
    if time_use is None:
        time = all_times
    else:
        # Make sure all simulations are the same length, and then choose one time axis to use
        if any([t.size != all_times[0].size for t in all_times]):
            print('Error (read_plot_timeseries_ensemble): not all the simulations are the same length.')
            sys.exit()
        time = all_times[time_use]

    if annual_average:
        # Make sure it's an integer number of 30-day months
        calendar = netcdf_time(file_paths[0], return_units=True)[2]
        if calendar != '360_day' or not monthly or time.size%12 != 0:
            print('Error (read_plot_timeseries_ensemble): can only do true annual averages if there are an integer number of 30-day months.')
            sys.exit()
        time, all_datas = calc_annual_averages(time, all_datas)

    if smooth != 0:
        for n in range(len(all_datas)):
            if time_use is None:
                data_tmp, time_tmp = moving_average(all_datas[n], smooth, time=time[n])
                time[n] = time_tmp
            else:
                data_tmp, time_tmp = moving_average(all_datas[n], smooth, time=time)
            all_datas[n] = data_tmp
        if time_use is not None:
            time = time_tmp

    # Set other things for plot
    if len(var_name)==1:
        title, units = set_parameters(var_name[0])[2:4]
    elif title is None or units is None:
        print('Error (read_plot_timeseries_ensemble): must set title and units')
        sys.exit()
    if percent:
        units = '% of '+str(base_year_start)+'-'+str(base_year_end)+' mean'
    if plot_anomaly:
        title += ' \n(anomaly from '+str(base_year_start)+'-'+str(base_year_end)+' mean)'
    if colours is None:
        colours = default_colours(len(file_paths))
    if alpha:
        alphas = [1] + [0.5 for n in range(len(file_paths)-1)]
    else:
        alphas = None

    if plot_mean:
        if first_in_mean:
            n0 = 0
        else:
            n0 = 1
        if time_use is None and any([t.size != all_times[n0].size for t in all_times[n0:]]):
            print('Error (read_plot_timeseries_ensemble): can only calculate mean if simulations are same length.')
            sys.exit()            
        # Calculate the mean
        all_datas.append(np.mean(all_datas[n0:], axis=0))
        all_times.append(all_times[n0])
        if len(colours) != len(all_datas):
            # Choose a colour
            # First replace any black in the colours array
            if 'black' in colours:
                colours[colours.index('black')] = (0.4, 0.4, 0.4)
            colours.append('black')
        if alphas is not None:
            alphas.append(1)
        if sim_names is not None:
            sim_names.append('Mean')

    if print_mean:
        print(('Mean values for ' + title + ':'))
        for data, sim in zip(all_datas, sim_names):
            print((sim + ': ' + str(np.mean(data)) + ' ' + units))

    timeseries_multi_plot(time, all_datas, sim_names, colours, title=title, units=units, monthly=monthly, fig_name=fig_name, dpi=dpi, legend_in_centre=legend_in_centre, thick_last=plot_mean, thick_first=(plot_mean and not first_in_mean), linestyles=linestyles, alphas=alphas, first_on_top=(plot_mean and not first_in_mean), vline=vline, year_ticks=year_ticks)
