#######################################################
# Other figures you might commonly make
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import datetime

from grid import choose_grid, Grid
from file_io import check_single_time, find_variable, read_netcdf, netcdf_time, read_title_units
from plot_utils.labels import check_date_string, depth_axis, yearly_ticks, lon_label, lat_label
from plot_utils.windows import finished_plot, set_panels
from plot_utils.colours import get_extend, set_colours
from utils import mask_3d, xy_to_xyz, z_to_xyz, var_min_max_zt, mask_outside_box
from diagnostics import tfreeze
from constants import deg_string, rignot_melt
from interpolation import interp_bilinear
from calculus import area_average
from timeseries import trim_and_diff, timeseries_ismr


# Create a temperature vs salinity distribution plot. Temperature and salinity are split into NxN bins (default N=1000) and the colour of each bin shows the log of the volume of water masses in that bin.

# Arguments:
# file_path: path to NetCDF file containing the variable THETA and/or SALT. You can specify a second file for the second variable in second_file_path if needed.

# Optional keyword arguments:
# option: 'fris' (only plot water masses in FRIS cavity; default), 'cavities' (only plot water masses in all ice shelf cavities), or 'all' (plot water masses from all parts of the model domain).
# grid: a Grid object OR path to a grid directory OR path to a NetCDF file containing the grid variables. If you specify nothing, the grid will be read from file_path.
# time_index, t_start, t_end, time_average: as in function read_netcdf. You must either define time_index or set time_average=True, so it collapses to a single record.
# second_file_path: path to NetCDF file containing the variable THETA or SALT, if they are not both present in file_path
# tmin, tmax, smin, smax: bounds on temperature and salinity to plot
# num_bins: number of temperature and salinity bins used to categorise the water masses. Default is 1000, but if you're zooming in quite a lot using tmin etc., you might want to increase this.
# date_string: as in function latlon_plot
# figsize: size of figure you want
# fig_name: as in function finished_plot

# Suggested bounds for WSK simulation:
# option='fris': smin=34.2
# option='cavities': smin=33.5, tmax=1, num_bins=2000
# option='all': smin=33, tmax=1.5, num_bins=2000

def ts_distribution_plot (file_path, option='fris', grid=None, time_index=None, t_start=None, t_end=None, time_average=False, second_file_path=None, tmin=None, tmax=None, smin=None, smax=None, num_bins=1000, date_string=None, figsize=(8,6), fig_name=None):

    # Build the grid if needed
    grid = choose_grid(grid, file_path)
    # Make sure we'll end up with a single record in time
    check_single_time(time_index, time_average)
    # Determine what to write about the date
    date_string = check_date_string(date_string, file_path, time_index)

    # Quick inner function to read data (THETA or SALT)
    def read_data (var_name):
        # First choose the right file
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path)
        else:
            file_path_use = file_path
        data = read_netcdf(file_path_use, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        return data
    # Call this function for each variable
    temp = read_data('THETA')
    salt = read_data('SALT')

    # Select the points we care about
    if option == 'fris':
        # Select all points in the FRIS cavity
        loc_index = (grid.hfac > 0)*xy_to_xyz(grid.get_ice_mask(shelf='fris'), grid)
    elif option == 'cavities':
        # Select all points in ice shelf cavities
        loc_index = (grid.hfac > 0)*xy_to_xyz(grid.ice_mask, grid)
    elif option == 'all':
        # Select all unmasked points
        loc_index = grid.hfac > 0
    else:
        print 'Error (plot_misc): invalid option ' + option
        sys.exit()

    # Inner function to set up bins for a given variable (temp or salt)
    def set_bins (data):
        # Find the bounds on the data at the points we care about
        vmin = np.amin(data[loc_index])
        vmax = np.amax(data[loc_index])
        # Choose a small epsilon to add/subtract from the boundaries
        # This way nothing will be at the edge of a beginning/end bin
        eps = (vmax-vmin)*1e-3
        # Calculate boundaries of bins
        bins = np.linspace(vmin-eps, vmax+eps, num=num_bins)
        # Now calculate the centres of bins for plotting
        centres = 0.5*(bins[:-1] + bins[1:])
        return bins, centres
    # Call this function for each variable
    temp_bins, temp_centres = set_bins(temp)
    salt_bins, salt_centres = set_bins(salt)
    # Now set up a 2D array to increment with volume of water masses
    volume = np.zeros([temp_centres.size, salt_centres.size])

    # Loop over all cells to increment volume
    # This can't really be vectorised unfortunately
    fris_mask = grid.get_ice_mask(shelf='fris')
    for i in range(grid.nx):
        for j in range(grid.ny):
            if option=='fris' and not fris_mask[j,i]:
                # Disregard all points not in FRIS cavity
                continue
            if option=='cavities' and not grid.ice_mask[j,i]:
                # Disregard all points not in ice shelf cavities
                continue            
            for k in range(grid.nz):
                if grid.hfac[k,j,i] == 0:
                    # Disregard all masked points
                    continue
                # If we're still here, it's a point we care about
                # Figure out which bins it falls into
                temp_index = np.nonzero(temp_bins > temp[k,j,i])[0][0] - 1
                salt_index = np.nonzero(salt_bins > salt[k,j,i])[0][0] - 1
                # Increment volume array
                volume[temp_index, salt_index] += grid.dV[k,j,i]
    # Mask bins with zero volume
    volume = np.ma.masked_where(volume==0, volume)

    # Find the volume bounds for plotting
    min_vol = np.log(np.amin(volume))
    max_vol = np.log(np.amax(volume))
    # Calculate the surface freezing point for plotting
    tfreeze_sfc = tfreeze(salt_centres, 0)
    # Choose the plotting bounds if not set
    if tmin is None:
        tmin = temp_bins[0]
    if tmax is None:
        tmax = temp_bins[-1]
    if smin is None:
        smin = salt_bins[0]
    if smax is None:
        smax = salt_bins[-1]
    # Construct the title
    title = 'Water masses'
    if option=='fris':
        title += ' in FRIS cavity'
    elif option=='cavities':
        title += ' in ice shelf cavities'
    if date_string != '':
        title += ', ' + date_string

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # Use a log scale for visibility
    img = plt.pcolor(salt_centres, temp_centres, np.log(volume), vmin=min_vol, vmax=max_vol)
    # Add the surface freezing point
    plt.plot(salt_centres, tfreeze_sfc, color='black', linestyle='dashed', linewidth=2)
    ax.grid(True)
    ax.set_xlim([smin, smax])
    ax.set_ylim([tmin, tmax])
    plt.xlabel('Salinity (psu)')
    plt.ylabel('Temperature ('+deg_string+'C)')
    plt.colorbar(img)
    plt.text(.9, .6, 'log of volume', ha='center', rotation=-90, transform=fig.transFigure)
    plt.title(title)
    finished_plot(fig, fig_name=fig_name)


# Plot a Hovmoller plot of the given 2D data field.

# Arguments:
# data: 2D array of data (time x depth). Assumes it is not on the w-grid.
# time: array of Date objects corresponding to time axis.
# grid: Grid object.

# Optional keyword arguments:
# smooth: radius for moving average (0 means no smoothing)
# ax, make_cbar, ctype, vmin, vmax, title, titlesize, return_fig, fig_name, extend, fig_size, dpi: as in latlon_plot
# zmin, zmax: bounds on depth axis to plot (negative, in metres, zmin is the deep bound).
# monthly: as in netcdf_time
# contours: list of values to contour in black over top

def hovmoller_plot (data, time, grid, smooth=0, ax=None, make_cbar=True, ctype='basic', vmin=None, vmax=None, zmin=None, zmax=None, monthly=True, contours=None, title=None, titlesize=18, return_fig=False, fig_name=None, extend=None, figsize=(14,5), dpi=None):

    import cftime

    # Choose what the endpoints of the colourbar should do
    if extend is None:
        extend = get_extend(vmin=vmin, vmax=vmax)

    # If we're zooming, we need to choose the correct colour bounds
    if any([zmin, zmax]):
        vmin_tmp, vmax_tmp = var_min_max_zt(data, grid, zmin=zmin, zmax=zmax)
        if vmin is None:
            vmin = vmin_tmp
        if vmax is None:
            vmax = vmax_tmp
    # Get colourmap
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax)

    if monthly:
        # As in netcdf_time, the time axis will have been corrected so it is
        # marked with the beginning of each month. So to get the boundaries of
        # each time index, we just need to add one month to the end.
        if time[-1].month == 12:
            end_time = datetime.datetime(time[-1].year+1, 1, 1)
        else:
            end_time = datetime.datetime(time[-1].year, time[-1].month+1, 1)
        time_edges = np.concatenate((time, [end_time]))
    else:
        # Following MITgcm convention, the time axis will be stamped with the
        # first day of the next averaging period. So to get the boundaries of
        # each time index, we just need to extrapolate to the beginning,
        # assuming regularly spaced time intervals.
        dt = time[1]-time[0]
        start_time = time[0] - dt
        time_edges = np.concatenate(([start_time], time))

    # Smooth with a moving average
    # Will have to trim each end by one radius
    t_first = smooth
    t_last = data.shape[0]-smooth  # First one not selected as per python convection
    data_cumsum = np.ma.concatenate((np.zeros((1,data.shape[1])), np.ma.cumsum(data, axis=0)), axis=0)
    data = (data_cumsum[t_first+smooth+1:t_last+smooth+1,:] - data_cumsum[t_first-smooth:t_last-smooth,:])/(2*smooth+1)
    # Now trim the time axis too
    time_edges = time_edges[smooth:time_edges.size-smooth]            

    # Make the figure and axes, if needed
    existing_ax = ax is not None
    if not existing_ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the data
    img = ax.pcolormesh(time_edges, grid.z_edges, np.transpose(data), cmap=cmap, vmin=vmin, vmax=vmax)
    if contours is not None:
        # Overlay contours
        # Need time at the centres of each index
        # Have to do this with a loop unfortunately
        time_centres = []
        for t in range(time_edges.size-1):
            dt = (time_edges[t+1]-time_edges[t])/2
            time_centres.append(time_edges[t]+dt)
        plt.contour(time_centres, grid.z, np.transpose(data), levels=contours, colors='black', linestyles='solid')

    # Set depth limits
    if zmin is None:
        # Index of last masked cell
        k_bottom = np.argwhere(np.invert(data[0,:].mask))[-1][0]
        zmin = grid.z_edges[k_bottom+1]
    if zmax is None:
        # Index of first unmasked cell
        k_top = np.argwhere(np.invert(data[0,:].mask))[0][0]
        zmax = grid.z_edges[k_top]    
    ax.set_ylim([zmin, zmax])
    # Make nice axes labels
    depth_axis(ax)
    if make_cbar:
        # Add a colourbar
        plt.colorbar(img, extend=extend)
    if title is not None:
        # Add a title
        plt.title(title, fontsize=titlesize)

    if return_fig:
        return fig, ax
    elif existing_ax:
        return img
    else:
        finished_plot(fig, fig_name=fig_name, dpi=dpi)


# Creates a double Hovmoller plot with temperature on the top and salinity on the bottom.
def hovmoller_ts_plot (temp, salt, time, grid, smooth=0, tmin=None, tmax=None, smin=None, smax=None, zmin=None, zmax=None, monthly=True, t_contours=None, s_contours=None, ctype='basic', loc_string='', fig_name=None, figsize=(12,7), dpi=None):

    # Set panels
    fig, gs, cax_t, cax_s = set_panels('2x1C2')
    # Wrap things up in lists for easier iteration
    data = [temp, salt]
    vmin = [tmin, smin]
    vmax = [tmax, smax]
    contours = [t_contours, s_contours]
    title = ['Temperature ('+deg_string+'C)', 'Salinity (psu)']
    cax = [cax_t, cax_s]
    for i in range(2):
        ax = plt.subplot(gs[i,0])
        # Make the plot
        img = hovmoller_plot(data[i], time, grid, smooth=smooth, ax=ax, make_cbar=False, vmin=vmin[i], vmax=vmax[i], zmin=zmin, zmax=zmax, monthly=monthly, contours=contours[i], ctype=ctype, title=title[i])
        # Add a colourbar
        extend = get_extend(vmin=vmin[i], vmax=vmax[i])
        plt.colorbar(img, cax=cax[i], extend=extend)
        if i == 0:
            # Remove x-tick labels from top plot
            ax.set_xticklabels([])            
    plt.suptitle(loc_string, fontsize=22)
    finished_plot(fig, fig_name=fig_name, dpi=dpi)
    

# Read a precomputed Hovmoller file (from precompute_hovmoller in postprocess.py) and make the plot.

# Arguments:
# var: variable name in precomputed file, in the form loc_var, such as 'PIB_temp'
# hovmoller_file: path to precomputed Hovmoller file
# grid: Grid object or path to grid file/directory

# Optional keyword arguments:
# smooth, zmin, zmax, vmin, vmax, contours, monthly, fig_name, figsize: as in hovmoller_plot
def read_plot_hovmoller (var_name, hovmoller_file, grid, smooth=0, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, monthly=True, fig_name=None, figsize=(14,5)):

    data = read_netcdf(hovmoller_file, var_name)
    # Set monthly=False so we don't back up an extra month (because precomputed)
    time = netcdf_time(hovmoller_file, monthly=False)
    title, units = read_title_units(hovmoller_file, var_name)

    grid = choose_grid(grid, None)

    # Make the plot
    hovmoller_plot(data, time, grid, smooth=smooth, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax, monthly=monthly, contours=contours, title=title, fig_name=fig_name, figsize=figsize)


# Read precomputed data for temperature and salinity and make a T/S Hovmoller plot.
def read_plot_hovmoller_ts (hovmoller_file, loc, grid, smooth=0, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, t_contours=None, s_contours=None, fig_name=None, monthly=True, figsize=(12,7), dpi=None):

    grid = choose_grid(grid, None)
    temp = read_netcdf(hovmoller_file, loc+'_temp')
    salt = read_netcdf(hovmoller_file, loc+'_salt')
    time = netcdf_time(hovmoller_file, monthly=False)
    if loc == 'PIB':
        loc_string = 'Pine Island Bay '
    elif loc == 'Dot':
        loc_string = 'Dotson front '
    hovmoller_ts_plot(temp, salt, time, grid, smooth=smooth, tmin=tmin, tmax=tmax, smin=smin, smax=smax, zmin=zmin, zmax=zmax, monthly=monthly, t_contours=t_contours, s_contours=s_contours, loc_string=loc_string, fig_name=fig_name, figsize=figsize, dpi=dpi)


# Helper function for difference plots
# Returns time and difference in given variable over the same time indices
def read_and_trim_diff (file_1, file_2, var_name):

    time_1 = netcdf_time(file_1, monthly=False)
    time_2 = netcdf_time(file_2, monthly=False)
    data_1 = read_netcdf(file_1, var_name)
    data_2 = read_netcdf(file_2, var_name)
    time, data_diff = trim_and_diff(time_1, time_2, data_1, data_2)
    return time, data_diff


# Difference plots (2 minus 1)
def read_plot_hovmoller_diff (var_name, hovmoller_file_1, hovmoller_file_2, grid, smooth=0, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, monthly=True, fig_name=None, figsize=(14,5)):

    time, data_diff = read_and_trim_diff(hovmoller_file_1, hovmoller_file_2, var_name)
    title, units = read_title_units(hovmoller_file_1, var_name)
    grid = choose_grid(grid, None)
    hovmoller_plot(data_diff, time, grid, smooth=smooth, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax, monthly=monthly, contours=contours, ctype='plusminus', title='Change in '+title, fig_name=fig_name, figsize=figsize)


def read_plot_hovmoller_ts_diff (hovmoller_file_1, hovmoller_file_2, loc, grid, smooth=0, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, t_contours=None, s_contours=None, fig_name=None, monthly=True, figsize=(12,7), dpi=None):

    time, temp_diff = read_and_trim_diff(hovmoller_file_1, hovmoller_file_2, loc+'_temp')
    salt_diff = read_and_trim_diff(hovmoller_file_1, hovmoller_file_2, loc+'_salt')[1]
    grid = choose_grid(grid, None)
    if loc == 'PIB':
        loc_string = 'Pine Island Bay '
    elif loc == 'Dot':
        loc_string = 'Dotson front '
    hovmoller_ts_plot(temp_diff, salt_diff, time, grid, smooth=smooth, tmin=tmin, tmax=tmax, smin=smin, smax=smax, zmin=zmin, zmax=zmax, monthly=monthly, t_contours=t_contours, s_contours=s_contours, ctype='plusminus', loc_string=loc_string, fig_name=fig_name, figsize=figsize, dpi=dpi)


# Compare simulated mean melt rates for each Amundsen Sea ice shelf with the range given by Rignot 2013. The blue dots are the model output and the black error ranges are from Rignot.
# Input arguments:
# file_path: path to precomputed timeseries file (set precomputed=True) or a NetCDF model output file, containing SHIfwFlx data for the whole simulation (or a time-average - it will be averaged if it's not already).
# Optional keyword arguments:
# precomputed: set to True if file_path is a precomputed timeseries file with the melt rates for each ice shelf already there
# option: 'melting' (plot melt rates in m/y) or 'massloss' (plot basal mass loss)
# file_path_2: file_path for a second simulation, to plot on same axes.
# sim_names: list of length 2 containing simulation names for file_path and file_path_2.
# fig_name: as in function finished_plot
def amundsen_rignot_comparison (file_path, precomputed=False, option='melting', file_path_2=None, sim_names=None, fig_name=None):

    shelf_names = ['getz', 'dotson_crosson', 'thwaites', 'pig', 'cosgrove', 'abbot', 'venable']
    shelf_titles = ['Getz', 'Dotson &\nCrosson', 'Thwaites', 'Pine Island', 'Cosgrove', 'Abbot', 'Venable']
    num_shelves = len(shelf_names)

    if not precomputed:
        grid = Grid(file_path)
        
    second = file_path_2 is not None
    if second and (sim_names is None or not isinstance(sim_names, list) or len(sim_names) != 2):
        print 'Error (amundsen_rignot_comparison): must set sim_names as list of 2 simulation names if file_path_2 is set.'
        sys.exit()

    model_melt = []
    if second:
        model2_melt = []
    obs_melt = []
    obs_std = []
    for shelf in shelf_names:
        var_name = shelf+'_'+option
        if precomputed:
            model_melt.append(read_netcdf(file_path, var_name, time_average=True))
            if second:
                model2_melt.append(read_netcdf(file_path_2, var_name, time_average=True))
        else:
            model_melt.append(timeseries_ismr(file_path, grid, shelf=shelf, result=option, time_average=True))
            if second:
                model2_melt.append(timeseries_ismr(file_path_2, grid, shelf=shelf, result=option, time_average=True))
        obs = rignot_melt[shelf]
        if option == 'massloss':
            obs_melt.append(obs[0])
            obs_std.append(obs[1])
        elif option == 'melting':
            obs_melt.append(obs[2])
            obs_std.append(obs[3])
        else:
            print 'Error (amundsen_rignot_comparison): invalid option ' + option
            sys.exit()

    fig, ax = plt.subplots()
    if second:
        ax.plot(range(num_shelves), model_melt, 'o', color='blue', label=sim_names[0])
        ax.plot(range(num_shelves), model2_melt, 'o', color='green', label=sim_names[1])
    else:
        ax.plot(range(num_shelves), model_melt, 'o', color='blue', label='MITgcm')
    ax.errorbar(range(num_shelves), obs_melt, yerr=obs_std, fmt='none', color='black', capsize=4, label='Observations')
    ax.legend()
    ax.grid(True)
    plt.xticks(range(num_shelves), shelf_titles, rotation='vertical')
    plt.subplots_adjust(bottom=0.2)
    if option == 'massloss':
        title = 'Basal mass loss'
        units = 'Gt/y'
    elif option == 'melting':
        title = 'Average melt rate'
        units = 'm/y'
    plt.title(title, fontsize=16)
    plt.ylabel(units, fontsize=12)
    finished_plot(fig, fig_name=fig_name)
            
    
        
    

    
    
    
    
    
    
    
    


