#######################################################
# Other figures you might commonly make
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import datetime
import itertools

from .grid import choose_grid, Grid
from .file_io import check_single_time, find_variable, read_netcdf, netcdf_time, read_title_units
from .plot_utils.labels import check_date_string, depth_axis, yearly_ticks, lon_label, lat_label, reduce_cbar_labels
from .plot_utils.windows import finished_plot, set_panels
from .plot_utils.colours import get_extend, set_colours
from .plot_1d import timeseries_multi_plot
from .utils import mask_3d, xy_to_xyz, z_to_xyz, var_min_max_zt, mask_outside_box, moving_average, mask_2d_to_3d
from .diagnostics import tfreeze, potential_density
from .constants import deg_string, rignot_melt, region_bounds, region_names
from .interpolation import interp_bilinear
from .calculus import area_average
from .timeseries import trim_and_diff, timeseries_ismr, calc_annual_averages


# Helper function to split temperature and salinity in the given region (set by mask) into bins, to get the volume in m^3 of each bin. The arrays can be time-dependent if you want. You can set the bounds of the bins, but they must be at least as permissive as the bounds of the data in that region.
def ts_binning (temp, salt, grid, mask, time_dependent=False, num_bins=1000, tmin=None, tmax=None, smin=None, smax=None):

    if len(mask.shape)==2:
        # Get 3D version of 2D mask
        mask = mask_2d_to_3d(mask, grid)
    if time_dependent:
        num_time = time.size
    else:
        num_time = 1

    # Inner function to get min and max values in region
    def get_vmin_vmax (data):
        vmin = np.amax(data)
        vmax = np.amin(data)
        for t in range(num_time):
            if time_dependent:
                data_tmp = data[t,:]
            else:
                data_tmp = data
            vmin = min(vmin, np.amin(data_tmp[mask]))
            vmax = max(vmax, np.amax(data_tmp[mask]))
        return [vmin, vmax]
    print('Calculating bounds')
    temp_bounds = get_vmin_vmax(temp)
    salt_bounds = get_vmin_vmax(salt)
    if tmin is not None:
        if tmin > temp_bounds[0]:
            print('Error (ts_binning): tmin is too high')
            sys.exit()
        temp_bounds[0] = tmin
    if tmax is not None:
        if tmax < temp_bounds[-1]:
            print('Error (ts_binning): tmax is too low')
            sys.exit()
        temp_bounds[1] = tmax
    if smin is not None:
        if smin > salt_bounds[0]:
            print('Error (ts_binning): smin is too high')
            sys.exit()
        salt_bounds[0] = smin
    if smax is not None:
        if smax < salt_bounds[-1]:
            print('Error (ts_binning): smax is too low')
            sys.exit()
        salt_bounds[1] = smax            

    # Set up bins
    def set_bins (bounds):
        eps = (bounds[1]-bounds[0])*1e-3
        edges = np.linspace(bounds[0]-eps, bounds[1]+eps, num=num_bins+1)
        centres = 0.5*(edges[:-1] + edges[1:])
        return edges, centres
    temp_edges, temp_centres = set_bins(temp_bounds)
    salt_edges, salt_centres = set_bins(salt_bounds)
    if time_dependent:
        volume = np.zeros([num_time, num_bins, num_bins])
    else:
        volume = np.zeros([num_bins, num_bins])

    # Now categorise the values
    print('Binning T and S')
    for t in range(num_time):
        if time_dependent:
            print(('...time index '+str(t+1)+' of '+str(num_time)))
            temp_tmp = temp[t,:]
            salt_tmp = salt[t,:]
        else:
            temp_tmp = temp
            salt_tmp = salt
        for temp_val, salt_val, grid_val in zip(temp_tmp[mask], salt_tmp[mask], grid.dV[mask]):
            temp_index = np.nonzero(temp_edges > temp_val)[0][0]-1
            salt_index = np.nonzero(salt_edges > salt_val)[0][0]-1
            if time_dependent:
                volume[t, temp_index, salt_index] += grid_val
            else:
                volume[temp_index, salt_index] += grid_val
    # Mask bins with zero volume
    volume = np.ma.masked_where(volume==0, volume)
    return volume, temp_centres, salt_centres, temp_edges, salt_edges           


# Create a temperature vs salinity distribution plot. Temperature and salinity are split into NxN bins (default N=1000) and the colour of each bin shows the log of the volume of water masses in that bin.

# Arguments:
# file_path: path to NetCDF file containing the variable THETA and/or SALT. You can specify a second file for the second variable in second_file_path if needed.

# Optional keyword arguments:
# region: region key to plot (following constants.py); can also end with _cavity (eg fris_cavity) or be 'all' or 'cavities'
# grid: a Grid object OR path to a grid directory OR path to a NetCDF file containing the grid variables. If you specify nothing, the grid will be read from file_path.
# time_index, t_start, t_end, time_average: as in function read_netcdf. You must either define time_index or set time_average=True, so it collapses to a single record.
# second_file_path: path to NetCDF file containing the variable THETA or SALT, if they are not both present in file_path
# tmin, tmax, smin, smax: bounds on temperature and salinity to plot
# num_bins: number of temperature and salinity bins used to categorise the water masses. Default is 1000, but if you're zooming in quite a lot using tmin etc., you might want to increase this.
# date_string: as in function latlon_plot
# figsize: size of figure you want
# fig_name: as in function finished_plot

# Suggested bounds for WSK simulation:
# option='fris_cavity': smin=34.2
# option='cavities': smin=33.5, tmax=1, num_bins=2000
# option='all': smin=33, tmax=1.5, num_bins=2000

def ts_distribution_plot (file_path, region='all', grid=None, time_index=None, t_start=None, t_end=None, time_average=False, second_file_path=None, tmin=None, tmax=None, smin=None, smax=None, num_bins=1000, date_string=None, figsize=(8,6), fig_name=None):

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

    if region == 'all':
        mask = grid.hfac > 0
    elif region == 'cavities':
        mask = grid.ice_mask
    else:
        mask = grid.get_region_mask(region)

    # Make the bins
    volume, temp_centres, salt_centres, temp_edges, salt_edges = ts_binning(temp, salt, grid, mask, num_bins=num_bins)

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
    if region == 'all':
        pass
    elif region == 'cavities':
        title += ' in ice shelf cavities'
    elif region.endswith('cavity'):
        title += ' in ' + region_names[region[:region.index('_cavity')]]
    else:
        title += ' in ' + region_names[region]
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
# smooth: window for moving average (0 means no smoothing)
# ax, make_cbar, ctype, vmin, vmax, title, titlesize, return_fig, fig_name, extend, fig_size, dpi: as in latlon_plot
# zmin, zmax: bounds on depth axis to plot (negative, in metres, zmin is the deep bound).
# monthly: as in netcdf_time
# contours: list of values to contour in black over top

def hovmoller_plot (data, time, grid, smooth=0, ax=None, make_cbar=True, ctype='basic', vmin=None, vmax=None, zmin=None, zmax=None, monthly=True, contours=None, date_since_start=False, start=0, val0=None, title=None, titlesize=18, return_fig=False, fig_name=None, extend=None, figsize=(14,5), dpi=None, start_t=None, end_t=None, rasterized=False):

    # Choose what the endpoints of the colourbar should do
    if extend is None:
        extend = get_extend(vmin=vmin, vmax=vmax)

    if monthly:
        # As in netcdf_time, the time axis will have been corrected so it is
        # marked with the beginning of each month. So to get the boundaries of
        # each time index, we just need to add one month.
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
        
    if date_since_start:
        time_years = [t.year + t.month/12. for t in time_edges]
        time_edges = np.array([t - time_years[start] for t in time_years])

    # Smooth with a moving average
    data, time_edges = moving_average(data, smooth, time=time_edges)
    
    # If we're zooming, we need to choose the correct colour bounds
    if any([zmin, zmax]):
        vmin_tmp, vmax_tmp = var_min_max_zt(data, grid, zmin=zmin, zmax=zmax)
        if vmin is None:
            vmin = vmin_tmp
        if vmax is None:
            vmax = vmax_tmp
    # Get colourmap
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax, val0=val0)

    if start_t is None:
        start_t = time_edges[0]
    if end_t is None:
        end_t = time_edges[-1]

    # Make the figure and axes, if needed
    existing_ax = ax is not None
    if not existing_ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the data
    img = ax.pcolormesh(time_edges, grid.z_edges, np.transpose(data), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)
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
    ax.set_xlim([start_t, end_t])
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
def hovmoller_ts_plot (temp, salt, time, grid, smooth=0, split_year=None, tmin=None, tmax=None, smin=None, smax=None, zmin=None, zmax=None, monthly=True, t_contours=None, s_contours=None, title=None, date_since_start=False, start=0, t0=None, s0=None, ctype='basic', loc_string='', fig_name=None, figsize=(12,7), dpi=None, return_fig=False, ab_inside=False, rasterized=False):

    # Set panels
    fig, gs, cax_t, cax_s = set_panels('2x1C2', figsize=figsize)
    if split_year is not None:
        if date_since_start:
            first_year = time[0].year-time[start].year
            last_year = time[-1].year-time[start].year
        else:
            first_year = time[0].year
            last_year = time[-1].year
        width1 = (split_year-first_year)
        width2 = (last_year+1-split_year)
        gs = plt.GridSpec(2, 2, width_ratios=[width1, width2])
        gs.update(left=0.08, right=0.9, bottom=0.1, top=0.88, hspace=0.2, wspace=0.01)
        # Need to choose the correct colour bounds
        if any([zmin, zmax]):
            tmin_tmp, tmax_tmp = var_min_max_zt(temp, grid, zmin=zmin, zmax=zmax)
            smin_tmp, smax_tmp = var_min_max_zt(salt, grid, zmin=zmin, zmax=zmax)
            if tmin is None:
                tmin = tmin_tmp
            if tmax is None:
                tmax = tmax_tmp
            if smin is None:
                smin = smin_tmp
            if smax is None:
                smax = smax_tmp
        
    # Wrap things up in lists for easier iteration
    data = [temp, salt]
    vmin = [tmin, smin]
    vmax = [tmax, smax]
    val0 = [t0, s0]
    contours = [t_contours, s_contours]
    if ab_inside:
        titles = ['Temperature ('+deg_string+'C)', 'Salinity (psu)']
        ab = ['a', 'b']
    else:
        titles = ['a) Temperature ('+deg_string+'C)', 'b) Salinity (psu)']
    cax = [cax_t, cax_s]
    axs = []
    for i in range(2):
        ax = plt.subplot(gs[i,0])
        # Make the plot
        img = hovmoller_plot(data[i], time, grid, smooth=smooth, ax=ax, make_cbar=False, vmin=vmin[i], vmax=vmax[i], zmin=zmin, zmax=zmax, monthly=monthly, contours=contours[i], ctype=ctype, title=titles[i], date_since_start=date_since_start, start=start, val0=val0[i], end_t=split_year, rasterized=rasterized)
        if ab_inside:
            plt.text(0.01, 0.98, ab[i], weight='bold', ha='left', va='top', fontsize=16, transform=ax.transAxes)
        # Add a colourbar
        extend = get_extend(vmin=vmin[i], vmax=vmax[i])
        cbar = plt.colorbar(img, cax=cax[i], extend=extend)
        reduce_cbar_labels(cbar)
        if i == 0:
            # Remove x-tick labels from top plot
            ax.set_xticklabels([])                
        else:
            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('')
        axs.append(ax)
        if split_year is not None:
            # Now make another plot beside
            ax2 = plt.subplot(gs[i,1])
            img = hovmoller_plot(data[i], time, grid, smooth=smooth, ax=ax2, make_cbar=False, vmin=vmin[i], vmax=vmax[i], zmin=zmin, zmax=zmax, monthly=monthly, contours=contours[i], ctype=ctype, title='', date_since_start=date_since_start, start=start, val0=val0[i], start_t=split_year, rasterized=rasterized)
            ax2.set_yticklabels([])
            ax2.set_ylabel('')
            if i==0:
                ax2.set_xticklabels([])
            axs.append(ax2)
    if title is None:
        title = loc_string
    plt.suptitle(title, fontsize=22)
    if return_fig:
        return fig, axs
    else:
        finished_plot(fig, fig_name=fig_name, dpi=dpi)
    

# Read a precomputed Hovmoller file (from precompute_hovmoller in postprocess.py) and make the plot.

# Arguments:
# var: variable name in precomputed file, in the form loc_var, such as 'pine_island_bay_temp'
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
def read_plot_hovmoller_ts (hovmoller_file, loc, grid, smooth=0, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, t_contours=None, s_contours=None, date_since_start=False, ctype='basic', t0=None, s0=None, title=None, fig_name=None, monthly=True, figsize=(12,7), dpi=None, return_fig=False):

    grid = choose_grid(grid, None)
    temp = read_netcdf(hovmoller_file, loc+'_temp')
    salt = read_netcdf(hovmoller_file, loc+'_salt')
    time = netcdf_time(hovmoller_file, monthly=False)
    loc_string = region_names[loc]
    return hovmoller_ts_plot(temp, salt, time, grid, smooth=smooth, tmin=tmin, tmax=tmax, smin=smin, smax=smax, zmin=zmin, zmax=zmax, monthly=monthly, t_contours=t_contours, s_contours=s_contours, loc_string=loc_string, title=title, date_since_start=date_since_start, ctype=ctype, t0=t0, s0=s0, figsize=figsize, dpi=dpi, return_fig=return_fig, fig_name=fig_name)        


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
    loc_string = region_names[loc]
    hovmoller_ts_plot(temp_diff, salt_diff, time, grid, smooth=smooth, tmin=tmin, tmax=tmax, smin=smin, smax=smax, zmin=zmin, zmax=zmax, monthly=monthly, t_contours=t_contours, s_contours=s_contours, ctype='plusminus', loc_string=loc_string, fig_name=fig_name, figsize=figsize, dpi=dpi)


# Compare simulated mean melt rates for each Amundsen Sea ice shelf with the range given by Rignot 2013. The blue dots are the model output and the black error ranges are from Rignot.
# Input arguments:
# file_path: path to precomputed timeseries file (set precomputed=True) or a NetCDF model output file, containing SHIfwFlx data for the whole simulation (or a time-average - it will be averaged if it's not already).
# Optional keyword arguments:
# precomputed: set to True if file_path is a precomputed timeseries file with the melt rates for each ice shelf already there
# option: 'melting' (plot melt rates in m/y) or 'massloss' (plot basal mass loss)
# file_path_2: file_path for a second simulation, to plot on same axes. Or can be a list of file paths to do an ensemble.
# sim_names: list of length 2 containing simulation names for file_path and file_path_2.
# fig_name: as in function finished_plot
def amundsen_rignot_comparison (file_path, precomputed=False, option='melting', file_path_2=None, sim_names=None, fig_name=None):

    shelf_names = ['getz', 'dotson_crosson', 'thwaites', 'pig', 'cosgrove', 'abbot', 'venable']
    shelf_titles = ['Getz', 'Dotson &\nCrosson', 'Thwaites', 'Pine Island', 'Cosgrove', 'Abbot', 'Venable']
    num_shelves = len(shelf_names)

    if not precomputed:
        grid = Grid(file_path)
        
    second = file_path_2 is not None
    ensemble = second and isinstance(file_path_2, list)
    if ensemble:
        num_ens = len(file_path_2)
    if second and (sim_names is None or not isinstance(sim_names, list) or len(sim_names) != 2):
        print('Error (amundsen_rignot_comparison): must set sim_names as list of 2 simulation names if file_path_2 is set.')
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
                if ensemble:
                    for n in range(num_ens):
                        melt_tmp = read_netcdf(file_path_2[n], var_name, time_average=True)
                        if n == 0:
                            min_melt = melt_tmp
                            max_melt = melt_tmp
                        else:
                            min_melt = min(min_melt, melt_tmp)
                            max_melt = max(max_melt, melt_tmp)
                    model2_melt.append([min_melt, max_melt])
                else:
                    model2_melt.append(read_netcdf(file_path_2, var_name, time_average=True))
        else:
            model_melt.append(timeseries_ismr(file_path, grid, shelf=shelf, result=option, time_average=True))
            if second:
                if ensemble:
                    for n in range(num_ens):
                        melt_tmp = timeseries_ismr(file_path_2[n], grid, shelf=shelf, result=option, time_average=True)
                        if n == 0:
                            min_melt = melt_tmp
                            max_melt = melt_tmp
                        else:
                            min_melt = min(min_melt, melt_tmp)
                            max_melt = max(max_melt, melt_tmp)
                    model2_melt.append([min_melt, max_melt])
                else:
                    model2_melt.append(timeseries_ismr(file_path_2, grid, shelf=shelf, result=option, time_average=True))
        obs = rignot_melt[shelf]
        if option == 'massloss':
            obs_melt.append(obs[0])
            obs_std.append(obs[1])
        elif option == 'melting':
            obs_melt.append(obs[2])
            obs_std.append(obs[3])
        else:
            print(('Error (amundsen_rignot_comparison): invalid option ' + option))
            sys.exit()

    if second and ensemble:
        # Convert from min/max to central value and difference, for plotting
        model2_melt0 = []
        model2_melt_diff = []
        for n in range(num_shelves):
            model2_melt0.append(0.5*(model2_melt[n][0] + model2_melt[n][1]))
            model2_melt_diff.append(model2_melt0[n] - model2_melt[n][0])        

    fig, ax = plt.subplots()
    if second:
        ax.plot(list(range(num_shelves)), model_melt, 'o', color='blue', label=sim_names[0])
        if ensemble:
            ax.errorbar(list(range(num_shelves)), model2_melt0, yerr=model2_melt_diff, fmt='none', color='green', capsize=4, label=sim_names[1])
        else:
            ax.plot(list(range(num_shelves)), model2_melt, 'o', color='green', label=sim_names[1])
    else:
        if isinstance(sim_names, list):
            label = sim_names[0]
        elif isinstance(sim_names, str):
            label = sim_names
        else:
            label = 'MITgcm'
        ax.plot(list(range(num_shelves)), model_melt, 'o', color='blue', label=label)
    ax.errorbar(list(range(num_shelves)), obs_melt, yerr=obs_std, fmt='none', color='black', capsize=4, label='Observations')
    ax.legend()
    ax.grid(True)
    plt.xticks(list(range(num_shelves)), shelf_titles, rotation='vertical')
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
            

# Plot temperature and salinity casts from the given region against each year of the model output averaged over the same region. Also plot the mean CTD cast and the mean model cast. You can also plot obs versus 1 model versus ensemble (eg ERA5 and PACE ensemble), in which case you will see overlapping ranges rather than individual years.

# Arguments:
# loc: region name (anything in the "region_bounds" dictionary in constants.py)
# hovmoller_file: path to precomputed Hovmoller file for this region (from precompute_hovmoller in postprocess.py)
# obs_file: path to Matlab file with the CTD database
# grid: a Grid object OR path to a grid directory OR path to a NetCDF file containing the grid variables

# Optional keyword arguments:
# std: boolean (default False) to plot standard deviation instead of range and mean
# ens_hovmoller_files: list of paths to precomputed Hovmoller files for each member of a model ensemble
# month: month of model output to plot (1-12). Default is to plot each modelled February. To plot all months, set month=None.
# fig_name: as in finished_plot.

def ctd_cast_compare (loc, hovmoller_file, obs_file, grid, std=False, ens_hovmoller_files=None, month=2, fig_name=None):

    from scipy.io import loadmat

    ensemble = ens_hovmoller_files is not None

    grid = choose_grid(grid, None)

    # Get bounds on region
    [xmin, xmax, ymin, ymax] = region_bounds[loc]
    # Read obs
    f = loadmat(obs_file)
    obs_lat = np.squeeze(f['Lat'])
    obs_lon = np.squeeze(f['Lon'])
    obs_depth = -1*np.squeeze(f['P'])
    obs_temp = np.transpose(f['PT'])
    obs_salt = np.transpose(f['S'])
    # Convert NaNs into mask
    obs_temp = np.ma.masked_where(np.isnan(obs_temp), obs_temp)
    obs_salt = np.ma.masked_where(np.isnan(obs_salt), obs_salt)
    # Find casts within given region
    index = (obs_lon >= xmin)*(obs_lon <= xmax)*(obs_lat >= ymin)*(obs_lat <= ymax)
    obs_temp = obs_temp[index,:]
    obs_salt = obs_salt[index,:]
    num_obs = obs_temp.shape[0]

    # Read model output
    model_temp = read_netcdf(hovmoller_file, loc+'_temp')
    model_salt = read_netcdf(hovmoller_file, loc+'_salt')
    if month != 0:
        # Select the month we want
        time = netcdf_time(hovmoller_file, monthly=False)
        index = [t.month==month for t in time]
        model_temp = model_temp[index,:]
        model_salt = model_salt[index,:]
    num_model = model_temp.shape[0]

    if ensemble:
        # Read model ensemble output, all in one
        ens_temp = None
        ens_salt = None
        ens_time = None
        for file_path in ens_hovmoller_files:
            temp_tmp = read_netcdf(file_path, loc+'_temp')
            salt_tmp = read_netcdf(file_path, loc+'_salt')
            time_tmp = netcdf_time(file_path, monthly=False)
            if ens_temp is None:
                ens_temp = temp_tmp
                ens_salt = salt_tmp
                ens_time = time_tmp
            else:
                ens_temp = np.concatenate((ens_temp, temp_tmp))
                ens_salt = np.concatenate((ens_salt, salt_tmp))
                ens_time = np.concatenate((ens_time, time_tmp))
        if month != 0:
            index = [t.month==month for t in ens_time]
            ens_temp = ens_temp[index,:]
            ens_salt = ens_salt[index,:]

    # Set panels
    fig, gs = set_panels('1x2C0')
    # Wrap things up in lists for easier iteration
    obs_data = [obs_temp, obs_salt]
    model_data = [model_temp, model_salt]
    if ensemble:
        ens_data = [ens_temp, ens_salt]
        all_data = [obs_data, model_data, ens_data]
        depths = [obs_depth, grid.z, grid.z]
        colours = ['black', 'red', 'blue']
        num_ranges = len(colours)
    titles = ['Temperature', 'Salinity']
    if std:
        titles = [t+' std' for t in titles]
    units = [deg_string+'C', 'psu']
    if std:
        vmin = [None, None]
        vmax = [None, None]
    else:
        vmin = [-2, 33]
        vmax = [2, 34.75]
    for i in range(2):
        ax = plt.subplot(gs[0,i])
        if ensemble:
            # Plot transparent ranges, with means on top
            # OR just plot standard deviation
            for n in range(num_ranges):
                if std:
                    ax.plot(np.std(all_data[n][i], axis=0), depths[n], color=colours[n], linewidth=2)
                else:
                    ax.fill_betweenx(depths[n], np.amin(all_data[n][i], axis=0), x2=np.amax(all_data[n][i], axis=0), color=colours[n], alpha=0.3)
                    ax.plot(np.mean(all_data[n][i], axis=0), depths[n], color=colours[n], linewidth=2)
        else:
            # Plot obs
            if std:
                ax.plot(np.std(obs_data[i], axis=0), obs_depth, color='black', linewidth=2)
            else:
                # Plot individual lines
                for n in range(num_obs):
                    ax.plot(obs_data[i][n,:], obs_depth, color=(0.6, 0.6, 0.6), linewidth=1)
                # Plot obs mean in thicker dashedblack
                ax.plot(np.mean(obs_data[i], axis=0), obs_depth, color='black', linewidth=2, linestyle='dashed')
            # Plot model years
            if std:
                ax.plot(np.std(model_data[i], axis=0), grid.z, color='blue', linewidth=2)
            else:
                # Different colours for each year
                for n in range(num_model):
                    ax.plot(model_data[i][n,:], grid.z, linewidth=1)
                # Plot model mean in thicker black
                ax.plot(np.mean(model_data[i], axis=0), grid.z, color='black', linewidth=2)
        ax.set_xlim([vmin[i], vmax[i]])
        ax.grid(True)
        plt.title(titles[i], fontsize=16)
        plt.xlabel(units[i], fontsize=14)
        if i==0:
            plt.ylabel('Depth (m)', fontsize=14)
        else:
            ax.set_yticklabels([])
    if ensemble:
        plt.suptitle(loc + ': CTDs (black), ERA5 (red), PACE ensemble (blue)', fontsize=20)
    else:
        if std:
            plt.suptitle(loc + ': model (blue) vs CTDs (black)', fontsize=20)
        else:
            plt.suptitle(loc + ': model (colours) vs CTDs (grey)', fontsize=20)
    finished_plot(fig, fig_name=fig_name)


# Plot a timeseries of the number of cells grounded and ungrounded, and the maximum thinning and thickening, in a coupled run.
def plot_geometry_timeseries (output_dir='./', fig_name_1=None, fig_name_2=None):

    from .postprocess import segment_file_paths

    file_paths = segment_file_paths(output_dir)

    # Get the grid from the first one
    old_grid = Grid(file_paths[0])

    # Set up timeseries arrays
    time = []
    ground = []
    unground = []
    thin = []
    thick = []

    # Loop over the rest of the timeseries
    for file_path in file_paths[1:]:
        print(('Processing ' + file_path))
        # Save time index from the beginning of the run
        time.append(netcdf_time(file_path)[0])
        # Calculate geometry changes
        new_grid = Grid(file_path)
        ground.append(np.count_nonzero((old_grid.bathy!=0)*(new_grid.bathy==0)))
        unground.append(np.count_nonzero((old_grid.bathy==0)*(new_grid.bathy!=0)))
        ddraft = np.ma.masked_where(old_grid.draft==0, np.ma.masked_where(new_grid.draft==0, new_grid.draft-old_grid.draft))
        thin.append(np.amin(ddraft))
        thick.append(np.amax(ddraft))
        old_grid = new_grid
    time = np.array(time)
    ground = np.array(ground)
    unground = np.array(unground)
    thin = -1*np.array(thin)
    thick = np.array(thick)

    # Plot
    timeseries_multi_plot(time, [ground, unground], ['# Grounded', '# Ungrounded'], ['blue', 'red'], title='Changes in ocean cells', fig_name=fig_name_1)
    timeseries_multi_plot(time, [thin, thick], ['Maximum thinning', 'Maximum thickening'], ['red', 'blue'], title='Changes in ice shelf draft', fig_name=fig_name_2)


# Create an animated T/S diagram of the given annually-averaged temperature and salinity fields for each year, in the given region.
def ts_animation (temp, salt, time, grid, region, sim_title, tmin=None, tmax=None, smin=None, smax=None, num_bins=1000, mask=None, plot_tfreeze=False, rho_lev=None, mov_name=None):

    import matplotlib.animation as animation

    # Get years if needed
    if isinstance(time[0], datetime.datetime):
        time = np.array([t.year for t in time])
    num_years = time.size

    volume, temp_centres, salt_centres, temp_edges, salt_edges = ts_binning(temp, salt, grid, mask, time_dependent=True, num_bins=num_bins)
    
    # Calculate potential density of bins
    salt_2d, temp_2d = np.meshgrid(salt_centres, temp_centres)
    rho = potential_density('MDJWF', salt_2d, temp_2d)
    if plot_tfreeze:
        # Calculate surface freezing point
        tfreeze_sfc = tfreeze(salt_centres, 0)

    # Now make the animation
    # Set up some parameters
    min_vol = np.log(np.amin(volume))
    max_vol = np.log(np.amax(volume))
    if tmin is None:
        tmin = temp_edges[0]
    if tmax is None:
        tmax = temp_edges[-1]
    if smin is None:
        smin = salt_edges[0]
    if smax is None:
        smax = salt_edges[-1]
    if rho_lev is None:
        rho_lev = np.arange(np.ceil(np.amin(rho)*10)/10., np.ceil(np.amax(rho)*10)/10., 0.1)

    print('Plotting')
    fig, ax = plt.subplots(figsize=(8,6))

    # Inner function to plot one frame
    def plot_one_frame (t):
        img = ax.pcolormesh(salt_edges, temp_edges, np.log(volume[t,:]), vmin=min_vol, vmax=max_vol)
        ax.contour(salt_centres, temp_centres, rho, rho_lev, colors='black', linestyles='dotted')
        if plot_tfreeze:
            ax.plot(salt_centres, tfreeze_sfc, color='black', linestyle='dashed', linewidth=2)
        ax.set_xlim([smin, smax])
        ax.set_ylim([tmin, tmax])
        plt.xlabel('Salinity (psu)')
        plt.ylabel('Temperature ('+deg_string+'C)')
        plt.text(.9, .6, 'log of volume', ha='center', rotation=-90, transform=fig.transFigure)
        plt.title(sim_title+'\n'+region_names[region]+': '+str(int(time[t])))
        if t==0:
            return img

    # First frame
    img = plot_one_frame(0)
    plt.colorbar(img)

    # Function to update figure with the given frame
    def animate(t):
        print(('Frame ' + str(t+1) + ' of ' + str(num_years)))
        ax.cla()
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=list(range(num_years)))
    anim.save(mov_name, bitrate=2000, fps=2)

    
        

    
    
    
    
    
    
    
    


