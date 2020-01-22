#######################################################
# Other figures you might commonly make
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import datetime

from grid import choose_grid
from file_io import check_single_time, find_variable, read_netcdf, netcdf_time
from plot_utils.labels import check_date_string, depth_axis, yearly_ticks, lon_label, lat_label
from plot_utils.windows import finished_plot
from plot_utils.colours import get_extend, set_colours
from utils import mask_3d, xy_to_xyz, z_to_xyz, var_min_max_zt, mask_outside_box
from diagnostics import tfreeze
from constants import deg_string, bounds_PIB, bounds_Dot
from interpolation import interp_bilinear
from calculus import area_average


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
        loc_index = (grid.hfac > 0)*xy_to_xyz(grid.fris_mask, grid)
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
    for i in range(grid.nx):
        for j in range(grid.ny):
            if option=='fris' and not grid.fris_mask[j,i]:
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
# ax, make_cbar, ctype, vmin, vmax, title, titlesize, return_fig, fig_name, extend, fig_size, dpi: as in latlon_plot
# zmin, zmax: bounds on depth axis to plot (negative, in metres, zmin is the deep bound).
# monthly: as in netcdf_time
# contours: list of values to contour in black over top

def hovmoller_plot (data, time, grid, ax=None, make_cbar=True, ctype='basic', vmin=None, vmax=None, zmin=None, zmax=None, monthly=True, contours=None, title=None, titlesize=18, return_fig=False, fig_name=None, extend=None, figsize=(8,6), dpi=None):

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

    # Make the figure and axes, if needed
    existing_ax = ax is not None
    if not existing_ax:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Plot the data
    img = ax.pcolormesh(time_edges, grid.z_edges, data, cmap=cmap, vmin=vmin, vmax=vmax)
    if contours is not None:
        # Overlay contours
        # Need time at the centres of each index
        # Have to do this with a loop unfortunately
        time_centres = []
        for t in range(time_edges.size-1):
            dt = (time_edges[t+1]-time_edges[t])/2
            time_centres.append(time_edges[t]+dt)
        plt.contour(time_centres, grid.z, data, levels=contours, colors='black', linestyles='solid')
    
    # Set depth limits
    if zmin is None:
        zmin = grid.z_edges[-1]
    if zmax is None:
        zmax = grid.z_edges[0]
    ax.set_ylim([zmin, zmax])
    # Make nice axes labels
    yearly_ticks(ax)    
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


# Read a series of NetCDF files and create a Hovmoller plot of the given variable in the given region.

# Arguments:
# var: 'temp', 'salt', 'u', or 'v'
# file_paths: a list of paths to NetCDF output files, or a string with just one file

# Optional keyword arguments:
# option: 'box' (average over a region between lat-lon coordinates; default) or 'point' (interpolate to given lat-lon point)
# box: if option='box', can set to 'PIB' or 'Dot' to average over the predefined regions in Pine Island Bay or in front of Dotson
# xmin, xmax, ymin, ymax: if option='box' and box=None, manual bounds on longitude and latitude of box. If any of them aren't set, they will default to the edges of the domain.
# x0, y0: if option='point', lon and lat coordinates to interpolate to
# grid: Grid object, if it already exists (otherwise it will be calculated from the NetCDF files)
# zmin, zmax, vmin, vmax, contours, monthly, fig_name, figsize: as in hovmoller_plot
def read_plot_hovmoller (var, file_paths, option='box', box=None, xmin=None, xmax=None, ymin=None, ymax=None, x0=None, y0=None, grid=None, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, monthly=True, fig_name=None, figsize=(8,6)):
    
    if isinstance(file_paths, str):
        # Just one file path
        file_paths = [file_paths]

    # Build the grid if needed
    grid = choose_grid(grid, file_paths[0])

    # Set parameters for this variable
    gtype = 't'
    ctype = 'basic'
    if var == 'temp':
        var_name = 'THETA'
        title = 'Temperature ' + deg_string
    elif var == 'salt':
        var_name = 'SALT'
        title = 'Salinity (psu)'
    elif var == 'u':
        var_name = 'UVEL'
        title = 'Zonal velocity (m/s)'
        gtype = 'u'
    elif var == 'v':
        var_name = 'VVEL'
        title = 'Meridional velocity (m/s)'
        gtype = 'v'
    # Update the colourmap if needed
    if var in ['u', 'v']:
        ctype = 'plusminus'

    data = None
    time = None
    # Read and process data for each file.
    for file_path in file_paths:
        print 'Reading ' + file_path
        time_tmp = netcdf_time(file_path, monthly=monthly)
        data_3d = mask_3d(read_netcdf(file_path, var_name), grid, time_dependent=True)
        if option == 'box':
            print 'Averaging over box'
            if box is not None:
                # Preset box
                if box == 'PIB':
                    [xmin, xmax, ymin, ymax] = bounds_PIB
                elif box == 'Dot':
                    [xmin, xmax, ymin, ymax] = bounds_Dot
                else:
                    print 'Error (read_plot_hovmoller): invalid preset box ' + box + '. Valid options are PIB or Dot.'
                    sys.exit()
            data_3d = mask_outside_box(data_3d, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, time_dependent=True)
            data_tmp = area_average(data_3d, grid, gtype=gtype, time_dependent=True)
        elif option == 'point':
            if x0 is None or y0 is None:
                print "Error (read_plot_hovmoller): must set x0 and y0 for option='point'"
                sys.exit()
            data_tmp = interp_bilinear(data_3d, x0, y0, grid, gtype=gtype)
        else:
            print 'Error (read_plot_hovmoller): invalid option ' + option
            sys.exit()
        if data is None:
            data = data_tmp
            time = time_tmp
        else:
            data = np.concatenate((data, data_tmp))
            time = np.concatenate((time, time_tmp))  

    # Update the title
    if option == 'box':
        if box == 'PIB':
            title += ', Pine Island Bay'
        elif box == 'Dot':
            title += ', Dotson'
    else:
        title += ' at ' + lon_label(x0) + ', ' + lat_label(y0)

    # Make the plot
    hovmoller_plot(data, time, grid, ctype=ctype, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax, monthly=monthly, contours=contours, title=title, fig_name=fig_name, figsize=figsize)
        
        
    

    
    
    
    
    
    
    
    


