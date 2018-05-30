#######################################################
# Lat-lon shaded plots
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np

from grid import Grid
from io import read_netcdf, find_variable, netcdf_time
from utils import convert_ismr, mask_except_zice, mask_3d, mask_land_zice, mask_land, select_bottom, select_year, find_aice_min_max
from plot_utils import finished_plot, cell_boundaries, latlon_axes, set_colours, shade_land, shade_land_zice, contour_iceshelf_front, set_colour_bounds, parse_date, prepare_vel, overlay_vectors, set_panels, get_extend
from diagnostics import t_minus_tf


# Basic lat-lon plot of any variable.

# Arguments:
# data: 2D (lat x lon) array of data to plot, already masked as desired
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_lon_lat
# include_shelf: if True (default), plot the values beneath the ice shelf and contour the ice shelf front. If False, shade the ice shelf in grey like land.
# ctype: as in function set_colours
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: something to write on the bottom of the plot about the date
# title: a title to add to the plot
# return_fig: if True, return the figure and axis variables so that more work can be done on the plot (eg adding titles). Default False.
# fig_name: as in function finished_plot
# change_points: only matters if ctype='ismr'. As in function set_colours.

def latlon_plot (data, grid, gtype='t', include_shelf=True, ctype='basic', vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, title=None, return_fig=False, fig_name=None, change_points=None):
    
    # Choose what the endpoints of the colourbar should do
    extend = get_extend(vmin=vmin, vmax=vmax)

    # If we're zooming, we need to choose the correct colour bounds
    if zoom_fris or any([xmin, xmax, ymin, ymax]):
        vmin_tmp, vmax_tmp = set_colour_bounds(data, grid, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, gtype=gtype)
        # Don't override manually set bounds
        if vmin is None:
            vmin = vmin_tmp
        if vmax is None:
            vmax = vmax_tmp
    # Get colourmap
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points)

    # Prepare quadrilateral patches
    lon, lat, data_plot = cell_boundaries(data, grid, gtype=gtype)

    fig, ax = plt.subplots()
    if include_shelf:
        # Shade land in grey
        shade_land(ax, grid, gtype=gtype)
    else:
        # Shade land and ice shelves in grey
        shade_land_zice(ax, grid, gtype=gtype)
    # Plot the data    
    img = ax.pcolormesh(lon, lat, data_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    if include_shelf:
        # Contour ice shelf front
        contour_iceshelf_front(ax, grid)
    # Add a colourbar
    plt.colorbar(img, extend=extend)
    # Make nice axes
    latlon_axes(ax, lon, lat, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    if date_string is not None:
        # Add the date in the bottom right corner
        plt.text(.99, .01, date_string, fontsize=14, ha='right', va='bottom', transform=fig.transFigure)
    if title is not None:
        # Add a title
        plt.title(title, fontsize=18)

    if return_fig:
        return fig, ax
    else:
        finished_plot(fig, fig_name=fig_name)


# Plot ice shelf melt rate field.

# Arguments:
# shifwflx: 2D (lat x lon) array of ice shelf freshwater flux (variable SHIfwFlx), already masked
# grid: Grid object

# Optional keyword arguments:
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot
# change_points: as in function set_colours
# fig_name: as in function finished_plot

def plot_ismr (shifwflx, grid, vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, change_points=None, fig_name=None):

    # Convert to m/y
    ismr = convert_ismr(shifwflx)
    latlon_plot(ismr, grid, ctype='ismr', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, change_points=change_points, title='Ice shelf melt rate (m/y)', fig_name=fig_name)


# Plot bottom water temperature or salinity.

# Arguments:
# var: 'temp' or 'salt'
# data: 3D (depth x lat x lon) array of temperature in degC or salinity in psu, already masked with hfac
# grid: Grid object

# Optional keyword arguments:
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot
# fig_name: as in function finished_plot

def plot_bw (var, data, grid, vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, fig_name=None):

    if var == 'temp':
        title = r'Bottom water temperature ($^{\circ}$C)'
    elif var == 'salt':
        title = 'Bottom water salinity (psu)'
    latlon_plot(select_bottom(data), grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, title=title, fig_name=fig_name)


# Plot surface temperature or salinity.

# Arguments:
# var: 'temp' or 'salt'
# data: 3D (depth x lat x lon) array of temperature in degC or salinity in psu, already masked with hfac
# grid: Grid object

# Optional keyword arguments:
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot
# fig_name: as in function finished_plot

def plot_ss (var, data, grid, vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, fig_name=None):

    if var == 'temp':
        title = r'Sea surface temperature ($^{\circ}$C)'
    elif var == 'salt':
        title = 'Sea surface salinity (psu)'
    latlon_plot(data[0,:], grid, include_shelf=False, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, title=title, fig_name=fig_name)


# Plot miscellaneous 2D variables that do not include the ice shelf: sea ice concentration or thickness, mixed layer depth, free surface, surface salt flux.

# Arguments:
# var: 'aice', 'hice', 'mld', 'eta', 'saltflx'
# data: 2D (lat x lon) array of sea ice concentration (fraction), sea ice thickness, mixed layer depth, free surface (all m), or surface salt flux (kg/m^2/s) already masked with the land and ice shelf
# grid: Grid object

# Optional keyword arguments:
# ctype: as in function set_colours
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot
# fig_name: as in function finished_plot

def plot_2d_noshelf (var, data, grid, ctype='basic', vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, fig_name=None):

    if var == 'aice':
        title = 'Sea ice concentration (fraction)'
    elif var == 'hice':
        title = 'Sea ice effective thickness (m)'
    elif var == 'mld':
        title = 'Mixed layer depth (m)'
    elif var == 'eta':
        title = 'Free surface (m)'
    elif var == 'saltflx':
        title = r'Surface salt flux (kg/m$^2$/s)'
    latlon_plot(data, grid, include_shelf=False, ctype=ctype, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, title=title, fig_name=fig_name)


# Plot the difference from the in-situ freezing point.

# Arguments:
# temp, salt: 3D (depth x lat x lon) arrays of temprature and salinity, already masked with hfac
# grid: Grid object

# Optional keyword arguments:
# tf_option: 'bottom' (to plot difference from in-situ freezing point in the bottom layer), 'max' (to plot maximum at each point in the water column), 'min' (to plot minimum, default).
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot
# fig_name: as in function finished_plot

def plot_tminustf (temp, salt, grid, tf_option='min', vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, fig_name=None):

    # Calculate difference from freezing point
    tminustf = t_minus_tf(temp, salt, grid)
    # Do the correct vertical transformation
    if tf_option == 'bottom':
        tmtf_plot = select_bottom(tminustf)
        title_end = '\n(bottom layer)'
    elif tf_option == 'max':
        tmtf_plot = np.amax(tminustf, axis=0)
        title_end = '\n(maximum over depth)'
    elif tf_option == 'min':
        tmtf_plot = np.amin(tminustf, axis=0)
        title_end = '\n(minimum over depth)'
    latlon_plot(tmtf_plot, grid, ctype='plusminus', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, title=r'Difference from in-situ freezing point ($^{\circ}$C)'+title_end, fig_name=fig_name)


# Plot horizontal ocean or sea ice velocity: magnitude overlaid with vectors.

# Arguments:
# u, v: 3D (depth x lat x lon) arrays of u and v, on the u-grid and v-grid respectively, already masked with hfac
# grid: Grid object

# Optional keyword arguments:
# vel_option: as in function prepare_vel
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot
# fig_name: as in function finished_plot

def plot_vel (u, v, grid, vel_option='avg', vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, fig_name=None):

    # Do the correct vertical transformation, and interpolate to the tracer grid
    speed, u_plot, v_plot = prepare_vel(u, v, grid, vel_option=vel_option)

    include_shelf=True
    if vel_option == 'avg':
        title_beg = 'Vertically averaged '
    elif vel_option == 'sfc':
        title_beg = 'Surface '
    elif vel_option == 'bottom':
        title_beg = 'Bottom '
    elif vel_option == 'ice':
        title_beg = 'Sea ice '
        include_shelf = False

    # Make the plot but don't finish it yet
    fig, ax = latlon_plot(speed, grid, ctype='vel', include_shelf=include_shelf, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, date_string=date_string, title=title_beg+'velocity (m/s)', return_fig=True)

    # Overlay circulation
    if zoom_fris:
        chunk = 6
    else:
        chunk = 10
    if vel_option == 'avg':
        scale = 0.8
    elif vel_option == 'sfc':
        scale = 1.5
    elif vel_option == 'bottom':
        scale = 1
    elif vel_option == 'ice':
        scale = 4
    overlay_vectors(ax, u_plot, v_plot, grid, chunk=chunk, scale=scale)

    finished_plot(fig, fig_name=fig_name)


# NetCDF interface. Call this function with a specific variable key and information about the necessary NetCDF file, to get a nice lat-lon plot.

# Arguments:
# var: keyword indicating which special variable to plot. The options are:
#      'ismr': ice shelf melt rate
#      'bwtemp': bottom water temperature
#      'bwsalt': bottom water salinity
#      'sst': surface temperature
#      'sss': surface salinity
#      'aice': sea ice concentration
#      'hice': sea ice thickness
#      'mld': mixed layer depth
#      'eta': free surface
#      'saltflx': surface salt flux
#      'tminustf': difference from in-situ freezing point
#      'vel': horizontal velocity: magnitude overlaid with vectors
#      'velice': sea ice velocity: magnitude overlaid with vectors
# file_path: path to NetCDF file containing the necessary variable:
#            'ismr': SHIfwFlx
#            'bwtemp': THETA
#            'bwsalt': SALT
#            'sst': THETA
#            'sss': SALT
#            'aice': SIarea
#            'hice': SIheff
#            'mld': MXLDEPTH
#            'eta': ETAN
#            'saltflx': SIempmr
#            'tminustf': THETA and SALT
#            'vel': UVEL and VVEL
#            'velice': SIuice and SIvice
#            If there are two variables needed (eg THETA and SALT for 'tminustf') and they are stored in separate files, you can put the other file in second_file_path (see below).
# grid: either a Grid object, or the path to the NetCDF grid file

# Optional keyword arguments:
# time_index, t_start, t_end, time_average: as in function read_netcdf. You must either define time_index or set time_average=True, so it collapses to a single record.
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# date_string: as in function latlon_plot. If time_index is defined and date_string isn't, date_string will be automatically determined based on the calendar in file_path.
# fig_name: as in function finished_plot
# second_file_path: path to NetCDF file containing a second variable which is necessary and not contained in file_path. It doesn't matter which is which.
# change_points: only matters for 'ismr'. As in function set_colours.
# tf_option: only matters for 'tminustf'. As in function plot_tminustf.
# vel_option: only matters for 'vel'. As in function prepare_vel.

def read_plot_latlon (var, file_path, grid, time_index=None, t_start=None, t_end=None, time_average=False, vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, date_string=None, fig_name=None, second_file_path=None, change_points=None, tf_option='min', vel_option='avg'):

    # Make sure we'll end up with a single record in time
    if time_index is None and not time_average:
        print 'Error (read_plot_latlon): either specify time_index or set time_average=True.'
        sys.exit()

    if date_string is None and time_index is not None:
        # Determine what to write about the date
        date_string = parse_date(file_path=file_path, time_index=time_index)

    if not isinstance(grid, Grid):
        # This is the path to the NetCDF grid file, not a Grid object
        # Make a grid object from it
        grid = Grid(grid)

    # Read necessary variables from NetCDF file(s), and mask appropriately
    if var == 'ismr':
        shifwflx = mask_except_zice(read_netcdf(file_path, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var in ['bwtemp', 'sst', 'tminustf']:
        # Read temperature. Some of these variables need more than temperature and so second_file_path might be set.
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, 'THETA')
        else:
            file_path_use = file_path        
        temp = mask_3d(read_netcdf(file_path_use, 'THETA', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var in ['bwsalt', 'sss', 'tminustf']:
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, 'SALT')
        else:
            file_path_use = file_path
        salt = mask_3d(read_netcdf(file_path_use, 'SALT', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var == 'aice':
        aice = mask_land_zice(read_netcdf(file_path, 'SIarea', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var == 'hice':
        hice = mask_land_zice(read_netcdf(file_path, 'SIheff', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var == 'mld':
        mld = mask_land_zice(read_netcdf(file_path, 'MXLDEPTH', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var == 'eta':
        eta = mask_land_zice(read_netcdf(file_path, 'ETAN', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var == 'saltflx':
        saltflx = mask_land_zice(read_netcdf(file_path, 'SIempmr', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
    if var == 'vel':
        # First read u
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, 'UVEL')
        else:
            file_path_use = file_path
        u = mask_3d(read_netcdf(file_path_use, 'UVEL', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype='u')
        # Now read v
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, 'VVEL')
        else:
            file_path_use = file_path
        v = mask_3d(read_netcdf(file_path_use, 'VVEL', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype='v')
    if var == 'velice':
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, 'SIuice')
        else:
            file_path_use = file_path
        uice = mask_land_zice(read_netcdf(file_path_use, 'SIuice', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype='u')
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, 'SIvice')
        else:
            file_path_use = file_path
        vice = mask_land_zice(read_netcdf(file_path_use, 'SIvice', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype='v')
        
    # Plot
    if var == 'ismr':
        plot_ismr(shifwflx, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, change_points=change_points, date_string=date_string, fig_name=fig_name)
    elif var == 'bwtemp':
        plot_bw('temp', temp, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'bwsalt':
        plot_bw('salt', salt, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'sst':
        plot_ss('temp', temp, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'sss':
        plot_ss('salt', salt, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'aice':
        plot_2d_noshelf('aice', aice, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'hice':
        plot_2d_noshelf('hice', hice, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'mld':
        plot_2d_noshelf('mld', mld, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'eta':
        plot_2d_noshelf('eta', eta, grid, ctype='plusminus', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'saltflx':
        plot_2d_noshelf('saltflx', saltflx, grid, ctype='plusminus', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'tminustf':
        plot_tminustf(temp, salt, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, tf_option=tf_option, date_string=date_string, fig_name=fig_name)
    elif var == 'vel':
        plot_vel(u, v, grid, vel_option=vel_option, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    elif var == 'velice':
        plot_vel(uice, vice, grid, vel_option='ice', vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, date_string=date_string, fig_name=fig_name)
    else:
        print 'Error (read_plot_latlon): variable key ' + str(var) + ' does not exist'
        sys.exit()


# Plot topographic variables: bathymetry, ice shelf draft, water column thickness.

# Arguments:
# var: 'bathy', 'zice', 'wct'
# grid: either a Grid object, or the path to the NetCDF grid file

# Optional keyword arguments:
# vmin, vmax: as in function set_colours
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# fig_name: as in function finished_plot

def plot_topo (var, grid, vmin=None, vmax=None, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, fig_name=None):

    if not isinstance(grid, Grid):
        # This is the path to the NetCDF grid file, not a Grid object
        # Make a grid object from it
        grid = Grid(grid)

    if var == 'bathy':
        data = abs(mask_land(grid.bathy, grid))
        title = 'Bathymetry (m)'
    elif var == 'zice':
        data = abs(mask_except_zice(grid.zice, grid))
        title = 'Ice shelf draft (m)'
    elif var == 'wct':
        data = abs(mask_land(grid.wct, grid))
        title = 'Water column thickness (m)'

    latlon_plot(data, grid, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=title, fig_name=fig_name)


# 1x2 lat-lon plot showing sea ice area at the timesteps of minimum and maximum area in the given year.
def plot_aice_minmax (file_path, grid, year, fig_name=None):

    if not isinstance(grid, Grid):
        # This is the path to the NetCDF grid file, not a Grid object
        # Make a grid object from it
        grid = Grid(grid)

    # Read sea ice area and the corresponding dates
    aice = mask_land_zice(read_netcdf(file_path, 'SIarea'), grid, time_dependent=True)
    time = netcdf_time(file_path)
    # Find the range of dates we care about
    t_start, t_end = select_year(time, year)
    # Trim the arrays to these dates
    aice = aice[t_start:t_end,:]
    time = time[t_start:t_end]
    # Find the indices of min and max sea ice area
    t_min, t_max = find_aice_min_max(aice, grid)
    # Wrap up in lists for easy iteration
    aice_minmax = [aice[t_min,:], aice[t_max,:]]
    time_minmax = [time[t_min], time[t_max]]

    # Plot
    fig, gs, cbaxes = set_panels('1x2C1')
    for t in range(2):
        lon, lat, aice_plot = cell_boundaries(aice_minmax[t], grid)
        ax = plt.subplot(gs[0,t])
        shade_land_zice(ax, grid)
        img = ax.pcolormesh(lon, lat, aice_plot, vmin=0, vmax=1)
        latlon_axes(ax, lon, lat)
        if t == 1:
            # Don't need latitude labels a second time
            ax.set_yticklabels([])
        plt.title(parse_date(date=time_minmax[t]), fontsize=18)
    # Colourbar
    plt.colorbar(img, cax=cbaxes, orientation='horizontal')
    # Main title above
    plt.suptitle('Min and max sea ice area', fontsize=22)
    finished_plot(fig, fig_name=fig_name)

    
