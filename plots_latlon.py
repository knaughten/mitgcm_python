#######################################################
# Lat-lon shaded plots
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

from io import Grid, read_netcdf
from utils import convert_ismr, mask_except_zice
from plot_utils import finished_plot, cell_boundaries, latlon_axes, set_colours, shade_land, shade_land_zice, contour_iceshelf_front


# Basic lon-lat plot of any variable.

# Arguments:
# var: 2D (lat x lon) array of data to plot, already masked as desired
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function cell_boundaries
# ctype: as in function set_colours
# include_shelf: if True (default), plot the values beneath the ice shelf and contour the ice shelf front. If False, shade the ice shelf in grey like land.
# change_points: as in function set_colours (only matters if ctype='ismr')
# return_fig: if True, return the figure and axis variables so that more work can be done on the plot (eg adding titles). Default False.
# fig_name: as in function finished_plot

def lonlat_plot (var, grid, gtype='t', ctype='basic', include_shelf=True, change_points=None, return_fig=False, fig_name=None):

    # Prepare quadrilateral patches
    lon, lat, var_plot = cell_boundaries(var, grid, gtype=gtype)
    # Get colourmap
    cmap, vmin, vmax = set_colours(var, ctype=ctype)

    fig, ax = plt.subplots()
    if include_shelf:
        # Shade land in grey
        shade_land(ax, grid)
    else:
        # Shade land and ice shelves in grey
        shade_land_zice(ax, grid)
    # Plot the data    
    img = ax.pcolormesh(lon, lat, var_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    if include_shelf:
        # Contour ice shelf front
        contour_iceshelf_front(ax, grid)
    plt.colorbar(img)
    latlon_axes(ax)

    if return_fig:
        return fig, ax
    else:
        finished_plot(fig, fig_name=fig_name)


# Plot ice shelf melt rate field.

# Arguments:
# ismr: 2D (lat x lon) array of ice shelf melt rate in m/y, already masked
# grid: Grid object

# Optional keyword arguments:
# fig_name: as in function finished_plot
# change_points: as in function set_colours

def plot_ismr (ismr, grid, fig_name=None, change_points=None):

    fig, ax = lonlat_plot(ismr, grid, ctype='ismr', include_shelf=True, change_points=change_points, return_fig=True)
    ax.set_title('Ice shelf melt rate (m/y)', fontsize=18)
    finished_plot(fig, fig_name=fig_name)
        

# NetCDF interface for plot_ismr
# Later, make this more general for all types of special 2D plots!
def read_plot_ismr (file_path, grid_path, change_points=None, time_index=None, t_start=None, t_end=None, time_average=False, fig_name=None):

    # Make sure we'll end up with a single record in time
    if time_index is None and not time_average:
        print 'Error (read_plot_ismr): either specify time_index or set time_average=True.'
        sys.exit()

    # Read grid and data
    grid = Grid(grid_path)
    ismr = convert_ismr(read_netcdf(file_path, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))
    # Mask out land and open ocean
    ismr = mask_except_zice(ismr, grid)

    # Make the plot
    plot_ismr(ismr, grid, fig_name=fig_name, change_points=change_points)
    

    
