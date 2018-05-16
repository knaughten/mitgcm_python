#######################################################
# Lat-lon shaded plots
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

from io import Grid, read_netcdf
from utils import convert_ismr, mask_except_zice
from plot_utils import finished_plot, cell_boundaries, latlon_axes, set_colours

# Basic lon-lat plot of any variable, with no titles or special colourmaps.
# To do: nice colourmap
def quick_plot (var, grid, gtype='t', fig_name=None):

    lon, lat, var_plot = cell_boundaries(var, grid, gtype=gtype)
    fig, ax = plt.subplots()
    img = ax.pcolormesh(lon, lat, var_plot)
    plt.colorbar(img)
    latlon_axes(ax, lon, lat)
    finished_plot(fig, fig_name=fig_name)


# Plot ice shelf melt rate field
def plot_ismr (ismr, grid, fig_name=None, change_points=None):

    # To do:
    # Shade land in grey
    # Contour ice shelf front
    # Title

    lon, lat, var_plot = cell_boundaries(ismr, grid)

    fig, ax = plt.subplots()
    vmin, vmax, cmap = set_colours(ismr, ctype='ismr', change_points=change_points)
    img = ax.pcolormesh(lon, lat, var_plot, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(img)
    latlon_axes(ax, lon, lat)
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
    

    
