#######################################################
# Lat-lon shaded plots
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

from io import Grid, read_netcdf
from utils import convert_ismr
from plot_utils import finished_plot, cell_boundaries, set_colours

# Basic lon-lat plot of any variable, with no titles or special colourmaps.
def quick_plot (var, grid, gtype='t', fig_name=None):

    lon, lat, var_plot = cell_boundaries(var, grid, gtype=gtype)

    fig, ax = plt.subplots()
    img = ax.pcolormesh(lon, lat, var_plot)
    plt.colorbar(img)
    finished_plot(fig, fig_name=fig_name)


# Plot ice shelf melt rate field
'''def plot_ismr (ismr, grid, fig_name=None):

    lon, lat, var_plot = cell_boundaries(ismr, grid)

    fig, ax = plt.subplots()
    cmap = set_colours(ismr, ctype='ismr')'''
    
    
    


# NetCDF interface for plot_ismr
def read_plot_ismr (file_path, grid_path, time_index=None, t_start=None, t_end=None, time_average=False, fig_name=None):

    # Make sure we'll end up with a single record in time
    if time_index is not None and not time_average:
        print 'Error (read_plot_ismr): either specify time_index or set time_average=True.'
        sys.exit()

    # Read grid and data
    grid = Grid(grid_path)
    ismr = convert_ismr(read_netcdf(file_path, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))

    # Make the plot
    plot_ismr(ismr, grid, fig_name=fig_name)
    

    
