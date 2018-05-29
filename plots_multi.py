#######################################################
# Special plots with multiple panels
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from grid import Grid
from io import read_netcdf, netcdf_time
from utils import select_year, find_aice_min_max
from plot_utils import set_panels, parse_date

def plot_aice_minmax (file_path, grid, year, fig_name=None):

    if not isinstance(grid, Grid):
        # This is the path to the NetCDF grid file, not a Grid object
        # Make a grid object from it
        grid = Grid(grid)

    # Read sea ice area and the corresponding dates
    aice = read_netcdf(file_path, 'SIarea')
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
        plt.title(parse_date(time_minmax[t]), fontsize=18)
    # Colourbar
    plt.colorbar(img, cax=cbaxes)
    # Main title above
    plt.title('Sea ice area, ' + str(year)+ ' min and max', fontsize=20)
    finished_plot(fig, fig_name=fig_name)
    
    
    
