#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from grid import Grid
from io import netcdf_time
from timeseries import fris_melt
from plot_utils import monthly_ticks, finished_plot


# Plot timeseries of FRIS' basal mass balance components (melting, freezing, total) at every time index in the given file.

# Arguments:
# file_path: path to NetCDF file containing variable "SHIfwFlx"
# grid: either a Grid object, or the path to NetCDF grid file

# Optional keyword argument:
# fig_name: as in function finished_plot

def plot_fris_massbalance (file_path, grid, fig_name=None):

    if not isinstance(grid, Grid):
        # This is the path to the NetCDF grid file, not a Grid object
        # Make a grid object from it
        grid = Grid(grid)

    # Calculate timeseries
    melt, freeze = fris_melt(file_path, grid, mass_balance=True)
    # Read time axis
    time = netcdf_time(file_path)

    # Plot
    fig, ax = plt.subplots()
    ax.plot_date(time, melt, '-', color='red', linewidth=1.5, label='Melting')
    ax.plot_date(time, freeze, '-', color='blue', linewidth=1.5, label='Freezing')
    ax.plot_date(time, melt+freeze, '-', color='black', linewidth=1.5, label='Total')
    ax.axhline(color='black')
    ax.grid(True)
    monthly_ticks(ax)
    plt.title('Basal mass balance of FRIS', fontsize=18)
    plt.ylabel('Gt/y', fontsize=16)
    ax.legend()
    finished_plot(fig, fig_name=fig_name)


    
