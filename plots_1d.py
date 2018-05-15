#######################################################
# 1D plots, e.g. timeseries
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from io import Grid, netcdf_time
from timeseries import fris_melt
from plot_utils import monthly_ticks


# Plot timeseries of FRIS' basal mass balance components (melting, freezing, total) at every time index in the given file.
# Arguments:
# file_path: path to NetCDF file containing variable "SHIfwFlx"
# grid_path: path to NetCDF grid file
# Optional keyword argument:
# fig_name: if defined, the figure will be saved to a file with the given name. Otherwise, it will be displayed on the screen.
def plot_fris_massbalance (file_path, grid_path, fig_name=None):

    melt, freeze = fris_melt(file_path, Grid(grid_path), mass_balance=True)
    time = netcdf_time(file_path)

    fig, ax = plt.subplots()
    ax.plot_date(time, melt, '-', color='red', linewidth=1.5, label='Melting')
    ax.plot_date(time, freeze, '-', color='blue', linewidth=1.5, label='Freezing')
    ax.plot_date(time, melt+freeze, '-', color='black', linewidth=1.5, label='Total')
    ax.axhline(color='black')
    monthly_ticks(ax)
    plt.title('Basal mass balance of FRIS', fontsize=18)
    plt.ylabel('Gt/y', fontsize=16)
    ax.grid(True)
    ax.legend()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        fig.show()


    
