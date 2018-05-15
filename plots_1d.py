import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as dt

from io import Grid, netcdf_time
from timeseries import fris_melt


def plot_fris_massbalance (file_path, grid_path, fig_name=None):

    melt, freeze = fris_melt(file_path, Grid(grid_path), mass_balance=True)
    time = netcdf_time(file_path)

    fig, ax = plt.subplots()
    ax.plot_date(time, melt, '-', color='red', linewidth=1.5, label='Melting')
    ax.plot_date(time, freeze, '-', color='blue', linewidth=1.5, label='Freezing')
    ax.plot_date(time, melt+freeze, '-', color='black', linewidth=1.5, label='Total')
    ax.axhline(color='black')
    ax.xaxis.set_major_locator(dt.MonthLocator())
    ax.xaxis.set_major_formatter(dt.DateFormatter('%b %Y'))
    plt.title('Basal mass balance of FRIS', fontsize=18)
    plt.ylabel('Gt/y', fontsize=16)
    ax.grid(True)
    ax.legend()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        fig.show()


    
