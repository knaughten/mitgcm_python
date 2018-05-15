#######################################################
# Helper functions for plotting
#######################################################

import matplotlib.dates as dt


# On a timeseries plot, label every month
def monthly_ticks (ax):

    ax.xaxis.set_major_locator(dt.MonthLocator())
    ax.xaxis.set_major_formatter(dt.DateFormatter('%b %y'))


# On a timeseries plot, label every year
def yearly_ticks (ax):

    ax.xaxis.set_major_locator(dt.YearLocator())
    ax.xaxis.set_major_formatter(dt.DateFormatter('%Y'))


# If a figure name is defined, save the figure to that file. Otherwise, display the figure on screen.
def finished_plot (fig, fig_name=None):

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        fig.show()
