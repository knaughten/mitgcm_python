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
