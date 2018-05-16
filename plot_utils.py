#######################################################
# Helper functions for plotting
#######################################################

import numpy as np
import matplotlib.dates as dt
import matplotlib.colors as cl
import sys


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


# Determine longitude and latitude on the boundaries of cells for the given grid type (tracer, u, v, psi), and throw away one row and one column of the given data field so that every remaining point has latitude and longitude boundaries defined on 4 sides.
# This is needed for pcolormesh so that the coordinates of the quadrilateral patches are correctly defined.
# Arguments:
# data: array of at least 2 dimensions, where the second last dimension is latitude (size M), and the last dimension is longitude (size N).
# grid: Grid object (see io.py)
# Optional keyword argument:
# gtype: 't' (tracer grid, default), 'u' (U-grid), 'v' (V-grid), or 'psi' (psi-grid)
# Output:
# lon: longitude at the boundary of each cell (size MxN)
# lat: latitude at the boundary of each cell (size MxN)
# data: data within each cell (size ...x(M-1)x(N-1), note one row and one column have been removed depending on the grid type)
def cell_boundaries (data, grid, gtype='t'):

    if gtype == 't':
        # Tracer grid: at centres of cells
        # Boundaries are corners of cells
        # Throw away eastern and northern edges
        return grid.lon_corners_2d, grid.lat_corners_2d, data[...,:-1,:-1]
    elif gtype == 'u':
        # U-grid: on left edges of cells
        # Boundaries are centres of cells in X, corners of cells in Y
        # Throw away western and northern edges
        return grid.lon_2d, grid.lat_corners_2d, data[...,:-1,1:]
    elif gtype == 'v':
        # V-grid: on bottom edges of cells
        # Boundaries are corners of cells in X, centres of cells in Y
        # Throw away eastern and southern edges
        return grid.lon_corners_2d, grid.lat_2d, data[...,1:,:-1]
    elif gtype == 'psi':
        # Psi-grid: on southwest corners of cells
        # Boundaries are centres of cells
        # Throw away western and southern edges
        return grid.lon_2d, grid.lat_2d, data[...,1:,1:]


def latlon_axes (ax, lon, lat, xmin=None, xmax=None, ymin=None, ymax=None):

    # Set limits on axes
    if [xmin, xmax, ymin, ymax].count(None) == 0:
        # Special zooming
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    else:
        # Just set to the boundaries of the lon and lat axes
        ax.axis('tight')

    # Check location of ticks
    lon_ticks = ax.get_xticks()
    lat_ticks = ax.get_yticks()
    # Often there are way more longitude ticks than latitude ticks
    if float(len(lon_ticks))/float(len(lat_ticks)) > 1.5:
        # Remove every second one
        lon_ticks = lon_ticks[1::2]
        ax.set_xticks(lon_ticks)

    # Set nice tick labels
    lon_labels = []
    for x in lon_ticks:
        # Decide whether it's west or east
        if x <= 0:
            x = -x
            suff = r'$^{\circ}$W'
        else:
            suff = r'$^{\circ}$E'
        # Decide how to format the number
        if round(x) == x:
            # No decimal places needed
            label = str(int(round(x)))
        elif round(x,1) == x:
            # One decimal place
            label = '{0:.1f}'.format(x)
        else:
            # Round to two decimal places
            label = '{0:.2f}'.format(round(x,2))
        lon_labels.append(label+suff)
    ax.set_xticklabels(lon_labels)
    # Repeat for latitude
    lat_labels = []
    for y in lat_ticks:
        if y <= 0:
            y = -y
            suff = r'$^{\circ}$S'
        else:
            suff = r'$^{\circ}$N'
        if round(y) == y:
            label = str(int(round(y)))
        elif round(y,1) == y:
            label = '{0:.1f}'.format(y)
        else:
            label = '{0:.2f}'.format(round(y,2))
        lat_labels.append(label+suff)
    ax.set_yticklabels(lat_labels)

    
        
