#######################################################
# Helper functions for plotting
#######################################################

import matplotlib.dates as dt
import matplotlib.colors as cl


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


#def set_colours (data, ctype):
