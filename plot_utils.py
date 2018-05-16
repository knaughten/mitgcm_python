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


# Separate function for finding vmin and vmax between several different arrays, and/or in specific region
#def 

#lon_min=None, lon_max=None, lat_min=None, lat_max=None, grid=None, gtype='t'

    
# Optional keyword arguments:
# ctype: 'basic' (default) is a rainbow colour map
#        'plusminus' is a red/blue colour map where 0 is white
#        'ismr' is a special colour map for ice shelf melting/refreezing, with negative values in blue, 0 in white, and positive values moving from yellow to orange to red to pink
# vmin, vmax: if defined, enforce these minimum and/or maximum values for the colour map
# change_points: list of size 3 containing values where the 'ismr' colourmap should hit the colours yellow, orange, and red. It should not include the minimum value, 0, or the maximum value. Setting these parameters allows for a nonlinear transition between colours, and enhanced visibility of the melt rate. If it is not defined, the change points will be determined linearly.
def set_colours (data, ctype='basic', vmin=None, vmax=None, change_points=None):

    # Work out bounds
    if vmin is None:
        vmin = np.amin(data)
    if vmax is None:
        vmax = np.amax(data)

    if ctype == 'basic':
        pass
    elif ctype == 'plusminus':
        pass
    elif ctype == 'ismr':
        # Fancy colourmap for ice shelf melting and refreezing
        
        # First define the colours we'll use
        ismr_blue = (0.26, 0.45, 0.86)
        ismr_white = (1, 1, 1)
        ismr_yellow = (1, 0.9, 0.4)
        ismr_orange = (0.99, 0.59, 0.18)
        ismr_red = (0.5, 0.0, 0.08)
        ismr_pink = (0.96, 0.17, 0.89)
        
        if change_points is None:            
            # Set change points to yield a linear transition between colours
            change_points = 0.25*vmax*np.arange(1,3+1)
        if len(change_points) != 3:
            print 'Error (set_colours): wrong size for change_points list'
            sys.exit()
            
        if vmin < 0:
            # There is refreezing here; include blue for elements < 0
            cmap_vals = np.concatenate(([vmin], [0], change_points, [vmax]))
            cmap_colours = [ismr_blue, ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]            
            cmap_vals_norm = (cmap_vals-vmin)/(vmax-vmin)
        else:
            # No refreezing; start at 0
            cmap_vals = np.concatenate(([0], change_points, [vmax]))
            cmap_colours = [ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]
            cmap_vals_norm = cmap_vals/vmax
        cmap_vals_norm[-1] = 1
        cmap_list = []
        for i in range(cmap_vals.size):
            cmap_list.append((cmap_vals_norm[i], cmap_colours[i]))

        return cl.LinearSegmentedColormap.from_list('ismr', cmap_list)


# Shade the land in grey
def shade_land (ax, grid):

    # Convert 0s to proper masked values, so only 1s are unmasked
    land = np.ma.masked_where(np.invert(grid.land_mask), grid.land_mask)
    # Prepare quadrilateral patches
    lon, lat, land_plot = cell_boundaries(land, grid)
    # Plot
    ax.pcolormesh(lon, lat, land_plot, cmap=cl.ListedColormap([(0.6, 0.6, 0.6)]))
    
            
        
        

    
        
