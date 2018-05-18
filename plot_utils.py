#######################################################
# Helper functions for plotting
#######################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import matplotlib.colors as cl
import sys

from utils import mask_except_zice
import constants as const
from io import netcdf_time


# On a timeseries plot, label every month
def monthly_ticks (ax):

    ax.xaxis.set_major_locator(dt.MonthLocator())
    ax.xaxis.set_major_formatter(dt.DateFormatter("%b '%y"))


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


# Determine longitude and latitude on the boundaries of cells for the given grid type (tracer, u, v, psi), and throw away one row and one column of the given data field so that every remaining point has latitude and longitude boundaries defined on 4 sides. This is needed for pcolormesh so that the coordinates of the quadrilateral patches are correctly defined.

# Arguments:
# data: array of at least 2 dimensions, where the second last dimension is latitude (size M), and the last dimension is longitude (size N).
# grid: Grid object

# Optional keyword argument:
# gtype: as in function Grid.get_lon_lat

# Output:
# lon: longitude at the boundary of each cell (size MxN)
# lat: latitude at the boundary of each cell (size MxN)
# data: data within each cell (size ...x(M-1)x(N-1), note one row and one column have been removed depending on the grid type)

def cell_boundaries (data, grid, gtype='t'):

    if gtype in ['t', 'w']:
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


# Set the limits of the longitude and latitude axes, and give them nice labels.

# Arguments:
# ax: Axes object
# lon, lat: values on x and y axes

# Optional keyword arguments:
# zoom_fris: zoom into the FRIS cavity (bounds set in constants.py)
# xmin, xmax, ymin, ymax: specific limits on longitude and latitude

def latlon_axes (ax, lon, lat, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None):
    
    # Set limits on axes
    if zoom_fris:
        xmin = const.fris_bounds[0]
        xmax = const.fris_bounds[1]
        ymin = const.fris_bounds[2]
        ymax = const.fris_bounds[3]
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # Check location of ticks
    lon_ticks = ax.get_xticks()
    lat_ticks = ax.get_yticks()
    # Often there are way more longitude ticks than latitude ticks
    if float(len(lon_ticks))/float(len(lat_ticks)) > 1.5:
        # Automatic tick locations can disagree with limits of axes, but this doesn't change the axes limits unless you get and then set the tick locations. So make sure there are no disagreements now.
        lon_ticks = lon_ticks[(lon_ticks >= ax.get_xlim()[0])*(lon_ticks <= ax.get_xlim()[1])]
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


# Truncate colourmap function from https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n== -1:
        n = cmap.N
    new_cmap = cl.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

    
# Create colourmaps.

# Arguments:
# data: array of data the colourmap will apply to

# Optional keyword arguments:
# ctype: 'basic' is just the 'jet' colourmap
#        'plusminus' creates a red/blue colour map where 0 is white
#        'ismr' creates a special colour map for ice shelf melting/refreezing, with negative values in blue, 0 in white, and positive values moving from yellow to orange to red to pink
# vmin, vmax: if defined, enforce these minimum and/or maximum values for the colour map. vmin might get modified for 'ismr' colour map if there is no refreezing (i.e. set to 0).
# change_points: list of size 3 containing values where the 'ismr' colourmap should hit the colours yellow, orange, and red. It should not include the minimum value, 0, or the maximum value. Setting these parameters allows for a nonlinear transition between colours, and enhanced visibility of the melt rate. If it is not defined, the change points will be determined linearly.

# Output:
# vmin, vmax: min and max values for colourmap
# cmap: colourmap to plot with

def set_colours (data, ctype='basic', vmin=None, vmax=None, change_points=None):

    # Work out bounds
    if vmin is None:
        vmin = np.amin(data)
    else:
        # Make sure it's not an integer
        vmin = float(vmin)
    if vmax is None:
        vmax = np.amax(data)
    else:
        vmax = float(vmax)

    if ctype == 'basic':
        return plt.get_cmap('jet'), vmin, vmax

    elif ctype == 'plusminus':
        # Truncate the RdBu_r colourmap as needed, so that 0 is white and no unnecessary colours are shown
        if abs(vmin) > vmax:
            min_colour = 0
            max_colour = 0.5*(1 - vmax/vmin)
        else:
            min_colour = 0.5*(1 + vmin/vmax)
            max_colour = 1
        return truncate_colormap(plt.get_cmap('RdBu_r'), min_colour, max_colour), vmin, vmax

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

        # Make sure vmin isn't greater than 0
        return cl.LinearSegmentedColormap.from_list('ismr', cmap_list), min(vmin,0), vmax


# Shade the given boolean mask in grey on the plot.
def shade_mask (ax, mask, grid):

    # Properly mask all the False values, so that only True values are unmasked
    mask_plot = np.ma.masked_where(np.invert(mask), mask)
    # Prepare quadrilateral patches
    lon, lat, mask_plot = cell_boundaries(mask_plot, grid)
    # Add to plot
    ax.pcolormesh(lon, lat, mask_plot, cmap=cl.ListedColormap([(0.6, 0.6, 0.6)]))


# Shade the land in grey
def shade_land (ax, grid):

    shade_mask(ax, grid.land_mask, grid)


# Shade the land and ice shelves in grey
def shade_land_zice (ax, grid):

    shade_mask(ax, grid.land_mask+grid.zice_mask, grid)
    

# Contour the ice shelf front in black
def contour_iceshelf_front (ax, grid):

    # Find the shallowest non-zero ice shelf draft
    zice0 = np.amax(grid.zice[grid.zice!=0])
    # Add to plot
    ax.contour(grid.lon_2d, grid.lat_2d, grid.zice, levels=[zice0], colors=('black'), linestyles='solid')


# Find the minimum and maximum values of an array in the given region.

# Arguments:
# data: 2D array (lat x lon), already masked as desired
# grid: Grid object

# Optional keyword arguments:
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# gtype: as in function Grid.get_lon_lat

# Output:
# vmin, vmax: min and max values of data in the given region

def set_colour_bounds (data, grid, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, gtype='t'):

    # Choose the correct longitude and latitude arrays
    lon, lat = grid.get_lon_lat(gtype=gtype)

    # Set limits on axes
    if zoom_fris:
        xmin = const.fris_bounds[0]
        xmax = const.fris_bounds[1]
        ymin = const.fris_bounds[2]
        ymax = const.fris_bounds[3]
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lon)
    if ymax is None:
        ymax = np.amax(lon)

    # Select the correct indices
    loc = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)
    # Find the min and max values
    return np.amin(data[loc]), np.amax(data[loc])


# Get the date in file_path at time_index, and return a nice string that can be added to plots.
def parse_date (file_path, time_index):

    date = netcdf_time(file_path)[time_index]
    return date.strftime('%d %b %Y')

    

    
            
        
        

    
        
