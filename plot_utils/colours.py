#######################################################
# Create nice colourmaps for plots.
#######################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import sys


# Set up colourmaps of type ctype. Options for ctype are:
# 'basic': just the 'jet' colourmap
# 'plusminus': a red/blue colour map where 0 is white
# 'vel': the 'cool' colourmap starting at 0; good for plotting velocity
# 'ismr': a special colour map for ice shelf melting/refreezing, with negative values in blue, 0 in white, and positive values moving from yellow to orange to red to pink.
# Other keyword arguments:
# vmin, vmax: min and max values to enforce for the colourmap. Sometimes they will be modified (to make sure 'vel' starts at 0, and 'ismr' includes 0). If you don't specify them, they will be determined based on the entire array of data. If your plot is zooming into this array, you should use get_colour_bounds (../utils.py) to determine the correct bounds in the plotted region.
# change_points: only matters for 'ismr'. List of size 3 containing values where the 'ismr' colourmap should hit the colours yellow, orange, and red. It should not include the minimum value, 0, or the maximum value. Setting these parameters allows for a nonlinear transition between colours, and enhanced visibility of the melt rate. If it is not defined, the change points will be determined linearly.

# truncate_colourmap, plusminus_cmap, and ismr_cmap are helper functions; set_colours is the API. It returns a colourmap and the minimum and maximum values.

def truncate_colourmap (cmap, minval=0.0, maxval=1.0, n=-1):
    
    # From https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib    
    if n== -1:
        n = cmap.N
    new_cmap = cl.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plusminus_cmap (vmin, vmax):

    # Truncate the RdBu_r colourmap as needed, so that 0 is white and no unnecessary colours are shown.    
    if abs(vmin) > vmax:
        min_colour = 0
        max_colour = 0.5*(1 - vmax/vmin)
    else:
        min_colour = 0.5*(1 + vmin/vmax)
        max_colour = 1
    return truncate_colourmap(plt.get_cmap('RdBu_r'), min_colour, max_colour)


def ismr_cmap (vmin, vmax, change_points=None):

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
        print 'Error (ismr_cmap): wrong size for change_points list'
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
        return plusminus_cmap(vmin, vmax), vmin, vmax

    elif ctype == 'vel':
        # Make sure it starts at 0
        return plt.get_cmap('cool'), 0, vmax

    elif ctype == 'ismr':
        # Make sure vmin isn't larger than 0
        return ismr_cmap(vmin, vmax, change_points=change_points), min(vmin,0), vmax        

    
# Choose what the endpoints of the colourbar should do. If they're manually set, they should extend. The output can be passed to plt.colorbar with the keyword argument 'extend'.
def get_extend (vmin=None, vmax=None):

    if vmin is None and vmax is None:
        return 'neither'
    elif vmin is not None and vmax is None:
        return 'min'
    elif vmin is None and vmax is not None:
        return 'max'
    elif vmin is not None and vmax is not None:
        return 'both'
