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
# 'ratio': as above, but 1 is white and the data does not go below 0
# 'centered': as above, but centered on the given value with white
# 'vel': the 'cool' colourmap starting at 0; good for plotting velocity
# 'ismr': a special colour map for ice shelf melting/refreezing, with negative values in blue, 0 in white, and positive values moving from yellow to orange to red to pink.
# 'psi': a special colour map for streamfunction contours, with negative values in blue and positive values in red, but small values more visible than regular plus-minus.
# Other keyword arguments:
# vmin, vmax: min and max values to enforce for the colourmap. Sometimes they will be modified (to make sure 'vel' starts at 0, and 'ismr' includes 0). If you don't specify them, they will be determined based on the entire array of data. If your plot is zooming into this array, you should use get_colour_bounds (../utils.py) to determine the correct bounds in the plotted region.
# change_points: only matters for 'ismr' or 'psi'. List of size 3 (for 'ismr') or 5 (for 'psi') containing values where the colourmap should hit the colours yellow, orange, and red (for 'ismr') or medium blue, light blue, light red, medium red, and dark red (for 'psi'). It should not include the minimum value, 0, or the maximum value. Setting these parameters allows for a nonlinear transition between colours, and enhanced visibility of the melt rate. If it is not defined, the change points will be determined linearly.

# truncate_colourmap, plusminus_cmap, ismr_cmap, and psi_cmap are helper functions; set_colours is the API. It returns a colourmap and the minimum and maximum values.

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


# Create a linear segmented colourmap from the given values and colours. Helper function for ismr_cmap and psi_cmap.
def special_cmap (cmap_vals, cmap_colours, vmin, vmax, name):

    vmin_tmp = min(vmin, np.amin(cmap_vals))
    vmax_tmp = max(vmax, np.amax(cmap_vals))

    cmap_vals_norm = (cmap_vals-vmin_tmp)/(vmax_tmp-vmin_tmp)
    cmap_vals_norm[-1] = 1
    cmap_list = []
    for i in range(cmap_vals.size):
        cmap_list.append((cmap_vals_norm[i], cmap_colours[i]))
    cmap = cl.LinearSegmentedColormap.from_list(name, cmap_list)

    if vmin > vmin_tmp or vmax < vmax_tmp:
        min_colour = (vmin - vmin_tmp)/(vmax_tmp - vmin_tmp)
        max_colour = (vmax - vmin_tmp)/(vmax_tmp - vmin_tmp)
        cmap = truncate_colourmap(cmap, min_colour, max_colour)

    return cmap


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
        return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'ismr')
    else:
        # No refreezing; start at 0
        cmap_vals = np.concatenate(([0], change_points, [vmax]))
        cmap_colours = [ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]
        return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'ismr')


def psi_cmap (vmin, vmax, change_points=None):

    # Note this assumes vmin < 0 and vmax > 0

    psi_dkblue = (0, 0, 0.3)
    psi_medblue = (0, 0, 1)
    psi_ltblue = (0.6, 0.6, 1)
    psi_white = (1, 1, 1)
    psi_ltred = (1, 0.6, 0.6)
    psi_medred = (1, 0, 0)
    psi_dkred = (0.3, 0, 0)
    psi_black = (0, 0, 0)

    if change_points is None:
        # Set change points to yield a linear transition between colours
        change_points = [vmin/3, 2*vmin/3, vmax/4, vmax/2, 3*vmax/4]
    if len(change_points) != 5:
        print 'Error (psi_cmap): wrong size for change_points list'
        sys.exit()

    cmap_vals = np.concatenate(([vmin], change_points[:2], [0], change_points[2:], [vmax]))
    cmap_colours = [psi_dkblue, psi_medblue, psi_ltblue, psi_white, psi_ltred, psi_medred, psi_dkred, psi_black]
    return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'psi')


def ratio_cmap (vmin, vmax):
    # 0 is dark blue, 1 is white, vmax is dark red
    cmap_vals = np.array([0, 1, vmax])
    cmap_colours = [(0, 0, 0.5), (1, 1, 1), (0.5, 0, 0)]
    return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'ratio')


def centered_cmap (vmin, vmax, val0):

    ncolours = 256
    half_colours = ncolours/2
    
    cmap_vals = []
    cmap_colours = []

    set1 = np.linspace(vmin, val0, num=half_colours, endpoint=False)
    for n in range(half_colours):
        cmap_vals.append(set1[n])
        cmap_colours.append(plt.get_cmap('RdBu_r')(n))
    set2 = np.linspace(val0, vmax, num=half_colours)
    for n in range(half_colours, ncolours):
        cmap_vals.append(set2[n-half_colours])
        cmap_colours.append(plt.get_cmap('RdBu_r')(n))
    return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'centered')


def set_colours (data, ctype='basic', vmin=None, vmax=None, change_points=None, val0=None):

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

    elif ctype == 'centered':
        if val0 is None or vmin > val0 or vmax < val0:
            print 'Error (set_colours): invalid val0'
            sys.exit()
        return centered_cmap(vmin, vmax, val0), vmin, vmax

    elif ctype == 'vel':
        # Make sure it starts at 0
        return plt.get_cmap('cool'), 0, vmax

    elif ctype == 'ismr':
        return ismr_cmap(vmin, vmax, change_points=change_points), vmin, vmax

    elif ctype == 'psi':
        if vmin >= 0 or vmax <= 0:
            print 'Error (set_colours): streamfunction limits do not cross 0.'
            sys.exit()
        return psi_cmap(vmin, vmax, change_points=change_points), vmin, vmax

    elif ctype == 'ratio':
        if vmin < 0:
            print 'Error (set_colours): ratio colourmap only accepts positive values.'
            sys.exit()
        if vmax < 1:
            print 'Error (set_colours): ratio colourmap needs values greater than 1'
            sys.exit()
        return ratio_cmap(vmin, vmax), vmin, vmax

    else:
        print 'Error (set_colours): invalid ctype ' + ctype
        sys.exit()            

    
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
