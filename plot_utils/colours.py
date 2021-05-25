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
# 'parula': using the Matlab-type "parula" colourmap rather than jet
# 'grey': from white to dark grey
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


def plusminus_cmap (vmin, vmax, val0, reverse=False):

    if val0 is None:
        val0 = 0

    # Truncate the RdBu_r colourmap as needed, so that val0 is white and no unnecessary colours are shown.    
    if abs(vmin-val0) > vmax-val0:
        min_colour = 0
        max_colour = 0.5*(1 - (vmax-val0)/(vmin-val0))
    else:
        min_colour = 0.5*(1 + (vmin-val0)/(vmax-val0))
        max_colour = 1
    if reverse:
        cmap = plt.get_cmap('RdBu')
    else:
        cmap = plt.get_cmap('RdBu_r')
    return truncate_colourmap(cmap, min_colour, max_colour)


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


def centered_cmap (vmin, vmax, val0):

    ncolours = 256
    half_colours = ncolours/2
    set1 = np.linspace(vmin, val0, num=half_colours, endpoint=False)
    set2 = np.linspace(val0, vmax, num=half_colours)
    cmap_vals = np.concatenate((set1, set2))
    cmap_colours = []
    for n in range(ncolours):
        cmap_colours.append(plt.get_cmap('RdBu_r')(n))
    return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'centered')


def ratio_cmap (vmin, vmax):
    return centered_cmap(vmin, vmax, 1)


# From https://github.com/BIDS/colormap/blob/master/parula.py
def parula_cmap ():

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
     [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
     [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
      0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
     [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
      0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
     [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
      0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
     [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
      0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
     [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
      0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
     [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
      0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
      0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
     [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
      0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
     [0.0589714286, 0.6837571429, 0.7253857143], 
     [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
     [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
      0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
     [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
      0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
     [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
      0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
     [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
      0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
     [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
     [0.7184095238, 0.7411333333, 0.3904761905], 
     [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
      0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
     [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
     [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
      0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
     [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
      0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
     [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
     [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
     [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
      0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
     [0.9763, 0.9831, 0.0538]]

    return cl.LinearSegmentedColormap.from_list('parula', cm_data)



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

    elif ctype == 'parula':
        return parula_cmap(), vmin, vmax

    elif ctype == 'grey':
        return plt.get_cmap('Greys'), vmin, vmax

    elif ctype == 'plusminus':
        return plusminus_cmap(vmin, vmax, val0), vmin, vmax
    elif ctype == 'plusminus_r':
        return plusminus_cmap(vmin, vmax, val0, reverse=True), vmin, vmax

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

# Evenly choose n colours from the 'jet' colourmap.
def choose_n_colours (n):

    cmap = plt.get_cmap('jet')
    loc = np.linspace(0, 1, num=n)
    return cmap(loc)
