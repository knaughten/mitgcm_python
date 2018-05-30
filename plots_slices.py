#######################################################
# Zonal or meridional slices (lat-depth or lon-depth)
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

from plot_utils import slice_patches, plot_slice_patches, finished_plot, slice_axes, plusminus_cmap, lon_label, lat_label


# Velocity interface
# NetCDF interface for temp, salt, u, v
# Comments!!!
def slice_plot (data, grid, gtype='t', lon0=None, lat0=None, hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, ctype='basic', title=None, date_string=None, fig_name=None):

    # Choose what the endpoints of the colourbar should do
    extend = get_extend(vmin=vmin, vmax=vmax)

    # Build the patches and get the bounds
    patches, values, loc0, hmin_tmp, hmax_tmp, zmin_tmp, zmax_tmp, vmin_tmp, vmax_tmp = slice_patches(data, grid, gtype=gtype, lon0=lon0, lat0=lat0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax)
    
    # Bit of padding on the left and bottom to show mask
    hmin_tmp -= 0.015*(hmax_tmp-hmin_tmp)
    zmin_tmp -= 0.015*(zmax_tmp-zmin_tmp)
    # Update any bounds which aren't already set
    if hmin is None:
        hmin = hmin_tmp
    if hmax is None:
        hmax = hmax_tmp
    if zmin is None:
        zmin = zmin_tmp
    if zmax is None:
        zmax = zmax_tmp
    if vmin is None:
        vmin = vmin_tmp
    if vmax is None:
        vmax = vmax_tmp
    # Set colour map
    if ctype == 'basic':
        cmap = 'jet'
    elif ctype == 'plusminus':
        cmap = plusminus_cmap(vmin, vmax)
    else:
        print 'Error (slice_plot): invalid ctype=' + ctype
        sys.exit()

    # Figure out orientation and format slice location
    if lon0 is not None:
        h_axis = 'lat'
        loc_string = lon_label(loc0, 3)
    elif lat0 is not None:
        h_axis = 'lon'
        loc_string = lat_label(loc0, 3)
    # Set up the title
    if title is None:
        title = ''
    title += ' at ' + loc_string    

    # Plot
    fig, ax = plt.subplots()
    # Add patches
    img = plot_slice_patches(ax, patches, values, hmin, hmax, zmin, zmax, vmin, vmax, cmap=cmap)
    # Make nice axis labels
    slice_axes(ax, h_axis=h_axis)
    # Add a colourbar
    plt.colorbar(img, extend=extend)
    # Add a title
    plt.title(title, fontsize=18)
    if date_string is not None:
        # Add the date in the bottom right corner
        plt.text(.99, .01, date_string, fontsize=14, ha='right', va='bottom', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_name)

    
