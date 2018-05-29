#######################################################
# Zonal or meridional slices (lat-depth or lon-depth)
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from plot_utils import slice_patches, plot_slice_patches, finished_plot


# Bells and whistles to add later:
# Alternate colourmaps
# Set vmin and vmax
# Set spatial bounds
# Date string
# Title
# Return figure
# Nice axes
# NetCDF interface for temp, salt, u, v
def slice_plot (data, grid, gtype='t', lon0=None, lat0=None, fig_name=None):

    # Build the patches and get the bounds
    patches, values, loc0, hmin, hmax, zmin, zmax, vmin, vmax = slice_patches(data, grid, gtype=gtype, lon0=lon0, lat0=lat0)

    # Plot
    fig, ax = plt.subplots()
    # Add patches
    img = plot_slice_patches(ax, patches, values, hmin, hmax, zmin, zmax, vmin, vmax)
    # Add a colourbar
    plt.colorbar(img)
    finished_plot(fig, fig_name=fig_name)

    
