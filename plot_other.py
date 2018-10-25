#######################################################
# Other figures you might commonly make
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np

from grid import choose_grid
from file_io import check_single_time, find_variable
from plot_utils.labels import check_date_string
from utils import mask_3d, xy_to_xyz
from diagnostics import tfreeze
from constants import deg_string


def ts_distribution_plot (file_path, grid=None, time_index=None, t_start=None, t_end=None, time_average=False, second_file_path=None, tmin=None, tmax=None, smin=None, smax=None, only_cavities=True, only_fris=True, num_bins=1000, date_string=None, figsize=(8,6), fig_name=None):

    # Build the grid if needed
    grid = choose_grid(grid, file_path)
    # Make sure we'll end up with a single record in time
    check_single_time(time_index, time_average)
    # Determine what to write about the date
    date_string = check_date_string(date_string, file_path, time_index)

    # Quick inner function to read data (THETA or SALT)
    def read_data (var_name):
        # First choose the right file
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path)
        else:
            file_path_use = file_path
        data = read_netcdf(file_path_use, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        return data
    # Call this function for each variable
    temp = read_data('THETA')
    salt = read_data('SALT')

    # Select the points we care about
    if only_fris:
        # Select all points in the FRIS cavity
        index = (grid.hfac > 0)*xy_to_xyz(grid.fris_mask, grid)
    elif only_cavities:
        # Select all points in ice shelf cavities
        index = (grid.hfac > 0)*xy_to_xyz(grid.ice_mask, grid)
    else:
        # Select all unmasked points
        index = grid.hfac > 0

    # Inner function to set up bins for a given variable (temp or salt)
    def set_bins (data):
        # Find the bounds on the data at the points we care about
        vmin = np.amin(data[index])
        vmax = np.amax(data[index])
        # Choose a small epsilon to add/subtract from the boundaries
        # This way nothing will be at the edge of a beginning/end bin
        eps = (vmax-vmin)*1e-3
        # Calculate boundaries of bins
        bins = np.linspace(vmin-eps, vmax+eps, num=num_bins)
        # Now calculate the centres of bins for plotting
        centres = 0.5*(bins[:-1] + bins[1:])
        return bins, centres
    # Call this function for each variable
    temp_bins, temp_centres = set_bins(temp)
    salt_bins, salt_centres = set_bins(salt)
    # Now set up a 2D array to increment with volume of water masses
    volume = np.zeros([temp_centres.size, salt_centres.size])

    # Loop over all cells to increment volume
    # This can't really be vectorised unfortunately
    for i in range(grid.nx):
        for j in range(grid.ny):            
            if only_fris and not grid.fris_mask[j,i]:
                # Disregard all points not in FRIS cavity
                continue
            if only_cavities and not grid.ice_mask[j,i]:
                # Disregard all points not in ice shelf cavities
                continue            
            for k in range(grid.nz):
                if grid.hfac[k,j,i] == 0:
                    # Disregard all masked points
                    continue
                # If we're still here, it's a point we care about
                # Figure out which bins it falls into
                temp_index = np.nonzero(temp_bins > temp[k,j,i])[0][0] - 1
                salt_index = np.nonzero(salt_bins > salt[k,j,i])[0][0] - 1
                # Calculate volume of this cell, taking partial cells into account
                dV = grid.dA[j,i]*grid.dz[k]*grid.hfac[k,j,i]
                # Increment volume array
                volume[temp_index, salt_index] += dV
    # Mask bins with zero volume
    volume = np.ma.masked_where(volume==0, volume)

    # Find the volume bounds for plotting
    min_vol = np.log(np.amin(volume))
    max_vol = np.log(np.amax(volume))
    # Calculate the surface freezing point for plotting
    tfreeze_sfc = tfreeze(salt_centres, 0)
    # Choose the plotting bounds if not set
    if tmin is not None:
        tmin = temp_bins[0]
    if tmax is not None:
        tmax = temp_bins[-1]
    if smin is not None:
        smin = salt_bins[0]
    if smax is not None:
        smax = salt_bins[-1]
    # Construct the title
    title = 'Water masses'
    if only_fris:
        title += ' in FRIS cavity'
    elif only_cavities:
        title += ' in ice shelf cavities'
    if date_string != '':
        title += ', ' + date_string

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # Use a log scale for visibility
    img = pcolor(salt_centres, temp_centres, np.log(volume), vmin=min_vol, vmax=max_vol)
    # Add the surface freezing point
    plt.plot(salt_centres, tfreeze_sfc, color='black', linestyle='dashed', linewidth=2)
    grid(True)
    ax.set_xlim([smin, smax])
    ax.set_ylim([tmin, tmax])
    ax.xlabel('Salinity (psu)')
    ax.ylabel('Temperature ('+deg_string+'C)')
    plt.colorbar(img)
    plt.text(.98, .5, 'log of volume', ha='center', transform=fig.transFigure)
    plt.title(title)
    finished_plot(fig, fig_name=fig_name)
    
    
    
    
    


