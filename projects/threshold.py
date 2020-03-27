##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..file_io import read_netcdf, NCfile
from ..utils import real_dir, var_min_max, select_bottom, mask_3d, mask_except_ice, convert_ismr
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.latlon import shade_land_ice, prepare_vel
from ..plot_utils.labels import latlon_axes, parse_date
from ..postprocess import segment_file_paths
from ..constants import deg_string
from ..plot_latlon import latlon_plot

# Analyse the coastal winds in UKESM vs ERA5:
#   1. Figure out what percentage of points have winds in the opposite directions
#   2. Suggest possible caps on the ERA5/UKESM ratio
#   3. Make scatterplots of both components
#   4. Plot the wind vectors and their differences along the coast
def analyse_coastal_winds (grid_dir, ukesm_file, era5_file, save_fig=False, fig_dir='./'):

    fig_name = None
    fig_dir = real_dir(fig_dir)

    print 'Selecting coastal points'
    grid = Grid(grid_dir)
    coast_mask = grid.get_coast_mask(ignore_iceberg=True)
    var_names = ['uwind', 'vwind']

    ukesm_wind_vectors = []
    era5_wind_vectors = []
    for n in range(2):
        print 'Processing ' + var_names[n]
        # Read the data and select coastal points only
        ukesm_wind = (read_netcdf(ukesm_file, var_names[n])[coast_mask]).ravel()
        era5_wind = (read_netcdf(era5_file, var_names[n])[coast_mask]).ravel()
        ratio = np.abs(era5_wind/ukesm_wind)
        # Save this component
        ukesm_wind_vectors.append(ukesm_wind)
        era5_wind_vectors.append(era5_wind)

        # Figure out how many are in opposite directions
        percent_opposite = float(np.count_nonzero(ukesm_wind*era5_wind < 0))/ukesm_wind.size*100
        print str(percent_opposite) + '% of points have ' + var_names[n] + ' components in opposite directions'

        print 'Analysing ratios'
        print 'Minimum ratio of ' + str(np.amin(ratio))
        print 'Maximum ratio of ' + str(np.amax(ratio))
        print 'Mean ratio of ' + str(np.mean(ratio))
        percent_exceed = np.empty(20)
        for i in range(20):
            percent_exceed[i] = float(np.count_nonzero(ratio > i+1))/ratio.size*100
        # Find first value of ratio which includes >90% of points
        i_cap = np.nonzero(percent_exceed < 10)[0][0] + 1
        print 'A ratio cap of ' + str(i_cap) + ' will cover ' + str(100-percent_exceed[i_cap]) + '%  of points'
        # Plot the percentage of points that exceed each threshold ratio
        fig, ax = plt.subplots()
        ax.plot(np.arange(20)+1, percent_exceed, color='blue')
        ax.grid(True)
        ax.axhline(y=10, color='red')
        plt.xlabel('Ratio', fontsize=16)
        plt.ylabel('%', fontsize=16)
        plt.title('Percentage of ' + var_names[n] + ' points exceeding given ratios', fontsize=18)
        if save_fig:
            fig_name = fig_dir + 'ratio_caps.png'
        finished_plot(fig, fig_name=fig_name)

        print 'Making scatterplot'
        fig, ax = plt.subplots()
        ax.scatter(era5_wind, ukesm_wind, color='blue')
        xlim = np.array(ax.get_xlim())
        ylim = np.array(ax.get_ylim())
        ax.axhline(color='black')
        ax.axvline(color='black')
        # Plot the y=x diagonal line in red
        ax.plot(xlim, xlim, color='red')
        # Plot the ratio cap in green
        ax.plot(i_cap*xlim, xlim, color='green')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.xlabel('ERA5', fontsize=16)
        plt.ylabel('UKESM', fontsize=16)
        plt.title(var_names[n] + ' (m/s) at coastal points, 1979-2014 mean', fontsize=18)
        # Construct figure name, if needed
        if save_fig:
            fig_name = fig_dir + 'scatterplot_' + var_names[n] + '.png'
        finished_plot(fig, fig_name=fig_name)

    print 'Plotting coastal wind vectors'
    scale = 30
    lon_coast = grid.lon_2d[coast_mask].ravel()
    lat_coast = grid.lat_2d[coast_mask].ravel()
    fig, gs = set_panels('1x3C0')
    # Panels for UKESM, ERA5, and ERA5 minus UKESM
    [uwind, vwind] = [[ukesm_wind_vectors[i], era5_wind_vectors[i], era5_wind_vectors[i]-ukesm_wind_vectors[i]] for i in range(2)]
    titles = ['UKESM', 'ERA5', 'ERA5 minus UKESM']
    for i in range(3):
        ax = plt.subplot(gs[0,i])
        shade_land_ice(ax, grid)
        q = ax.quiver(lon_coast, lat_coast, uwind[i], vwind[i], scale=scale)
        latlon_axes(ax, grid.lon_corners_2d, grid.lat_corners_2d)
        plt.title(titles[i], fontsize=16)
        if i > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.suptitle('Coastal winds', fontsize=20)
    if save_fig:
        fig_name = fig_dir + 'coastal_vectors.png'
    finished_plot(fig, fig_name=fig_name)


# Calculate the fields used for animate_cavity and save to a NetCDF file.
def precompute_animation_fields (output_dir='./', out_file='animation_fields.nc'):

    var_names = ['bwtemp', 'bwsalt', 'ismr', 'vel']
    num_vars = len(var_names)
    mask_names = ['land_mask', 'ice_mask']
    num_masks = len(mask_names)

    # Get all the model output files
    file_paths = segment_file_paths(real_dir(output_dir))

    time = None
    masks = [None for n in range(num_masks)]
    data = [None for n in range(num_vars)]
    vmin = [None for n in range(num_vars)]
    vmax = [None for n in range(num_vars)]
    # Loop over files
    for file_path in file_paths:
        print 'Processing ' + file_path
        time_tmp, units, calendar = netcdf_time(file_path, return_date=False, return_units=True)
        if time is None:
            # First file - initialise array
            time = time_tmp
        else:
            time = np.concatenate((time, time_tmp), axis=0)
        # Get masks as time-dependent fields
        grid = Grid(file_path)
        for n in range(num_masks):
            if mask_names[n] == 'land_mask':
                mask_tmp = add_time_dim(grid.land_mask, time_tmp.size)
            elif mask_names[n] == 'ice_mask':
                mask_tmp = add_time_dim(grid.ice_mask, time_tmp.size)
            if masks[n] is None:
                masks[n] = mask_tmp
            else:
                masks[n] = np.concatenate((masks[n], mask_tmp), axis=0)
        # Loop over proper variables and process data
        for n in range(num_vars):
            print '...'+var_names[n]
            if var_names[n] == 'bwtemp':
                data_tmp = select_bottom(mask_3d(read_netcdf(file_path, 'THETA'), grid, time_dependent=True))
            elif var_names[n] == 'bwsalt':
                data_tmp = select_bottom(mask_3d(read_netcdf(file_path, 'SALT'), grid, time_dependent=True))
            elif var_names[n] == 'ismr':
                data_tmp = convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx'), grid, time_dependent=True))
            elif var_names[n] == 'vel':
                u = mask_3d(read_netcdf(file_path, 'UVEL'), grid, gtype='u', time_dependent=True)
                v = mask_3d(read_netcdf(file_path, 'VVEL'), grid, gtype='v', time_dependent=True)
                data_tmp = prepare_vel(u, v, grid, vel_option='avg', time_dependent=True)[0]
            if data[n] is None:
                data[n] = data_tmp
            else:
                data[n] = np.concatenate((data[n], data_tmp), axis=0)
            # Find the min and max over the region
            for t in range(num_time):
                vmin_tmp, vmax_tmp = var_min_max(data_tmp[t], grid, zoom_fris=True, pster=True)
                if vmin[n] is None:
                    # First timestep - initialise
                    vmin[n] = vmin_tmp
                    vmax[n] = vmax_tmp
                else:
                    vmin[n] = min(vmin[n], vmin_tmp)
                    vmax[n] = max(vmax[n], vmax_tmp)

    # Write to NetCDF
    ncfile = NCfile(output_file, grid, 'xyt')
    ncfile.add_time(time, units=units, calendar=calendar)
    for n in range(num_masks):
        ncfile.add_variable(mask_names[n], masks[n], 'xyt')
    for n in range(num_vars):
        ncfile.add_variable(var_names[n], data[n], 'xyt', vmin=vmin[n], vmax=vmax[n])
    ncfile.close()    


# Make animations of bottom water temperature, bottom water salinity, ice shelf melt rate, and barotropic velocity in the FRIS cavity for the given simulation.
# Type "load_animations" in the shell before calling this function.
# The grid is just for grid sizes, so pass it any valid grid regardless of coupling status.
def animate_cavity (animation_file, grid, mov_name=None):

    import matplotlib.animation as animation

    grid = choose_grid(grid, None)

    var_names = ['bwtemp', 'bwsalt', 'ismr', 'vel']
    var_titles = ['Bottom water temperature ('+deg_string+'C)', 'Bottom water salinity (psu)', 'Ice shelf melt rate (m/y)', 'Barotropic velocity (m/s)']
    ctype = ['basic', 'basic', 'ismr', 'vel']
    # These min and max values will be overrided if they're not restrictive enough
    vmin = [-2.5, 33.4, None, None]
    vmax = [2.5, 34.75, None, None]
    num_vars = len(var_names)

    # Read data from precomputed file
    time = netcdf_time(animation_file)
    # Parse dates
    for date in time:
        dates.append(parse_date(date=date))
    land_mask = read_netcdf(animation_file, 'land_mask')
    ice_mask = read_netcdf(animation_file, 'ice_mask')
    data = []
    extend = []
    dates = []
    for n in range(num_vars):
        data_tmp, vmin_tmp, vmax_tmp = read_netcdf(animation_file, var_names[n], return_minmax=True)
        data.append(data_tmp)
        # Figure out what to do with bounds
        if vmin[n] is None or vmin[n] < vmin_tmp:
            extend_min = False
            vmin[n] = vmin_tmp
        else:
            extend_min = True
        if vmax[n] is None or vmax[n] > vmax_tmp:
            extend_max = False
            vmax[n] = vmax_tmp
        else:
            extend_max = True
        if extend_min and extend_max:
            extend.append('both')
        elif extend_min:
            extend.append('min')
        elif extend_max:
            extend.append('max')
        else:
            extend.append('neither')    

    # Initialise the plot
    fig, gs, cax1, cax2, cax3, cax4 = set_panels('2x2C4')
    cax = [cax1, cax2, cax3, cax4]
    ax = []
    for n in range(num_vars):
        ax.append(plt.subplot(gs[n/2,n%2]))
        ax[n].axis('equal')

    # Inner function to plot a frame
    def plot_one_frame (t):
        img = []
        for n in range(num_vars):
            img.append(latlon_plot(data[n][t,:], grid, ax=ax[n], make_cbar=False, ctype=ctype[n], vmin=vmin[n], vmax=vmax[n], zoom_fris=True, pster=True, title=var_titles[n], titlesize=16, land_mask=land_mask[t,:], ice_mask=ice_mask[t,:]))
        plt.suptitle(dates[t], fontsize=20)
        if t == 0:
            return img

    # First frame
    img = plot_one_frame(0)
    for n in range(num_vars):
        plt.colorbar(img[n], cax=cax[n], extend=extend[n])

    # Function to update figure with the given frame
    def animate(t):
        print 'Frame ' + str(t) + ' of ' + str(num_time)
        for n in range(num_vars):
            ax[n].cla()
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=range(num_time))
    writer = animation.FFMpegWriter(bitrate=500, fps=12)
    anim.save(mov_name, writer=writer)
    
