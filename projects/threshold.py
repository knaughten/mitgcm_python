##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..file_io import read_netcdf
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
def precompute_animation_fields (output_dir='./', file_path='animation_fields.nc'):

    var_names = ['bwtemp', 'bwsalt', 'ismr', 'vel']
    num_vars = len(var_names)
    
    file_paths = segment_file_paths(real_dir(output_dir))


# Make animations of bottom water temperature, bottom water salinity, ice shelf melt rate, and barotropic velocity in the FRIS cavity for the given simulation.
# Run "load_animations" before calling this function.
def animate_cavity (output_dir='./', mov_name=None):

    import matplotlib.animation as animation

    var_names = ['bwtemp', 'bwsalt', 'ismr', 'vel']
    var_titles = ['Bottom water temperature ('+deg_string+'C)', 'Bottom water salinity (psu)', 'Ice shelf melt rate (m/y)', 'Barotropic velocity (m/s)']
    ctype = ['basic', 'basic', 'ismr', 'vel']
    # These min and max values will be overrided if they're not restrictive enough
    vmin = [-2.5, 33.4, None, None]
    vmax = [2.5, 34.75, None, None]
    num_vars = len(var_names)
    
    file_paths = segment_file_paths(real_dir(output_dir))    

    # Read and process all the data from each segment
    all_grids = []
    all_dates = []
    all_data = [[] for n in range(num_vars)]
    for file_path in file_paths:
        print 'Processing ' + file_path
        grid = Grid(file_path)
        # Loop over variables
        for n in range(num_vars):
            print '...'+var_names[n]
            if var_names[n] == 'bwtemp':
                data = select_bottom(mask_3d(read_netcdf(file_path, 'THETA'), grid, time_dependent=True))
            elif var_names[n] == 'bwsalt':
                data = select_bottom(mask_3d(read_netcdf(file_path, 'SALT'), grid, time_dependent=True))
            elif var_names[n] == 'ismr':
                data = convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx'), grid, time_dependent=True))
            elif var_names[n] == 'vel':
                u = mask_3d(read_netcdf(file_path, 'UVEL'), grid, gtype='u', time_dependent=True)
                v = mask_3d(read_netcdf(file_path, 'VVEL'), grid, gtype='v', time_dependent=True)
                data = prepare_vel(u, v, grid, vel_option='avg', time_dependent=True)[0]
            # Loop over timesteps and append to master lists
            for t in range(data.shape[0]):
                if n == 0:
                    all_grids.append(grid)
                    all_dates.append(parse_date(file_path=file_path, time_index=t))
                all_data[n].append(data[t,:])
    num_time = len(all_data[0])

    # Now find true min and max values, and what the endpoints of each colourbar should do
    extend = []
    for n in range(num_vars):
        # Initialise to something crazy
        vmin_tmp = np.amax(all_data[n][0])
        vmax_tmp = np.amin(all_data[n][0])
        for t in range(num_time):
            vmin_2, vmax_2 = var_min_max(all_data[n][t], all_grids[t], zoom_fris=True, pster=True)
            vmin_tmp = min(vmin_tmp, vmin_2)
            vmax_tmp = max(vmax_tmp, vmax_2)
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
            img.append(latlon_plot(all_data[n][t], all_grids[t], ax=ax[n], make_cbar=False, ctype=ctype[n], vmin=vmin[n], vmax=vmax[n], zoom_fris=True, pster=True, title=var_titles[n], titlesize=18))
        plt.suptitle(all_dates[n], fontsize=24)
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
    
