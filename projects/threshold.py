##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import netCDF4 as nc
from scipy import stats

from ..grid import Grid, choose_grid, UKESMGrid
from ..file_io import read_netcdf, NCfile, netcdf_time, read_iceprod, read_binary
from ..utils import real_dir, var_min_max, select_bottom, mask_3d, mask_except_ice, convert_ismr, add_time_dim, mask_land, xy_to_xyz, moving_average, mask_land_ice, fix_lon_range, split_longitude
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.latlon import shade_land_ice, prepare_vel, overlay_vectors
from ..plot_utils.labels import latlon_axes, parse_date, slice_axes
from ..plot_utils.slices import transect_patches, transect_values, plot_slice_patches
from ..plot_utils.colours import set_colours
from ..postprocess import segment_file_paths
from ..constants import deg_string, vaf_to_gmslr, temp_C2K
from ..plot_latlon import latlon_plot, read_plot_latlon_comparison
from ..plot_1d import read_plot_timeseries_ensemble, timeseries_multi_plot
from ..plot_misc import read_plot_hovmoller_ts
from ..plot_slices import get_loc
from ..timeseries import calc_annual_averages
from ..plot_ua import read_ua_difference, check_read_gl, read_ua_bdry, ua_plot
from ..diagnostics import density, parallel_vector, tfreeze
from ..interpolation import interp_reg_xy


# Global variables
sim_keys = ['ctO', 'ctIO', 'abO', 'abIO', '1pO', '1pIO']
sim_dirs = ['WSFRIS_'+key+'/output/' for key in sim_keys]
sim_names = ['piControl-O','piControl-IO','abrupt-4xCO2-O','abrupt-4xCO2-IO','1pctCO2-O','1pctCO2-IO']
coupled = [False, True, False, True, False, True]
timeseries_file = 'timeseries.nc'
timeseries_file_2 = 'timeseries2_annual.nc'
hovmoller_file = 'hovmoller.nc'
ua_post_file = 'ua_postprocessed.nc'
end_file = 'last_10y.nc'
mid_file = 'years_26_35.nc'
num_sim = len(sim_keys)
grid_path = 'WSFRIS_999/ini_grid/'  # Just for dimensions etc.


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

    var_names = ['bwtemp', 'bwsalt'] #, 'ismr', 'vel']
    num_vars = len(var_names)

    # Get all the model output files
    file_paths = segment_file_paths(real_dir(output_dir))

    time = None
    land_mask = None
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
        # Get land mask as time-dependent field
        grid = Grid(file_path)
        mask_tmp = add_time_dim(grid.land_mask, time_tmp.size)
        if land_mask is None:
            land_mask = mask_tmp
        else:
            land_mask = np.concatenate((land_mask, mask_tmp), axis=0)
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
            for t in range(time_tmp.size):
                vmin_tmp, vmax_tmp = var_min_max(data_tmp[t], grid, zoom_fris=True, pster=True)
                if vmin[n] is None:
                    # First timestep - initialise
                    vmin[n] = vmin_tmp
                    vmax[n] = vmax_tmp
                else:
                    vmin[n] = min(vmin[n], vmin_tmp)
                    vmax[n] = max(vmax[n], vmax_tmp)

    # Write to NetCDF
    ncfile = NCfile(out_file, grid, 'xyt')
    ncfile.add_time(time, units=units, calendar=calendar)
    ncfile.add_variable('land_mask', land_mask, 'xyt')
    for n in range(num_vars):
        ncfile.add_variable(var_names[n], data[n], 'xyt', vmin=vmin[n], vmax=vmax[n])
    ncfile.close()    


# Make animations of bottom water temperature and salinity in the FRIS cavity for the given simulation.
# Type "load_animations" in the shell before calling this function.
# The grid is just for grid sizes, so pass it any valid grid regardless of coupling status.
def animate_cavity (animation_file, grid, mov_name='cavity.mp4'):

    import matplotlib.animation as animation

    grid = choose_grid(grid, None)

    var_names = ['bwtemp', 'bwsalt'] #, 'ismr', 'vel']
    var_titles = ['Bottom water temperature ('+deg_string+'C)', 'Bottom water salinity (psu)'] #, 'Ice shelf melt rate (m/y)', 'Barotropic velocity (m/s)']
    ctype = ['basic', 'basic'] #, 'ismr', 'vel']
    # These min and max values will be overrided if they're not restrictive enough
    vmin = [-2.5, 33.4] #, None, None]
    vmax = [2.5, 34.75] #, None, None]
    num_vars = len(var_names)

    # Read data from precomputed file
    time = netcdf_time(animation_file)
    num_time = time.size
    # Parse dates
    dates = []
    for date in time:
        dates.append(parse_date(date=date))
    land_mask = read_netcdf(animation_file, 'land_mask') == 1
    data = []
    extend = []    
    for n in range(num_vars):
        data_tmp, vmin_tmp, vmax_tmp = read_netcdf(animation_file, var_names[n], return_minmax=True)
        data_tmp = np.ma.masked_where(land_mask, data_tmp)
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
    fig, gs, cax1, cax2 = set_panels('1x2C2', figsize=(24,12))
    cax = [cax1, cax2]
    ax = []
    for n in range(num_vars):
        ax.append(plt.subplot(gs[n/2,n%2]))
        ax[n].axis('equal')

    # Inner function to plot a frame
    def plot_one_frame (t):
        img = []
        for n in range(num_vars):
            img.append(latlon_plot(data[n][t,:], grid, ax=ax[n], make_cbar=False, ctype=ctype[n], vmin=vmin[n], vmax=vmax[n], zoom_fris=True, pster=True, title=var_titles[n], titlesize=36, land_mask=land_mask[t,:]))
        plt.suptitle(dates[t], fontsize=40)
        if t == 0:
            return img

    # First frame
    img = plot_one_frame(0)
    for n in range(num_vars):
        cbar = plt.colorbar(img[n], cax=cax[n], extend=extend[n], orientation='horizontal')
        cbar.ax.tick_params(labelsize=18)

    # Function to update figure with the given frame
    def animate(t):
        print 'Frame ' + str(t+1) + ' of ' + str(num_time)
        for n in range(num_vars):
            ax[n].cla()
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=range(num_time))
    writer = animation.FFMpegWriter(bitrate=2000, fps=12)
    anim.save(mov_name, writer=writer)
    

# Plot all the timeseries variables, showing all simulations on the same axes for each variable.
def plot_all_timeseries (base_dir='./', fig_dir='./'):
    
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    timeseries_types = ['fris_massloss', 'fris_temp', 'fris_salt', 'fris_density', 'sws_shelf_temp', 'sws_shelf_salt', 'sws_shelf_density', 'filchner_trough_temp', 'filchner_trough_salt', 'filchner_trough_density', 'wdw_core_temp', 'wdw_core_salt', 'wdw_core_density', 'seaice_area', 'wed_gyre_trans', 'filchner_trans', 'sws_shelf_iceprod']  # Everything except the mass balance (doesn't work as ensemble)
    # Now the Ua timeseries for the coupled simulations
    ua_timeseries = ['iceVAF', 'iceVolume', 'groundedArea', 'slr_contribution']
    ua_titles = ['Ice volume above floatation', 'Total ice volume', 'Total grounded area', 'Sea level rise contribution']
    ua_units = ['m^3','m^3','m^2','m']
    file_paths = [base_dir + d + timeseries_file for d in sim_dirs]
    colours = ['blue', 'blue', 'red', 'red', 'green', 'green']
    linestyles = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
    ua_files = [base_dir + d + ua_post_file for d in sim_dirs]

    for var in timeseries_types:
        for annual_average in [False, True]:
            fig_name = fig_dir + var
            if annual_average:
                fig_name += '_annual.png'
            else:
                fig_name += '.png'
            read_plot_timeseries_ensemble(var, file_paths, sim_names, precomputed=True, colours=colours, linestyles=linestyles, annual_average=annual_average, time_use=2, fig_name=fig_name)

    # Now the Ua timeseries
    sim_names_ua = []
    colours_ua = []
    # Read time from an ocean file
    time = netcdf_time(file_paths[2], monthly=False)
    # Read data from each simulation
    for i in range(len(ua_timeseries)):
        var = ua_timeseries[i]
        datas = []
        for n in range(num_sim):
            if coupled[n]:
                sim_names_ua.append(sim_names[n])
                colours_ua.append(colours[n])
                if var == 'slr_contribution':
                    # Calculate from iceVAF
                    data_tmp = read_netcdf(ua_files[n], 'iceVAF')
                    data_tmp = (data_tmp-data_tmp[0])*vaf_to_gmslr
                else:
                    data_tmp = read_netcdf(ua_files[n], var)
                datas.append(data_tmp)
        timeseries_multi_plot(time, datas, sim_names_ua, colours_ua, title=ua_titles[i], units=ua_units[i], fig_name=fig_dir+var+'.png')


# For each of temperature, salinity, and density, plot all four key regions on the same axis, for a single simulation.
def plot_timeseries_regions (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    file_paths = [base_dir + d + timeseries_file for d in sim_dirs]
    var_names = ['temp', 'salt', 'density']
    var_titles = ['Temperature', 'Salinity', 'Density']
    units = [deg_string+'C', 'psu', r'kg/m$^3$']
    regions = ['fris', 'filchner_trough', 'sws_shelf', 'wdw_core']
    region_labels = ['FRIS', 'Filchner Trough', 'Continental Shelf', 'WDW core']
    colours = ['blue', 'magenta', 'green', 'red']

    for n in range(num_sim):
        for m in range(len(var_names)):
            time = netcdf_time(file_paths[n], monthly=False)
            datas = []
            for loc in regions:
                datas.append(read_netcdf(file_paths[n], loc+'_'+var_names[m]))
            time, datas = calc_annual_averages(time, datas)                
            timeseries_multi_plot(time, datas, region_labels, colours, title=var_titles[m]+', '+sim_names[n], units=units[m], fig_name=fig_dir+var_names[m]+'_'+sim_keys[n]+'.png')


# Plot grounding line at the beginning of the ctIO simulation, and the end of each coupled simulation.
def gl_plot (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    ua_files = [base_dir + d + ua_post_file for d in sim_dirs]
    colours = ['black', 'blue', 'red', 'green']
    labels = ['Initial']

    xGL_all = []
    yGL_all = []
    for n in range(num_sim):
        if coupled[n]:
            xGL = read_netcdf(ua_files[n], 'xGL')
            yGL = read_netcdf(ua_files[n], 'yGL')
            labels.append(sim_names[n])
            if len(xGL_all)==0:
                # Initial grounding line for the first simulation
                xGL_all.append(xGL[0,:])
                yGL_all.append(yGL[0,:])
            # Final grounding line for all simulations
            xGL_all.append(xGL[-1,:])
            yGL_all.append(yGL[-1,:])
    fig, ax = plt.subplots(figsize=(7,6))
    for n in range(len(xGL_all)):
        ax.plot(xGL_all[n], yGL_all[n], '-', color=colours[n], label=labels[n])
    ax.legend(loc='upper left')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Final grounding line position')
    finished_plot(fig, fig_name=fig_dir+'gl_final.png')


# Make a Hovmoller plot of temperature and salinity in the Filchner Trough for each simulation.
def filchner_trough_hovmollers (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    file_paths = [base_dir + d + hovmoller_file for d in sim_dirs]
    grid = Grid(base_dir+grid_path)

    for n in range(num_sim):
        read_plot_hovmoller_ts(file_paths[n], 'filchner_trough', grid, smooth=6, t_contours=[-1.9], fig_name=fig_dir+'hovmoller_ft_'+sim_keys[n]+'.png')


# Plot anomalies in Ua variables (ice thickness and velocity) for four scenarios:
# 1. Drift in piControl (last year minus first year)
# 2. abrupt-4xCO2 minus piControl, after 75 years
# 3. abrupt-4xCO2 minus piControl, after 150 years
# 4. 1pctCO2 minus piControl, after 150 years
def plot_ua_changes (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    var_names = ['h', 'velb']
    var_titles = ['Change in ice thickness (m)', 'Change in basal velocity (m/y)']
    vmin = [-80, -50]
    vmax = [80, 50]

    # Construct file paths
    num_sims = 4
    years = [[2910, 2984, 3059, 3059], [3059, 1924, 1999, 1999]]
    sims = [['ctIO' for n in range(num_sims)], ['ctIO', 'abIO', 'abIO', '1pIO']]
    titles = ['Drift in piControl (150 y)', 'abrupt-4xCO2 minus piControl (75 y)', 'abrupt-4xCO2 minus piControl (150 y)', '1pctCO2 minus piControl (150 y)']
    gl_time_index = [150, 75, 150, 150]
    file_paths = [[], []]
    gl_files = []
    for n in range(num_sims):
        for m in range(2):
            file_paths[m].append(base_dir+'WSFRIS_'+sims[m][n]+'/output/'+str(years[m][n])+'01/Ua/WSFRIS_'+sims[m][n]+'_'+str(years[m][n])+'01_0360.mat')
        gl_files.append(base_dir+'WSFRIS_'+sims[1][n]+'/output/'+ua_post_file)

    # Read grounding line data
    xGL = []
    yGL = []
    for n in range(num_sims):
        xGL_tmp, yGL_tmp = check_read_gl(gl_files[n], gl_time_index[n]-1)
        xGL.append(xGL_tmp)
        yGL.append(yGL_tmp)
    x_bdry, y_bdry = read_ua_bdry(file_paths[0][0])

    # Loop over variables
    for i in range(len(var_names)):
        print 'Processing ' + var_names[i]
        data = []
        for n in range(num_sims):
            x, y, data_diff = read_ua_difference(var_names[i], file_paths[0][n], file_paths[1][n])
            data.append(data_diff)
        # Set up plot
        fig, gs, cax = set_panels('2x2C1', figsize=(9.5,10))
        for n in range(num_sims):
            ax = plt.subplot(gs[n/2, n%2])
            img = ua_plot('reg', data[n], x, y, xGL=xGL[n], yGL=yGL[n], x_bdry=x_bdry, y_bdry=y_bdry, ax=ax, make_cbar=False, ctype='plusminus', vmin=vmin[i], vmax=vmax[i], zoom_fris=True, title=titles[n], titlesize=16, extend='both')
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        plt.suptitle(var_titles[i], fontsize=24)
        finished_plot(fig, fig_name=fig_dir+'ua_changes_'+var_names[i]+'.png')


# Plot bottom water temperature, salinity, density, and velocity zoomed into the Filchner Trough region of the continental shelf break.
def plot_inflow_zoom (base_dir='./', fig_dir='./'):

    var_names = ['bwtemp', 'bwsalt', 'density', 'vel']
    ctype = ['basic', 'basic', 'basic', 'vel']
    var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity (psu)', r'Bottom potential density (kg/m$^3$-1000)', 'Bottom velocity (m/s)']
    sim_numbers = [0, 2, 4]
    file_paths = [base_dir + sim_dirs[n] + end_file for n in sim_numbers]
    sim_names_plot = [sim_names[n] for n in sim_numbers]
    num_sim_plot = len(sim_numbers)
    [xmin, xmax, ymin, ymax] = [-50, -20, -77, -73]
    h0 = -1250
    chunk = 4
    vmin_impose = [None, 33.6, 27, None]

    grid = Grid(grid_path)
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    # Loop over variables
    bwtemp = []
    bwsalt = []
    for m in range(len(var_names)):
        # Read the data
        data = []
        u = []
        v = []
        for n in range(num_sim_plot):
            if var_names[m] == 'bwtemp':
                data_tmp = select_bottom(mask_3d(read_netcdf(file_paths[n], 'THETA', time_index=0), grid))
                # Save bottom water temperature for later
                bwtemp.append(data_tmp.data)
            elif var_names[m] == 'bwsalt':
                data_tmp = select_bottom(mask_3d(read_netcdf(file_paths[n], 'SALT', time_index=0), grid))
                bwsalt.append(data_tmp.data)
            elif var_names[m] == 'density':
                data_tmp = mask_land(density('MDJWF', bwsalt[n], bwtemp[n], 0), grid)-1e3
            elif var_names[m] == 'vel':
                u_tmp = mask_3d(read_netcdf(file_paths[n], 'UVEL', time_index=0), grid, gtype='u')
                v_tmp = mask_3d(read_netcdf(file_paths[n], 'VVEL', time_index=0), grid, gtype='v')
                data_tmp, u_tmp, v_tmp = prepare_vel(u_tmp, v_tmp, grid, vel_option='bottom')
                u.append(u_tmp)
                v.append(v_tmp)
            data.append(data_tmp)
                
        # Get the colour bounds
        vmin = np.amax(data[0])
        vmax = np.amin(data[0])
        extend = 'neither'
        for n in range(num_sim_plot):
            vmin_tmp, vmax_tmp = var_min_max(data[n], grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            vmin = min(vmin, vmin_tmp)
            vmax = max(vmax, vmax_tmp)
        if vmin_impose[m] is not None:
            vmin = vmin_impose[m]
            extend = 'min'

        # Make the plot
        fig, gs, cax = set_panels('1x3C1')
        for n in range(num_sim_plot):
            ax = plt.subplot(gs[0,n])
            img = latlon_plot(data[n], grid, ax=ax, make_cbar=False, ctype=ctype[m], vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=sim_names_plot[n])
            if n != 0:
                # Remove axis labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            # Contour 1250 m isobath
            ax.contour(grid.lon_2d, grid.lat_2d, grid.bathy, levels=[h0], colors='black', linestyles='dashed')
            if var_names[m] == 'vel':
                # Overlay velocity vectors
                overlay_vectors(ax, u[n], v[n], grid, chunk=chunk)
        plt.colorbar(img, cax=cax, orientation='horizontal', extend=extend)
        plt.suptitle(var_titles[m]+' over last 10 years', fontsize=24)
        finished_plot(fig, fig_name=fig_dir+'filchner_trough_zoom_'+var_names[m]+'.png')


# Plot slices through the Filchner Trough for temperature, salinity, density, and along-transect velocity.
def filchner_trough_slices (base_dir='./', fig_dir='./'):

    point0 = (-43, -81)
    point1 = (-25, -72)
    # Variables to plot
    var_names = ['temp', 'salt', 'rho', 'valong']
    var_titles = ['Temperature ('+deg_string+'C)', 'Salinity (psu)', 'Potential density (kg/m$^3$-1000', 'Along-transect velocity (m/s)']
    ctype = ['basic', 'basic', 'basic', 'plusminus']
    num_vars = len(var_names)
    # Simulations to plot
    sim_numbers = [0, 2, 4]
    file_paths = [base_dir + sim_dirs[n] + end_file for n in sim_numbers]
    sim_names_plot = [sim_names[n] for n in sim_numbers]
    num_sim_plot = len(sim_numbers)
    zmin = -1500

    grid = Grid(grid_path)
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    patches = None
    temp = []
    salt = []
    for m in range(num_vars):
        # Read data
        data = []
        for n in range(num_sim_plot):
            if var_names[m] == 'temp':
                data_tmp = mask_3d(read_netcdf(file_paths[n], 'THETA', time_index=0), grid)
                # Save temperature for density calculation
                temp.append(data_tmp.data)
            elif var_names[m] == 'salt':
                data_tmp = mask_3d(read_netcdf(file_paths[n], 'SALT', time_index=0), grid)
                # Save temperature for density calculation
                salt.append(data_tmp.data)
            elif var_names[m] == 'rho':
                data_tmp = mask_3d(density('MDJWF', salt[n], temp[n], 0), grid)-1000
            elif var_names[m] == 'valong':
                u = mask_3d(read_netcdf(file_paths[n], 'UVEL', time_index=0), grid, gtype='u')
                v = mask_3d(read_netcdf(file_paths[n], 'VVEL', time_index=0), grid, gtype='v')
                data_tmp = parallel_vector(u, v, grid, point0, point1)
            data.append(data_tmp)

        # Make the patches and find the min and max values
        vmin = np.amax(data[0])
        vmax = np.amin(data[0])
        values = []
        for n in range(num_sim_plot):
            if patches is None:
                patches, values_tmp, hmin, hmax, zmin, zmax, vmin_tmp, vmax_tmp, left, right, below, above = transect_patches(data[n], grid, point0, point1, zmin=zmin, return_bdry=True)
            else:
                values_tmp, vmin_tmp, vmax_tmp = transect_values(data[n], grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax)
            values.append(values_tmp)
            vmin = min(vmin, vmin_tmp)
            vmax = max(vmax, vmax_tmp)

        # Make the plot
        cmap, vmin, vmax = set_colours(values[0], ctype=ctype[m], vmin=vmin, vmax=vmax)
        loc_string = get_loc(None, point0=point0, point1=point1)[1]
        fig, gs, cax = set_panels('1x3C1')
        for n in range(num_sim_plot):
            ax = plt.subplot(gs[0,n])
            img = plot_slice_patches(ax, patches, values[n], hmin, hmax, zmin, zmax, vmin, vmax, cmap=cmap)
            slice_axes(ax, h_axis='trans')
            if n != 0:
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.set_yticklabels([])
                ax.set_ylabel('')
            plt.title(sim_names_plot[n], fontsize=18)
        plt.colorbar(img, cax=cax, orientation='horizontal')
        plt.suptitle(var_titles[m] + ' from ' + loc_string + ', last 10 years', fontsize=24)
        finished_plot(fig, fig_name=fig_dir+'filchner_trough_slice_'+var_names[m]+'.png')


# Plot changes in atmospheric forcing variables between simulations.
def plot_forcing_changes (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    # ctO is baseline, abO and 1pO are sensitivity tests
    sim_numbers = [0, 2, 4]
    num_sim_plot = len(sim_numbers)
    directories = [base_dir + sim_dirs[n] for n in [0,2,4]]
    sim_names_plot = [sim_names[n] for n in sim_numbers]
    sim_keys_plot = [sim_keys[n] for n in sim_numbers]
    # Variables to plot
    var_names = ['atemp', 'wind', 'precip', 'iceprod']
    [xmin, xmax, ymin, ymax] = [-70, -24, -79, -72]

    grid = Grid(base_dir+grid_path)
    for n in range(1, num_sim_plot):
        for var in var_names:
            fig_name = fig_dir+var+'_'+sim_keys_plot[n]+'.png'
            read_plot_latlon_comparison(var, sim_names_plot[0], sim_names_plot[n], directories[0], directories[n], end_file, grid=grid, time_index=0, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_name=fig_name)


# Plot anomalies in bottom density in and around the FRIS cavity at two stages of the abrupt-4xCO2 simulation.
def plot_density_stages (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    sim_dirs_plot = [base_dir+sim_dirs[n] for n in [0,2]]
    fnames = [mid_file, end_file]
    titles = ['Years 26-35', 'Years 141-150']

    grid = Grid(base_dir+grid_path)
    data_diff = []
    for n in range(len(fnames)):
        data_abs = []
        for m in range(len(sim_dirs_plot)):
            file_path = sim_dirs_plot[m]+fnames[n]
            bwtemp = select_bottom(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid))
            bwsalt = select_bottom(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid))
            data_abs.append(mask_land(density('MDJWF', bwsalt, bwtemp, 0),grid)-1e3)
        data_diff.append(data_abs[1]-data_abs[0])

    fig, gs, cax1, cax2 = set_panels('1x2C2', figsize=(12,7))
    cax = [cax1, cax2]
    for n in range(len(fnames)):
        ax = plt.subplot(gs[0,n])
        img = latlon_plot(data_diff[n], grid, ax=ax, make_cbar=False, zoom_fris=True, pster=True, title=titles[n])
        plt.colorbar(img, cax=cax[n], orientation='horizontal')
    plt.suptitle(r'Bottom density anomaly (kg/m$^3$-1000), abrupt-4xCO2 minus piControl', fontsize=24)
    finished_plot(fig, fig_name=fig_dir+'density_stages.png')


# Plot streamfunction in and around the cavity in the piControl simulation and at two stages of the abrupt-4xCO2 simulations.
def plot_psi_stages (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    file_paths = [base_dir+sim_dirs[0]+end_file, base_dir+sim_dirs[2]+mid_file, base_dir+sim_dirs[2]+end_file]
    titles = ['piControl', 'abrupt-4xCO2 (years 26-35)', 'abrupt-4xCO2 (years 141-150)']
    vmin = -0.25
    vmax = 0.25

    grid = Grid(base_dir+grid_path)
    fig, gs, cax = set_panels('1x3C1', figsize=(16,7))
    for n in range(len(file_paths)):
        data = np.sum(mask_3d(read_netcdf(file_paths[n], 'PsiVEL'), grid), axis=0)*1e-6
        ax = plt.subplot(gs[0,n])
        img = latlon_plot(data, grid, ax=ax, make_cbar=False, ctype='plusminus', vmin=vmin, vmax=vmax, zoom_fris=True, pster=True, title=titles[n])
    plt.colorbar(img, cax=cax, orientation='horizontal')
    plt.suptitle('Horizontal velocity streamfunction (Sv), vertically integrated', fontsize=24)
    finished_plot(fig, fig_name=fig_dir+'psi_stages.png')


# Calculate the min and max annually-averaged temperature and salinity on the continental shelf and FRIS cavity over one simulation, to set the bounds for TS bins.
def precompute_ts_bounds (output_dir='./'):

    var_names = ['THETA', 'SALT']
    vmin = [100, 100]
    vmax = [-100, -100]

    # Loop over segments
    all_file_paths = segment_file_paths(real_dir(output_dir))
    for file_path in all_file_paths:
        print 'Reading ' + file_path
        grid = Grid(file_path)
        # Get the indices on the continental shelf and FRIS cavity
        loc_index = (grid.hfac > 0)*xy_to_xyz(grid.get_region_mask('sws_shelf') + grid.get_ice_mask(shelf='fris'), grid)
        # Loop over variables
        for n in range(len(var_names)):
            print '...' + var_names[n]
            data = read_netcdf(file_path, var_names[n], time_average=True)
            vmin[n] = min(vmin[n], np.amin(data[loc_index]))
            vmax[n] = max(vmax[n], np.amax(data[loc_index]))

    # Print the results
    for n in range(len(var_names)):
        print var_names[n] + ' bounds: ' + str(vmin[n]) + ', ' + str(vmax[n])


# Precompute the T/S distribution for the animation in the next function, and save to a NetCDF file.
def precompute_ts_animation_fields (expt, output_dir='./', out_file='ts_animation_fields.nc'):

    if expt == 'abIO':
        temp_bounds = [-3.203, 2.74]
        salt_bounds = [32.025, 34.847]
        start_year = 1850
    elif expt == '1pIO':
        temp_bounds = [-3.226, 1.879]
        salt_bounds = [32.234, 34.881]
        start_year = 1850
    else:
        print 'Error (precompute_ts_animation_fields): unknown expt ' + expt
        sys.exit()

    file_paths = segment_file_paths(real_dir(output_dir))
    num_years = len(file_paths)
    num_bins = 1000
    
    # Set up bins
    def set_bins (bounds):
        eps = (bounds[1]-bounds[0])*1e-3
        edges = np.linspace(bounds[0]-eps, bounds[1]+eps, num=num_bins+1)
        centres = 0.5*(edges[:-1] + edges[1:])
        return edges, centres
    temp_edges, temp_centres = set_bins(temp_bounds)
    salt_edges, salt_centres = set_bins(salt_bounds)
    volume = np.zeros([num_years, num_bins, num_bins])

    # Loop over years
    for t in range(num_years):
        print 'Processing ' + file_paths[t]
        # Set up the masks
        grid = Grid(file_paths[t])
        loc_index = (grid.hfac > 0)*xy_to_xyz(grid.get_region_mask('sws_shelf') + grid.get_ice_mask(shelf='fris'), grid)
        # Read data
        temp = read_netcdf(file_paths[t], 'THETA', time_average=True)
        salt = read_netcdf(file_paths[t], 'SALT', time_average=True)
        # Loop over valid cells and categorise them into bins
        for temp_val, salt_val, grid_val in itertools.izip(temp[loc_index], salt[loc_index], grid.dV[loc_index]):
            temp_index = np.nonzero(temp_edges > temp_val)[0][0]-1
            salt_index = np.nonzero(salt_edges > salt_val)[0][0]-1
            volume[t, temp_index, salt_index] += grid_val
    # Mask bins with zero volume
    volume = np.ma.masked_where(volume==0, volume)

    # Write to NetCDF
    id = nc.Dataset(out_file, 'w')
    id.createDimension('time', None)
    id.createVariable('time', 'f8', ('time'))
    id.variables['time'][:] = np.arange(num_years)+start_year
    def add_dimension (data, dim_name):
        id.createDimension(dim_name, data.size)
        id.createVariable(dim_name, 'f8', (dim_name))
        id.variables[dim_name][:] = data
    add_dimension(temp_centres, 'temp_centres')
    add_dimension(salt_centres, 'salt_centres')
    add_dimension(temp_edges, 'temp_edges')
    add_dimension(salt_edges, 'salt_edges')
    id.createVariable('volume', 'f8', ('time', 'temp_centres', 'salt_centres'))
    id.variables['volume'][:] = volume
    id.close()


# Make an animated T/S diagram through the simulation.
# Type "load_animations" in the shell before calling this function.
def ts_animation (file_path='ts_animation_fields.nc', mov_name='ts_diagram.mp4'):

    import matplotlib.animation as animation
    smin = 32.5

    # Read data
    time = read_netcdf(file_path, 'time')
    temp_edges = read_netcdf(file_path, 'temp_edges')
    salt_edges = read_netcdf(file_path, 'salt_edges')
    temp_centres = read_netcdf(file_path, 'temp_centres')
    salt_centres = read_netcdf(file_path, 'salt_centres')
    volume = read_netcdf(file_path, 'volume')
    # Get volume bounds for plotting
    min_vol = np.log(np.amin(volume))
    max_vol = np.log(np.amax(volume))
    # Calculate surface freezing point
    tfreeze_sfc = tfreeze(salt_centres, 0)
    # Calculate potential density of bins
    salt_2d, temp_2d = np.meshgrid(salt_centres, temp_centres)
    rho = density('MDJWF', salt_2d, temp_2d, 0)
    # Density contours to plot
    rho_lev = np.arange(1025.4, 1028.4, 0.2)
    
    # Initialise the plot
    fig, ax = plt.subplots(figsize=(8,6))

    # Inner function to plot one frame
    def plot_one_frame (t):
        img = ax.pcolormesh(salt_edges, temp_edges, np.log(volume[t,:]), vmin=min_vol, vmax=max_vol)
        ax.contour(salt_centres, temp_centres, rho, rho_lev, colors='black', linestyles='dotted')
        ax.plot(salt_centres, tfreeze_sfc, color='black', linestyle='dashed', linewidth=2)        
        ax.grid(True)
        ax.set_xlim([smin, salt_edges[-1]])
        ax.set_ylim([temp_edges[0], temp_edges[-1]])
        plt.xlabel('Salinity (psu)')
        plt.ylabel('Temperature ('+deg_string+'C)')
        plt.text(.9, .6, 'log of volume', ha='center', rotation=-90, transform=fig.transFigure)
        plt.title(str(time[t]))
        if t == 0:
            return img

    # First frame
    img = plot_one_frame(0)
    plt.colorbar(img)

    # Function to update figure with the given frame
    def animate(t):
        print 'Frame ' + str(t+1) + ' of ' + str(time.size)
        ax.cla()
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=range(time.size))
    writer = animation.FFMpegWriter(bitrate=2000, fps=2)
    anim.save(mov_name, writer=writer)


# Plot timeseries of changes in sea ice formation compared to changes in P-E over the continental shelf for the given simulation. Smooth with the given radius.
def plot_iceprod_pminuse (sim_key, smooth=0, base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    if sim_key in ['abO', '1pO']:
        ctrl_key = 'ctO'
    elif sim_key in ['abIO', '1pIO']:
        ctrl_key = 'ctIO'
    else:
        print 'Error (plot_iceprod_pminuse): invalid sim_key ' + sim_key
        sys.exit()
    sim_numbers = [sim_keys.index(key) for key in [ctrl_key, sim_key]]
    file_paths = [base_dir + sim_dirs[n] + timeseries_file_2 for n in sim_numbers]
    var_names = ['sws_shelf_pminuse', 'sws_shelf_iceprod']
    var_titles = ['Precipitation\nminus evaporation', 'Sea ice\nproduction']
    colours = ['green', 'blue']

    time = netcdf_time(file_paths[1], monthly=False)
    data = []
    for var in var_names:
        # Get average over control simulation
        base_val = read_netcdf(file_paths[0], var, time_average=True)
        # Now read the transient simulation and subtract the baseline value
        data_diff = read_netcdf(file_paths[1], var) - base_val
        # Smooth
        data_smoothed, time_trimmed = moving_average(data_diff, smooth, time=time)
        if var == var_names[0]:
            # Replace the time array only once
            time = time_trimmed
        data.append(data_smoothed)

    title = 'Anomalies on the continental shelf'
    if smooth > 0:
        title += ' ('+str(2*smooth+1)+'-year smoothed)'
    title += ':\n'+sim_names[sim_numbers[1]]+' minus average of '+sim_names[sim_numbers[0]]

    timeseries_multi_plot(time, data, var_titles, colours, title=title, units=r'10$^3$ m$^3$/y', fig_name=fig_dir+'timeseries_iceprod_pminuse_'+sim_key+'.png')
    

# Plot a map of the average number of months per year with net sea ice formation during the piControl-IO simulation, as well as the trend over the abrupt-4xCO2-IO and 1pctCO2-IO simulations.
def plot_freezing_months (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    sim_numbers = [1, 3, 5]
    directories = [sim_dirs[n] for n in sim_numbers]
    titles = ['Average freezing months per year\n('+sim_names[sim_numbers[0]]+')', 'Trend over 150 years\n('+sim_names[sim_numbers[1]]+')', 'Trend over 150 years\n('+sim_names[sim_numbers[2]]+')']
    [xmin, xmax, ymin, ymax] = [-70, -24, -79, -72]
    vmin = -3
    vmax = 0

    # Grid, just for open ocean mask
    grid = Grid(base_dir+grid_path)

    data_plot = []
    for n in range(len(directories)):
        print 'Processing ' + sim_names[sim_numbers[n]]
        output_dir = directories[n]
        file_paths = segment_file_paths(output_dir)
        num_time = len(file_paths)
        data = np.empty([num_time, grid.ny, grid.nx])
        for t in range(num_time):
            print '...year ' + str(t+1) + ' of ' + str(num_time)
            # Read sea ice production and add up all the freezing months this year
            iceprod = read_iceprod(file_paths[t])
            is_freezing = (iceprod > 0).astype(float)
            data[t,:] = np.sum(is_freezing, axis=0)
        if n == 0:
            # Calculate the average
            data_plot.append(mask_land_ice(np.mean(data, axis=0),grid))
        else:
            print 'Calculating trend'
            # Calculate the trend - have to do this individually for each data point
            time = np.arange(num_time)
            trend = np.empty([grid.ny, grid.nx])
            for j in range(grid.ny):
                for i in range(grid.nx):
                    trend[j,i] = stats.linregress(time, data[:,j,i])[0]*num_time
            data_plot.append(mask_land_ice(trend, grid))

    # Make the plot
    fig, gs, cax1, cax2 = set_panels('1x3C2')
    vmin_plot = [None, vmin, vmin]
    vmax_plot = [None, vmax, vmax]
    cax = [cax1, None, cax2]
    extend = ['neither', 'both', 'both']
    for n in range(len(sim_numbers)):
        ax = plt.subplot(gs[0,n])
        img = latlon_plot(data_plot[n], grid, ax=ax, make_cbar=False, vmin=vmin_plot[n], vmax=vmax_plot[n], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=titles[n])
        if n != 0:
            # Remove axis labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if cax[n] is not None:
            plt.colorbar(img, cax=cax[n], extend=extend[n])            
    finished_plot(fig, fig_name=fig_dir+'freezing_months.png')


# Plot timeseries of changes in atmospheric temperature and wind speed, averaged over several different regions. Smooth with the given radius.
def plot_atm_timeseries (sim_key, smooth=0, base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    if sim_key in ['abO', '1pO']:
        ctrl_key = 'ctO'
    elif sim_key in ['abIO', '1pIO']:
        ctrl_key = 'ctIO'
    else:
        print 'Error (plot_atm_timeseries): invalid sim_key ' + sim_key
        sys.exit()
    sim_numbers = [sim_keys.index(key) for key in [ctrl_key, sim_key]]
    file_paths = [base_dir + sim_dirs[n] + timeseries_file_2 for n in sim_numbers]
    var_names = ['atemp_avg', 'wind_avg']
    var_titles = ['Average surface air temperature', 'Average wind speed']
    var_units = [deg_string+'C', 'm/s']
    region_names = ['sws_shelf', 'ronne_depression']
    region_titles = ['Continental shelf', 'Ronne Depression']
    colours = ['green', 'blue']
    title_tail = '\n'+sim_names[sim_numbers[1]]+' minus average of '+sim_names[sim_numbers[0]]

    time = netcdf_time(file_paths[1], monthly=False)
    for v in range(len(var_names)):
        var = var_names[v]
        data = []
        for region in region_names:
            full_var = region + '_' + var
            # Get average over control simulation
            base_val = read_netcdf(file_paths[0], full_var, time_average=True)
            # Now read the transient simulation and subtract the baseline value
            data_diff = read_netcdf(file_paths[1], full_var) - base_val
            # Smooth
            data_smoothed, time_trimmed = moving_average(data_diff, smooth, time=time)
            if var == var_names[0] and region == region_names[0]:
                # Replace the time array only once
                time = time_trimmed
            data.append(data_smoothed)
        title = var_titles[v]
        if smooth > 0:
            title += ' ('+str(2*smooth+1)+'-year smoothed)'
        title += title_tail
        timeseries_multi_plot(time, data, region_titles, colours, title=title, units=var_units[v], fig_name=fig_dir+'timeseries_'+var[:var.index('_avg')]+'_'+sim_key+'.png')


# Plot maps of the trends in different atmospheric forcing variables over the transient simulations.
def plot_atm_trend_maps (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    sim_numbers = [3, 5]
    directories = [sim_dirs[n] for n in sim_numbers]
    var_names = ['atemp', 'windspeed', 'pminuse']
    var_titles = ['surface air temperature', 'wind speed', 'precipitation minus evaporation']
    var_units = [deg_string+'C', 'm/s', 'm/s']
    ctype = ['basic', 'plusminus', 'basic']
    [xmin, xmax, ymin, ymax] = [-70, -24, -79, -72]

    # Grid, just for open ocean mask
    grid = Grid(base_dir+grid_path)

    # Loop over variables
    for v in range(len(var_names)):
        # Read all the data
        data_plot = []
        for n in range(len(directories)):
            print 'Processing ' + sim_names[sim_numbers[n]]
            file_paths = segment_file_paths(directories[n])
            num_years = len(file_paths)
            data = np.empty([num_years, grid.ny, grid.nx])
            for t in range(num_years):
                print '...year ' + str(t+1) + ' of ' + str(num_years)
                if var_names[v] == 'atemp':
                    data[t,:] = read_netcdf(file_paths[t], 'EXFatemp', time_average=True) - temp_C2K
                elif var_names[v] == 'windspeed':
                    # Calculate speed at each month and then time-average at the end
                    u = read_netcdf(file_paths[t], 'UVEL')
                    v = read_netcdf(file_paths[t], 'VVEL')
                    speed = np.sqrt(u**2 + v**2)
                    data[t,:] = np.mean(speed, axis=0)
                elif var_names[v] == 'pminuse':
                    data[t,:] = read_netcdf(file_paths[t], 'EXFpreci', time_average=True) - read_netcdf(file_paths[t], 'EXFevap', time_average=True)
            print 'Calculating trend'
            time = np.arange(num_years)
            trend = np.empty([grid.ny, grid.nx])
            for j in range(grid.ny):
                for i in range(grid.nx):
                    trend[j,i] = stats.linregress(time, data[:,j,i])[0]*num_years
            data_plot.append(mask_land_ice(trend, grid))

        # Get bounds on the trend
        vmin = np.amax(data_plot[0])
        vmax = np.amin(data_plot[0])
        for n in range(len(directories)):
            vmin_tmp, vmax_tmp = var_min_max(data_plot[n], grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            vmin = min(vmin, vmin_tmp)
            vmax = max(vmax, vmax_tmp)

        # Make the plot
        fig, gs, cax = set_panels('1x2C1')
        for n in range(len(directories)):
            ax = plt.subplot(gs[0,n])
            img = latlon_plot(data_plot[n], grid, ax=ax, make_cbar=False, ctype=ctype[v], vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=sim_names[sim_numbers[n]])
            if n != 0:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        plt.colorbar(img, cax=cax, orientation='horizontal')
        plt.suptitle('Trend in '+var_titles[v]+' ('+var_units[v]+' per 150 years)', fontsize=24)
        finished_plot(fig, fig_name=fig_dir+var_names[v]+'_trend.png')


# Deep dive into wind anomalies before and after correction.
def plot_wind_changes (sim_key, var='windspeed', base_dir='./', fig_dir='./', forcing_dir='/work/n02/n02/shared/baspog/MITgcm/UKESM/'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    forcing_dir = real_dir(forcing_dir)
    [xmin, xmax, ymin, ymax] = [-70, -24, -79, -72]
    model_grid = Grid(base_dir+grid_path)
    ukesm_grid = UKESMGrid()
    ctrl_year_start = 3050
    ctrl_year_end = 3059
    trans_year_start = 1990
    trans_year_end = 1999
    ctype = ['basic', 'basic', 'plusminus', 'plusminus']

    ctrl_key = 'ctO'
    ctrl_name = 'piControl'
    if sim_key in ['abO', 'abIO']:
        sim_name = 'abrupt-4xCO2'
    elif sim_key in ['1pO', '1pIO']:
        sim_name = '1pctCO2'
    else:
        print 'Error (plot_wind_changes): invalid sim_key ' + sim_key
        sys.exit()

    ukesm_lon, ukesm_lat = ukesm_grid.get_lon_lat(dim=1)
    ukesm_lon = fix_lon_range(ukesm_lon)
    i_split = np.nonzero(ukesm_lon < 0)[0][0]
    ukesm_lon = split_longitude(ukesm_lon, i_split)

    # First read uncorrected data, straight from the forcing fields
    data_uncorr = []
    for name, year_start, year_end in zip([ctrl_name, sim_name], [ctrl_year_start, trans_year_start], [ctrl_year_end, trans_year_end]):
        data = None
        for year in range(year_start, year_end+1):
            u = read_binary(forcing_dir+name+'/'+name+'_uas_'+str(year), [ukesm_grid.nx, ukesm_grid.ny], 'xyt')
            v = read_binary(forcing_dir+name+'/'+name+'_vas_'+str(year), [ukesm_grid.nx, ukesm_grid.ny_v], 'xyt')
            # Average in 30-day blocks to match model output
            u = np.mean(np.reshape(u, (30, u.shape[0]/30, u.shape[1], u.shape[2]), order='F'), axis=0)
            v = np.mean(np.reshape(v, (30, v.shape[0]/30, v.shape[1], v.shape[2]), order='F'), axis=0)
            # Interpolate to tracer grid
            u_t = np.empty(u.shape)
            u_t[:,:-1,:] = 0.5*(u[:,:-1,:] + u[:,1:,:])
            u_t[:,-1,:] = 0.5*(u[:,-1,:] + u[:,0,:])
            v_t = 0.5*(v[:,:-1,:] + v[:,1:,:])            
            if var == 'windspeed':
                data_tmp = np.sqrt(u_t**2 + v_t**2)
            elif var == 'uwind':
                data_tmp = u_t
            elif var == 'vwind':
                data_tmp = v_t
            else:
                print 'Error (plot_wind_changes): invalid variable ' + var
                sys.exit()
            data_tmp = np.mean(data_tmp, axis=0)
            if data is None:
                data = data_tmp
            else:
                data += data_tmp
        # Divide by number of years
        data /= (year_end-year_start+1)
        # Interpolate to MITgcm grid
        data = split_longitude(data, i_split)
        data = interp_reg_xy(ukesm_lon, ukesm_lat, data, model_grid.lon_1d, model_grid.lat_1d)
        # Mask land and ice shelf
        data = mask_land_ice(data, model_grid)
        data_uncorr.append(data)
    # Now get the anomaly and percent anomaly
    data_uncorr.append(data_uncorr[1]-data_uncorr[0])
    data_uncorr.append(data_uncorr[2]/data_uncorr[0]*100)

    # Repeat for the corrected data, in model output
    data_corr = []
    for key, year_start, year_end in zip([ctrl_key, sim_key], [ctrl_year_start, trans_year_start], [ctrl_year_end, trans_year_end]):
        data = None
        for year in range(year_start, year_end+1):
            file_path = base_dir+'WSFRIS_'+key+'/output/'+str(year)+'01/MITgcm/output.nc'
            print 'Reading ' + file_path
            u = read_netcdf(file_path, 'EXFuwind')
            v = read_netcdf(file_path, 'EXFvwind')
            if var == 'windspeed':
                var_title = 'Wind speed (m/s)'
                data_tmp = np.sqrt(u**2 + v**2)
            elif var == 'uwind':
                var_title = 'Zonal wind (m/s)'
                data_tmp = u
            elif var == 'vwind':
                var_title = 'Meridional wind (m/s)'
                data_tmp = v
            data_tmp = np.mean(data_tmp, axis=0)
            if data is None:
                data = data_tmp
            else:
                data += data_tmp
        data /= (year_end-year_start+1)
        data = mask_land_ice(data, model_grid)
        data_corr.append(data)
    data_corr.append(data_corr[1]-data_corr[0])
    data_corr.append(data_corr[2]/data_corr[0]*100)

    # Get the min and max values
    vmin = []
    vmax = []
    for n in range(len(data_uncorr)):
        vmin1, vmax1 = var_min_max(data_uncorr[n], model_grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        vmin2, vmax2 = var_min_max(data_corr[n], model_grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        vmin.append(min(vmin1, vmin2))
        vmax.append(max(vmax1, vmax2))
    # First and second should be the same
    vmin0 = min(vmin[0], vmin[1])
    vmax0 = max(vmax[0], vmax[1])
    vmin[0] = vmin0
    vmin[1] = vmin0
    vmax[0] = vmax0
    vmax[1] = vmax0

    # Make the plot
    fig, gs, cax1, cax2, cax3 = set_panels('2x4C3')
    titles = [ctrl_name, sim_name, 'Anomaly', 'Percent anomaly']
    cax = [None, cax1, cax2, cax3]
    for n in range(len(data_uncorr)):
        # Plot uncorrected
        ax = plt.subplot(gs[0,n])
        img = latlon_plot(data_uncorr[n], model_grid, ax=ax, make_cbar=False, ctype=ctype[n], vmin=vmin[n], vmax=vmax[n], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=titles[n])
        if n == 0:
            plt.text(0.07, 0.7, 'uncorrected', fontsize=18, ha='center', va='center', transform=fig.transFigure)
            plt.text(0.07, 0.3, 'corrected', fontsize=18, ha='center', va='center', transform=fig.transFigure)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.colorbar(img, cax=cax[n], orientation='horizontal')
        # Plot corrected
        ax = plt.subplot(gs[1,n])
        latlon_plot(data_corr[n], model_grid, ax=ax, make_cbar=False, ctype=ctype[n], vmin=vmin[n], vmax=vmax[n], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.suptitle(var_title+' (last 10 years)', fontsize=24)
    finished_plot(fig, fig_name=fig_dir+var+'_anomalies_'+sim_key+'.png')
                    
            
        
    
 
