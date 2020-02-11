##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime

from ..grid import Grid, UKESMGrid, ERA5Grid
from ..file_io import read_binary, find_cmip6_files, NCfile, NCfile_basiclatlon, read_netcdf, write_binary, read_netcdf_list, netcdf_time
from ..interpolation import interp_reg_xy, smooth_xy, interp_grid
from ..utils import fix_lon_range, split_longitude, real_dir, dist_btw_points, mask_land_ice, polar_stereo, wrap_periodic
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.latlon import shade_land_ice, overlay_vectors
from ..plot_utils.labels import latlon_axes
from ..plot_latlon import latlon_plot
from ..plot_1d import timeseries_multi_plot
from ..constants import temp_C2K, rho_fw, deg2rad
from ..postprocess import segment_file_paths
from ..timeseries import set_parameters

# Functions to build a katabatic wind correction file between UKESM and ERA5, following the method of Mathiot et al 2010.

# Read the wind forcing output from either UKESM's historical simulation (option='UKESM') or ERA5 (option='ERA5') over the period 1979-2014, and time-average. Interpolate to the MITgcm grid (if interpolate=True) and save the output to a NetCDF file.
# If var='thermo' instead of 'wind' (default), do the same for surface air temperature, specific humidity, and precipitation.
def process_forcing (option, mit_grid_dir, out_file, source_dir=None, var='wind', interpolate=True):

    start_year = 1979
    end_year = 2014
    if var == 'wind':
        var_names = ['uwind', 'vwind']
        units = ['m/s', 'm/s']
    elif var == 'thermo':
        var_names = ['atemp', 'aqh', 'precip']
        units = ['degC', '1', 'm/s']
    else:
        print 'Error (process_forcing): invalid var ' + var
        sys.exit()
    if option == 'UKESM':
        if source_dir is None:
            source_dir = '/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/'
        expt = 'historical'
        ensemble_member = 'r1i1p1f2'
        if var == 'wind':
            var_names_in = ['uas', 'vas']
            gtype = ['u', 'v']
        elif var == 'thermo':
            var_names_in = ['tas', 'huss', 'pr']
            gtype = ['t', 't', 't']
        days_per_year = 12*30
    elif option == 'ERA5':
        if source_dir is None:
            source_dir = '/work/n02/n02/shared/baspog/MITgcm/reanalysis/ERA5/'
        file_head = 'ERA5_'
        gtype = ['t', 't', 't']
    else:
        print 'Error (process_forcing); invalid option ' + option
        sys.exit()

    mit_grid_dir = real_dir(mit_grid_dir)
    source_dir = real_dir(source_dir)

    print 'Building grids'
    if option == 'UKESM':
        forcing_grid = UKESMGrid()
    elif option == 'ERA5':
        forcing_grid = ERA5Grid()
    if interpolate:
        mit_grid = Grid(mit_grid_dir)

    # Open NetCDF file
    if interpolate:
        ncfile = NCfile(out_file, mit_grid, 'xy')

    # Loop over variables
    for n in range(len(var_names)):
        print 'Processing variable ' + var_names[n]
        # Read the data, time-integrating as we go
        data = None
        num_time = 0
        
        if option == 'UKESM':
            in_files, start_years, end_years = find_cmip6_files(source_dir, expt, ensemble_member, var_names_in[n], 'day')
            # Loop over each file
            for t in range(len(in_files)):
                file_path = in_files[t]
                print 'Processing ' + file_path
                print 'Covers years ' + str(start_years[t]) + ' to ' + str(end_years[t])
                # Loop over years
                t_start = 0  # Time index in file
                t_end = t_start+days_per_year
                for year in range(start_years[t], end_years[t]+1):
                    if year >= start_year and year <= end_year:
                        print 'Processing ' + str(year)
                        # Read data
                        print 'Reading ' + str(year) + ' from indices ' + str(t_start) + '-' + str(t_end)
                        data_tmp = read_netcdf(file_path, var_names_in[n], t_start=t_start, t_end=t_end)
                        if data is None:
                            data = np.sum(data_tmp, axis=0)
                        else:
                            data += np.sum(data_tmp, axis=0)
                        num_time += days_per_year
                    # Update time range for next time
                    t_start = t_end
                    t_end = t_start + days_per_year
            if var_names[n] == 'atemp':
                # Convert from K to C
                data -= temp_C2K
            elif var_names[n] == 'precip':
                # Convert from kg/m^2/s to m/s
                data /= rho_fw

        elif option == 'ERA5':
            # Loop over years
            for year in range(start_year, end_year+1):
                file_path = source_dir + file_head + var_names[n] + '_' + str(year)
                data_tmp = read_binary(file_path, [forcing_grid.nx, forcing_grid.ny], 'xyt')
                if data is None:
                    data = np.sum(data_tmp, axis=0)
                else:
                    data += np.sum(data_tmp, axis=0)
                num_time += data_tmp.shape[0]

        # Now convert from time-integral to time-average
        data /= num_time

        if not interpolate and option == 'UKESM':
            # Need to interpolate to the tracer grid still
            forcing_lon, forcing_lat = forcing_grid.get_lon_lat(gtype='t', dim=1)
            data = interp_grid(data, forcing_grid, gtype[n], 't', periodic=True, mask=False)
            if gtype[n] == 'v':
                # Delete the northernmost row (which was extended)
                data = data[:-1,:]
        else:
            forcing_lon, forcing_lat = forcing_grid.get_lon_lat(gtype=gtype[n], dim=1)
        # Get longitude in the range -180 to 180, then split and rearrange so it's monotonically increasing        
        forcing_lon = fix_lon_range(forcing_lon)
        i_split = np.nonzero(forcing_lon < 0)[0][0]
        forcing_lon = split_longitude(forcing_lon, i_split)
        data = split_longitude(data, i_split)
        if interpolate:
            # Now interpolate to MITgcm tracer grid        
            mit_lon, mit_lat = mit_grid.get_lon_lat(gtype='t', dim=1)
            print 'Interpolating'
            data_interp = interp_reg_xy(forcing_lon, forcing_lat, data, mit_lon, mit_lat)            
        elif n==0:
            # Just set up new file
            ncfile = NCfile_basiclatlon(out_file, forcing_lon, forcing_lat)
        print 'Saving to ' + out_file
        if interpolate:
            ncfile.add_variable(var_names[n], data_interp, 'xy', units=units[n])
        else:
            ncfile.add_variable(var_names[n], data, units=units[n])

    ncfile.close()


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


# Build a katabatic wind correction.
# New (22 Nov 2019): Correct on band around coast, rather than just coastal points extended outwards.
# New (29 Nov 2019): Correct in polar coordinates, with a scale factor for the magnitude and a rotation for the angle.
def katabatic_correction (grid_dir, ukesm_file, era5_file, out_file_scale, out_file_rotate, scale_cap=3, prec=64):

    var_names = ['uwind', 'vwind']
    scale_dist = 150.
    # Radius for smoothing
    sigma = 2

    print 'Building grid'
    grid = Grid(grid_dir)
    print 'Selecting coastal points'
    coast_mask = grid.get_coast_mask(ignore_iceberg=True)
    lon_coast = grid.lon_2d[coast_mask].ravel()
    lat_coast = grid.lat_2d[coast_mask].ravel()

    print 'Calculating winds in polar coordinates'
    magnitudes = []
    angles = []
    for fname in [ukesm_file, era5_file]:
        u = read_netcdf(fname, var_names[0])
        v = read_netcdf(fname, var_names[1])
        magnitudes.append(np.sqrt(u**2 + v**2))
        angle = np.arctan2(v,u)
        angles.append(angle)

    print 'Calculating corrections'
    # Take minimum of the ratio of ERA5 to UKESM wind magnitude, and the scale cap
    scale = np.minimum(magnitudes[1]/magnitudes[0], scale_cap)
    # Smooth and mask the land and ice shelf
    scale = mask_land_ice(smooth_xy(scale, sigma=sigma), grid)
    # Take difference in angles
    rotate = angles[1] - angles[0]
    # Take mod 2pi when necessary
    index = rotate < -np.pi
    rotate[index] += 2*np.pi
    index = rotate > np.pi
    rotate[index] -= 2*np.pi
    # Smoothing would be weird with the periodic angle, so just mask
    rotate = mask_land_ice(rotate, grid)

    print 'Calculating distance from the coast'
    min_dist = None
    # Loop over all the coastal points
    for i in range(lon_coast.size):
        # Calculate distance of every point in the model grid to this specific coastal point, in km
        dist_to_pt = dist_btw_points([lon_coast[i], lat_coast[i]], [grid.lon_2d, grid.lat_2d])*1e-3
        if min_dist is None:
            # Initialise the array
            min_dist = dist_to_pt
        else:
            # Figure out which cells have this coastal point as the closest one yet, and update the array
            index = dist_to_pt < min_dist
            min_dist[index] = dist_to_pt[index]

    print 'Tapering function offshore'
    # Cosine function moving from scaling factor to 1 over distance of 300 km offshore
    scale_tapered = scale #(min_dist < scale_dist)*(scale - 1)*np.cos(np.pi/2*min_dist/scale_dist) + 1
    # For the rotation, move from scaling factor to 0
    rotate_tapered = rotate #(min_dist < scale_dist)*rotate*np.cos(np.pi/2*min_dist/scale_dist)    

    print 'Plotting'
    data_to_plot = [min_dist, scale_tapered, rotate_tapered]
    titles = ['Distance to coast (km)', 'Scaling factor', 'Rotation factor']
    ctype = ['basic', 'ratio', 'plusminus']
    fig_names = ['min_dist.png', 'scale.png', 'rotate.png']
    for i in range(len(data_to_plot)):
        for fig_name in [None, fig_names[i]]:
            latlon_plot(data_to_plot[i], grid, ctype=ctype[i], include_shelf=False, title=titles[i], figsize=(10,6), fig_name=fig_name)

    print 'Writing to file'
    fields = [scale_tapered, rotate_tapered]
    out_files = [out_file_scale, out_file_rotate]
    for n in range(len(fields)):
        # Replace mask with zeros
        mask = fields[n].mask
        data = fields[n].data
        data[mask] = 0
        write_binary(data, out_files[n], prec=prec)


# Make figures of the winds over Antarctica (polar stereographic projection) in ERA5 and UKESM, with vectors and streamlines.
# First create era5_file and ukesm_file in process_forcing, with interpolate=False.
def plot_continent_wind (era5_file, ukesm_file, fig_name=None):

    # Read files
    var_list = ['lon', 'lat', 'uwind', 'vwind']
    era5_lon, era5_lat, era5_uwind, era5_vwind = read_netcdf_list(era5_file, var_list)
    ukesm_lon, ukesm_lat, ukesm_uwind, ukesm_vwind = read_netcdf_list(ukesm_file, var_list)

    # Interpolate UKESM to ERA5 grid
    # First wrap the longitude axis around so there are no missing values in the gap
    ukesm_lon = wrap_periodic(ukesm_lon, is_lon=True)
    for data in [ukesm_uwind, ukesm_vwind]:
        data = wrap_periodic(data)
        data = interp_reg_xy(ukesm_lon, ukesm_lat, data, era5_lon, era5_lat)

    # Get stereographic coordinates
    x, y = polar_stereo(era5_lon, era5_lat)

    # Plot outline of Antarctica? How? Need mask file from ERA5?
    # Vector plots - how many points? Every second point?
    # Also vector anomaly.
    # Streamplots - every fifth point?


# Plot a timeseries of the number of cells grounded and ungrounded, and the maximum thinning and thickening, in a coupled run.
def plot_geometry_timeseries (output_dir='./', fig_name_1=None, fig_name_2=None):

    file_paths = segment_file_paths(output_dir)

    # Get the grid from the first one
    old_grid = Grid(file_paths[0])

    # Set up timeseries arrays
    time = []
    ground = []
    unground = []
    thin = []
    thick = []

    # Loop over the rest of the timeseries
    for file_path in file_paths[1:]:
        print 'Processing ' + file_path
        # Save time index from the beginning of the run
        time.append(netcdf_time(file_path)[0])
        # Calculate geometry changes
        new_grid = Grid(file_path)
        ground.append(np.count_nonzero((old_grid.bathy!=0)*(new_grid.bathy==0)))
        unground.append(np.count_nonzero((old_grid.bathy==0)*(new_grid.bathy!=0)))
        ddraft = np.ma.masked_where(old_grid.draft==0, np.ma.masked_where(new_grid.draft==0, new_grid.draft-old_grid.draft))
        thin.append(np.amin(ddraft))
        thick.append(np.amax(ddraft))
        old_grid = new_grid
    time = np.array(time)
    ground = np.array(ground)
    unground = np.array(unground)
    thin = -1*np.array(thin)
    thick = np.array(thick)

    # Plot
    timeseries_multi_plot(time, [ground, unground], ['# Grounded', '# Ungrounded'], ['blue', 'red'], title='Changes in ocean cells', fig_name=fig_name_1)
    timeseries_multi_plot(time, [thin, thick], ['Maximum thinning', 'Maximum thickening'], ['red', 'blue'], title='Changes in ice shelf draft', fig_name=fig_name_2)


# Make timeseries plots of the 3 simulations (piControl, abrupt-4xCO2, 1pctCO2) for FRIS mass loss, temperature, and salinity.
def threshold_timeseries (ctrl_dir, abrupt_dir, onepct_dir, timeseries_file='timeseries.nc', fig_dir='./'):

    var_names = ['fris_massloss', 'fris_temp', 'fris_salt']
    labels = ['piControl', 'abrupt-4xCO2', '1pctCO2']
    colours = ['black', 'blue', 'green']
    file_paths = [real_dir(dir_path)+timeseries_file for dir_path in [ctrl_dir, abrupt_dir, onepct_dir]]
    num_sim = len(file_paths)
    
    # Read time axes
    times = [netcdf_time(file_path, monthly=False) for file_path in file_paths]
    # Need to shift the years in piControl so they match the other simulations
    ctrl_year0 = times[0][0].year
    year0 = times[1][0].year
    dyear = ctrl_year0 - year0
    ctrl_time = [datetime.date(t.year-dyear, t.month, t.day) for t in times[0]]
    times[0] = ctrl_time

    for var in var_names:
        print 'Processing ' + var
        datas = []
        for n in range(num_sim):
            if var == 'fris_massloss':
                datas.append(read_netcdf(file_paths[n], 'fris_total_melt')+read_netcdf(file_paths[n], 'fris_total_freeze'))
            else:
                datas.append(read_netcdf(file_paths[n], var))
        if var == 'fris_massloss':
            title = 'FRIS net basal mass loss'
            units = 'Gt/y'
        else:
            title, units = set_parameters(var)[2:4]
        timeseries_multi_plot(times, datas, labels, colours, title=title, units=units, fig_name=real_dir(fig_dir)+'timeseries_'+var+'_compare.png')
    
            
