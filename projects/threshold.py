##################################################################
# Weddell Sea threshold paper
##################################################################

import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid, UKESMGrid, ERA5Grid
from ..file_io import read_binary, find_cmip6_files, NCfile, NCfile_basiclatlon, read_netcdf, write_binary
from ..interpolation import interp_reg_xy, smooth_xy, interp_grid
from ..utils import fix_lon_range, split_longitude, real_dir, dist_btw_points, mask_land_ice, polar_stereo
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.latlon import shade_land_ice, overlay_vectors
from ..plot_utils.labels import latlon_axes
from ..plot_latlon import latlon_plot

# Functions to build a katabatic wind correction file between UKESM and ERA5, following the method of Mathiot et al 2010.

# Read the wind forcing output from either UKESM's historical simulation (option='UKESM') or ERA5 (option='ERA5') over the period 1979-2014, and time-average. Interpolate to the MITgcm grid (if interpolate=True) and save the output to a NetCDF file.
# If var='atemp' instead of 'wind' (default), do the same for surface air temperature.
def process_forcing (option, mit_grid_dir, out_file, source_dir=None, var='wind', interpolate=True):

    start_year = 1979
    end_year = 2014
    if var == 'wind':
        var_names = ['uwind', 'vwind']
        units = 'm/s'
    elif var == 'atemp':
        var_names = ['atemp']
        units = 'degC'
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
        elif var == 'atemp':
            var_names_in = ['tas']
            gtype = ['t']
        days_per_year = 12*30
    elif option == 'ERA5':
        if source_dir is None:
            source_dir = '/work/n02/n02/shared/baspog/MITgcm/reanalysis/ERA5/'
        file_head = 'ERA5_'
        gtype = ['t', 't']
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
                data = data[:,:-1]
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
            ncfile.add_variable(var_names[n], data_interp, 'xy', units=units)
        else:
            ncfile.add_variable(var_names[n], data, units=units)

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


# Build a katabatic wind correction, with scale factors for each wind component. Save to binary files for MITgcm to read, and also plot the results.
# New (22 Nov 2019): Correct on band around coast, rather than just coastal points extended outwards.
def katabatic_correction (grid_dir, ukesm_file, era5_file, out_file_head, scale_cap=3, prec=64):

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
    
    print 'Calculating scale factors'
    scale = []
    for n in range(2):
        # Read data
        ukesm_wind = read_netcdf(ukesm_file, var_names[n])
        era5_wind = read_netcdf(era5_file, var_names[n])
        # Take minimum of the ratio of ERA5 to UKESM wind mangitude (in this coordinate), and the scale cap
        scale_tmp = np.minimum(np.abs(era5_wind/ukesm_wind), scale_cap)
        # Smooth and mask the land and ice shelf
        scale.append(mask_land_ice(smooth_xy(scale_tmp, sigma=sigma), grid))

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
    scale_tapered = []
    for n in range(2):
        scale_tapered.append((min_dist < scale_dist)*(scale[n] - 1)*np.cos(np.pi/2*min_dist/scale_dist) + 1)
    # Combine (just for plotting purposes), and scale by sqrt(2) so all 1s map to 1
    scale_combined = np.sqrt((scale_tapered[0]**2 + scale_tapered[1]**2)/2)

    print 'Plotting'
    data_to_plot = [min_dist, scale_tapered[0], scale_tapered[1], scale_combined]
    titles = ['Distance to coast (km)', 'u-scaling factor', 'v-scaling factor', 'Combined scaling factor']
    ctype = ['basic', 'ratio', 'ratio', 'ratio']
    fig_names = ['min_dist.png', 'uscale.png', 'vscale.png', 'scale.png']
    for i in range(len(data_to_plot)):
        for fig_name in [None, fig_names[i]]:
            latlon_plot(data_to_plot[i], grid, ctype=ctype[i], include_shelf=False, title=titles[i], figsize=(10,6), fig_name=fig_name)

    print 'Writing to file'
    for n in range(2):
        scale_data = scale_tapered[n]
        # Replace mask with zeros
        mask = scale_data.mask
        scale_data = scale_data.data
        scale_data[mask] = 0
        write_binary(scale_data, out_file_head+'_'+var_names[n], prec=prec)


# Make figures of the winds over Antarctica (polar stereographic projection) in ERA5 and UKESM, with vectors and streamlines.
# First create era5_file and ukesm_file in process_forcing, with 
def plot_continent_wind (era5_file, ukesm_file, 
