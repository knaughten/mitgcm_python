##################################################################
# PACE paper from JSPS Amundsen Sea simulations
##################################################################

import numpy as np
from itertools import compress, cycle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress, ttest_1samp, pearsonr
from scipy.io import loadmat
import os

from ..grid import ERA5Grid, PACEGrid, Grid, dA_from_latlon, pierre_obs_grid
from ..file_io import read_binary, write_binary, read_netcdf, netcdf_time, read_title_units, read_annual_average, NCfile
from ..utils import real_dir, daily_to_monthly, fix_lon_range, split_longitude, mask_land_ice, moving_average, index_year_start, index_year_end, index_period, mask_2d_to_3d, days_per_month, add_time_dim, z_to_xyz, select_bottom, convert_ismr, mask_except_ice, xy_to_xyz, apply_mask, var_min_max, mask_3d, average_12_months
from ..plot_utils.colours import set_colours, choose_n_colours
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.labels import reduce_cbar_labels, round_to_decimals
from ..plot_utils.latlon import shade_background
from ..plot_1d import default_colours, make_timeseries_plot_2sided, timeseries_multi_plot, make_timeseries_plot
from ..plot_latlon import latlon_plot
from ..plot_slices import slice_plot
from ..constants import sec_per_year, kg_per_Gt, dotson_melt_years, getz_melt_years, pig_melt_years, region_names, deg_string, sec_per_day, region_bounds, Cp_sw, rad2deg
from ..plot_misc import hovmoller_plot, ts_animation, ts_binning
from ..timeseries import calc_annual_averages, set_parameters
from ..postprocess import get_output_files, segment_file_paths
from ..diagnostics import adv_heat_wrt_freezing, potential_density
from ..calculus import time_derivative, time_integral
from ..interpolation import interp_reg_xy


# Global variables
file_head_era5 = 'ERA5_'
file_head_pace = 'PACE_ens'
file_tail = '_clim'
days_per_year = 365
months_per_year = 12
per_day = 4  # ERA5 records per day
num_ens = 20 # Number of PACE ensemble members
var_era5 = ['atemp', 'aqh', 'apressure', 'uwind', 'vwind', 'precip', 'swdown', 'lwdown']
var_pace = ['TREFHT', 'QBOT', 'PSL', 'UBOT', 'VBOT', 'PRECT', 'FSDS', 'FLDS']


# Calculate the climatologies (daily or monthly) of each forcing variable in ERA5 (interpolated to the PACE grid) and in each PACE ensemble member.
def calc_climatologies (era5_dir, pace_dir, out_dir, wind_speed=False):

    # Day of the year that's 29 Feb (0-indexed)
    leap_day = 31+28
    # Climatology over the years that both products have data (not counting the RCP8.5 extension)
    start_year = 1979
    end_year = 2005
    num_years = end_year-start_year+1

    if wind_speed:
        # Overwrite variable lists
        var_era5 = ['speed']
        var_pace = ['speed']
        wind_comp_era5 = ['uwind', 'vwind']
        wind_comp_pace = ['UBOT', 'VBOT']
    monthly = [var in ['FSDS', 'FLDS'] for var in var_pace]
    var_era5_monthly = list(compress(var_era5, monthly))
    var_era5_daily = list(compress(var_era5, np.invert(monthly)))
    var_pace_monthly = list(compress(var_pace, monthly))
    var_pace_daily = list(compress(var_pace, np.invert(monthly)))
    num_vars = len(var_era5)
    num_vars_monthly = len(var_era5_monthly)
    num_vars_daily = len(var_era5_daily)  
    
    era5_grid = ERA5Grid()
    pace_grid = PACEGrid()

    # Get right edges of PACE grid (for binning)
    pace_lon, pace_lat = pace_grid.get_lon_lat(dim=1)
    def right_edges (A):
        edges_mid = 0.5*(A[:-1] + A[1:])
        edges_end = 2*edges_mid[-1] - edges_mid[-2]
        return np.concatenate((edges_mid, [edges_end]), axis=0)
    pace_lon_bins = right_edges(pace_lon)
    pace_lat_bins = right_edges(pace_lat)
    # Figure out i and j indices for ERA5 to PACE binning
    i_bins = np.digitize(era5_grid.lon, pace_lon_bins)
    # Wrap the periodic boundary
    i_bins[i_bins==pace_grid.nx] = 0
    j_bins = np.digitize(era5_grid.lat, pace_lat_bins)

    # Inner function to make climatology of an ERA5 variable
    def era5_process_var (var_name_pace, var_name_era5, monthly):
        print 'Processing ' + var_name_pace
        if monthly:
            per_year = months_per_year
        else:
            per_year = days_per_year        
        # Accumulate data over each year
        data_accum = np.zeros([per_year, era5_grid.ny, era5_grid.nx])
        for year in range(start_year, end_year+1):
            if var_name_pace == 'speed':
                # Read wind components and calculate magnitude
                def read_comp_era5 (var_comp):
                    file_path = real_dir(era5_dir) + file_head_era5 + var_comp + '_' + str(year)
                    return read_binary(file_path, [era5_grid.nx, era5_grid.ny], 'xyt')
                data_u = read_comp_era5(wind_comp_era5[0])
                data_v = read_comp_era5(wind_comp_era5[1])
                data = np.sqrt(data_u**2 + data_v**2)
            else:
                file_path = real_dir(era5_dir) + file_head_era5 + var_name_era5 + '_' + str(year)
                data = read_binary(file_path, [era5_grid.nx, era5_grid.ny], 'xyt')
            if monthly:
                # Monthly averages
                data = daily_to_monthly(data, year=year, per_day=per_day)
            else:
                # Average over each day
                data = np.mean(np.reshape(data, (per_day, data.shape[0]/per_day, era5_grid.ny, era5_grid.nx), order='F'), axis=0)
                if data.shape[0] == days_per_year+1:
                    # Remove leap day
                    data = np.concatenate((data[:leap_day,:], data[leap_day+1:,:]), axis=0)
            data_accum += data
        # Convert from integral to average
        return data_accum/num_years

    # Loop over daily and monthly variables
    print 'Processing ERA5'
    era5_clim_daily = np.empty([num_vars_daily, days_per_year, era5_grid.ny, era5_grid.nx])
    for n in range(num_vars_daily):
        era5_clim_daily[n,:] = era5_process_var(var_pace_daily[n], var_era5_daily[n], False)
    if num_vars_monthly > 0:
        era5_clim_monthly = np.empty([num_vars_monthly, months_per_year, era5_grid.ny, era5_grid.nx])
        for n in range(num_vars_monthly):
            era5_clim_monthly[n,:] = era5_process_var(var_pace_monthly[n], var_era5_monthly[n], True)

    # Now do all the binning at once to save memory
    era5_clim_regrid_daily = np.zeros([num_vars_daily, days_per_year, pace_grid.ny, pace_grid.nx])
    era5_clim_regrid_monthly = np.zeros([num_vars_monthly, months_per_year, pace_grid.ny, pace_grid.nx])
    print 'Regridding from ERA5 to PACE grid'
    for j in range(pace_grid.ny):
        for i in range(pace_grid.nx):
            if np.any(i_bins==i) and np.any(j_bins==j):
                index = (i_bins==i)*(j_bins==j)
                era5_clim_regrid_daily[:,:,j,i] = np.mean(era5_clim_daily[:,:,index], axis=-1)
                if num_vars_monthly > 0:
                    era5_clim_regrid_monthly[:,:,j,i] = np.mean(era5_clim_monthly[:,:,index], axis=-1)
    # Write each variable to binary
    for n in range(num_vars_daily):   
        file_path = real_dir(out_dir) + file_head_era5 + var_pace_daily[n] + file_tail
        write_binary(era5_clim_regrid_daily[n,:], file_path)
    for n in range(num_vars_monthly):   
        file_path = real_dir(out_dir) + file_head_era5 + var_pace_monthly[n] + file_tail
        write_binary(era5_clim_regrid_monthly[n,:], file_path)

    print 'Processing PACE'
    for n in range(num_vars):
        print 'Processing ' + var_pace[n]
        if monthly[n]:
            per_year = months_per_year
        else:
            per_year = days_per_year
        for ens in range(1, num_ens+1):
            ens_str = str(ens).zfill(2)
            print 'Processing PACE ensemble member ' + ens_str
            # As before, but simpler because no leap days and no need to regrid 
            data_accum = np.zeros([per_year, pace_grid.ny, pace_grid.nx])
            for year in range(start_year, end_year+1):
                if var_pace[n] == 'speed':
                    def read_comp_pace (var_comp):
                        file_path = real_dir(pace_dir) + file_head_pace + ens_str + '_' + var_comp + '_' + str(year)
                        return read_binary(file_path, [pace_grid.nx, pace_grid.ny], 'xyt')
                    data_u = read_comp_pace(wind_comp_pace[0])
                    data_v = read_comp_pace(wind_comp_pace[1])
                    data = np.sqrt(data_u**2 + data_v**2)
                else:
                    file_path = real_dir(pace_dir) + file_head_pace + ens_str + '_' + var_pace[n] + '_' + str(year)
                    data = read_binary(file_path, [pace_grid.nx, pace_grid.ny], 'xyt')
                data_accum += data
            data_clim = data_accum/num_years
            file_path = real_dir(out_dir) + file_head_pace + ens_str + '_' + var_pace[n] + file_tail
            write_binary(data_clim, file_path)


# For the given variable, make two plots of the PACE bias with respect to ERA5:
# 1. A time-averaged, ensemble-averaged lat-lon bias plot
# 2. An area-averaged (over the Amundsen Sea region) timeseries of each ensemble member, the ensemble mean, ERA5, and the ensemble-mean bias
# Also print out the annual and monthly values of the bias in #2.
def plot_biases (var_name, clim_dir, monthly=False, fig_dir='./', ratio=False):

    # Latitude bounds on ERA5 data
    ylim_era5 = [-90, -30]
    # Bounds on box to average over for seasonal climatology
    [xmin, xmax, ymin, ymax] = [240, 260, -75, -72]
    if monthly:
        per_year = months_per_year
        time_label = 'month of year'
    else:
        per_year = days_per_year
        time_label = 'day of year'
    # 19 visually distinct colours (from http://phrogz.net/css/distinct-colors.html)
    ens_colours = [(0.224,0.902,0.584), (0.475,0.537,0.949), (0.451,0.114,0.384), (1.0,0.267,0.0), (1.0,0.933,0.0), (0.275,0.549,0.459), (0.667,0.639,0.851), (0.949,0.239,0.522), (0.549,0.145,0.0), (0.467,0.502,0.0), (0.0,1.0,0.933), (0.349,0.275,0.549), (0.302,0.224,0.255), (0.949,0.6,0.475), (0.6,0.8,0.2), (0.412,0.541,0.549), (0.38,0.0,0.949), (1.0,0.0,0.267), (0.2,0.078,0.0)]

    grid = PACEGrid()
    data = np.empty([num_ens, per_year, grid.ny, grid.nx])
    # Read data
    for ens in range(1, num_ens+1):
        ens_str = str(ens).zfill(2)
        if ens < 13:
            ens_index = ens-1
        else:
            ens_index = ens-2
        file_path = real_dir(clim_dir) + file_head_pace + ens_str + '_' + var_name + file_tail
        data[ens_index,:] = read_binary(file_path, [grid.nx, grid.ny], 'xyt')
    # Also need ERA5 data
    file_path = real_dir(clim_dir) + file_head_era5 + var_name + file_tail
    data_era5 = read_binary(file_path, [grid.nx, grid.ny], 'xyt')

    # Plot spatial map
    # Ensemble-mean and time-mean bias
    if ratio:
        bias_xy = np.mean(data, axis=(0,1))/np.mean(data_era5, axis=0)
    else:
        bias_xy = np.mean(data, axis=(0,1)) - np.mean(data_era5, axis=0)
    # Mask out everything north of 30S so it doesn't get counted in min/max
    bias_xy[grid.lat > ylim_era5[-1]] = 0
    fig, ax = plt.subplots(figsize=(10,6))
    if ratio:
        ctype = 'ratio'
        vmax = 1.5
    else:
        ctype = 'plusminus'
        vmax = None
    cmap, vmin, vmax = set_colours(bias_xy, ctype=ctype, vmax=vmax)
    img = ax.pcolormesh(grid.lon, grid.lat, bias_xy, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(ylim_era5)
    plt.colorbar(img)
    plt.title(var_name, fontsize=18)
    finished_plot(fig, fig_name=real_dir(fig_dir)+var_name+'_xy.png')

    # Plot seasonal climatology
    # Area-mean bias over Amundsen region, for each ensemble
    index = (grid.lon >= xmin)*(grid.lon <= xmax)*(grid.lat >= ymin)*(grid.lat <=ymax)
    data_et = np.mean(data[:,:,index], axis=-1)
    data_era5_t = np.mean(data_era5[:,index], axis=-1)
    # Get mean bias over ensemble members
    bias_t = np.mean(data_et, axis=0) - data_era5_t
    # And mean bias over time
    bias = np.mean(bias_t)
    fig, ax = plt.subplots(figsize=(8,6))
    time = np.arange(per_year)+1
    # One line for each ensemble member
    for i in range(num_ens-1):
        ax.plot(time, data_et[i,:], '-', color=ens_colours[i], alpha=0.25)
    # Blue line for ERA5 on top
    ax.plot(time, data_era5_t, '-', color='blue', label='ERA5')
    # Black line for ensemble mean on top
    ax.plot(time, np.mean(data_et,axis=0), '-', color='black', label='PACE mean')    
    # Dashed red line for ensemble-mean bias (over line at 0)
    ax.axhline(color='black')
    ax.plot(time, bias_t, '--', color='red', label='Mean bias')
    ax.grid(True)
    ax.set_xlim([1, per_year+1])
    plt.title(var_name+': mean bias '+str(bias), fontsize=18)
    plt.xlabel(time_label, fontsize=16)
    ax.legend()
    finished_plot(fig, fig_name=real_dir(fig_dir)+var_name+'_et.png')

    print 'Annual bias: ' + str(bias)
    if monthly:
        bias_t_monthly = bias_t
    else:
        bias_t_monthly = daily_to_monthly(bias_t)
    print 'Monthly biases: '
    for month in range(months_per_year):
        print str(bias_t_monthly[month])
        

# Call plot_biases for all variables.
def plot_all_biases (clim_dir, fig_dir='./'):

    for var in var_pace:
        monthly = var in ['FSDS', 'FLDS']
        plot_biases(var, clim_dir, monthly=monthly, fig_dir=fig_dir)


# Ground the Abbot Ice Shelf in the given topography files.
def ground_abbot (grid_path, bathy_file_in, draft_file_in, pload_file_in, bathy_file_out, draft_file_out, pload_file_out):

    grid = Grid(grid_path)
    bathy = read_binary(bathy_file_in, [grid.nx, grid.ny], 'xy', prec=64)
    draft = read_binary(draft_file_in, [grid.nx, grid.ny], 'xy', prec=64)
    pload = read_binary(pload_file_in, [grid.nx, grid.ny], 'xy', prec=64)
    ice_mask = draft != 0  # Differs slightly from ice_mask in finished Grid because of hFacMinDr requirements applied at run-time
    abbot_mask = grid.restrict_mask(draft!=0, 'abbot')
    bathy[abbot_mask] = 0
    draft[abbot_mask] = 0
    pload[abbot_mask] = 0
    write_binary(bathy, bathy_file_out, prec=64)
    write_binary(draft, draft_file_out, prec=64)
    write_binary(draft, pload_file_out, prec=64)


# Plot timeseries of 2-year running means of a bunch of variables for the given list of simulations.
def plot_timeseries_2y (sim_dir, sim_names, timeseries_types=None, plot_mean=True, first_in_mean=False, fig_dir='./', hindcast=True, colours=None, plot_anomaly=False, base_year_start=1920, base_year_end=1949, base_year_start_first=None, trim_before=True, ismr_percent=True):

    from ..plot_1d import read_plot_timeseries_ensemble

    if timeseries_types is None:
        timeseries_types = ['dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'getz_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting', 'eta_avg', 'hice_max', 'pine_island_bay_temp_below_500m', 'pine_island_bay_salt_below_500m', 'dotson_bay_temp_below_500m', 'dotson_bay_salt_below_500m', 'inner_amundsen_shelf_temp_below_500m', 'inner_amundsen_shelf_salt_below_500m', 'amundsen_shelf_break_uwind_avg', 'dotson_massloss', 'pig_massloss', 'getz_massloss']
    timeseries_file = 'timeseries.nc'
    timeseries_paths = [real_dir(d) + 'output/' + timeseries_file for d in sim_dir]
    smooth = 24
    if hindcast:
        year_start = 1920
        year_ticks = np.arange(1920, 2010+1, 10)
    else:
        year_start = 1979
        year_ticks = np.arange(1980, 2010+1, 10)
    if trim_before:
        vline = None
    else:
        vline = year_start

    for var_name in timeseries_types:
        percent = ismr_percent and (var_name.endswith('melting') or var_name.endswith('massloss'))
        read_plot_timeseries_ensemble(var_name, timeseries_paths, sim_names=sim_names, precomputed=True, colours=colours, smooth=smooth, vline=vline, time_use=None, alpha=(colours is None), plot_mean=plot_mean, first_in_mean=first_in_mean, plot_anomaly=plot_anomaly, base_year_start=base_year_start, base_year_end=base_year_end, trim_before=trim_before, base_year_start_first=base_year_start_first, percent=percent, year_ticks=year_ticks, fig_name=fig_dir+'timeseries_'+var_name+'_2y.png')


# Try with pig_melting, thwaites_melting, dotson_crosson_melting, pine_island_bay_temp_bottom, dotson_bay_temp_bottom
def wind_melt_coherence (sim_dirs, sim_names, var='pig_melting', fig_name=None):

    from scipy.signal import coherence

    if isinstance(sim_dirs, str):
        sim_dirs = [sim_dirs]
    if isinstance(sim_names, str):
        sim_names = [sim_names]
    num_sims = len(sim_dirs)

    freq = []
    cxy = []
    for n in range(num_sims):
        file_path = real_dir(sim_dirs[n]) + 'output/timeseries.nc'
        wind = read_netcdf(file_path, 'amundsen_shelf_break_uwind_avg')
        data = read_netcdf(file_path, var)
        f, c = coherence(wind, data, fs=12, detrend='linear')
        freq.append(f)
        cxy.append(c)
    fig, ax = plt.subplots(figsize=(11,6))
    for n in range(num_sims):
        ax.plot(freq[n], cxy[n], label=sim_names[n])
    ax.set_xlim([0,1])
    ax.grid(True)
    xtick_labels = [10, 5, 3, 2, 1]
    xticks = [1./tick for tick in xtick_labels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel('Period (years)', fontsize=14)
    ax.set_ylabel('Correlation', fontsize=14)
    ax.set_title('Coherence between '+var+' and shelf break winds', fontsize=16)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    finished_plot(fig, fig_name=fig_name)


def plot_psd (sim_dirs, sim_names, var='pine_island_bay_temp_bottom', colours=None, alpha=False, fig_name=None):

    from scipy.signal import welch
    year0 = 1955
    start_year = 1979
    num_spinup = (start_year-year0)*12

    if isinstance(sim_dirs, str):
        sim_dirs = [sim_dirs]
    if isinstance(sim_names, str):
        sim_names = [sim_names]
    num_sims = len(sim_dirs)
    if colours is None:
        colours = default_colours(num_sims)
    if alpha:
        alphas = [1] + [0.5 for n in range(num_sims-1)]
    else:
        alphas = [1 for n in range(num_sims)]

    freq = []
    pxx = []
    for n in range(num_sims):
        file_path = real_dir(sim_dirs[n]) + 'output/timeseries.nc'
        data = read_netcdf(file_path, var)[num_spinup:]
        f, p = welch(data, fs=12, detrend='linear')
        freq.append(f)
        pxx.append(p)
    fig, ax = plt.subplots(figsize=(11,6))
    for n in range(num_sims):
        ax.plot(freq[n], pxx[n], label=sim_names[n], color=colours[n], alpha=alphas[n])
    ax.set_xlim([0,1])
    ax.grid(True)
    xtick_labels = [20, 10, 5, 3, 2, 1]
    xticks = [1./tick for tick in xtick_labels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel('Period (years)', fontsize=14)
    ax.set_ylabel('Power spectral density', fontsize=14)
    ax.set_title(var, fontsize=16)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    finished_plot(fig, fig_name=fig_name)


# Compare existing addMass file (idealised distribution of iceberg flux) to the estimate of Merino et al.
def plot_addmass_merino (merino_file, addmass_file, grid_dir):

    # Read model grid and forcing file
    grid = Grid(grid_dir)
    addmass = read_binary(addmass_file, [grid.nx, grid.ny, grid.nz], 'xyz', prec=64)
    # Sum in vertical and mask
    addmass = mask_land_ice(np.sum(addmass, axis=0), grid)
    # Scale by area to get kg/m^2/s
    addmass = addmass/grid.dA

    mlon = read_netcdf(merino_file, 'longitude')[0,:]
    # Cut off the last two indices because the grid is wrapped
    mlon = mlon[:-2]
    # Need to do the split-rearrange thing like with the SOSE grid
    i_split = np.nonzero(mlon < 0)[0][0]
    mlon = split_longitude(mlon, i_split)
    mlat = read_netcdf(merino_file, 'latitude')[:,0]
    mflux = np.mean(read_netcdf(merino_file, 'Icb_flux'), axis=0)  # Annual mean from monthly climatology
    # Deal with the longitude nonsense
    mflux = split_longitude(mflux[:,:-2], i_split)   
    # Mask out zeros (will catch land as well as open ocean regions with zero flux)
    mflux = np.ma.masked_where(mflux==0, mflux)
    # Get more grid variables
    mdA, mlon_e, mlat_e = dA_from_latlon(mlon, mlat, periodic=True, return_edges=True)
    # Remesh the lat/lon edges
    mlon_e, mlat_e = np.meshgrid(mlon_e[0,:], mlat_e[:,0])    

    # Get boundaries of Merino data that align with model
    i_start = np.where(mlon >= np.amin(grid.lon_2d))[0][0]
    i_end = np.where(mlon > np.amax(grid.lon_2d))[0][0] - 1
    j_start = np.where(mlat >= np.amin(grid.lat_2d))[0][0]
    j_end = np.where(mlat > np.amax(grid.lat_2d))[0][0] - 1

    # Calculate total flux in region for both datasets (Gt/y)
    addmass_total = np.sum(addmass*grid.dA)*sec_per_year/kg_per_Gt
    merino_total = np.sum(mflux[j_start:j_end,i_start:i_end]*mdA[j_start:j_end,i_start:i_end])*sec_per_year/kg_per_Gt
    print 'Total Amundsen Sea melt flux in Gt/y:'
    print 'Existing setup: ' + str(addmass_total)
    print 'Merino et al: ' + str(merino_total)

    # Multiply by 1e4 for readability
    addmass *= 1e4
    mflux *= 1e4

    # Plot spatial distribution
    fig, gs, cax = set_panels('1x2C1', figsize=(15,6))
    vmin = 0
    vmax = 1 #max(np.amax(addmass), np.amax(mflux[j_start:j_end,i_start:i_end]))
    ymax = -70
    for n in range(2):
        ax = plt.subplot(gs[0,n])
        if n == 0:
            img = latlon_plot(addmass, grid, ax=ax, make_cbar=False, include_shelf=False, vmin=vmin, vmax=vmax, ymax=ymax, title='Idealised addmass ('+str(addmass_total)+' Gt/y)')
        elif n == 1:
            img = ax.pcolormesh(mlon_e, mlat_e, mflux, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_xlim([np.amin(grid.lon_2d), np.amax(grid.lon_2d)])
            ax.set_ylim([np.amin(grid.lat_2d), ymax])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title('Merino et al. ('+str(merino_total)+' Gt/y)', fontsize=18)
    plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
    plt.suptitle(r'Iceberg meltwater flux (10$^{-4}$ kg/m$^2$/s)', fontsize=24)
    finished_plot(fig, fig_name='addmass.png')


# Determine the most spiky, least spiky, and medium ensemble members (from the first 10) based on their PIG and Dotson melt rates.
def order_ensemble_std (base_dir='./'):

    # Run IDs for each member, in order
    run_id = ['018', '027', '028', '029', '030', '033', '034', '035', '036', '037']
    # ERA5 ID for comparison
    era5_id = ['031']
    run_names = ['PACE '+str(n+1) for n in range(10)] + ['ERA5']
    ts_file = 'timeseries.nc'
    smooth = 24
    base_dir = real_dir(base_dir)

    for var_name in ['pig_massloss', 'dotson_massloss']:
        print var_name
        std_list = []
        for rid in run_id+era5_id:
            data = read_netcdf(base_dir+'PAS_'+rid+'/output/'+ts_file, var_name)
            data_smooth = moving_average(data, smooth)
            std_list.append(np.std(data_smooth))
        sort_index = np.argsort(std_list)
        print 'Members, from flattest to spikiest:'
        for n in sort_index:
            print run_names[n]


# Make a massive plot of Hovmollers in all PACE ensemble members (pass in order), for a given location and variable.
def hovmoller_ensemble_tiles (loc, var, sim_dir, hovmoller_file='hovmoller.nc', grid='PAS_grid/', fig_name=None):

    year_start = 1920  # Trim the spinup before this
    year_end = 2013
    num_members = len(sim_dir)
    if num_members not in [10]:
        print 'Error (hovmoller_ensemble_tiles): need to write an entry in set_panels for ' + str(num_members) + ' members'
        sys.exit()
    sim_names = ['PACE '+str(n+1) for n in range(num_members)]
    file_paths = [real_dir(d)+'/output/'+hovmoller_file for d in sim_dir]
    smooth = 12
    grid = choose_grid(grid, None)
    
    if loc == 'amundsen_west_shelf_break':
        title = 'Shelf break'
        if var == 'temp':
            vmin = -0.5
            vmax = 1.4
        elif var == 'salt':
            vmin = 34.4
            vmax = 34.72
    elif loc == 'dotson_bay':
        title = 'Dotson front'
        if var == 'temp':
            vmin = -1.8
            vmax = 1
        elif var == 'salt':
            vmin = 34.2
            vmax = 34.65
    elif loc == 'pine_island_bay':
        title = 'Pine Island Bay'
        if var == 'temp':
            vmin = -1.5
            vmax = 1.5
        elif var == 'salt':
            vmin = 34.2
            vmax = 34.72
    else:
        print 'Error (hovmoller_ensemble_tiles): invalid location ' + loc
        sys.exit()            
    if var == 'temp':
        title += ' temperature ('+deg_string+'C)'
        contours = [0, 1]
    elif var == 'salt':
        title += ' salinity (psu)'
        contours = [34.5, 34.7]
    else:
        print 'Error (hovmoller_ensemble_tiles): invalid variable ' + var
        sys.exit()

    fig, gs, cax = set_panels(str(num_members)+'x1C1')
    for n in range(num_members):
        # Select the axes
        ax = plt.subplot(gs[n,0])
        # Read the data
        data = read_netcdf(file_paths[n], loc+'_'+var)
        time = netcdf_time(file_paths[n], monthly=False)
        # Trim everything before the spinup
        t_start = index_year_start(time, year_start)
        data = data[t_start:]
        time = time[t_start:]
        # Plot the Hovmoller
        img = hovmoller_plot(data, time, grid, smooth=smooth, ax=ax, make_cbar=False, vmin=vmin, vmax=vmax, contours=contours)
        # Set limits on time axes so they all line up
        ax.set_xlim([datetime.date(year_start, 1, 1), datetime.date(year_end+1, 1, 1)])
        ax.set_xticks([datetime.date(year, 1, 1) for year in np.arange(year_start, year_end, 10)])
        if n != 0:
            # Hide the depth labels
            ax.set_yticklabels([])
        if n != num_members-1:
            # Hide the time labels
            ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Ensemble name on the right
        plt.text(1.02, 0.5, sim_names[n], ha='left', va='center', transform=ax.transAxes, fontsize=12)
    # Main title
    plt.suptitle(title, fontsize=16, x=0.05, ha='left')
    # Colourbar on top right
    cbar = plt.colorbar(img, cax=cax, extend='both', orientation='horizontal')
    reduce_cbar_labels(cbar)
    finished_plot(fig, fig_name=fig_name)


# Call hovmoller_ensemble_tiles for all combinations of 3 locations and 2 variables.
def all_hovmoller_tiles (sim_dir, hovmoller_file='hovmoller.nc', grid='PAS_grid/', fig_dir='./'):

    grid = choose_grid(grid, None)
    fig_dir = real_dir(fig_dir)
    for loc in ['pine_island_bay', 'dotson_bay']: #, 'amundsen_west_shelf_break']:
        for var in ['temp', 'salt']:
            fig_name = fig_dir+'hov_ens_'+loc+'_'+var+'.png'
            hovmoller_ensemble_tiles(loc, var, sim_dir, hovmoller_file=hovmoller_file, grid=grid, fig_name=fig_name)


# Read a variable and calculate the trend, with a bunch of options. Returns the trend per decade, and a boolean indicating whether or not the trend is significant.
def read_calc_trends (var, file_path, option, percent=False, year_start=1920, year_end=1949, smooth=24, p0=0.05):

    data = read_netcdf(file_path, var)
    time = netcdf_time(file_path, monthly=False)
    if percent:
        # Express as percentage of mean over baseline
        t_start, t_end = index_period(time, year_start, year_end)
        data_mean = np.mean(data[t_start:t_end])
        data = data/data_mean*100
    # Trim everything before year_start
    t0 = index_year_start(time, year_start)
    time = time[t0:]
    data = data[t0:]
    # Get time in decades
    time_sec = np.array([(t-time[0]).total_seconds() for t in time])
    time = time_sec/(365*sec_per_day*10)
    if option == 'smooth':
        # 2-year running mean to filter out seasonal cycle
        data, time = moving_average(data, smooth, time=time)
    elif option == 'annual':
        # Annual average to filter out seasonal cycle
        # First trim to the nearest complete year
        new_size = len(time)/12*12
        time = time[:new_size]
        data = data[:new_size]
        time, data = calc_annual_averages(time, data)
    # Calculate trends per decade
    slope, intercept, r_value, p_value, std_err = linregress(time, data)
    sig = p_value < p0
    return slope, sig


# Helper function to set some common variables for the ensemble members
def setup_ensemble (sim_dir, timeseries_file='timeseries.nc'):

    num_members = len(sim_dir)
    sim_names = ['PACE '+str(n+1) for n in range(num_members)]
    file_paths = [real_dir(d)+'/output/'+timeseries_file for d in sim_dir]
    colours = default_colours(num_members)
    return num_members, sim_names, file_paths, colours


# Calculate the trends in the given variable, and their significance, for the given variable in each ensemble member.
def ensemble_trends (var, sim_dir, timeseries_file='timeseries.nc', fig_name=None, option='smooth'):

    num_members, sim_names, file_paths, colours = setup_ensemble(sim_dir, timeseries_file)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axhline()
    ax.axvline()
    not_sig = 0
    title, units = read_title_units(file_paths[0], var)
    values = []
    for n in range(num_members):
        percent = var.endswith('_melting') or var.endswith('_massloss')
        if percent:
            units = '%'
        slope, sig = read_calc_trends(var, file_paths[n], option, percent=percent)
        if True: #sig:
            # Add to plot
            ax.plot(slope, 0, 'o', color=colours[n], label=sim_names[n])
            values.append(slope)
        else:
            not_sig += 1
    if not_sig > 0:
        ax.text(0.95, 0.05, str(not_sig)+' members had\nno significant trend', ha='right', va='bottom', fontsize=12, transform=ax.transAxes)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel('Trend ('+units+'/decade)')
    ax.set_title('Trend in '+title)
    finished_plot(fig, fig_name=fig_name)
    t_val, p_val = ttest_1samp(values, 0)
    confidence = (1-p_val)*100
    print 'Confidence trend is nonzero: '+str(confidence)+'%'


# Call for a bunch of variables.
def plot_all_trends (sim_dir, fig_dir=None):
    for var in ['abbot_melting', 'cosgrove_melting', 'dotson_crosson_melting', 'getz_melting', 'pig_melting', 'thwaites_melting', 'venable_melting', 'pine_island_bay_temp_below_500m', 'pine_island_bay_salt_below_500m', 'dotson_bay_temp_below_500m', 'dotson_bay_salt_below_500m', 'inner_amundsen_shelf_temp_below_500m', 'inner_amundsen_shelf_salt_below_500m', 'amundsen_shelf_break_uwind_avg', 'inner_amundsen_shelf_sss_avg', 'amundsen_shelf_break_adv_heat_ns_300_1500m']:
        if fig_dir is None:
            fig_name = None
        else:
            fig_name = real_dir(fig_dir) + var + '_trends.png'
        ensemble_trends(var, sim_dir, fig_name=fig_name)


# Call plot_timeseries_2y for the PACE ensemble, ensemble mean, and ERA5 using the right colours.
def plot_timeseries_ensemble_era5 (era5_dir, pace_dir, timeseries_types=None, fig_dir='./', ismr_percent=False, plot_anomaly=False):

    num_ens = len(pace_dir)
    sim_dir = [era5_dir] + pace_dir
    sim_names = ['ERA5', 'PACE ensemble'] + [None for n in range(num_ens-1)]
    colours = ['red'] + [(0.6, 0.6, 0.6) for n in range(num_ens)] + ['black']
    first_in_mean = False
    base_year_start_first = 1979
    if ismr_percent or plot_anomaly:
        if era5_dir is not None:
            print 'Warning: removing ERA5'
        # Remove ERA5 so we can plot percent/anomaly
        sim_dir = sim_dir[1:]
        sim_names = sim_names[1:]
        colours = colours[1:]
        first_in_mean = True
        base_year_start_first = None
    plot_timeseries_2y(sim_dir, sim_names, timeseries_types=timeseries_types, plot_mean=True, first_in_mean=first_in_mean, fig_dir=fig_dir, colours=colours, ismr_percent=ismr_percent, plot_anomaly=plot_anomaly, base_year_start_first=base_year_start_first)


# Plot a T/S diagram for a given (single) simulation and region, with each decade plotted in a different colour. You can restrict the depth to everything deeper than z0 (negative, in metres).
def plot_ts_decades (sim_dir, region, z0=None, year_start=1920, smin=None, smax=None, tmin=None, tmax=None, multi_region=False, fig_name=None):

    output_dir = real_dir(sim_dir) + 'output/'
    fnames = get_output_files(output_dir)
    file_paths = [output_dir + f for f in fnames]
    grid = Grid(file_paths[0])

    # Read temperature and salinity data, annually averaged
    temp, years = read_annual_average('THETA', file_paths, return_years=True)
    salt = read_annual_average('SALT', file_paths)
    # Trim before year_start
    t_start = years.index(year_start)
    temp = temp[t_start:,:]
    salt = salt[t_start:,:]
    years = years[t_start:]
    # Count the number of decades and set up that many colours
    num_decades = 0
    for t in years:
        if t % 10 == 0:
            num_decades += 1
    colours = choose_n_colours(num_decades)

    # Check if it's only one region
    if multi_region:
        num_region = len(region)
        if smin is None:
            smin = [None for n in range(num_region)]
        if smax is None:
            smax = [None for n in range(num_region)]
        if tmin is None:
            tmin = [None for n in range(num_region)]
        if tmax is None:
            tmax = [None for n in range(num_region)]
        if z0 is None:
            z0 = [None for n in range(num_region)]
        if fig_name is None:
            fig_name = [None for n in range(num_region)]
    else:
        num_region = 1
        region = [region]
        smin = [smin]
        smax = [smax]
        tmin = [tmin]
        tmax = [tmax]
        z0 = [z0]
        fig_name = [fig_name]

    for n in range(num_region):
        # Get the mask for the region
        if region[n].endswith('_front'):
            # Ice shelf front
            shelf = region[n][:region[n].index('_front')]
            mask = grid.get_icefront_mask(shelf)
            title = region_names[shelf] + ' front'
        else:
            mask = grid.get_region_mask(region[n])
            title = region_names[region[n]]
        if z0[n] is None:
            # Choose default value for z0
            if region[n] in ['pine_island_bay', 'dotson_bay']:
                # Approximate the thermocline
                z0[n] = -500
            else:
                if not region[n].endswith('_front'):
                    print 'Warning (plot_ts_decades): using default value of z0=0 for ' + region + ', is this what you want?'
                z0[n] = 0
        if z0[n] is not None and z0[n] != 0:
            title += ', below ' + str(abs(z0[n])) + 'm'
        # Now make the mask 3D and cut off anything shallower than this
        mask = mask_2d_to_3d(mask, grid, zmax=z0[n])

        # Set up plot
        fig, ax = plt.subplots(figsize=(10,7))
        # Loop over decades
        for t in range(0, len(years), 10):
            # Choose decade (to determine colour)
            decade = (years[t]-years[0])/10
            label = None
            if years[t] % 10 == 0:
                label = str(years[t]) + 's'
            # Average over the decade
            t_end = min(t+10, len(years))
            temp_decade = np.mean(temp[t:t_end,:], axis=0)
            salt_decade = np.mean(salt[t:t_end,:], axis=0)
            # Plot one point for each cell in the mask
            ax.plot(salt_decade[mask], temp_decade[mask], 'o', color=colours[decade], label=label, markersize=2)
        # Finish the rest of the plot
        ax.set_xlim([smin[n], smax[n]])
        ax.set_ylim([tmin[n], tmax[n]])
        ax.legend()
        ax.set_xlabel('Salinity (psu)', fontsize=14)
        ax.set_ylabel('Temperature ('+deg_string+'C)', fontsize=14)
        ax.set_title(title, fontsize=20)
        finished_plot(fig, fig_name=fig_name[n])


# Call plot_ts_decades for 2 regions and every ensemble member.
def plot_all_ts_decades (sim_dir, fig_dir='./'):

    num_ens = len(sim_dir)
    sim_names = ['ens'+str(n+1).zfill(2) for n in range(num_ens)]
    regions = ['pine_island_bay', 'dotson_bay']
    num_regions = len(regions)
    smin = [34.45, None]
    tmin = [-0.75, None]
    fig_dir = real_dir(fig_dir)
    for n in range(num_ens):
        fig_name = [fig_dir+'ts_decades_'+r+'_'+sim_names[n]+'.png' for r in regions]
        plot_ts_decades(sim_dir[n], regions, smin=smin, tmin=tmin, multi_region=True, fig_name=fig_name)


# Make a scatterplot of wind trends vs. temperature trends in the PACE ensemble.
def wind_temp_trend_scatterplot (sim_dir, temp_var='inner_amundsen_shelf_temp_below_500m', timeseries_file='timeseries.nc', fig_name=None, option='smooth'):

     num_members, sim_names, file_paths, colours = setup_ensemble(sim_dir, timeseries_file)
     wind_var = 'amundsen_shelf_break_uwind_avg'
     wind_title, wind_units = read_title_units(file_paths[0], wind_var)
     temp_title, temp_units = read_title_units(file_paths[0], temp_var)

     fig, ax = plt.subplots(figsize=(10,6))
     ax.axhline()
     ax.axvline()
     wind_trends = []
     temp_trends = []
     not_sig = 0
     for n in range(num_members):
         wind_slope, wind_sig = read_calc_trends(wind_var, file_paths[n], option)
         temp_slope, temp_sig = read_calc_trends(temp_var, file_paths[n], option)
         if wind_sig and temp_sig:
             # Plot, and save trends for line of best fit later
             ax.plot(wind_slope, temp_slope, 'o', color=colours[n], label=sim_names[n])
             wind_trends.append(wind_slope)
             temp_trends.append(temp_slope)
         else:
             not_sig += 1
     # Add line of best fit
     slope, intercept, r_value, p_value, std_err = linregress(np.array(wind_trends), np.array(temp_trends))
     if p_value < 0.05:
         [x0, x1] = ax.get_xlim()
         [y0, y1] = slope*np.array([x0, x1]) + intercept
         ax.plot([x0, x1], [y0, y1], '-', color='black', linewidth=1, zorder=0)
         trend_title = 'r$^2$='+str(round_to_decimals(r_value**2,3))
     else:
         trend_title = 'no significant relationship'
     ax.text(0.05, 0.95, trend_title, ha='left', va='top', fontsize=12, transform=ax.transAxes)
     # Add titles
     ax.set_xlabel('Trend in '+wind_title+' ('+wind_units+'/decade)')
     ax.set_ylabel('Trend in '+temp_title+' ('+temp_units+'/decade)')
     ax.set_title('Temperature versus wind trends in PACE', fontsize=18)
     
     if not_sig > 0:
         ax.text(0.95, 0.05, str(not_sig)+' members had\nno significant trend', ha='right', va='bottom', fontsize=12, transform=ax.transAxes)
     # Add legend
     box = ax.get_position()
     ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
     ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
     finished_plot(fig, fig_name=fig_name)


# Test the correlation between the time-integral of winds at the shelf break and melt rate anomalies for the given ice shelf. Plot one two-sided timeseries for each ensemble member, and then a big scatterplot showing the correlation.
def wind_melt_correlation (sim_dir, shelf, timeseries_file='timeseries.nc', fig_dir='./'):

    num_members, sim_names, file_paths, colours = setup_ensemble(sim_dir, timeseries_file)
    smooth = 24
    base_year_start = 1920
    base_year_end = 1949
    fig_dir = real_dir(fig_dir)

    all_wind = None
    all_ismr = None
    for n in range(num_members):
        # Read timeseries
        time = netcdf_time(file_paths[n], monthly=False)
        ismr = read_netcdf(file_paths[n], shelf+'_melting')
        wind = read_netcdf(file_paths[n], 'amundsen_shelf_break_uwind_avg')
        # Take anomalies from 1920-1949 mean
        t_start, t_end = index_period(time, base_year_start, base_year_end)
        ismr_mean = np.mean(ismr[t_start:t_end])
        ismr -= ismr_mean
        wind_mean = np.mean(wind[t_start:t_end])
        wind -= wind_mean
        # Trim the spinup
        ismr = ismr[t_start:]
        wind = wind[t_start:]
        time = time[t_start:]
        # Calculate 2 year running means of both timeseries
        ismr, time = moving_average(ismr, smooth, time=time)
        wind = moving_average(wind, smooth)
        # Now take time-integral of wind
        dt = 365./12*sec_per_day  # Assume constant month length, no leap years
        wind = np.cumsum(wind*dt)
        # Plot with twin y-axes
        make_timeseries_plot_2sided(time, wind, ismr, 'PACE '+str(n+1).zfill(2), 'time-integral of shelf break winds (m)', region_names[shelf]+' melt rate (m/y)', fig_name=fig_dir+'wind_'+shelf+'_ens'+str(n+1).zfill(2)+'.png')
        # Save to long arrays for correlation later
        if all_wind is None:
            all_wind = wind
            all_ismr = ismr
        else:
            all_wind = np.concatenate((all_wind, wind), axis=0)
            all_ismr = np.concatenate((all_ismr, ismr), axis=0)
    # Now make the scatterplot
    fig, ax = plt.subplots(figsize=(10,6))
    ax.axhline(color='black')
    ax.axvline(color='black')
    ax.plot(all_wind, all_ismr, 'o', color='blue', markersize=2)
    ax.set_xlabel('time-integral of shelf break winds (m)', fontsize=12)
    ax.set_ylabel('melt rate anomaly of '+region_names[shelf]+' (m/y)', fontsize=12)
    # Add line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(np.array(all_wind), np.array(all_ismr))
    [x0, x1] = ax.get_xlim()
    [y0, y1] = slope*np.array([x0, x1]) + intercept
    ax.plot([x0, x1], [y0, y1], '-', color='black', linewidth=1)
    ax.text(0.05, 0.95, 'r$^2$='+str(r_value**2), ha='left', va='top', fontsize=12, transform=ax.transAxes)
    finished_plot(fig, fig_name=fig_dir+'wind_'+shelf+'_scatterplot.png')


# Find the best number of years to use for a moving average, such that the correlation between time-integrated winds and melt rates (both anomalies from their given moving average) is maximised.
def find_correlation_timescale(sim_dir, shelf, timeseries_file='timeseries.nc'):

    num_members, sim_names, file_paths, colours = setup_ensemble(sim_dir, timeseries_file)
    smooth_short = 24
    year0 = 1920
    test_smooth = range(20, 50+1)
    num_tests = len(test_smooth)

    r2 = np.empty(num_tests)
    for m in range(num_tests):
        all_wind = None
        all_ismr = None
        for n in range(num_members):
            # Read timeseries
            time = netcdf_time(file_paths[n], monthly=False)
            ismr = read_netcdf(file_paths[n], shelf+'_melting')
            wind = read_netcdf(file_paths[n], 'amundsen_shelf_break_uwind_avg')
            t_start = index_year_start(time, year0)
            # Calculate long-term running mean
            smooth_long = test_smooth[m]*12
            ismr_tmp = moving_average(ismr, smooth_long)
            wind_tmp = moving_average(wind, smooth_long)
            # Pad to be the same size as the original arrays
            def pad_array (A):
                A_pad = np.empty(ismr.size)
                A_pad[:smooth_long] = A[0]
                A_pad[smooth_long:-smooth_long] = A
                A_pad[-smooth_long:] = A[-1]
                return A_pad
            ismr_avg = pad_array(ismr_tmp)
            wind_avg = pad_array(wind_tmp)
            # Now get the anomaly from the moving average
            ismr_anom = ismr - ismr_avg
            wind_anom = wind - wind_avg
            # Trim the spinup
            ismr_anom = ismr_anom[t_start:]
            wind_anom = wind_anom[t_start:]
            # Calculate 2-year running means of both timeseries
            ismr_anom = moving_average(ismr_anom, smooth_short)
            wind_anom = moving_average(wind_anom, smooth_short)
            # Now take time-integral of wind
            dt = 365./12*sec_per_day
            wind_anom = np.cumsum(wind_anom*dt)
            # Save to long arrays for correlation
            if all_wind is None:
                all_wind = wind_anom
                all_ismr = ismr_anom
            else:
                all_wind = np.concatenate((all_wind, wind_anom), axis=0)
                all_ismr = np.concatenate((all_ismr, ismr_anom), axis=0)
        # Calculate correlation
        slope, intercept, r_value, p_value, std_err = linregress(np.array(all_wind), np.array(all_ismr))
        r2[m] = r_value**2
        print 'Timescale of '+str(test_smooth[m])+' years gives r^2='+str(r2[m])
    m0 = np.argmax(r2)
    print 'Best correlation is with timescale of '+str(test_smooth[m0])+' years: r^2='+str(r2[m0])


# Test the correlation in 4 stages:
# 1) time-integral of winds at the shelf break and southward heat flux at the shelf break
# 2) southward heat flux at the shelf break and thermocline depth on the shelf
# 3) thermocline depth on the shelf and temperatures below 500 m on the shelf
# 4) temperatures below 500 m on the shelf and melt rate of each ice shelf (Dotson/Crosson, Thwaites, PIG)
def correlation_4pt (sim_dir, timeseries_file='timeseries.nc', fig_dir='./'):

    num_members, sim_names, file_paths, colours = setup_ensemble(sim_dir, timeseries_file)
    smooth = 24
    base_year_start = 1920
    base_year_end = 1949
    fig_dir = real_dir(fig_dir)

    # Inner function to analyse one set of variables
    def do_one_correlation (var1, var2, fig_name_head, int_first=False):
        all_data1 = []
        all_data2 = []
        for n in range(num_members):
            # Read timeseries
            time = netcdf_time(file_paths[n], monthly=False)
            data1 = read_netcdf(file_paths[n], var1)
            data2 = read_netcdf(file_paths[n], var2)
            # Take anomalies from 1920-1949 mean
            t_start, t_end = index_period(time, base_year_start, base_year_end)
            data1 -= np.mean(data1[t_start:t_end])
            data2 -= np.mean(data2[t_start:t_end])
            # Trim the spinup
            data1 = data1[t_start:]
            data2 = data2[t_start:]
            time = time[t_start:]
            # Calculate 2 year running means of both timeseries
            data1, time = moving_average(data1, smooth, time=time)
            data2 = moving_average(data2, smooth)
            if int_first:
                # Now take time-integral of first variable
                dt = 365./12*sec_per_day
                data1 = np.cumsum(data1*dt)
                str1 = 'time-integral of '+var1
            else:
                str1 = var1
            str2 = var2
            #if n==0:
            # Plot the first ensemble member with twin y-axes
            make_timeseries_plot_2sided(time, data1, data2, 'PACE '+str(n+1).zfill(2), str1, str2, fig_name=fig_name_head+'_ens'+str(n+1).zfill(2)+'.png')
            # Save to long arrays for correlation later
            if all_data1 is None:
                all_data1 = data1
                all_data2 = data2
            else:
                all_data1 = np.concatenate((all_data1, data1), axis=0)
                all_data2 = np.concatenate((all_data2, data2), axis=0)
        # Now make the scatterplot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.axhline(color='black')
        ax.axvline(color='black')
        ax.plot(all_data1, all_data2, 'o', color='blue', markersize=2)
        ax.set_xlabel(str1, fontsize=12)
        ax.set_ylabel(str2, fontsize=12)
        # Add line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(np.array(all_data1), np.array(all_data2))
        [x0, x1] = ax.get_xlim()
        [y0, y1] = slope*np.array([x0, x1]) + intercept
        ax.plot([x0, x1], [y0, y1], '-', color='black', linewidth=1)
        ax.text(0.05, 0.95, 'r$^2$='+str(r_value**2), ha='left', va='top', fontsize=12, transform=ax.transAxes)
        finished_plot(fig, fig_name=fig_name_head+'_scatterplot.png')

    # Now call this function for each set of variables
    var_names = ['amundsen_shelf_break_uwind_avg', 'amundsen_shelf_break_adv_heat_s', 'inner_amundsen_shelf_thermocline', 'inner_amundsen_shelf_temp_below_500m']
    abbrv = ['wind', 'hflx', 'thmc', 'temp']
    shelves = ['dotson_crosson_melting', 'thwaites_melting', 'pig_melting']
    abbrv_shelves = ['melt_dot', 'melt_thw', 'melt_pig']
    for n in range(len(var_names)-1):
        do_one_correlation(var_names[n], var_names[n+1], fig_dir+'correlation_'+abbrv[n]+'_'+abbrv[n+1], int_first=(abbrv[n]=='wind'))
    for m in range(len(shelves)):
        do_one_correlation(var_names[-1], shelves[m], fig_dir+'correlation_'+abbrv[-1]+'_'+abbrv_shelves[m])


# Calculate monthly climatologies from daily climatologies for ERA5 and PACE.
def monthly_from_daily_climatologies (clim_dir):

    models = ['ERA5'] + ['PACE_ens'+str(n+1).zfill(2) for n in range(num_ens)]
    # Get PACE grid just for the sizes
    grid = PACEGrid()
    clim_dir = real_dir(clim_dir)
    
    # Start and end days for each month
    start_days = [0]
    end_days = []
    for month in range(months_per_year-1):
        ndays = days_per_month(month+1, 1979)  # Random non-leap-year
        day_index = start_days[-1]+ndays
        start_days.append(day_index)
        end_days.append(day_index)
    end_days.append(days_per_year)

    for var in var_pace:
        if var in ['FSDS', 'FLDS']:
            # Already monthly
            continue
        print 'Processing ' + var
        for model in models:
            print 'Processing ' + model
            data_daily = read_binary(clim_dir+model+'_'+var+'_clim', [grid.nx, grid.ny], 'xyt')
            data_monthly = np.empty([months_per_year, grid.ny, grid.nx])
            for month in range(months_per_year):
                data_monthly[month,:] = np.mean(data_daily[start_days[month]:end_days[month],:], axis=0)
            write_binary(data_monthly, clim_dir+model+'_'+var+'_clim_monthly')


# As in plot_biases, but showing a big 3x4+1 plot showing the monthly biases as well as annual mean
def plot_monthly_biases (var_name, clim_dir, grid_dir, fig_dir='./'):

    # Build grid to get bounds
    model_grid = Grid(grid_dir, max_lon=360)
    xmin = model_grid.lon_corners_1d[0]
    xmax = 2*model_grid.lon_1d[-1] - model_grid.lon_corners_1d[-1]
    ymin = model_grid.lat_corners_1d[0]
    ymax = 2*model_grid.lat_1d[-1] - model_grid.lat_corners_1d[-1]
    # Also build PACE grid to get sizes
    pace_grid = PACEGrid()
    clim_dir = real_dir(clim_dir)
    fig_dir = real_dir(fig_dir)

    # Read data
    pace_data = np.empty([num_ens, months_per_year, pace_grid.ny, pace_grid.nx])
    for ens in range(num_ens):
        file_path = clim_dir + 'PACE_ens' + str(ens+1).zfill(2) + '_' + var_name + '_clim'
        if var_name not in ['FLDS', 'FSDS']:
            file_path += '_monthly'
        pace_data[ens,:] = read_binary(file_path, [pace_grid.nx, pace_grid.ny], 'xyt')
    # Take ensemble mean
    pace_data = np.mean(pace_data, axis=0)
    # Read ERA5 data too (already interpolated to PACE grid)
    file_path = file_path.replace('PACE_ens'+str(num_ens), 'ERA5')
    era5_data = read_binary(file_path, [pace_grid.nx, pace_grid.ny], 'xyt')

    # Get bounds across all months (annual mean is implicitly within this)
    mask = add_time_dim((pace_grid.lon >= xmin)*(pace_grid.lon <= xmax)*(pace_grid.lat >= ymin)*(pace_grid.lat <= ymax), months_per_year)
    vmin = np.amin(pace_data[mask] - era5_data[mask])
    vmax = np.amax(pace_data[mask] - era5_data[mask])
    cmap, vmin, vmax = set_colours(pace_data-era5_data, ctype='plusminus', vmin=vmin, vmax=vmax)

    # Set up figure
    fig, gs, cax = set_panels('3x4+1C1')
    titles = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    for n in range(months_per_year+1):
        if n == months_per_year:
            # Plot annual mean in top right
            ax = plt.subplot(gs[0,3])
            img = ax.pcolormesh(pace_grid.lon, pace_grid.lat, np.mean(pace_data, axis=0)-np.mean(era5_data, axis=0), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title('Annual')
        else:
            # Plot individual month
            ax = plt.subplot(gs[n/4+1, n%4])
            img = ax.pcolormesh(pace_grid.lon, pace_grid.lat, pace_data[n,:]-era5_data[n,:], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(titles[n])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xticks([])
        ax.set_yticks([])
    # Make colourbar
    plt.colorbar(img, cax=cax, orientation='horizontal')
    # Main title
    plt.text(0.05, 0.95, var_name+': PACE ensemble mean minus ERA5', transform=fig.transFigure, fontsize=20, ha='left', va='top')
    finished_plot(fig, fig_name=fig_dir+'monthly_bias_'+var_name+'.png')


# Calculate the trend at every point in the given region, for the given variable, and all ensemble members. Save to a NetCDF file (3D or 4D depending on whether variable is 2D or 3D).
def make_trend_file (var_name, region, sim_dir, grid_dir, out_file, dim=3, gtype='t', start_year=1920):

    num_ens = len(sim_dir)

    grid = Grid(grid_dir)
    units = None
    # Construct region mask
    if region == 'all':
        # All ocean points, 2D or 3D
        if dim == 3:
            mask = grid.get_hfac(gtype=gtype) != 0
        elif dim == 2:
            mask = np.invert(grid.get_land_mask(gtype=gtype))
    elif region == 'ice':
        # All cavity points, 2D or 3D
        mask = grid.get_ice_mask(gtype=gtype)
        if dim == 3:
            mask = xy_to_xyz(mask, grid)*(grid.get_hfac(gtype=gtype) != 0)
    else:
        # Specific region
        mask = grid.get_region_mask(region, is_3d=(dim==3))
    # Save number of indices in the region
    num_pts = np.count_nonzero(mask)
    # Set up an array to hold trends for each ensemble member; 0 outside the given region
    if dim == 2:
        trends = np.zeros([num_ens, grid.ny, grid.nx])
    elif dim == 3:
        trends = np.zeros([num_ens, grid.nz, grid.ny, grid.nx])

    # Loop over ensemble member
    for m in range(num_ens):
        print 'Processing ' + sim_dir[m]
        # Get all the file paths (assume one for each year)
        file_paths = segment_file_paths(sim_dir[m]+'/output/')
        t_start = file_paths.index(sim_dir[m]+'/output/'+str(start_year)+'01/MITgcm/output.nc')
        file_paths = file_paths[t_start:]
        num_years = len(file_paths)
        # Read annually-averaged data at each point in the mask
        data_save = np.empty([num_years, num_pts])
        for t in range(num_years):
            print '...Reading ' + file_paths[t]
            if var_name == 'advection_3d':
                data_x, long_name, units = read_netcdf(file_paths[t], 'ADVx_TH', time_average=True, return_info=True)
                data_y = read_netcdf(file_paths[t], 'ADVy_TH', time_average=True)
                data_z = read_netcdf(file_paths[t], 'ADVr_TH', time_average=True)
                data = np.ma.zeros(data_x.shape)
                data[:-1,:-1,:-1] = data_x[:-1,:-1,:-1] - data_x[:-1,:-1,1:] + data_y[:-1,:-1,:-1] - data_y[:-1,1:,:-1] + data_z[1:,:-1,:-1] - data_z[:-1,:-1,:-1]
                long_name = 'net advection of heat'
            elif var_name == 'diffusion_kpp':
                data1, long_name, units = read_netcdf(file_paths[t], 'DFrI_TH', time_average=True, return_info=True)
                data2 = read_netcdf(file_paths[t], 'KPPg_TH', time_average=True)
                data = np.ma.zeros(data1.shape)
                data[:-1,:] = data1[1:,:] - data1[:-1,:] + data2[1:,:] - data2[:-1,:]
                long_name = 'net vertical implicit diffusion and KPP transport of heat'
            else:
                data, long_name, units = read_netcdf(file_paths[t], var_name, time_average=True, return_info=True)
            if len(data.shape) != dim:
                print 'Error (make_trend_file): wrong dimension for this variable.'
                sys.exit()
            if var_name == 'ADVx_TH':
                # Need to convert to heat advection relative to freezing point
                u = read_netcdf(file_paths[t], 'UVEL', time_average=True)
                [data, tmp] = adv_heat_wrt_freezing([data, None], [u, None], grid)
            elif var_name == 'ADVy_TH':
                v = read_netcdf(file_paths[t], 'VVEL', time_average=True)
                [tmp, data] = adv_heat_wrt_freezing([None, data], [None, v], grid)                
            data_save[t,:] = data[mask]
        # Now calculate the trend at each point
        print 'Calculating trends'
        trends_tmp = np.empty(num_pts)
        for p in range(num_pts):
            slope, intercept, r_value, p_value, std_err = linregress(np.arange(num_years), data_save[:,p])
            trends_tmp[p] = slope
        # Save to master array
        trends[m,mask] = trends_tmp

    # Mask master array outside the given region (where it's still zero)
    trends = np.ma.masked_where(trends==0, trends)
    
    # Save to NetCDF - treat ensemble axis as time for convenience
    if dim == 2:
        file_dim = 'xyt'
    elif dim == 3:
        file_dim = 'xyzt'
    ncfile = NCfile(out_file, grid, file_dim)
    ncfile.add_time(np.arange(num_ens)+1, units='ensemble member')
    ncfile.add_variable(var_name+'_trend', trends, file_dim, gtype=gtype, long_name='trend in '+long_name, units=units+'/y')
    ncfile.close()

    
# Make plots from the trend file created above.
def trend_region_plots (in_file, var_name, region, grid_dir, fig_dir='./', dim=3, gtype='t', zmin=None, zmax=None, sign='positive', lon0_slices=[]):

    fig_dir = real_dir(fig_dir)
    grid = Grid(grid_dir)
    lon, lat = grid.get_lon_lat(gtype=gtype)
    if dim == 3:
        z_3d = z_to_xyz(grid.z, grid)
    # Get x-y bounds on region
    if region == 'all':
        [xmin, xmax, ymin, ymax] = [None, None, None, None]
    elif region == 'ice':
        ice_mask = grid.get_ice_mask(gtype=gtype)
        xmin = np.amin(lon[ice_mask])
        xmax = np.amax(lon[ice_mask])
        ymin = np.amin(lat[ice_mask])
        ymax = np.amax(lat[ice_mask])
    else:
        [xmin, xmax, ymin, ymax] = region_bounds[region]

    # Read data
    trends, long_name, units = read_netcdf(in_file, var_name+'_trend', return_info=True)
    # Calculate mean trend and significance along first axis (ensemble member)
    mean_trend = np.mean(trends, axis=0)
    t_val, p_val = ttest_1samp(trends, 0, axis=0)
    sig = (1-p_val)*100
    # For any trends which aren't significant, fill with zeros for now
    mean_trend[sig < 95] = 0
    if dim == 3:
        # Also mask out depths we don't care about
        if zmin is not None:
            mean_trend = np.ma.masked_where(z_3d<zmin, mean_trend)
        if zmax is not None:
            mean_trend = np.ma.masked_where(z_3d>zmax, mean_trend)

    if dim == 2:
        # Plot the mean trend that's significant
        latlon_plot(mean_trend, grid, ctype='plusminus', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=long_name+',\n'+region_names[region]+' ('+units+')', titlesize=14, fig_name=fig_dir+var_name+'_trend.png')
    else:
        # Select maximum significant trend over depth, and the depth at which this occurs
        if sign == 'positive':
            k_max = np.argmax(mean_trend, axis=0)
        elif sign == 'negative':
            k_max = np.argmin(mean_trend, axis=0)
        k, j, i = k_max, np.arange(grid.ny)[:,None], np.arange(grid.nx)
        max_trend = mean_trend[k,j,i]
        max_trend_depth = z_3d[k,j,i]
        # Now mask out anything with 0 trend
        max_trend = np.ma.masked_where(max_trend==0, max_trend)
        max_trend_depth = np.ma.masked_where(max_trend==0, max_trend_depth)
        # Plot both of them
        latlon_plot(max_trend, grid, ctype='plusminus', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title='Maximum '+long_name+' over depth,\n'+region_names[region]+' ('+units+')', titlesize=14, fig_name=fig_dir+var_name+'_trend_max.png')
        latlon_plot(max_trend_depth, grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmin=zmin, vmax=zmax, title='Depth of maximum '+long_name+',\n'+region_names[region]+' (m)', titlesize=14, fig_name=fig_dir+var_name+'_trend_depth.png')
    
    # Now plot trend at every integer longitude within the domain (lat-depth slices)
    for lon0 in lon0_slices:
        slice_plot(mean_trend, grid, gtype=gtype, lon0=lon0, ctype='plusminus', zmin=zmin, zmax=zmax, title=long_name+' \n('+units+')', titlesize=14, fig_name=fig_dir+var_name+'_trend_'+str(lon0)+'.png')


def trend_sensitivity_to_convection (sim_dir, timeseries_file='timeseries.nc', fig_dir='./'):

    loc = ['pine_island_bay', 'dotson_bay', 'inner_amundsen_shelf']
    ismr = ['pig_massloss', 'dotson_massloss', None]
    var_ts = ['_temp_below_500m', '_salt_below_500m']
    max_cutoff = 0
    num_cutoff = 50
    year_start = 1920
    smooth = 24
    num_ens = len(sim_dir)
    file_paths = [d + '/output/' + timeseries_file for d in sim_dir]

    # Read one time array, assume it's the same everywhere (i.e. all ensemble members have finished)
    time = netcdf_time(file_paths[0], monthly=False)
    t0 = index_year_start(time, year_start)
    time = time[t0:]
    # Now smooth a dummy array so we can trim the time correctly
    time = moving_average(np.arange(time.size), smooth, time=time)[1]
    num_time = time.size
    # Overwrite the time array with scalars of unit years (monthly averaged, assume evenly spaced for simplicity)
    time = np.arange(num_time)/12.

    # Loop over regions
    for l in range(len(loc)):
        
        # Read temperature for this region, for all ensemble members
        temp = np.empty([num_ens, num_time])
        for n in range(num_ens):
            temp[n,:] = moving_average(read_netcdf(file_paths[n], loc[l]+var_ts[0])[t0:], smooth)
        # Get range of cutoff temperatures
        cutoff_temp = np.linspace(np.amin(temp), max_cutoff, num=num_cutoff)
        
        # Now loop over all variables for this region: temp, salt, and maybe ismr
        var_names = [loc[l]+v for v in var_ts]
        if ismr[l] is not None:
            var_names += [ismr[l]]
        for var in var_names:
            # Get the title and units
            var_title, var_units = set_parameters(var)[2:4]

            # Read this variable for all ensemble members
            var_data = np.empty([num_ens, num_time])
            for n in range(num_ens):
                var_data[n,:] = moving_average(read_netcdf(file_paths[n], var)[t0:], smooth)

            # Now calculate trends and significance for each cutoff temp
            all_trends = np.empty([num_ens, num_cutoff])
            sig = np.empty(num_cutoff)
            for m in range(num_cutoff):
                # Calculate trend for each ensemble member (convert to per decade)
                for n in range(num_ens):
                    # Extract values where temperature exceeds this cutoff
                    index = temp[n,:] > cutoff_temp[m]
                    all_trends[n, m] = linregress(time[index], var_data[n,index])[0]*10
                # Now calculate significance of ensemble
                p_val = ttest_1samp(all_trends[:,m], 0)[1]
                sig[m] = (1-p_val)*100

            # Plot cutoff temperature versus mean trend, and cutoff temperature versus significance
            data_plot = [all_trends, sig]
            titles = ['Trends in\n'+var_title, 'Significance of ensemble trend in\n'+var_title]
            units = [var_units+'/decade', '%']
            file_tail = ['_cutoff_trend', '_cutoff_sig']
            for p in range(len(data_plot)):
                fig, ax = plt.subplots()
                if p == 0:
                    for n in range(num_ens):
                        ax.plot(cutoff_temp, data_plot[p][n,:], linewidth=1.5)
                else:
                    ax.plot(cutoff_temp, data_plot[p], '-', linewidth=1.5)
                if p==0:
                    # Add horizontal line at 0 trend
                    ax.axhline(color='black')
                elif p==1:
                    # Add dashed lines at 90% and 95% threshold
                    for y in [90, 95]:
                        ax.axhline(y, color='black', linestyle='dashed')
                ax.grid(True)
                plt.title(titles[p])
                plt.xlabel('Cutoff temperature for convection ('+deg_string+'C)')
                plt.ylabel(units[p])
                finished_plot(fig, fig_name=fig_dir+var+file_tail[p]+'.png')


def correlate_ismr_forcing (pace_dir, grid_path, timeseries_file='timeseries.nc', fig_dir='./'):

    forcing_names = ['amundsen_shelf_break_uwind_avg', 'inner_amundsen_shelf_atemp_avg', 'inner_amundsen_shelf_aqh_avg', 'inner_amundsen_shelf_precip_avg']
    forcing_titles = ['time-integral of winds at shelf break', 'air temperature over inner shelf', 'humidity over inner shelf', 'precipitation over inner shelf']
    forcing_abbrv = ['uwind', 'atemp', 'aqh', 'precip']
    title_head = 'Correlation of ice shelf melting and '
    num_forcing = len(forcing_names)
    base_year_start = 1920
    base_year_end = 1949    
    num_ens = len(pace_dir)
    output_dir = [real_dir(d) + 'output/' for d in pace_dir]
    ts_paths = [od + timeseries_file for od in output_dir]
    smooth = 24
    max_lag = 24
    ymax = -70
    fig_dir = real_dir(fig_dir)
    vmax = 0.5

    grid = Grid(grid_path)
    mask = grid.get_ice_mask()
    num_ice_pts = np.count_nonzero(mask)

    # Get time indices of the base period
    time = netcdf_time(ts_paths[0], monthly=False)
    t_start, t_end = index_period(time, base_year_start, base_year_end)

    # Read and process the ice shelf melt rates
    print 'Reading ice shelf melt rates'
    ismr_data = None    
    for n in range(num_ens):
        print '...' + pace_dir[n]
        ismr_tmp = None
        file_paths = segment_file_paths(output_dir[n])
        for f in file_paths:
            ismr_1y = mask_except_ice(convert_ismr(read_netcdf(f, 'SHIfwFlx')), grid, time_dependent=True)
            if ismr_tmp is None:
                ismr_tmp = ismr_1y
            else:
                ismr_tmp = np.concatenate((ismr_tmp, ismr_1y), axis=0)
        # Trim the spinup
        ismr_tmp = ismr_tmp[t_start:,:]
        # Get 2 year running mean
        ismr_tmp = moving_average(ismr_tmp, smooth)
        # Centre the data by subtracting the time-mean at each point
        ismr_tmp -= np.mean(ismr_tmp, axis=0)
        # Save to master array, but only the ice shelf points
        if ismr_data is None:
            num_time = ismr_tmp.shape[0]
            ismr_data = np.ma.empty([num_ens, num_time, num_ice_pts])
        ismr_data[n,:] = ismr_tmp[:,mask]

    # Now process one forcing at a time
    for m in range(num_forcing):
        print 'Processing ' + forcing_names[m]
        print 'Reading forcing timeseries and calculating lags'
        forcing_data = np.empty([num_ens, num_time])
        avg_lag = np.empty(num_ens)

        for n in range(num_ens):
            print '...'+pace_dir[n]
            # Read and process the timeseries
            data_tmp = read_netcdf(ts_paths[n], forcing_names[m])            
            if forcing_names[m] == 'amundsen_shelf_break_uwind_avg':
                # Time-integral of wind anomaly from 1920-1949 mean
                data_tmp -= np.mean(data_tmp[t_start:t_end])
                dt = 365./12*sec_per_day
                data_tmp = np.cumsum(data_tmp*dt)
            # Trim the spinup
            data_tmp = data_tmp[t_start:]
            # Get 2 year running mean
            data_tmp = moving_average(data_tmp, smooth)
            # Now centre the data by subtracting the mean
            data_tmp -= np.mean(data_tmp)
            # Save to master array
            forcing_data[n,:] = data_tmp

            # Loop over ice shelf points
            lag = np.empty(num_ice_pts)
            for p in range(num_ice_pts):
                # Extract timeseries of ice shelf melting at this point
                ismr_ts = ismr_data[n,:,p]
                # Calculate cross-correlation with forcing
                corr = np.correlate(forcing_data[n,:], ismr_ts, mode='full')
                # Get best lag period: peak in correlation, shifted to account for different sized arrays (as in https://stackoverflow.com/questions/49372282/find-the-best-lag-from-the-numpy-correlate-output)
                # Do the shift first by trimming all indices that would lead to a negative lag, or a lag larger than the maximum
                corr = corr[num_time-1:]
                corr = corr[:max_lag+1]
                lag[p] = np.argmax(corr)
            # Now calculate average lag over all points, and save to master array
            avg_lag[n] = np.mean(lag)

        # Calculate average lag over all ensemble members and round to nearest int
        final_lag = int(np.round(np.mean(avg_lag)))
        print 'Optimum lag of ' + str(final_lag) + ' months'

        # Now get ensemble-mean correlation coefficients for each point and each member
        print 'Calculating correlation coefficients'
        mean_r = np.empty(num_ice_pts)
        for p in range(num_ice_pts):
            r_values = np.empty(num_ens)
            for n in range(num_ens):
                # Extract the two timeseries and shift by the correct lag
                forcing_ts = forcing_data[n,:-final_lag]
                ismr_ts = ismr_data[n,final_lag:,p]
                # Now do the linear regression
                r_values[n] = linregress(forcing_ts, ismr_ts)[2]
            # Take mean over ensemble members
            mean_r[p] = np.mean(r_values)
        # Convert this array to lon-lat and fill in the mask
        r_data = np.zeros([grid.ny, grid.nx])
        r_data[mask] = mean_r
        r_data = mask_except_ice(r_data, grid)
        print 'Mean r over all points and ensemble members = '+str(np.mean(r_data))

        # Plot this map
        latlon_plot(r_data, grid, ctype='plusminus', ymax=ymax, vmin=-vmax, vmax=vmax, title=title_head+forcing_titles[m]+'\nat optimum lag of '+str(final_lag)+' months', titlesize=14, figsize=(14,5), fig_name=fig_dir+'correlation_map_ismr_'+forcing_abbrv[m]+'.png')


# Precompute the ensemble mean, annual mean temperature and salinity for each year of the PACE ensemble (excluding spinup).
def precompute_ts_ensemble_mean (sim_dir, grid_dir, out_file, start_year=1920, end_year=2013):

    var_names = ['THETA', 'SALT']
    units = ['degC', 'psu']
    num_var = len(var_names)
    num_ens = len(sim_dir)
    num_years = end_year-start_year+1
    grid = Grid(grid_dir)

    final_data = np.ma.zeros([num_var, num_years, grid.nz, grid.ny, grid.nx])
    for d in sim_dir:
        for year in range(start_year, end_year+1):
            file_path = real_dir(d) + 'output/' + str(year)+ '01/MITgcm/output.nc'
            print 'Reading ' + file_path
            for n in range(num_var):
                final_data[n,year-start_year,:] += read_netcdf(file_path, var_names[n], time_average=True)
    # Divide by number of simulations to get ensemble mean
    final_data /= num_ens

    print 'Writing ' + out_file
    ncfile = NCfile(out_file, grid, 'xyzt')
    ncfile.add_time(np.arange(start_year, end_year+1), units='year')
    for n in range(num_var):
        ncfile.add_variable(var_names[n], final_data[n,:], 'xyzt', units=units)
    ncfile.close()


# Helper function to read and process OHC terms
def read_process_ohc (sim_dir, region='amundsen_shelf', option='d/dt', timeseries_file='timeseries_full.nc', smooth=24, depth=300, base_year_start=1920, base_year_end=1949):

    num_ens = len(sim_dir)
    for n in range(num_ens):
        file_path = real_dir(sim_dir[n]) + 'output/' + timeseries_file
        ohc_tmp = read_netcdf(file_path, region+'_ohc_below_'+str(depth)+'m')
        dohc_adv_tmp = read_netcdf(file_path, region+'_dohc_adv_below_'+str(depth)+'m')
        if n==0:
            # Read the time axis and initialise the arrays
            time = netcdf_time(file_path, monthly=False)
            ohc = ohc_tmp
            dohc_adv = dohc_adv_tmp
        else:
            # Update the arrays
            ohc += ohc_tmp
            dohc_adv += dohc_adv_tmp
    # Now take ensemble mean
    ohc /= num_ens
    dohc_adv /= num_ens

    # Subtract 1920-1949 mean from advective convergence
    t_start, t_end = index_period(time, base_year_start, base_year_end)
    dohc_adv -= np.mean(dohc_adv[t_start:t_end])
    if option == 'd/dt':
        # Differentiate OHC
        dohc = time_derivative(ohc, time)
        # Smooth
        dohc_smooth, time_smooth = moving_average(dohc, smooth, time=time)
        dohc_adv_smooth = moving_average(dohc_adv, smooth)
        return time_smooth, dohc_smooth, dohc_adv_smooth
    elif option == 'trend':
        # Get trend in OHC
        # Need time in seconds
        time_sec = np.array([(t-time[0]).total_seconds() for t in time])
        # Remove spinup
        time_sec = time_sec[t_start:]
        ohc = ohc[t_start:]
        dohc_adv = dohc_adv[t_start:]
        ohc_trend = linregress(time_sec, ohc)[0]
        # Now take average value of dohc_adv
        dohc_adv_avg = np.mean(dohc_adv)
        return ohc_trend, dohc_adv_avg


# Plot the ensemble mean rate of change of ocean heat content below 300 m in the inner shelf, and the contribution from horizontal advection.
def plot_ohc_adv (sim_dir, region='amundsen_shelf', timeseries_file='timeseries_ohc.nc', smooth=24, base_year_start=1920, base_year_end=1949, fig_dir='./'):

    fig_dir = real_dir(fig_dir)

    time_smooth, dohc_smooth, dohc_adv_smooth = read_process_ohc(sim_dir, region=region, timeseries_file=timeseries_file, smooth=smooth, depth=300, base_year_start=base_year_start, base_year_end=base_year_end)
    
    # Get the correlation coefficient between the two timeseries
    r, p = pearsonr(dohc_smooth, dohc_adv_smooth)
    print 'Sum of dOHC: '+str(np.sum(dohc_smooth))
    print 'Sum of dOHC_adv: '+str(np.sum(dohc_adv_smooth))

    # Plot
    timeseries_multi_plot(time_smooth, [dohc_smooth, dohc_adv_smooth], ['Total', 'Horizontal advection'], ['blue', 'red'], title='Ensemble mean rate of change of ocean heat content\nbelow 300m in '+region_names[region]+' (r='+round_to_decimals(r,3)+')', units='GJ/s', vline=base_year_start, fig_name=fig_dir+'timeseries_'+region+'_dohc_adv.png')

    # Now plot the residual
    residual = dohc_smooth-dohc_adv_smooth
    r, p = pearsonr(dohc_smooth, residual)
    print 'Sum of residual: '+str(np.sum(residual))
    timeseries_multi_plot(time_smooth, [dohc_smooth, residual], ['Total', 'Residual'], ['blue', 'green'], title='Ensemble mean rate of change of ocean heat content\nbelow 300m in '+region_names[region]+' (r='+round_to_decimals(r,3)+')', units='GJ/s', vline=base_year_start, fig_name=fig_dir+'timeseries_'+region+'_dohc_residual.png')


# Create an animated T/S diagram of the precomputed ensemble mean conditions on the continental shelf, over the 20th century. 
def ts_animation_pace_shelf (precompute_file, grid_path, region='amundsen_shelf', sim_title='PACE ensemble mean', fig_dir='./', mov_name='ts_diagram.mp4'):    

    fig_dir = real_dir(fig_dir)
    grid = Grid(grid_path)
    time = read_netcdf(precompute_file, 'time')  # In years, not Date objects
    print 'Reading precomputed T and S'
    temp = read_netcdf(precompute_file, 'THETA')
    salt = read_netcdf(precompute_file, 'SALT')
    ts_animation(temp, salt, time, grid, region, sim_title, mov_name=fig_dir+mov_name)


# Plot a T/S diagram of the change in volume between the beginning and end of the simulation (ensemble mean).
def ts_volume_change (precompute_file, grid_path, region='amundsen_shelf', num_years=10, num_bins=1000, sim_title='PACE ensemble mean', fig_name=None):

    grid = Grid(grid_path)
    mask = mask_2d_to_3d(grid.get_region_mask(region), grid)
    print 'Reading precomputed T and S'
    temp = read_netcdf(precompute_file, 'THETA')
    salt = read_netcdf(precompute_file, 'SALT')
    temp_start = np.mean(temp[:num_years,:], axis=0)
    salt_start = np.mean(salt[:num_years,:], axis=0)
    temp_end = np.mean(temp[-num_years:,:], axis=0)
    salt_end = np.mean(salt[-num_years:,:], axis=0)

    # Get bounds for both time periods together
    def get_vmin_vmax (data1, data2):
        vmin = min(np.amin(data1[mask]), np.amin(data2[mask]))
        vmax = max(np.amax(data1[mask]), np.amax(data2[mask]))
        return [vmin, vmax]
    [tmin, tmax] = get_vmin_vmax(temp_start, temp_end)
    [smin, smax] = get_vmin_vmax(salt_start, salt_end)

    # Get the binned volume for both time periods, on the same bins
    volume_start, temp_centres, salt_centres, temp_edges, salt_edges = ts_binning(temp_start, salt_start, grid, mask, num_bins=num_bins, tmin=tmin, tmax=tmax, smin=smin, smax=smax)
    volume_end = ts_binning(temp_end, salt_end, grid, mask, num_bins=num_bins, tmin=tmin, tmax=tmax, smin=smin, smax=smax)[0]
    volume_diff = volume_end - volume_start
    # Take log multiplied by sign
    volume_diff = np.sign(volume_diff)*np.log(np.abs(volume_diff))

    # Get density contours
    salt_2d, temp_2d = np.meshgrid(salt_centres, temp_centres)
    rho = potential_density('MDJWF', salt_2d, temp_2d)
    rho_lev = np.arange(np.ceil(np.amin(rho)*10)/10., np.ceil(np.amax(rho)*10)/10., 0.1)

    # Plot the change in volume
    print 'Plotting'
    cmap, vmin, vmax = set_colours(volume_diff, ctype='plusminus')
    fig, ax = plt.subplots(figsize=(8,6))
    img = ax.pcolormesh(salt_edges, temp_edges, volume_diff, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(img)
    ax.contour(salt_centres, temp_centres, rho, rho_lev, colors='black', linestyles='dotted')
    ax.set_xlim([33.5, 34.74])
    plt.xlabel('Salinity (psu)')
    plt.ylabel('Temperature ('+deg_string+'C)')
    plt.text(.9, .6, r'log of change in volume', ha='center', rotation=-90, transform=fig.transFigure)
    plt.title(sim_title+', last 10y minus first 10y\n'+region_names[region])
    finished_plot(fig, fig_name=fig_name)


# Look at all the heat budget terms in one simulation, below the given depth in the given region.
def heat_budget_analysis (output_dir, region='amundsen_shelf', z0=-300, smooth=24, fig_name=None):

    var = ['ADVx_TH', 'ADVr_TH', 'DFxE_TH', 'DFrE_TH', 'DFrI_TH', 'KPPg_TH', 'oceQsw', 'total', 'TOTTTEND']
    titles = ['advection_xy', 'advection_z', 'diffusion_xy', 'diffusion_z_explicit', 'diffusion_z_implicit', 'kpp', 'shortwave', 'total', 'tendency']
    num_var = len(var)
    file_paths = segment_file_paths(output_dir)
    num_time_total = len(file_paths)*12
    data_int = np.empty([num_var, num_time_total])

    grid = Grid(file_paths[0])
    mask = mask_2d_to_3d(grid.get_region_mask(region), grid, zmax=z0)
    z_edges_3d = z_to_xyz(grid.z_edges, grid)
    dA_3d = xy_to_xyz(grid.dA, grid)
    swfrac = 0.62*np.exp(z_edges_3d[:-1,:]/0.6) + (1-0.62)*np.exp(z_edges_3d[:-1,:]/20.)
    swfrac1 = 0.62*np.exp(z_edges_3d[1:,:]/0.6) + (1-0.62)*np.exp(z_edges_3d[1:,:]/20.)
    rhoConst = 1028.5

    time = None
    for n in range(len(file_paths)):
        print 'Processing ' + file_paths[n]
        time_tmp = netcdf_time(file_paths[n])
        num_time = time_tmp.size
        if time is None:
            time = time_tmp
        else:
            time = np.concatenate((time, time_tmp), axis=0)
        for v in range(num_var):
            print '...'+titles[v]
            if var[v] == 'total':
                # Sum of all previous entries
                data_int[v, n*12:(n+1)*12] = np.sum(data_int[:,n*12:(n+1)*12], axis=0)
            else:
                # Read the variable
                data = read_netcdf(file_paths[n], var[v])
                if var[v] in ['ADVx_TH', 'DFxE_TH']:
                    # There is a second component
                    data_x = data
                    data_y = read_netcdf(file_paths[n], var[v].replace('x', 'y'))
                # Loop over timesteps
                for t in range(num_time):
                    if var[v] in ['ADVx_TH', 'DFxE_TH']:
                        # Get x and y fluxes
                        data_tmp = np.ma.zeros(data_x.shape[1:])
                        data_tmp[:,:-1,:-1] = data_x[t,:,:-1,:-1] - data_x[t,:,:-1,1:] + data_y[t,:,:-1,:-1] - data_y[t,:,1:,:-1]
                    elif var[v] in ['ADVr_TH', 'DFrE_TH', 'DFrI_TH', 'KPPg_TH']:
                        # Get z fluxes
                        data_tmp = np.ma.zeros(data.shape[1:])
                        data_tmp[:-1,:] = data[t,1:,:] - data[t,:-1,:]
                    elif var[v] == 'oceQsw':
                        # Get penetration of SW radiation
                        data_tmp = xy_to_xyz(data[t,:], grid)*(swfrac-swfrac1)*dA_3d/(rhoConst*Cp_sw)
                    elif var[v] == 'TOTTTEND':
                        # Get tendency in correct units
                        data_tmp = data[t,:]/sec_per_day*grid.dV
                    # Mask everywhere outside region
                    data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
                    # Sum over all cells
                    data_int[v, n*12+t] = np.sum(data_tmp)

    # Smooth
    # Start with a dummy variable so we get time trimmed to the right size
    tmp, time_smoothed = moving_average(np.zeros(time.size), smooth, time=time)
    data_smoothed = np.empty([num_var, time_smoothed.size])
    for v in range(num_var):
        data_smoothed[v,:] = moving_average(data_int[v,:], smooth)

    # Plot
    timeseries_multi_plot(time_smoothed, [data_smoothed[v,:] for v in range(num_var)], titles, default_colours(num_var), title='Heat budget in '+region_names[region]+' below '+str(int(-z0))+'m', units=deg_string+r'C m$^3$/s', fig_name=fig_name)


# Plot how the correlation between d/dt OHC and the advective component changes depending on the threshold depth.
def ohc_adv_correlation_vs_depth (sim_dir, timeseries_file='timeseries_ohc_full.nc', smooth=24, base_year_start=1920, base_year_end=1949, fig_name=None):

    regions = ['amundsen_shelf', 'inner_amundsen_shelf']
    colours = ['red', 'blue']
    num_regions = len(regions)
    depths = np.arange(200,700+100,100)
    num_depths = depths.size
    correlation = np.empty([num_regions, num_depths])
    num_ens = len(sim_dir)    

    # Loop over regions
    for n in range(num_regions):
        # Loop over depths
        for k in range(num_depths):
            time_smooth, dohc_smooth, dohc_adv_smooth = read_process_ohc(sim_dir, region=regions[n], timeseries_file=timeseries_file, smooth=smooth, depth=depths[k], base_year_start=base_year_start, base_year_end=base_year_end)
            r, p = pearsonr(dohc_smooth, dohc_adv_smooth)
            correlation[n,k] = r

    fig, ax = plt.subplots()
    for n in range(num_regions):
        ax.plot(depths, correlation[n,:], '-', linewidth=1.5, color=colours[n], label=regions[n])
    ax.legend()
    ax.grid(True)
    plt.xlabel('Cutoff depth (m)')
    plt.ylabel('Correlation')
    plt.title('Correlation between d/dt OHC and the advective contribution')
    finished_plot(fig, fig_name=fig_name)


# Compare the trend in OHC to the time-averaged anomaly of dOHC_adv
# Compare the individual members and the ensemble mean.
def compare_ohc_trends (sim_dir, region='amundsen_shelf', timeseries_file='timeseries_ohc_full.nc', depth=300, base_year_start=1920, base_year_end=1949, fig_name=None):

    num_ens = len(sim_dir)

    # Get trend and anomaly in each member
    ohc_trend = []
    dohc_adv_anom = []
    for d in sim_dir:
        ohc, dohc_adv = read_process_ohc([d], region=region, option='trend', timeseries_file=timeseries_file, depth=depth, base_year_start=base_year_start, base_year_end=base_year_end)
        ohc_trend.append(ohc)
        dohc_adv_anom.append(dohc_adv)
    # Get mean of trends and mean of anomalies
    ohc_trend.append(np.mean(ohc_trend))
    dohc_adv_anom.append(np.mean(dohc_adv_anom))
    # Calculate percent explained by advection
    adv_percent = dohc_adv_anom[-1]/ohc_trend[-1]*100
    print 'Horizontal advection can explain '+str(adv_percent)+'% of OHC trend (ensemble mean)'

    # Plot
    index = -1*np.arange(num_ens+1)
    sim_labels = ['PACE '+str(n+1).zfill(2) for n in range(num_ens)] + ['Mean']
    fig, ax = plt.subplots(figsize=(6,8))
    ax.plot(ohc_trend, index, '*', color='blue', markersize=10, label='Trend in ocean heat content')
    ax.plot(dohc_adv_anom, index, '*', color='red', markersize=10, label='Mean anomaly in horizontal advection')
    ax.set_yticks(index)
    ax.set_yticklabels(sim_labels)
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.13))        
    ax.grid(linestyle='dashed')
    plt.xlabel('GJ/s')
    plt.title(region_names[region]+' below '+str(depth)+'m')
    finished_plot(fig, fig_name=fig_name)


# Plot timeseries of the standard deviation across the ensemble of the given set of variables.
def plot_timeseries_std (var_type, sim_dir, smooth=24, timeseries_file='timeseries.nc', start_year=1920, fig_name=None):

    num_ens = len(sim_dir)
    if var_type == 'ismr':
        var_names = ['getz_melting', 'dotson_crosson_melting', 'thwaites_melting', 'pig_melting', 'cosgrove_melting', 'abbot_melting', 'venable_melting']
        labels = ['Getz', 'Dotson & Crosson', 'Thwaites', 'PIG', 'Cosgrove', 'Abbot', 'Venable']
        title = 'Standard deviation in ice shelf melt rates'
        units = 'm/y'
    elif var_type == 'temp':
        var_names = ['inner_amundsen_shelf_temp_below_500m', 'pine_island_bay_temp_below_500m', 'dotson_bay_temp_below_500m']
        labels = ['Inner shelf', 'Pine Island Bay', 'front of Dotson']
        title = 'Standard deviation in temperature below 500m'
        units = deg_string + 'C'
    elif var_type == 'salt':
        var_names = ['inner_amundsen_shelf_salt_below_500m', 'pine_island_bay_salt_below_500m', 'dotson_bay_salt_below_500m']
        labels = ['Inner shelf', 'Pine Island Bay', 'front of Dotson']
        title = 'Standard deviation in salinity below 500m'
        units = 'psu'
    elif var_type == 'wind':
        var_names = ['amundsen_shelf_break_uwind_avg']
        labels = ['Shelf break']
        title = 'Standard deviation in zonal wind'
        units = 'm/s'

    data_std = []
    for var in var_names:
        data_var = None
        for n in range(num_ens):
            # Read and smooth the data
            file_path = real_dir(sim_dir[n])+'output/'+timeseries_file
            time_tmp = netcdf_time(file_path)
            data_tmp = read_netcdf(real_dir(sim_dir[n])+'output/'+timeseries_file, var)
            data_tmp, time_tmp = moving_average(data_tmp, smooth, time=time_tmp)
            t_start = index_year_start(time_tmp, start_year)
            data_tmp = data_tmp[t_start:]
            time_tmp = time_tmp[t_start:]
            if data_var is None:
                data_var = np.empty([num_ens, data_tmp.size])
                time = time_tmp
            data_var[n,:] = data_tmp
        # Now calculate and save the standard deviation across ensemble members
        data_std.append(np.std(data_var, axis=0))

    # Calculate trends in the standard deviation timeseries
    for v in range(len(var_names)):
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(time.size)/12., data_std[v])
        print var_names[v]+': '+str(slope)+' '+units+'/y, p='+str(p_value)

    # Plot
    timeseries_multi_plot(time, data_std, labels, default_colours(len(var_names)), title=title, units=units, fig_name=fig_name)


# Make a Hovmoller plot of the ensemble mean and standard deviation of the given variable (temperature or salinity) at the given region.
def hovmoller_mean_std (sim_dir, var, region, grid_path, smooth=24, start_year=1920, hovmoller_file='hovmoller.nc', fig_name=None):

    num_ens = len(sim_dir)
    var_name = region+'_'+var
    grid = Grid(grid_path)
    if var == 'temp':
        suptitle = 'Temperature ('+deg_string+'C)'
    elif var == 'salt':
        suptitle = 'Salinity (psu)'
    suptitle += ' at '+region_names[region]

    # Read data
    data = None
    for n in range(num_ens):
        file_path = real_dir(sim_dir[n])+'output/'+hovmoller_file
        time_tmp = netcdf_time(file_path)
        data_tmp = read_netcdf(file_path, var_name)
        if data is None:
            data = np.ma.empty([num_ens, data_tmp.shape[0], data_tmp.shape[1]])
            time = time_tmp
            t_start = index_year_start(time, start_year)
        data[n,:] = data_tmp
    # Calculate mean and standard deviation
    data_mean = np.ma.mean(data, axis=0)
    data_std = np.ma.std(data, axis=0)

    # Plot
    fig, gs, cax_1, cax_2 = set_panels('2x1C2', figsize=(12,7))
    data = [data_mean, data_std]
    titles = ['Ensemble mean', 'Ensemble standard deviation']
    cax = [cax_1, cax_2]
    for i in range(2):
        ax = plt.subplot(gs[i,0])
        img = hovmoller_plot(data[i], time, grid, smooth=smooth, ax=ax, make_cbar=False, title=titles[i], start_t=time[t_start])
        cbar = plt.colorbar(img, cax=cax[i])
        if i == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('')
    plt.suptitle(suptitle, fontsize=22)
    finished_plot(fig, fig_name=fig_name)


# Plot all the bias correction fields used for PACE forcing variables.
def plot_bias_correction_fields (input_dir, grid_dir, fig_dir='./'):

    input_dir = real_dir(input_dir)
    fig_dir = real_dir(fig_dir)
    fnames = ['atemp_offset_PAS', 'aqh_offset_PAS', 'precip_offset_PAS', 'swdown_offset_PAS', 'lwdown_offset_PAS', 'katabatic_scale_PAS_90W', 'katabatic_rotate_PAS_90W']
    num_var = len(fnames)
    ctype = ['plusminus', 'plusminus', 'plusminus', 'plusminus', 'plusminus', 'ratio', 'plusminus']
    titles = [r'$\bf{a}$. Temperature ('+deg_string+'C)', r'$\bf{b}$. Humidity (10$^{-3}$ kg/kg)', r'$\bf{c}$. Precipitation (10$^{-9}$ m/s)', r'$\bf{d}$. SW radiation (W/m$^2$)', r'$\bf{e}$. LW radiation (W/m$^2$)', r'$\bf{f}$. Wind scaling factor (1)', r'$\bf{g}$. Wind rotation angle ($^{\circ}$)']
    factor = [1, 1e3, 1e9, -1, -1, 1, rad2deg]

    grid = Grid(grid_dir)
    data = []
    for n in range(num_var):
        data_tmp = read_binary(input_dir+fnames[n], [grid.nx, grid.ny], 'xy', prec=64)
        data_tmp = mask_land_ice(data_tmp, grid)*factor[n]
        data.append(data_tmp)

    fig, gs, cax = set_panels('2x4-1C7')
    for n in range(num_var):
        ax = plt.subplot(gs[(n+1)/4, (n+1)%4])
        img = latlon_plot(data[n], grid, ax=ax, make_cbar=False, ctype=ctype[n], include_shelf=False, title=titles[n], titlesize=13)
        cbar = plt.colorbar(img, cax=cax[n], orientation='horizontal')
        alternate = n==5
        reduce_cbar_labels(cbar, alternate=alternate)
        if n == 3:
            # Reduce label size
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
        else:
            # Remove ticks from all panels except bottom left
            ax.set_xticks([])
            ax.set_yticks([])
    plt.text(0.15, 0.85, 'Bias correction\nfields for\nPACE forcing', transform=fig.transFigure, ha='center', va='top', fontsize=18)
    finished_plot(fig, fig_name=fig_dir+'bias_corrections.png', dpi=300)


# Plot ensemble mean, spread, and trend for winds, temperature, and ice shelf melting.
def plot_timeseries_3var (base_dir='./', timeseries_file='timeseries_final.nc', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    num_ens = 20
    var_names = ['amundsen_shelf_break_uwind_avg', 'amundsen_shelf_temp_btw_200_700m', 'all_massloss']
    var_titles = [r'$\bf{a}$. Zonal wind over shelf break', r'$\bf{b}$. Temperature on shelf (200-700m)', r'$\bf{c}$. Total basal mass loss from ice shelves']
    var_units = [' m/s', deg_string+'C', ' Gt/y']
    num_var = len(var_names)
    sim_dir = [base_dir+'PAS_PACE'+str(n+1).zfill(2)+'/output/' for n in range(num_ens)] + [base_dir+'PAS_ERA5/output/']
    year_start_pace = 1920
    year_start_era5 = 1979
    year_end = 2013
    base_period = 30*months_per_year
    smooth = 24

    # Read all the data and take 2-year running means
    pace_data = None
    pace_mean = None
    era5_data = None
    for n in range(num_ens+1):
        file_path = sim_dir[n] + timeseries_file
        time_tmp = netcdf_time(file_path, monthly=False)  # It is actually monthly but this keyword would otherwise trigger a 1-month correction which we don't want
        if n == num_ens:
            year_start = year_start_era5
        else:
            year_start = year_start_pace
        t0, tf = index_period(time_tmp, year_start, year_end)
        # Trim
        time_tmp = time_tmp[t0:tf]
        for v in range(num_var):
            data_tmp = read_netcdf(file_path, var_names[v])
            data_tmp = data_tmp[t0:tf]
            data_smooth, time_smooth = moving_average(data_tmp, smooth, time=time_tmp)
            if pace_data is None:
                # Set up master array
                num_time = time_smooth.size
                pace_data = np.empty([num_var, num_ens, num_time])
                pace_mean = np.empty([num_var, num_time])
                pace_time = time_smooth
            if n < num_ens:
                # This is a PACE ensemble member
                pace_data[v,n,:] = data_smooth
                if n == num_ens - 1:
                    # Also save PACE ensemble mean
                    pace_mean[v,:] = np.mean(pace_data[v,:,:], axis=-2)
            elif n == num_ens:
                # This is ERA5
                if era5_data is None:
                    num_time = time_smooth.size
                    era5_data = np.empty([num_var, num_time])
                    era5_time = time_smooth
                era5_data[v,:] = data_smooth

    # Calculate mean trend for each variable (equivalent to trend of mean, but check this)
    slopes = []
    intercepts = []
    # Get time in decades
    time_sec = np.array([(t-pace_time[0]).total_seconds() for t in pace_time])
    time_decades = time_sec/(365*sec_per_day*10)
    for v in range(num_var):
        slope, intercept, r_value, p_value, std_err = linregress(time_decades, pace_mean[v,:])
        slopes.append(slope)
        intercepts.append(intercept)
        print '\n'+var_names[v]
        print 'Trend of mean = '+str(slope)
        # Now do some checking
        slope_members = []        
        for n in range(num_ens):
            slope, intercept, r_value, p_value, std_err = linregress(time_decades, pace_data[v,n,:])
            slope_members.append(slope)
        print 'Mean of trend = '+str(np.mean(slope_members))

    # Calculate mean and standard deviation over first 30 years (across all ensemble members) for each variable
    base_mean = []
    base_std = []
    for v in range(num_var):
        base_mean.append(np.mean(pace_data[v,:,:base_period]))
        base_std.append(np.std(pace_data[v,:,:base_period]))    

    # Set up plot
    fig = plt.figure(figsize=(5.5,12))
    gs = plt.GridSpec(3,1)
    gs.update(left=0.14, right=0.86, bottom=0.08, top=0.97, hspace=0.25)
    for v in range(num_var):
        ax = plt.subplot(gs[v,0])
        ax.tick_params(direction='in')
        # Plot ensemble members in thinner light blue
        labels = ['PACE ensemble'] + [None for n in range(num_ens-1)]
        for n in range(num_ens):
            ax.plot_date(pace_time, pace_data[v,n,:], '-', color='DodgerBlue', label=labels[n], linewidth=1, alpha=0.5)
        # Plot ensemble mean in thicker solid blue, but make sure it will be on above ERA5 at the end
        ax.plot_date(pace_time, pace_mean[v,:], '-', color='blue', label='PACE mean', linewidth=2, zorder=(num_ens+1))
        # Plot ERA5 in thicker solid red
        ax.plot_date(era5_time, era5_data[v,:], '-', color='red', label='ERA5', linewidth=1.5, zorder=(num_ens))
        # Plot trend in thin black on top
        trend_vals = slopes[v]*time_decades + intercepts[v]
        ax.plot_date(pace_time, trend_vals, '-', color='black', linewidth=1, zorder=(num_ens+2))
        # Print trend and r^2
        if v==2:
            trend_str = round_to_decimals(slopes[v],1)
        else:
            trend_str = round_to_decimals(slopes[v],3)
        plt.text(0.02, 0.97, '+'+trend_str+var_units[v]+'/decade', ha='left', va='top', fontsize=12, transform=ax.transAxes)
        ax.set_xlim([pace_time[0], pace_time[-1]])
        ax.set_xticks([datetime.date(y,1,1) for y in np.arange(1930, 2010+1, 10)])
        for label in ax.get_xticklabels()[1::2]:
            label.set_visible(False)
        ax.grid(linestyle='dotted')
        plt.ylabel(var_units[v], fontsize=12)
        if v==2:
            plt.xlabel('Year', fontsize=12)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.33), ncol=3, fontsize=12)
        plt.title(var_titles[v], fontsize=15)
        # Now add second y-axis showing difference in standard deviations from the mean
        limits = ax.get_ylim()
        std_limits = [(l-base_mean[v])/base_std[v] for l in limits]
        ax2 = ax.twinx()
        ax2.tick_params(direction='in')
        ax2.set_ylim(std_limits)
        if v==0:
            ax2.set_ylabel('anomaly in standard deviations', fontsize=12)    
    finished_plot(fig, fig_name=fig_dir+'timeseries_3var.png', dpi=300)


# Calculate the mean trend and ensemble significance for a whole bunch of variables.
def calc_all_trends (base_dir='./', timeseries_file='timeseries_final.nc'):

    num_ens = 20
    var_names = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'getz_massloss', 'dotson_massloss', 'thwaites_massloss', 'pig_massloss', 'cosgrove_massloss', 'abbot_massloss', 'venable_massloss', 'amundsen_shelf_temp_btw_200_700m', 'pine_island_bay_temp_btw_200_700m', 'dotson_bay_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'pine_island_bay_salt_btw_200_700m', 'dotson_bay_salt_btw_200_700m', 'amundsen_shelf_thermocline', 'pine_island_bay_thermocline', 'dotson_bay_thermocline', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg']
    units = ['m/s', 'Gt/y', 'Gt/y', 'Gt/y', 'Gt/y', 'Gt/y', 'Gt/y', 'Gt/y', 'Gt/y', 'degC', 'degC', 'degC', 'psu', 'psu', 'psu', 'm', 'm', 'm', 'degC', 'psu']
    num_var = len(var_names)
    base_dir = real_dir(base_dir)
    sim_dir = [base_dir+'PAS_PACE'+str(n+1).zfill(2)+'/output/' for n in range(num_ens)]
    year_start = 1920
    base_year_end = 1949
    smooth = 24

    time = None
    for v in range(num_var):
        trends = []
        for n in range(num_ens):
            # Read data
            file_path = sim_dir[n] + timeseries_file
            if time is None:
                time_tmp = netcdf_time(file_path, monthly=False)
                t0 = index_year_start(time_tmp, year_start)
                time_tmp = time_tmp[t0:]
                t_base = index_year_end(time_tmp, base_year_end)
            data_tmp = read_netcdf(file_path, var_names[v])[t0:]
            if var_names[v].endswith('massloss'):
                # Get mean value over baseline period
                data_base_mean = np.mean(data_tmp[:t_base])
            # Smooth
            if time is None:
                data, time = moving_average(data_tmp, smooth, time=time_tmp)
                # Get time in decades
                time_sec = np.array([(t-time[0]).total_seconds() for t in time])
                time_decades = time_sec/(365*sec_per_day*10)
            else:
                data = moving_average(data_tmp, smooth)
            if var_names[v].endswith('massloss'):
                # Convert to percent of baseline value
                data = data/data_base_mean*100
                units[v] = '%'
            # Calculate trend
            slope, intercept, r_value, p_value, std_err = linregress(time_decades, data)
            trends.append(slope)
        # Calculate significance
        p_val = ttest_1samp(trends, 0)[1]
        sig = (1-p_val)*100
        print var_names[v]+': trend='+str(np.mean(trends))+' '+units[v]+'/decade, significance='+str(sig)


# Plot sensitivity of temperature trend to convection.
def plot_temp_trend_vs_cutoff (base_dir='./', timeseries_file='timeseries_final.nc', fig_dir='./'):

    num_ens = 20
    var_name = 'amundsen_shelf_temp_btw_200_700m'
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    sim_dir = [base_dir+'PAS_PACE'+str(n+1).zfill(2)+'/output/' for n in range(num_ens)]
    year_start = 1920
    smooth = 24
    max_cutoff = -0.2
    num_cutoff = 50

    time = None
    temp = None
    for n in range(num_ens):
        # Read data
        file_path = sim_dir[n] + timeseries_file
        if time is None:
            time_tmp = netcdf_time(file_path, monthly=False)
            t0 = index_year_start(time_tmp, year_start)
            time_tmp = time_tmp[t0:]
        temp_tmp = read_netcdf(file_path, var_name)[t0:]
        # Smooth
        if time is None:
            temp_smooth, time = moving_average(temp_tmp, smooth, time=time_tmp)
            num_time = time.size
            temp = np.empty([num_ens, num_time])
            temp[n,:] = temp_smooth
            # Get time in decades
            time_sec = np.array([(t-time[0]).total_seconds() for t in time])
            time_decades = time_sec/(365*sec_per_day*10)
        else:
            temp[n,:] = moving_average(temp_tmp, smooth)

    # Get range of cutoff temperatures
    cutoff_temp = np.linspace(np.amin(temp), max_cutoff, num=num_cutoff)
    # Calculate trends and significance for each cutoff
    all_trends = np.empty([num_ens, num_cutoff])
    sig = np.empty(num_cutoff)
    for m in range(num_cutoff):
        # Calculate trend for each ensemble member
        for n in range(num_ens):
            # Extract values where temperature exceeds this cutoff
            index = temp[n,:] > cutoff_temp[m]
            all_trends[n,m] = linregress(time_decades[index], temp[n,index])[0]
        # Now calculate significance of ensemble
        p_val = ttest_1samp(all_trends[:,m], 0)[1]
        sig[m] = (1-p_val)*100
    mean_trends = np.mean(all_trends, axis=0)

    # Plot
    fig = plt.figure(figsize=(7,5))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.12, right=0.97, bottom=0.12, top=0.8)
    # Cutoff temperature vs trends (individual and mean)
    ax = plt.subplot(gs[0,0])
    for n in range(num_ens):
        ax.plot(cutoff_temp, all_trends[n,:], linewidth=1)
    ax.plot(cutoff_temp, mean_trends, linewidth=2, color='black')
    ax.set_xlim([cutoff_temp[0], cutoff_temp[-1]])
    plt.xlabel('Cutoff temperature ('+deg_string+'C)', fontsize=12)
    plt.ylabel(deg_string+'C/decade', fontsize=12)
    plt.title('Trends (colours) and ensemble mean (black)', fontsize=14)
    ax.grid(linestyle='dotted')
    # Cutoff temperature vs significance
    '''ax = plt.subplot(gs[0,1])
    ax.plot(cutoff_temp, sig, linewidth=2, color='black')
    ax.set_xlim([cutoff_temp[0], cutoff_temp[-1]])
    ax.set_ylim([99.99, 100])
    plt.ylabel('%', fontsize=12)
    plt.title(r'$\bf{b}$. Significance of ensemble trend', fontsize=14)
    ax.grid(linestyle='dotted')'''
    plt.suptitle('Sensitivity of temperature trend on shelf\n(200-700m) to convection', fontsize=18)
    finished_plot(fig, fig_name=fig_dir+'temp_trend_vs_cutoff.png', dpi=300)


# Helper function to find the years with observations in Pierre's climatology
def years_with_obs (obs_dir):

    file_head = 'ASEctd_griddedMean'
    file_tail = '.mat'
    obs_years = []
    for fname in os.listdir(obs_dir):
        if fname.startswith(file_head) and fname.endswith(file_tail) and not (fname == file_head+file_tail):
            # This is a file with 1 year of data. Extract the year
            year = int(fname[len(file_head):-len(file_tail)])
            obs_years.append(year)
    obs_years.sort()
    return obs_years


# Make a 3x2 plot showing temperature and salinity casts in 3 regions, comparing ERA5-forced output to Pierre's climatology.
def plot_ts_casts_obs (obs_dir, base_dir='./', fig_dir='./'):

    obs_dir = real_dir(obs_dir)
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    model_file = base_dir + 'PAS_ERA5/output/hovmoller.nc'
    grid_path = base_dir + 'PAS_grid/'
    grid = Grid(grid_path)
    obs_file_head = obs_dir + 'ASEctd_griddedMean'
    obs_file_tail = '.mat'
    obs_years = years_with_obs(obs_dir)
    obs_num_years = len(obs_years)
    regions = ['amundsen_west_shelf_break', 'pine_island_bay', 'dotson_bay']
    region_titles = [r'$\bf{a}$. Shelf break trough', r'$\bf{b}$. Pine Island Bay', r'$\bf{c}$. Dotson front']
    num_regions = len(regions)
    model_var = ['temp', 'salt']
    obs_var = ['PTmean', 'Smean']
    var_titles = ['Temperature', 'Salinity']
    var_units = [deg_string+'C', 'psu']
    num_var = len(model_var)
    model_year0 = 1947
    model_start_year = 1979
    model_end_year = 2019
    model_num_time = (model_end_year-model_year0+1)*months_per_year
    model_years = np.arange(model_start_year, model_end_year+1)
    model_num_years = model_years.size
    obs_smooth = 51
    obs_smooth_below = -100

    # Read precomputed Hovmollers from file
    print 'Reading model output'
    model_hov = np.ma.empty([num_regions, num_var, model_num_time, grid.nz])
    for r in range(num_regions):
        for v in range(num_var):
            model_hov[r,v,:] = read_netcdf(model_file, regions[r]+'_'+model_var[v])
    # For each of the years with observations, average over January-February
    model_data = np.ma.empty([num_regions, num_var, model_num_years, grid.nz])
    for t in range(model_num_years):
        # Find time index of that January
        t_jan = (model_years[t]-model_year0)*months_per_year
        # Weight with days per month
        ndays = np.array([days_per_month(m+1,model_years[t]) for m in range(2)])
        model_data[:,:,t,:] = np.sum(model_hov[:,:,t_jan-1:t_jan+1,:]*ndays[None,None,:,None], axis=2)/np.sum(ndays)

    # Now read observations
    print 'Reading observations'
    obs_data = None
    for t in range(obs_num_years):
        print '...'+str(obs_years[t])
        f = loadmat(obs_file_head+str(obs_years[t])+obs_file_tail)
        if obs_data is None:
            # This is the first year: read the grid and set up array
            obs_lon, obs_lat, obs_depth, obs_dA, obs_dV = pierre_obs_grid(f, xy_dim=2, z_dim=1, dA_dim=3)
            # Get MITgcm's ice mask on this grid
            obs_ice_mask = interp_reg_xy(grid.lon_1d, grid.lat_1d, grid.ice_mask.astype(float), obs_lon, obs_lat)
            obs_ice_mask[obs_ice_mask < 0.5] = 0
            obs_ice_mask[obs_ice_mask >= 0.5] = 1
            obs_ice_mask = obs_ice_mask.astype(bool)
            obs_data = np.ma.empty([num_regions, num_var, obs_num_years, obs_depth.size])
        for v in range(num_var):
            # Read 3D temp or salinity
            obs_var_3d = np.transpose(f[obs_var[v]])
            obs_var_3d = np.ma.masked_where(np.isnan(obs_var_3d), obs_var_3d)
            for r in range(num_regions):
                # Area-average over the given region
                [xmin, xmax, ymin, ymax] = region_bounds[regions[r]]
                # Make a mask which is 1 only within these bounds where there is data, and excluding cavities
                mask = (obs_lon >= xmin)*(obs_lon <= xmax)*(obs_lat >= ymin)*(obs_lat <= ymax)*np.invert(obs_ice_mask)
                mask = xy_to_xyz(mask, [obs_lat.size, obs_lon.size, obs_depth.size]).astype(float)
                mask[obs_var_3d.mask] = 0
                obs_profile = np.sum(obs_var_3d*obs_dA*mask, axis=(1,2))/np.sum(obs_dA*mask, axis=(1,2))
                # Make a smoothed version and overwrite with it below 100m depth
                obs_profile_smoothed = moving_average(obs_profile, obs_smooth, keep_edges=True)
                index = obs_depth < obs_smooth_below
                obs_profile[index] = obs_profile_smoothed[index]
                obs_data[r,v,t,:] = obs_profile

    # Calculate time-mean and standard deviation from each source
    model_mean = np.mean(model_data, axis=-2)
    obs_mean = np.mean(obs_data, axis=-2)
    model_std = np.std(model_data, axis=-2)
    obs_std = np.std(obs_data, axis=-2)
    # Also make depth positive
    obs_depth *= -1
    model_depth = -grid.z

    # Plot
    fig = plt.figure(figsize=(7,12))
    gs = plt.GridSpec(3,25)
    gs.update(left=0.1, right=0.98, bottom=0.12, top=0.93, wspace=0.2, hspace=0.4)
    for r in range(num_regions):
        for v in range(num_var):
            # Choose first 8 panels and merge them (leaving 1 empty between variables)
            ax = plt.subplot(gs[r,v*13:v*13+8])
            ax.tick_params(direction='in')
            ax.grid(linestyle='dotted')
            # Plot each year of observations in thin grey
            for t in range(obs_num_years):
                ax.plot(obs_data[r,v,t,:], obs_depth, color='DimGrey', linewidth=0.5, label=('Observations (each year)' if t==0 else None))
            # Plot observational mean in thick black, but make sure it will be second from the top
            ax.plot(obs_mean[r,v,:], obs_depth, color='black', linewidth=1.5, label='Observations (mean/std)', zorder=obs_num_years+model_num_years)
            # Plot each year of model output in thin light blue
            for t in range(model_num_years):
                ax.plot(model_data[r,v,t,:], model_depth, color='DodgerBlue', linewidth=0.5, label=('Model (each year)' if t==0 else None))
            # Plot model mean in thick blue        
            ax.plot(model_mean[r,v,:], model_depth, color='blue', linewidth=1.5, label='Model (mean/std)')
            # Find the deepest unmasked depth where there is data from both model and obs
            y_deep = min(np.amax(np.ma.masked_where(obs_mean[r,v,:].mask, obs_depth)), np.amax(np.ma.masked_where(model_mean[r,v,:].mask, model_depth)))
            ax.set_ylim([y_deep,0])
            if v==0 and r==0:
                plt.ylabel('Depth (m)', fontsize=12)
            if v==1:
                ax.set_yticklabels([])
            if r == num_regions-1:
                plt.xlabel(var_units[v], fontsize=12)
            plt.title(var_titles[v], fontsize=14)
            if v==1 and r==2:
                plt.legend(loc='lower center', bbox_to_anchor=(-0.1, -0.53), ncol=2, fontsize=12)
            if v==0 and r==2:
                # Remove the last tick label so it doesn't get too close
                label = ax.get_xticklabels()[-1]
                label.set_visible(False)
            # Now plot standard deviations
            # Choose next 4 panels and merge them
            ax2 = plt.subplot(gs[r,v*13+8:v*13+12])
            ax2.tick_params(direction='in')
            ax2.grid(linestyle='dotted')
            ax2.plot(obs_std[r,v,:], obs_depth, color='black', linewidth=1.5)
            ax2.plot(model_std[r,v,:], model_depth, color='blue', linewidth=1.5)
            ax2.set_yticklabels([])
            ax2.set_ylim([y_deep,0])
            # Overwrite the labels so there are no unnecessary decimals - otherwise you get an overlap of labels at 0
            xticks = ax2.get_xticks()
            ax2.set_xticklabels([round_to_decimals(tick,1) for tick in xticks])
            plt.title('std', fontsize=12)        
        plt.text(0.5, 0.985-0.3*r, region_titles[r], ha='center', va='top', fontsize=18, transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_dir+'ts_casts_obs.png', dpi=300)


# Make a 2-panel timeseries plot showing ERA5-forced temperature (200-700m) in Pine Island Bay and Dotson, with Pierre's obs as markers on top.
def plot_temp_timeseries_obs (obs_dir, base_dir='./', fig_dir='./'):

    obs_dir = real_dir(obs_dir)
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    model_file = base_dir + 'PAS_ERA5/output/timeseries.nc'
    grid_path = base_dir + 'PAS_grid/'
    grid = Grid(grid_path)
    obs_file_head = obs_dir + 'ASEctd_griddedMean'
    obs_file_tail = '.mat'
    obs_years = years_with_obs(obs_dir)
    obs_num_years = len(obs_years)
    regions = ['pine_island_bay', 'dotson_bay']
    region_titles = [r'$\bf{a}$. Pine Island Bay', r'$\bf{b}$. Dotson front']
    depth_key = ['btw_200_700m', 'below_700m']
    depth_titles = [' (200-700m)', ' (below 700m)']
    num_depths = len(depth_key)
    depth_colours = ['blue', 'red']
    num_regions = len(regions)
    start_year = 1979
    z_shallow = -200
    z_deep = -700

    # Read model timeseries
    print 'Reading model timeseries'
    model_temp = None
    time = netcdf_time(model_file, monthly=False)  # It is actually monthly but we don't want to advance by a month because that's already been done in the precomputation
    t0 = index_year_start(time, start_year)
    time = time[t0:]
    for n in range(num_regions):
        for k in range(num_depths):
            temp_tmp = read_netcdf(model_file, regions[n]+'_temp_'+depth_key[k])[t0:]
            if model_temp is None:
                model_temp = np.empty([num_regions, num_depths, temp_tmp.size])
            model_temp[n,k,:] = temp_tmp

    # Read observations for each year, averaged over each region
    print 'Reading observations'
    obs_temp = None
    obs_date = []
    for t in range(obs_num_years):
        print '...'+str(obs_years[t])
        f = loadmat(obs_file_head+str(obs_years[t])+obs_file_tail)
        if obs_temp is None:
            # This is the first year: read the grid and set up array
            obs_lon, obs_lat, obs_depth, obs_dA, obs_dV = pierre_obs_grid(f, xy_dim=3, z_dim=3)
            # Get MITgcm's ice mask on this grid
            obs_ice_mask = interp_reg_xy(grid.lon_1d, grid.lat_1d, grid.ice_mask.astype(float), obs_lon[0,:], obs_lat[0,:])
            obs_ice_mask[obs_ice_mask < 0.5] = 0
            obs_ice_mask[obs_ice_mask >= 0.5] = 1
            obs_ice_mask = xy_to_xyz(obs_ice_mask.astype(bool), [obs_lon.shape[2], obs_lon.shape[1], obs_lon.shape[0]])
            obs_temp = np.ma.empty([num_regions, num_depths, obs_num_years])
        # Read 3D temperature
        obs_temp_3d = np.transpose(f['PTmean'])
        obs_temp_3d = np.ma.masked_where(np.isnan(obs_temp_3d), obs_temp_3d)
        for n in range(num_regions):
            # Volume-average over the given region and depth bounds, excluding cavities and regions with no data
            [xmin, xmax, ymin, ymax] = region_bounds[regions[n]]
            mask = (obs_lon >= xmin)*(obs_lon <= xmax)*(obs_lat >= ymin)*(obs_lat <= ymax)*np.invert(obs_ice_mask)
            mask = mask.astype(float)
            mask[obs_temp_3d.mask] = 0
            masks = [mask*(obs_depth >= z_deep)*(obs_depth <= z_shallow), mask*(obs_depth < z_deep)]
            for k in range(num_depths):
                if np.count_nonzero(masks[k])==0:
                    # There are no data points for this region in this year.
                    obs_temp[n,k,t] = np.ma.masked
                else:
                    obs_temp[n,k,t] = np.sum(obs_temp_3d*obs_dV*masks[k])/np.sum(obs_dV*masks[k])
        # Assume the date is 1 February
        obs_date.append(datetime.date(obs_years[t],2,1))

    # Plot
    fig = plt.figure(figsize=(7,7))
    gs = plt.GridSpec(2,1)
    gs.update(left=0.1, right=0.98, bottom=0.15, top=0.95, hspace=0.25)
    for n in range(num_regions):
        ax = plt.subplot(gs[n,0])
        ax.grid(linestyle='dotted')
        for k in range(num_depths):
            # Plot model timeseries
            ax.plot_date(time, model_temp[n,k,:], '-', color=depth_colours[k], linewidth=1, label='Model'+depth_titles[k])
        for k in range(num_depths):  # Second loop so we get the right legend order
            # Plot observations as points on top
            ax.plot_date(obs_date, obs_temp[n,k,:], 'o', color=depth_colours[k], markersize=4, label='Observations'+depth_titles[k])
        ax.set_xlim([time[0], time[-1]])
        ax.set_xticks([datetime.date(y,1,1) for y in np.arange(1980,2020,5)])
        plt.title(region_titles[n], fontsize=16)
        if n==0:
            plt.ylabel('Temperature ('+deg_string+'C)', fontsize=12)
        if n==num_regions-1:
            plt.xlabel('Year', fontsize=12)        
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.44), ncol=2, fontsize=12)
    finished_plot(fig, fig_name=fig_dir+'temp_timeseries_obs.png', dpi=300)
    
         
# Plot timeseries of mass loss from PIG, Dotson, and Getz for the given simulation, with observational estimates overlaid on top.
def plot_ismr_timeseries_obs (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    model_file = base_dir + 'PAS_ERA5/output/timeseries.nc'
    start_year = 1979
    shelf = ['pig', 'dotson', 'getz']
    shelf_titles = [r'$\bf{a}$. Pine Island Ice Shelf', r'$\bf{b}$. Dotson Ice Shelf', r'$\bf{c}$. Getz Ice Shelf']
    num_shelves = len(shelf)
    obs = [pig_melt_years, dotson_melt_years, getz_melt_years]

    # Read data and trim the spinup
    time = netcdf_time(model_file, monthly=False)
    t0 = index_year_start(time, start_year)
    time = time[t0:]
    model_melt = np.empty([num_shelves, time.size])
    for n in range(num_shelves):
        model_melt[n,:] = read_netcdf(model_file, shelf[n]+'_massloss')[t0:]

    # Set up the plot
    fig = plt.figure(figsize=(8,9))
    gs = plt.GridSpec(3,1)
    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.95, hspace=0.3)
    for n in range(num_shelves):
        ax = plt.subplot(gs[n,0])
        # Plot the model timeseries
        ax.plot_date(time, model_melt[n,:], '-', color='blue', label='Model')
        # Loop over observational years and plot the range
        num_obs = len(obs[n]['year'])
        for t in range(num_obs):
            # Plot all observations on 1 Feb
            obs_date = datetime.date(obs[n]['year'][t], 2, 1)
            ax.errorbar(obs_date, obs[n]['melt'][t], yerr=obs[n]['err'][t], fmt='none', color='black', capsize=3, label=('Observations' if t==0 else None))
        ax.grid(linestyle='dotted')
        ax.set_xlim([time[0],time[-1]])
        ax.set_xticks([datetime.date(y,1,1) for y in np.arange(1980,2020,5)])
        ax.set_title(shelf_titles[n], fontsize=16)
        if n == 0:
            ax.set_ylabel('Basal mass loss (Gt/y)', fontsize=14)
        if n == num_shelves-1:
            ax.set_xlabel('Year', fontsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.44), ncol=2, fontsize=12)
    finished_plot(fig, fig_name=fig_dir+'ismr_timeseries_obs.png', dpi=300)


# Helper function to construct the NSDIC file name for the given year and month.
def nsidc_fname (year, month):

    fname = 'seaice_conc_monthly_sh'
    if year < 1992:
        fname += '_f08_'
    elif year < 1995 or (year == 1995 and month < 10):
        fname += '_f11_'
    elif year < 2008:
        fname += '_f13_'
    else:
        fname += '_f17_'
    fname += str(year) + str(month).zfill(2) + '_v03r01.nc'
    return fname


# Plot seasonal averages of sea ice concentration in the ERA5-forced run versus NSIDC observations.
def plot_aice_seasonal_obs (nsidc_dir, base_dir='./', fig_dir='./'):

    nsidc_dir = real_dir(nsidc_dir)
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    model_dir = base_dir + 'PAS_ERA5/output/'
    grid_path = base_dir + 'PAS_grid/'
    grid = Grid(grid_path)
    start_year = 1988
    end_year = 2019
    season_months = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    season_titles = ['DJF', 'MAM', 'JJA', 'SON']
    num_seasons = len(season_titles)
    [xmin, xmax, ymin, ymax] = [grid.lon_1d[0], grid.lon_1d[-1], grid.lat_1d[0], grid.lat_1d[-1]]
    [vmin, vmax] = [0, 1]

    # Read and seasonally average model output
    print 'Reading model output'
    model_aice = np.zeros([num_seasons, grid.ny, grid.nx])
    ndays_int = np.zeros(num_seasons)
    for year in range(start_year, end_year+1):
        file_path = model_dir + str(year) + '01/MITgcm/output.nc'
        model_aice_tmp = read_netcdf(file_path, 'SIarea')
        for n in range(num_seasons):
            for month in season_months[n]:
                ndays = days_per_month(month, year)
                model_aice[n,:] += model_aice_tmp[month-1,:]*ndays
                ndays_int[n] += ndays
    model_aice /= ndays_int[:,None,None]
    model_aice = mask_land_ice(model_aice, grid, time_dependent=True)

    # Read and seasonally average NSIDC data
    print 'Reading NSIDC obs'
    nsidc_aice = None
    ndays_int = np.zeros(num_seasons)
    for year in range(start_year, end_year+1):
        print '...'+str(year)
        for n in range(num_seasons):
            for month in season_months[n]:
                if year == 1988 and month == 1:
                    # Missing data for this month
                    continue
                file_path = nsidc_dir + nsidc_fname(year, month)
                if nsidc_aice is None:
                    # Read grid and set up the master array
                    nsidc_lon = read_netcdf(file_path, 'longitude')
                    nsidc_lat = read_netcdf(file_path, 'latitude')
                    # Mask outside the MITgcm region, with a little bit of leeway so there's no gaps
                    mask = np.invert((nsidc_lon >= xmin-1)*(nsidc_lon <= xmax+1)*(nsidc_lat >= ymin-1)*(nsidc_lat <= ymax+1))
                    nsidc_aice = np.ma.zeros([num_seasons, nsidc_lat.shape[0], nsidc_lat.shape[1]])
                nsidc_aice_tmp = np.ma.masked_where(mask, np.squeeze(read_netcdf(file_path, 'seaice_conc_monthly_cdr')))
                ndays = days_per_month(month, year)
                nsidc_aice[n,:] += nsidc_aice_tmp*ndays
                ndays_int[n] += ndays
    nsidc_aice /= ndays_int[:,None,None]

    # Plot
    fig = plt.figure(figsize=(6,8))
    gs = plt.GridSpec(4,2)
    gs.update(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.02, wspace=0.02)
    cax = fig.add_axes([0.3, 0.03, 0.4, 0.02])
    for n in range(num_seasons):
        # Plot model
        ax = plt.subplot(gs[n,0])
        img = latlon_plot(model_aice[n,:], grid, ax=ax, include_shelf=False, make_cbar=False, vmin=vmin, vmax=vmax)
        if n == 0:
            ax.set_title('Model', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        # Plot obs
        ax = plt.subplot(gs[n,1])
        shade_background(ax)
        ax.pcolormesh(nsidc_lon, nsidc_lat, nsidc_aice[n,:], vmin=vmin, vmax=vmax, cmap='jet')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        if n == 0:
            ax.set_title('Observations', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        # Season name on left
        plt.text(0.11, 0.815-0.21*n, season_titles[n], fontsize=16, ha='right', va='center', transform=fig.transFigure)
        # Colourbar below
        if n == num_seasons - 1:
            cbar = plt.colorbar(img, cax=cax, orientation='horizontal', ticks=np.arange(0, 1.25, 0.25))
    plt.suptitle('Sea ice concentration ('+str(start_year)+'-'+str(end_year)+')', fontsize=18)
    finished_plot(fig, fig_name=fig_dir+'aice_seasonal_obs.png', dpi=300)

    
# Plot timeseries of the number of ensemble members which are unusually warm (in the top 25% for all members and all years) and unusually cold (in the bottom 25%).
def plot_warm_cold_years (base_dir='./', timeseries_file='timeseries_final.nc', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    num_ens = 20
    sim_dir = [base_dir+'PAS_PACE'+str(n+1).zfill(2)+'/output/' for n in range(num_ens)]
    var_name = 'amundsen_shelf_temp_btw_200_700m'
    percentile = 25
    start_year = 1920
    end_year = 2013
    time_years = np.arange(start_year, end_year+1)
    num_years = time_years.size

    temp = np.empty([num_ens, num_years])
    t0 = None
    for n in range(num_ens):
        # Read and annually-average data
        file_path = real_dir(sim_dir[n])+timeseries_file
        if t0 is None:
            time = netcdf_time(file_path, monthly=False)
            t0 = index_year_start(time, start_year)
        temp_tmp = read_netcdf(file_path, var_name)[t0:]
        for t in range(num_years):
            temp[n,t] = average_12_months(temp_tmp, t*12, calendar='noleap')

    # Calculate percentiles and number of cold and warm years
    cutoff_warm = np.percentile(temp, 100-percentile)
    cutoff_cold = np.percentile(temp, percentile)
    print 'warm cutoff='+str(cutoff_warm)+' degC'
    print 'cold cutoff='+str(cutoff_cold)+' degC'
    num_warm = np.sum((temp > cutoff_warm).astype(float), axis=0)
    num_cold = np.sum((temp < cutoff_cold).astype(float), axis=0)
    # Calculate trends
    for data, string in zip([num_warm, num_cold], ['warm', 'cold']):
        slope, intercept, r_value, p_value, std_err = linregress(time_years, data)
        print string+' trend: '+str(slope*10)+' members/decade, significance='+str((1-p_value)*100)

    # Plot
    data_plot = [num_cold, num_warm]
    titles = [r'$\bf{a}$. Chance of cold year', r'$\bf{b}$. Chance of warm year']
    ytitles = ['# members colder than\n25$^{\mathrm{th}}$ percentile', '# members warmer than\n75$^{\mathrm{th}}$ percentile']
    fig = plt.figure(figsize=(6,7))
    gs = plt.GridSpec(2,1)
    gs.update(left=0.15, right=0.9, bottom=0.08, top=0.95, hspace=0.25)
    for n in range(2):
        ax = plt.subplot(gs[n,0])
        ax.bar(time_years, data_plot[n])
        ax.set_xlim([start_year, end_year])
        ax.set_ylim([0, num_ens])
        ax.set_yticks(np.arange(0, num_ens+5, 5))
        if n==1:
            plt.xlabel('Year', fontsize=12)
        plt.ylabel(ytitles[n], fontsize=12)
        plt.title(titles[n], fontsize=16)
        ax.grid(linestyle='dotted')
        ax2 = ax.twinx()
        ax2.set_ylim([0, 100])
        ax2.set_yticks(np.arange(0, 125, 25))
        ax2.set_ylabel('%', fontsize=12)
    finished_plot(fig, fig_name=fig_dir+'warm_cold_years.png', dpi=300)


# Precompute trends in heat advection across the ensemble. Call for key='x', 'y'
def precompute_adv_trend (key, base_dir='./'):

    base_dir = real_dir(base_dir)
    num_ens = 20
    sim_dir = [base_dir+'PAS_PACE'+str(n+1).zfill(2) for n in range(num_ens)]
    grid_path = base_dir + 'PAS_grid/'
    var_name = 'ADV'+key+'_TH'
    make_trend_file(var_name, 'all', sim_dir, grid_path, base_dir+var_name+'_trend.nc')


# Precompute trends in heat budget terms across the ensemble.
def precompute_heat_budget_trend (key, base_dir='./'):

    base_dir = real_dir(base_dir)
    num_ens = 20
    sim_dir = [base_dir+'PAS_PACE'+str(n+1).zfill(2) for n in range(num_ens)]
    grid_path = base_dir + 'PAS_grid/'
    for var_name in ['advection_3d', 'diffusion_3d']:
        make_trend_file(var_name, 'all', sim_dir, grid_path, base_dir+var_name+'_trend.nc')


# Plot anomalies in the non-zero heat budget terms for the first ensemble member.
def plot_test_heat_budget (base_dir='./', fig_name=None):

    base_dir = real_dir(base_dir)
    grid_path = base_dir + 'PAS_grid/'
    start_year = 1920
    base_year_end = 1949
    end_year = 2013
    smooth = 24
    z0 = -200
    region = 'amundsen_shelf'
    output_dir = base_dir+'PAS_PACE01/output/'
    var = ['ADVx_TH', 'ADVr_TH', 'DFrI_TH', 'KPPg_TH', 'oceQsw', 'total']
    num_var = len(var)
    titles = ['3D advection', 'Vertical diffusion + KPP', 'Shortwave', 'Total']

    segment_dir = [str(year)+'01' for year in range(start_year,end_year+1)]
    file_paths = segment_file_paths(output_dir, segment_dir=segment_dir)
    num_time_total = len(file_paths)*12
    num_time_base = (base_year_end-start_year+1)*12
    data_int = np.empty([num_var, num_time_total])

    grid = Grid(grid_path)
    mask = mask_2d_to_3d(grid.get_region_mask(region), grid, zmax=z0)
    z_edges_3d = z_to_xyz(grid.z_edges, grid)
    dA_3d = xy_to_xyz(grid.dA, grid)
    swfrac = 0.62*np.exp(z_edges_3d[:-1,:]/0.6) + (1-0.62)*np.exp(z_edges_3d[:-1,:]/20.)
    swfrac1 = 0.62*np.exp(z_edges_3d[1:,:]/0.6) + (1-0.62)*np.exp(z_edges_3d[1:,:]/20.)
    rhoConst = 1028.5

    time = None
    for n in range(len(file_paths)):
        print 'Processing ' + file_paths[n]
        time_tmp = netcdf_time(file_paths[n])
        num_time = time_tmp.size
        if time is None:
            time = time_tmp
        else:
            time = np.concatenate((time, time_tmp), axis=0)
        for v in range(num_var):
            print '...'+titles[v]
            if var[v] == 'total':
                # Sum of all previous entries
                data_int[v, n*12:(n+1)*12] = np.sum(data_int[:,n*12:(n+1)*12], axis=0)
            else:
                # Read the variable
                data = read_netcdf(file_paths[n], var[v])
                if var[v] == 'ADVx_TH':
                    # There is a second component
                    data_x = data
                    data_y = read_netcdf(file_paths[n], var[v].replace('x', 'y'))
                # Loop over timesteps
                for t in range(num_time):
                    if var[v] == 'ADVx_TH':
                        # Get x and y fluxes
                        data_tmp = np.ma.zeros(data_x.shape[1:])
                        data_tmp[:,:-1,:-1] = data_x[t,:,:-1,:-1] - data_x[t,:,:-1,1:] + data_y[t,:,:-1,:-1] - data_y[t,:,1:,:-1]
                    elif var[v] in ['ADVr_TH', 'DFrI_TH', 'KPPg_TH']:
                        # Get z fluxes
                        data_tmp = np.ma.zeros(data.shape[1:])
                        data_tmp[:-1,:] = data[t,1:,:] - data[t,:-1,:]
                    elif var[v] == 'oceQsw':
                        # Get penetration of SW radiation
                        data_tmp = xy_to_xyz(data[t,:], grid)*(swfrac-swfrac1)*dA_3d/(rhoConst*Cp_sw)
                    # Mask everywhere outside region
                    data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
                    # Sum over all cells
                    data_int[v, n*12+t] = np.sum(data_tmp)
    # Subtract the mean over the base period
    data_mean = np.mean(data_int[:,:num_time_base], axis=1)
    data_int -= data_mean[:,None]
    # Time-integrate
    data_tint = np.empty(data_int.shape)
    for v in range(num_var):
        data_tint[v,:] = time_integral(data_int[v,:], time)

    # Smooth
    # Start with a dummy variable so we get time trimmed to the right size
    tmp, time_smoothed = moving_average(np.zeros(time.size), smooth, time=time)
    data_smoothed = np.empty([num_var, time_smoothed.size])
    for v in range(num_var):
        data_smoothed[v,:] = moving_average(data_tint[v,:], smooth)
    # Convert to EJ
    data_smoothed *= Cp_sw*rhoConst*1e-18

    # Sum the components as needed
    data_final = [np.sum(data_smoothed[0:2,:], axis=0), np.sum(data_smoothed[2:4,:], axis=0), data_smoothed[4,:], data_smoothed[5,:]]

    # Plot
    timeseries_multi_plot(time_smoothed, data_final, titles, default_colours(len(data_final)), title='Time-integrated heat budget anomalies\nin '+region_names[region]+' below '+str(int(-z0))+'m', units=deg_string+r'10$^{18} $J', fig_name=fig_name)

    
    
    

    
    
    
        
    
                    
            
                        
                
                
            

    

    
    
    
    
    

                

        
                
        
    
    
    

    
    

    
    
    
            

    
        
    
    
        
    
                          
    
    

    
        
    
             
            

    

    




    
        
                     
                                                 
    

    

    

    

    

    
