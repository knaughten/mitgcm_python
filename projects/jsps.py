##################################################################
# JSPS Amundsen Sea simulations
##################################################################

import numpy as np
from itertools import compress, cycle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress

from ..grid import ERA5Grid, PACEGrid, Grid, dA_from_latlon, choose_grid
from ..file_io import read_binary, write_binary, read_netcdf, netcdf_time, read_title_units, read_annual_average
from ..utils import real_dir, daily_to_monthly, fix_lon_range, split_longitude, mask_land_ice, moving_average, index_year_start, index_period, mask_2d_to_3d, days_per_month
from ..plot_utils.colours import set_colours, choose_n_colours
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.labels import reduce_cbar_labels, round_to_decimals
from ..plot_1d import default_colours, make_timeseries_plot_2sided
from ..plot_latlon import latlon_plot
from ..constants import sec_per_year, kg_per_Gt, dotson_melt_years, getz_melt_years, pig_melt_years, region_names, deg_string, sec_per_day
from ..plot_misc import hovmoller_plot
from ..timeseries import calc_annual_averages
from ..postprocess import get_output_files


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
    cmap, vmin, vmax = set_colours(bias_xy, ctype='plusminus')
    if ratio:
        vmax = min(vmax, 3)
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
    smooth = 12
    if hindcast:
        year_start = 1920
        year_ticks = np.arange(1930, 2010+1, 10)
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


# Plot timeseries of mass loss from PIG, Dotson, and Getz for the given simulation, with observational estimates overlaid on top.
def plot_ismr_timeseries_obs (timeseries_file, start_year=1979, fig_name=None):

    # Could do: option for ensemble with mean on top

    shelf = ['pig', 'dotson', 'getz']
    obs = [pig_melt_years, dotson_melt_years, getz_melt_years]
    obs_month = 2  # Assume all obs in February

    # Read data and trim the spinup
    time = netcdf_time(timeseries_file, monthly=False)
    for t in range(time.size):
        if time[t].year == start_year:
            t0 = t
            break
    time = time[t0:]
    model_melt = []
    for s in shelf:
        model_melt.append(read_netcdf(timeseries_file, s+'_massloss')[t0:])

    # Set up the plot
    fig, gs = set_panels('3x1C0')
    gs.update(bottom=0.05, top=0.9)
    for n in range(len(shelf)):
        ax = plt.subplot(gs[n,0])
        # Plot the model timeseries
        ax.plot_date(time, model_melt[n], '-', color='blue')
        # Loop over observational years and plot the range
        num_obs = len(obs[n]['year'])
        for t in range(num_obs):
            obs_date = datetime.date(obs[n]['year'][t], obs_month, 1)
            ax.errorbar(obs_date, obs[n]['melt'][t], yerr=obs[n]['err'][t], fmt='none', color='red', capsize=4)
        ax.grid(True)
        ax.set_title(region_names[shelf[n]], fontsize=18)
        if n == 0:
            ax.set_ylabel('Gt/y', fontsize=14)
        if n != len(shelf)-1:
            ax.set_xticklabels([])
    plt.suptitle('Ice shelf mass loss compared to observations', fontsize=24)
    finished_plot(fig, fig_name=fig_name)


# Determine the most spiky, least spiky, and medium ensemble members (from the first 10) based on their PIG and Dotson melt rates.
def order_ensemble_std (base_dir='./'):

    # Run IDs for each member, in order
    run_id = ['018', '027', '028', '029', '030', '033', '034', '035', '036', '037']
    # ERA5 ID for comparison
    era5_id = ['031']
    run_names = ['PACE '+str(n+1) for n in range(10)] + ['ERA5']
    ts_file = 'timeseries.nc'
    smooth = 12
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
    smooth = 6
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
def read_calc_trends (var, file_path, option, percent=False, year_start=1920, year_end=1949, smooth=12, p0=0.05):

    data = read_netcdf(file_path, var)
    time = netcdf_time(file_path, monthly=False)
    if percent:
        # Express as percentage of mean over baseline
        t_start, t_end = index_period(time, year_start, year_end)
        data_mean = np.mean(data[t_start:t_end])
        data = data/data_mean*100
    if option == 'smooth':
        # 2-year running mean to filter out seasonal cycle
        data = moving_average(data, smooth)
    elif option == 'annual':
        # Annual average to filter out seasonal cycle
        # First trim to the nearest complete year
        new_size = len(time)/12*12
        time = time[:new_size]
        data = data[:new_size]
        time, data = calc_annual_averages(time, data)
    # Calculate trends
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(data.size), data)
    # Multiply slope by 10 to get trend per decade
    slope *= 10
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

    from scipy.stats import ttest_1samp

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
    for var in ['abbot_melting', 'cosgrove_melting', 'dotson_crosson_melting', 'getz_melting', 'pig_melting', 'thwaites_melting', 'venable_melting', 'pine_island_bay_temp_below_500m', 'pine_island_bay_salt_below_500m', 'dotson_bay_temp_below_500m', 'dotson_bay_salt_below_500m', 'inner_amundsen_shelf_temp_below_500m', 'inner_amundsen_shelf_salt_below_500m', 'amundsen_shelf_break_uwind_avg']:
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
    smooth = 12
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
    smooth_short = 12
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
            smooth_long = test_smooth[m]*12/2
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
    smooth = 12
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
        
         

        

    

    
    

    
        
    
             
            

    

    




    
        
                     
                                                 
    

    

    

    

    

    
