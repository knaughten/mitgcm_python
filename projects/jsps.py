##################################################################
# JSPS Amundsen Sea simulations
##################################################################

import numpy as np
from itertools import compress, cycle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import ERA5Grid, PACEGrid, Grid
from ..file_io import read_binary, write_binary, read_netcdf
from ..utils import real_dir, daily_to_monthly
from ..plot_utils.colours import set_colours
from ..plot_utils.windows import finished_plot
from ..plot_1d import default_colours


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
def calc_climatologies (era5_dir, pace_dir, out_dir):

    # Day of the year that's 29 Feb (0-indexed)
    leap_day = 31+28
    # Climatology over the years that both products have data (not counting the RCP8.5 extension)
    start_year = 1979
    end_year = 2005
    num_years = end_year-start_year+1

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
            if ens == 13:
                continue
            ens_str = str(ens).zfill(2)
            print 'Processing PACE ensemble member ' + ens_str
            # As before, but simpler because no leap days and no need to regrid
            data_accum = np.zeros([per_year, pace_grid.ny, pace_grid.nx])
            for year in range(start_year, end_year+1):
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
def plot_biases (var_name, clim_dir, monthly=False, fig_dir='./'):

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
    data = np.empty([num_ens-1, per_year, grid.ny, grid.nx])
    # Read data
    for ens in range(1, num_ens+1):
        if ens == 13:
            continue
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
    bias_xy = np.mean(data, axis=(0,1)) - np.mean(data_era5, axis=0)
    # Mask out everything north of 30S so it doesn't get counted in min/max
    bias_xy[grid.lat > ylim_era5[-1]] = 0
    fig, ax = plt.subplots(figsize=(10,6))
    cmap, vmin, vmax = set_colours(bias_xy, ctype='plusminus')
    img = ax.contourf(grid.lon, grid.lat, bias_xy, 30, cmap=cmap, vmin=vmin, vmax=vmax)
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
def plot_timeseries_2y (sim_dir, sim_names, fig_dir='./'):

    from ..plot_1d import read_plot_timeseries_ensemble

    timeseries_types = ['thwaites_melting', 'dotson_crosson_melting', 'pig_melting', 'pine_island_bay_temp_bottom', 'dotson_bay_temp_bottom', 'amundsen_shelf_break_uwind_avg']
    timeseries_file = 'timeseries.nc'
    timeseries_paths = [real_dir(d) + 'output/' + timeseries_file for d in sim_dir]
    smooth = 12
    year_start = 1979

    for var_name in timeseries_types:
        read_plot_timeseries_ensemble(var_name, timeseries_paths, sim_names=sim_names, precomputed=True, smooth=smooth, alpha=True, vline=year_start, time_use=None, fig_name=fig_dir+'timeseries_'+var_name+'_2y.png')


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
    xtick_labels = [20, 10, 5, 3, 2, 1]
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
        data = read_netcdf(file_path, var)
        f, p = welch(data, fs=12, detrend='linear')
        freq.append(f)
        pxx.append(p)
    fig, ax = plt.subplots(figsize=(11,6))
    for n in range(num_sims):
        ax.plot(freq[n], pxx[n], label=sim_names[n], color=colours[n], alpha=alphas[n])
    ax.set_xlim([0,1])
    ax.grid(True)
    xtick_labels = [10, 5, 3, 2, 1]
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

    

    

    
