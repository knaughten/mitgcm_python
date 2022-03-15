##################################################################
# JSPS Amundsen Sea simulations forced with LENS
##################################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress, ttest_1samp

from ..plot_1d import read_plot_timeseries_ensemble
from ..utils import real_dir, fix_lon_range, add_time_dim, days_per_month
from ..grid import Grid
from ..ics_obcs import find_obcs_boundary
from ..file_io import read_netcdf, read_binary, netcdf_time
from ..constants import deg_string, months_per_year
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.colours import set_colours
from ..plot_utils.labels import reduce_cbar_labels
from ..interpolation import interp_slice_helper
from ..postprocess import precompute_timeseries_coupled


# Update the timeseries calculations from wherever they left off before.
def update_lens_timeseries (num_ens=5, base_dir='./'):

    timeseries_types = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m']
    base_dir = real_dir(base_dir)
    sim_dir = [base_dir + 'PAS_LENS' + str(n+1).zfill(3) + '/output/' for n in range(num_ens)]
    timeseries_file = 'timeseries.nc'

    for n in range(num_ens):
        # Work out the first year based on where the timeseries file left off
        start_year = netcdf_time(sim_dir[n]+timeseries_file, monthly=False)[-1].year+1
        # Work on the last year based on the contents of the output directory
        sim_years = []
        for fname in os.listdir(sim_dir[n]):
            if os.path.isdir(sim_dir[n]+fname) and fname.endswith('01'):
                sim_years.append(int(fname))
        sim_years.sort()
        end_year = sim_years[-1]//100
        print('Processing years '+str(start_year)+'-'+str(end_year))
        segment_dir = [str(year)+'01' for year in range(start_year, end_year+1)]
        precompute_timeseries_coupled(output_dir=sim_dir[n], segment_dir=segment_dir, timeseries_types=timeseries_types, hovmoller_loc=[], timeseries_file=timeseries_file, key='PAS')        


# Plot a bunch of precomputed timeseries from ongoing LENS-forced test simulations (ensemble of 5 to start), compared to the PACE-forced ensemble mean.
def check_lens_timeseries (num_ens=5, base_dir='./', fig_dir=None):

    var_names = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m']
    base_dir = real_dir(base_dir)
    pace_file = base_dir+'timeseries_pace_mean.nc'
    file_paths = ['PAS_LENS'+str(n+1).zfill(3)+'/output/timeseries.nc' for n in range(num_ens)] + [pace_file]
    sim_names = ['LENS ensemble'] + [None for n in range(num_ens-1)] + ['PACE mean']
    colours = ['DarkGrey' for n in range(num_ens)] + ['blue']
    smooth = 24
    start_year = 1920

    for var in var_names:
        if fig_dir is not None:
            fig_name = real_dir(fig_dir) + 'timeseries_LENS_' + var + '.png'
        else:
            fig_name=None
        read_plot_timeseries_ensemble(var, file_paths, sim_names=sim_names, precomputed=True, colours=colours, smooth=smooth, vline=start_year, time_use=None, fig_name=fig_name)
        

# Compare time-averaged temperature and salinity boundary conditions from PACE over 2006-2013 (equivalent to the first 20 ensemble members of LENS actually!) to the WOA boundary conditions used for the original simulations.
def compare_bcs_ts_mean (fig_dir='./'):

    # Bunch of file paths
    base_dir = '/data/oceans_output/shelf/kaight/'
    grid_dir = base_dir + 'mitgcm/PAS_grid/'
    obcs_dir = base_dir + 'ics_obcs/PAS/'
    obcs_file_head = 'OB'
    obcs_file_tail = '_woa_mon.bin'
    obcs_var = ['theta', 'salt']
    obcs_bdry = ['N', 'W', 'E']
    pace_dir = '/data/oceans_input/raw_input_data/CESM/PPACE/monthly/'
    pace_var = ['TEMP', 'SALT']
    num_ens = 20
    pace_file_head = '/b.e11.BRCP85LENS.f09_g16.SST.restoring.ens'
    pace_file_mid = '.pop.h.'
    pace_file_tail = '.200601-201312.nc'
    num_var = len(pace_var)
    var_titles = ['Temperature ('+deg_string+'C)', 'Salinity (psu)']
    fig_dir = real_dir(fig_dir)
    
    # Build the grids
    mit_grid = Grid(grid_dir)
    pace_grid_file = pace_dir + pace_var[0] + pace_file_head + '01' + pace_file_mid + pace_var[0] + pace_file_tail
    pace_lon = fix_lon_range(read_netcdf(pace_grid_file, 'TLONG'))
    pace_lat = read_netcdf(pace_grid_file, 'TLAT')
    pace_z = -1*read_netcdf(pace_grid_file, 'z_t')*1e-2
    pace_nz = pace_z.size
    pace_ny = pace_lat.shape[0]
    pace_nx = pace_lat.shape[1]

    # Read the PACE output to create a time-mean, ensemble-mean (no seasonal cycle for now)
    pace_data_mean_3d = np.ma.zeros([num_var, pace_nz, pace_ny, pace_nx])
    for v in range(num_var):
        for n in range(num_ens):
            pace_file = pace_dir + pace_var[v] + pace_file_head + str(n+1).zfill(2) + pace_file_mid + pace_var[v] + pace_file_tail
            print('Reading ' + pace_file)
            pace_data_mean_3d[v,:] += read_netcdf(pace_file, pace_var[v], time_average=True)
    # Convert from ensemble-integral to ensemble-mean
    pace_data_mean_3d /= num_ens

    # Now loop over boundaries
    for bdry in obcs_bdry:
        # Find the location of this boundary (lat/lon)
        loc0 = find_obcs_boundary(mit_grid, bdry)[0]
        # Find interpolation coefficients to the PACE grid - unfortunately not a regular grid in POP
        if bdry in ['N', 'S']:
            pace_n = pace_nx
            mit_h = mit_grid.lon_1d
        elif bdry in ['E', 'W']:
            pace_n = pace_ny
            mit_h = mit_grid.lat_1d
        i1 = np.empty(pace_n)
        i2 = np.empty(pace_n)
        c1 = np.empty(pace_n)
        c2 = np.empty(pace_n)
        for j in range(pace_n):
            if bdry in ['N', 'S']:
                i1[j], i2[j], c1[j], c2[j] = interp_slice_helper(pace_lat[:,j], loc0)
            elif bdry in ['E', 'W']:
                i1[j], i2[j], c1[j], c2[j] = interp_slice_helper(pace_lon[j,:], loc0, lon=True)
        for v in range(num_var):
            # Read and time-average the existing (WOA) boundary condition file
            obcs_file = obcs_dir + obcs_file_head + bdry + obcs_var[v] + obcs_file_tail
            if bdry in ['N', 'S']:
                dimensions = 'xzt'
            elif bdry in ['E', 'W']:
                dimensions = 'yzt'
            obcs_data_mean = np.mean(read_binary(obcs_file, [mit_grid.nx, mit_grid.ny, mit_grid.nz], dimensions), axis=0)
            # Apply land mask
            if bdry == 'N':
                bdry_hfac = mit_grid.hfac[:,-1,:]
            elif bdry == 'S':
                bdry_hfac = mit_grid.hfac[:,0,:]
            elif bdry == 'W':
                bdry_hfac = mit_grid.hfac[:,:,0]
            elif bdry == 'E':
                bdry_hfac = mit_grid.hfac[:,:,-1]
            obcs_data_mean = np.ma.masked_where(bdry_hfac==0, obcs_data_mean)
            # Interpolate to the PACE grid
            pace_data_mean = np.ma.empty([pace_nz, pace_n])
            pace_h = np.ma.empty([pace_n])
            for j in range(pace_n):
                if bdry in ['N', 'S']:
                    pace_data_mean[:,j] = c1[j]*pace_data_mean_3d[v,:,int(i1[j]),j] + c2[j]*pace_data_mean_3d[v,:,int(i2[j]),j]
                    pace_h[j] = c1[j]*pace_lon[int(i1[j]),j] + c2[j]*pace_lon[int(i2[j]),j]
                elif bdry in ['E', 'W']:
                    pace_data_mean[:,j] = c1[j]*pace_data_mean_3d[v,:,j,int(i1[j])] + c2[j]*pace_data_mean_3d[v,:,j,int(i2[j])]
                    pace_h[j] = c1[j]*pace_lat[j,int(i1[j])] + c2[j]*pace_lat[j,int(i2[j])]
            # Trim
            if bdry in ['N', 'S']:
                # Get limits on longitude in MITgcm
                hmin = find_obcs_boundary(mit_grid, 'W')[0]
                hmax = find_obcs_boundary(mit_grid, 'E')[0]
            elif bdry in ['E', 'W']:
                hmin = find_obcs_boundary(mit_grid, 'S')[0]
                hmax = find_obcs_boundary(mit_grid, 'N')[0]
            j1 = interp_slice_helper(pace_h, hmin, lon=(bdry in ['N', 'S']))[0]
            j2 = interp_slice_helper(pace_h, hmax, lon=(bdry in ['N', 'S']))[1]
            pace_data_mean = pace_data_mean[:,j1:j2+1]
            pace_h = pace_h[j1:j2+1]
            # Find bounds for colour scale
            vmin = min(np.amin(obcs_data_mean), np.amin(pace_data_mean))
            vmax = max(np.amax(obcs_data_mean), np.amax(pace_data_mean))
            # Plot
            fig, gs, cax = set_panels('1x2C1')
            # WOA OBCS
            ax = plt.subplot(gs[0,0])
            ax.pcolormesh(mit_h, mit_grid.z*1e-3, obcs_data_mean, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('WOA 2018', fontsize=14)
            # PACE
            ax = plt.subplot(gs[0,1])
            img = plt.pcolormesh(pace_h, pace_z*1e-3, pace_data_mean, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('LENS 20-member mean', fontsize=14)
            plt.colorbar(img, cax=cax, orientation='horizontal')
            plt.suptitle(var_titles[v]+', '+bdry+' boundary', fontsize=16)
            finished_plot(fig, fig_name=fig_dir+'pace_obcs_'+pace_var[v]+'_'+str(bdry)+'.png')


# Plot the LENS bias corrections for OBCS at all three boundaries with annual mean, for each variable.
def plot_lens_obcs_bias_corrections_annual (fig_dir=None):

    base_dir = '/data/oceans_output/shelf/kaight/'
    mit_grid_dir = base_dir + 'mitgcm/PAS_grid/'
    in_dir = base_dir + 'CESM_bias_correction/obcs/'
    bdry_loc = ['N', 'W', 'E']
    oce_var = ['TEMP', 'SALT', 'UVEL', 'VVEL']
    ice_var = ['aice', 'hi', 'hs', 'uvel', 'vvel']
    gtype_oce = ['t', 't', 'u', 'v']
    gtype_ice = ['t', 't', 't', 'u', 'v']
    oce_titles = ['Temperature ('+deg_string+'C)', 'Salinity (psu)', 'Zonal velocity (m/s)', 'Meridional velocity (m/s)']
    ice_titles = ['Sea ice concentration', 'Sea ice thickness (m)', 'Snow thickness (m)', 'Sea ice zonal velocity (m/s)', 'Sea ice meridional velocity (m/s)']
    file_head = in_dir + 'LENS_offset_'
    bdry_patches = [None, None, None]
    num_bdry = len(bdry_loc)

    grid = Grid(mit_grid_dir)

    # Loop over ocean and ice variables
    for var_names, gtype, titles, oce in zip([oce_var, ice_var], [gtype_oce, gtype_ice], [oce_titles, ice_titles], [True, False]):
        for v in range(len(var_names)):
            data = []
            h = []
            vmin = 0
            vmax = 0
            lon, lat = grid.get_lon_lat(gtype=gtype[v], dim=1)
            hfac = grid.get_hfac(gtype=gtype[v])
            for n in range(num_bdry):
                # Read and time-average the file (don't worry about different month lengths for plotting purposes)
                file_path = file_head + var_names[v] + '_' + bdry_loc[n]
                if bdry_loc[n] in ['N', 'S']:
                    dimensions = 'x'
                    h.append(lon)
                    h_label = 'Longitude'
                elif bdry_loc[n] in ['E', 'W']:
                    dimensions = 'y'
                    h.append(lat)
                    h_label = 'Latitude'
                if oce:
                    dimensions += 'z'
                dimensions += 't'
                data_tmp = read_binary(file_path, [grid.nx, grid.ny, grid.nz], dimensions)
                data_tmp = np.mean(data_tmp, axis=0)
                # Mask
                if bdry_loc[n] == 'N':
                    hfac_bdry = hfac[:,-1,:]
                elif bdry_loc[n] == 'S':
                    hfac_bdry = hfac[:,0,:]
                elif bdry_loc[n] == 'E':
                    hfac_bdry = hfac[:,:,-1]
                elif bdry_loc[n] == 'W':
                    hfac_bdry = hfac[:,:,0]
                if not oce:
                    hfac_bdry = hfac_bdry[0,:]
                data_tmp = np.ma.masked_where(hfac_bdry==0, data_tmp)
                data.append(data_tmp)
                # Keep track of min and max across all boundaries
                vmin = min(vmin, np.amin(data_tmp))
                vmax = max(vmax, np.amax(data_tmp))
            # Now plot each boundary
            if oce:
                fig, gs, cax = set_panels('1x3C1')
            else:
                fig, gs = set_panels('1x3C0')
                gs.update(left=0.05, wspace=0.1, bottom=0.1)
            cmap = set_colours(data[0], ctype='plusminus', vmin=vmin, vmax=vmax)[0]
            for n in range(num_bdry):
                ax = plt.subplot(gs[0,n])
                if oce:
                    # Simple slice plot (don't worry about partial cells or lat/lon edges vs centres)
                    img = plt.pcolormesh(h[n], grid.z*1e-3, data[n], cmap=cmap, vmin=vmin, vmax=vmax)
                    if n==0:
                        ax.set_ylabel('Depth (km)')
                    if n==num_bdry-1:
                        plt.colorbar(img, cax=cax, orientation='horizontal')
                else:
                    # Line plot
                    plt.plot(h[n], data[n])
                    ax.grid(linestyle='dotted')
                    ax.set_ylim([vmin, vmax])
                    ax.axhline(color='black')
                if n==0:
                    ax.set_xlabel(h_label)
                ax.set_title(bdry_loc[n], fontsize=12)
                plt.suptitle(titles[v], fontsize=16)
                if fig_dir is not None:
                    fig_name = real_dir(fig_dir) + 'obcs_offset_' + var_names[v] + '.png'
                else:
                    fig_name = None
                finished_plot(fig, fig_name=fig_name)


# Now plot all months and the annual mean, for one variable and one boundary at a time.
def plot_lens_obcs_bias_corrections_monthly (fig_dir=None):

    base_dir = '/data/oceans_output/shelf/kaight/'
    mit_grid_dir = base_dir + 'mitgcm/PAS_grid/'
    in_dir = base_dir + 'CESM_bias_correction/obcs/'
    bdry_loc = ['N', 'W', 'E']
    oce_var = ['TEMP', 'SALT', 'UVEL', 'VVEL']
    ice_var = ['aice', 'hi', 'hs', 'uvel', 'vvel']
    gtype_oce = ['t', 't', 'u', 'v']
    gtype_ice = ['t', 't', 't', 'u', 'v']
    oce_titles = ['Temperature ('+deg_string+'C)', 'Salinity (psu)', 'Zonal velocity (m/s)', 'Meridional velocity (m/s)']
    ice_titles = ['Sea ice concentration', 'Sea ice thickness (m)', 'Snow thickness (m)', 'Sea ice zonal velocity (m/s)', 'Sea ice meridional velocity (m/s)']
    file_head = in_dir + 'LENS_offset_'
    bdry_patches = [None, None, None]
    num_bdry = len(bdry_loc)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    grid = Grid(mit_grid_dir)

    # Loop over ocean and ice variables
    for var_names, gtype, titles, oce in zip([oce_var, ice_var], [gtype_oce, gtype_ice], [oce_titles, ice_titles], [True, False]):
        for v in range(len(var_names)):
            lon, lat = grid.get_lon_lat(gtype=gtype[v], dim=1)
            hfac = grid.get_hfac(gtype=gtype[v])
            for n in range(num_bdry):
                # Read the file but no time-averaging
                file_path = file_head + var_names[v] + '_' + bdry_loc[n]
                if bdry_loc[n] in ['N', 'S']:
                    dimensions = 'x'
                    h = lon
                elif bdry_loc[n] in ['E', 'W']:
                    dimensions = 'y'
                    h = lat
                if oce:
                    dimensions += 'z'
                dimensions += 't'
                data = read_binary(file_path, [grid.nx, grid.ny, grid.nz], dimensions)
                # Mask
                if bdry_loc[n] == 'N':
                    hfac_bdry = hfac[:,-1,:]
                elif bdry_loc[n] == 'S':
                    hfac_bdry = hfac[:,0,:]
                elif bdry_loc[n] == 'E':
                    hfac_bdry = hfac[:,:,-1]
                elif bdry_loc[n] == 'W':
                    hfac_bdry = hfac[:,:,0]
                if not oce:
                    hfac_bdry = hfac_bdry[0,:]
                hfac_bdry = add_time_dim(hfac_bdry, months_per_year)
                data = np.ma.masked_where(hfac_bdry==0, data)
                # Now plot each month
                if oce:
                    fig, gs, cax = set_panels('3x4+1C1')
                else:
                    fig, gs = set_panels('3x4+1C0')
                cmap, vmin, vmax = set_colours(data, ctype='plusminus')
                for t in range(months_per_year):
                    ax = plt.subplot(gs[t//4+1, t%4])
                    if oce:
                        ax.pcolormesh(h, grid.z*1e-3, data[t,:], cmap=cmap, vmin=vmin, vmax=vmax)
                    else:
                        ax.plot(h, data[t,:])
                        ax.set_ylim([vmin, vmax])
                        ax.grid(linestyle='dotted')
                        ax.axhline(color='black')
                    ax.set_title(month_names[t], fontsize=10)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                # Now plot annual mean
                ax = plt.subplot(gs[0,3])
                if oce:
                    img = ax.pcolormesh(h, grid.z*1e-3, np.mean(data, axis=0), cmap=cmap, vmin=vmin, vmax=vmax)
                    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
                    reduce_cbar_labels(cbar)
                    ax.set_yticklabels([])
                else:
                    ax.plot(h, np.mean(data, axis=0))
                    ax.set_ylim([vmin, vmax])
                    ax.grid(linestyle='dotted')
                    ax.axhline(color='black')
                ax.set_xticklabels([])
                ax.set_title('Annual', fontsize=10)
                plt.suptitle(titles[v] + ', ' + bdry_loc[n] + ' boundary', fontsize=14)
                if fig_dir is not None:
                    fig_name = real_dir(fig_dir) + 'obcs_offset_monthly_' + var_names[v] + '_' + bdry_loc[n] + '.png'
                else:
                    fig_name = None
                finished_plot(fig, fig_name=fig_name)


# Calculate and plot the ensemble mean trends in the LENS ocean for the given boundary and given variable.
def calc_obcs_trends_lens (var_name, bdry_loc, fig_name=None):

    in_dir = '/data/oceans_input/raw_input_data/CESM/LENS/monthly/'+var_name+'/'
    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/'
    file_head = 'b.e11.BRCP85C5CNBDRD.f09_g16.'
    file_mid = '.pop.h.'
    file_tail_1 = '.200601-208012.nc'
    file_tail_2 = '.208101-210012.nc'
    file_tail_alt = '.200601-210012.nc'
    num_ens = 40
    ens_break = 36
    ens_offset = 105
    start_year = 2006
    end_year = 2100
    break_year = 2081
    t_break = (break_year-start_year)*months_per_year
    num_years = end_year - start_year + 1
    p0 = 0.05
    if var_name == 'TEMP':
        units = deg_string+'C'
    elif var_name == 'SALT':
        units = 'psu'
    else:
        print 'Error (calc_obcs_trends_lens): unknown variable ' + var_name
        sys.exit()

    # Read POP grid
    grid_file = in_dir + file_head + '001' + file_tail_1
    lon = fix_lon_range(read_netcdf(grid_file, 'TLONG'))
    lat = read_netcdf(grid_file, 'TLAT')
    z = -1e-2*read_netcdf(grid_file, 'z_t')
    nz = z.size
    ny = lat.shape[0]
    nx = lon.shape[1]

    # Read MITgcm grid and get boundary location
    mit_grid = Grid(mit_grid_dir)
    loc0 = find_obcs_boundary(mit_grid, bdry)[0]
    # Find interpolation coefficients to the POP grid
    if bdry in ['N', 'S']:
        nh = nx
        hmin = find_obcs_boundary(mit_grid, 'W')[0]
        hmax = find_obcs_boundary(mit_grid, 'E')[0]
    elif bdry in ['E', 'W']:
        nh = ny
        hmin = find_obcs_boundary(mit_grid, 'S')[0]
        hmax = find_obcs_boundary(mit_grid, 'N')[0]
    i1 = np.empty(nh)
    i2 = np.empty(nh)
    c1 = np.empty(nh)
    c2 = np.empty(nh)
    for j in range(nh):
        if bdry in ['N', 'S']:
            i1[j], i2[j], c1[j], c2[j] = interp_slice_helper(lat[:,j], loc0)
        elif bdry in ['E', 'W']:
            i1[j], i2[j], c1[j], c2[j] = interp_slice_helper(lon[j,:], loc0, lon=True)

    # Loop over ensemble members
    for n in range(num_ens):
        print('Processing ensemble member ' + str(n+1))
        print('...reading data')
        data_full = np.ma.zeros([num_years*months_per_year, nz, ny, nx])
        if n+1 < ens_break:
            file_path_1 = in_dir + file_head + str(n+1).zfill(3) + file_tail_1
            data_full[:t_break,:] = read_netcdf(file_path_1, var_name)
            file_path_2 = in_dir + file_head + str(n+1).zfill(3) + file_tail_2
            data_full[t_break:,:] = read_netcdf(file_path_2, var_name)
        else:
            file_path = in_dir + file_head + str(n+1-ens_break+ens_offset).zfill(3) + file_tail_alt
            data_full = read_netcdf(file_path, var_name)
        print('...annually averaging')
        data_annual = np.ma.zeros([num_years, nz, ny, nx])
        for year in range(start_year, end_year+1):
            ndays = np.array([days_per_month(month+1, year) for month in range(12)])
            t0 = (year-start_year)*months_per_year
            data_annual = np.average(data_full[t0:t0+months_per_year,:], axis=0, weights=ndays)
        print('...extracting the boundary')
        data_bdry = np.ma.zeros([num_years, nz, nh])
        if n==0:
            h = np.ma.empty([nh])
        for j in range(nh):
            if bdry in ['N', 'S']:
                data_bdry[:,:,j] = c1[j]*data_annual[:,:,int(i1[j]),j] + c2[j]*data_annual[:,:,int(i2[j]),j]
                if n==0:
                    h[j] = c1[j]*lon[int(i1[j]),j] + c2[j]*lon[int(i2[j]),j]
            elif bdry in ['E', 'W']:
                data_bdry[:,:,j] = c1[j]*data_annual[:,:,j,int(i1[j])] + c2[j]*data_annual[:,:,j,int(i2[j])]
                if n==0:
                    h[j] = c1[j]*lat[j,int(i1[j])] + c2[j]*lat[j,int(i2[j])]
        # Mask out anything outside the MITgcm grid
        if n==0:            
            j1 = interp_slice_helper(h, hmin, lon=(bdry in ['N', 'S']))[0]
            j2 = interp_slice_helper(h, hmax, lon=(bdry in ['N', 'S']))[0]
            nh_trim = j2-j1+1
            h = h[j1:j2+1]
            # Now set up arrays for trends at each point and each ensemble member
            trends = np.ma.zeros([num_ens, nz, nh_trim])
        data_bdry = data_bdry[:,:,j1:j2+1]
        # Loop over each point and calculate trends
        for k in range(nz):
            for j in range(nh_trim):
                trends[n,k,j] = linregress(np.arange(num_years), data_bdry[:,k,j])[0]

    # Calculate the mean trend and significance
    mean_trend = np.mean(trends, axis=0)*1e-2
    p_val = ttest_1samp(trends, 0, axis=0)[1]
    # For any trends which aren't significant, fill with zeros
    mean_trend[p_val > p0] = 0

    # Plot
    fig, ax = plt.subplots()
    cmap, vmin, vmax = set_colours(mean_trend, ctype='plusminus')
    img = ax.pcolormesh(h, z, mean_trend, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img)
    plt.title(var_name+' trend at '+bdry_loc+' boundary ('+units+'/century)', fontsize=14)
    finished_plot(fig, fig_name=fig_name)
    
    


        
                    
                
            
            
            
        

    

    
                
                    
                

                    
                        
        
        

    
    
            
            
            
    
