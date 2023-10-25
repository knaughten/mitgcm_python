##################################################################
# JSPS Amundsen Sea simulations forced with CESM future scenarios
##################################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress, ttest_1samp, ttest_ind, norm
from scipy.ndimage.filters import gaussian_filter
import datetime
import itertools

from ..plot_1d import read_plot_timeseries_ensemble
from ..plot_latlon import latlon_plot
from ..plot_slices import make_slice_plot, slice_plot
from ..utils import real_dir, fix_lon_range, add_time_dim, days_per_month, xy_to_xyz, z_to_xyz, index_year_start, var_min_max, polar_stereo, mask_3d, moving_average, index_period, mask_land, mask_except_ice, mask_land_ice, distance_to_grounding_line, apply_mask, index_year_end, select_bottom
from ..grid import Grid, read_pop_grid, read_cice_grid, CAMGrid
from ..ics_obcs import find_obcs_boundary, trim_slice_to_grid, trim_slice, get_hfac_bdry, read_correct_cesm_ts_space, read_correct_cesm_non_ts, get_fill_mask
from ..file_io import read_netcdf, read_binary, netcdf_time, write_binary, find_cesm_file, NCfile, average_12_months
from ..constants import deg_string, months_per_year, Tf_ref, region_names, Cp_sw, rhoConst, sec_per_day, rho_ice, sec_per_year, region_bounds, rho_fw
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.colours import set_colours, get_extend, choose_n_colours
from ..plot_utils.labels import reduce_cbar_labels, lon_label, round_to_decimals, slice_axes, lat_label, latlon_axes
from ..plot_utils.slices import slice_patches, slice_values
from ..plot_utils.latlon import overlay_vectors, shade_land, contour_iceshelf_front, cell_boundaries, shade_mask, cell_boundaries, shade_background, average_blocks
from ..plot_misc import ts_binning, hovmoller_plot
from ..interpolation import interp_slice_helper, interp_slice_helper_nonreg, extract_slice_nonreg, interp_bdry, fill_into_mask, distance_weighted_nearest_neighbours, interp_to_depth, interp_grid, interp_reg_xy, interp_reg_xyz, discard_and_fill, interp_reg, extend_into_mask, interp_nonreg_xy
from ..postprocess import precompute_timeseries_coupled, make_trend_file
from ..diagnostics import potential_density
from ..make_domain import latlon_points
from ..timeseries import monthly_to_annual, calc_annual_averages
from ..calculus import area_average, vertical_integral, vertical_average


# Update the timeseries calculations from wherever they left off before.
def update_lens_timeseries (num_ens=5, base_dir='./', sim_dir=None):

    timeseries_types = ['amundsen_shelf_isotherm_0.5C_below_100m'] #['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m', 'eta_avg', 'seaice_area', 'PITE_trans', 'getz_massloss', 'dotson_massloss', 'crosson_massloss', 'thwaites_massloss', 'pig_massloss', 'cosgrove_massloss', 'abbot_massloss', 'venable_massloss'] #, 'getz_iceshelf_area', 'dotson_iceshelf_area', 'crosson_iceshelf_area', 'thwaites_iceshelf_area', 'pig_iceshelf_area', 'cosgrove_iceshelf_area', 'abbot_iceshelf_area', 'venable_iceshelf_area']
    base_dir = real_dir(base_dir)
    if sim_dir is None:
        sim_dir = [base_dir + 'PAS_LENS' + str(n+1).zfill(3) + '_O/output/' for n in range(num_ens)]
    else:
        num_ens = len(sim_dir)
    timeseries_file = 'timeseries_isotherm.nc' #'timeseries.nc'

    for n in range(num_ens):
        if not os.path.isfile(sim_dir[n]+timeseries_file):
            # Start fresh
            segment_dir = None
        else:
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
def check_lens_timeseries (num_ens=5, base_dir='./', fig_dir=None, sim_dir=None, finished=False):

    var_names = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m']
    base_dir = real_dir(base_dir)
    pace_file = base_dir+'timeseries_pace_mean.nc'
    if sim_dir is None:
        sim_dir = ['PAS_LENS'+str(n+1).zfill(3)+'_O/output/' for n in range(num_ens)]
    else:
        num_ens = len(sim_dir)
    file_paths = [pace_file] + [sd+'timeseries.nc' for sd in sim_dir]
    sim_names = ['PACE mean'] + ['LENS ensemble'] + [None for n in range(num_ens-1)]
    colours = ['blue'] + ['DarkGrey' for n in range(num_ens)]
    smooth = 24
    start_year = 1920

    for var in var_names:
        if fig_dir is not None:
            fig_name = real_dir(fig_dir) + 'timeseries_LENS_' + var + '.png'
        else:
            fig_name=None
        read_plot_timeseries_ensemble(var, file_paths, sim_names=sim_names, precomputed=True, colours=colours, smooth=smooth, vline=start_year, time_use=None, fig_name=fig_name, plot_mean=finished, first_in_mean=False)


# As above, but multiple ensembles for different scenarios.
def check_scenario_timeseries (base_dir='./', num_LENS=5, num_noOBCS=5, num_MENS=5, num_LW2=5, num_LW1=5, fig_dir=None):

    var_names = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m']
    base_dir = real_dir(base_dir)
    pace_file = base_dir+'timeseries_pace_mean.nc'
    smooth = 24
    start_year = 1920

    sim_dir = ['PAS_LENS'+str(n+1).zfill(3)+'_noOBC/' for n in range(num_noOBCS)] + ['PAS_LENS'+str(n+1).zfill(3)+'_O/' for n in range(num_LENS)] + ['PAS_MENS_'+str(n+1).zfill(3)+'_O/' for n in range(num_MENS)] + ['PAS_LW2.0_'+str(n+1).zfill(3)+'_O/' for n in range(num_LW2)] + ['PAS_LW1.5_'+str(n+1).zfill(3)+'_O/' for n in range(num_LW1)]
    file_paths = [base_dir+sd+'output/timeseries.nc' for sd in sim_dir] + [pace_file]
    sim_names = []
    for scenario, num_scenario in zip(['LENS no OBCS', 'LENS', 'MENS', 'LW2.0', 'LW1.5'], [num_noOBCS, num_LENS, num_MENS, num_LW2, num_LW1]):
        if num_scenario > 0:
            sim_names += [scenario]
            if num_scenario > 1:
                sim_names += [None for n in range(num_scenario-1)]
    sim_names += ['PACE mean']
    colours = ['BurlyWood' for n in range(num_noOBCS)] + ['DarkGrey' for n in range(num_LENS)] + ['IndianRed' for n in range(num_MENS)] + ['MediumSeaGreen' for n in range(num_LW2)] + ['DodgerBlue' for n in range(num_LW1)] + ['MediumOrchid']

    for var in var_names:
        if fig_dir is not None:
            fig_name = real_dir(fig_dir) + 'timeseries_scenario_' + var + '.png'
        else:
            fig_name = None
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
    pace_lon, pace_lat, pace_z, pace_nx, pace_ny, pace_nz = read_pop_grid(pace_grid_file)

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
            direction = 'lat'
            mit_h = mit_grid.lon_1d
            pace_h_2d = pace_lon
        elif bdry in ['E', 'W']:
            direction = 'lon'
            mit_h = mit_grid.lat_1d
            pace_h_2d = pace_lat
        i1, i2, c1, c2 = interp_slice_helper_nonreg(pace_lon, pace_lat, loc0, direction)
        for v in range(num_var):
            # Read and time-average the existing (WOA) boundary condition file
            obcs_file = obcs_dir + obcs_file_head + bdry + obcs_var[v] + obcs_file_tail
            if bdry in ['N', 'S']:
                dimensions = 'xzt'
            elif bdry in ['E', 'W']:
                dimensions = 'yzt'
            obcs_data_mean = np.mean(read_binary(obcs_file, [mit_grid.nx, mit_grid.ny, mit_grid.nz], dimensions), axis=0)
            # Apply land mask
            bdry_hfac = get_hfac_bdry(mit_grid, bdry)
            obcs_data_mean = np.ma.masked_where(bdry_hfac==0, obcs_data_mean)
            # Interpolate to the PACE grid
            pace_data_mean = extract_slice_nonreg(pace_data_mean_3d, direction, i1, i2, c1, c2)
            pace_h = extract_slice_nonreg(pace_h_2d, direction, i1, i2, c1, c2)
            # Trim
            pace_data_mean, pace_h = trim_slice_to_grid(pace_data_mean, pace_h, mit_grid, direction)
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


# Calculate a monthly climatology of each variable from the LENS simulations of CESM over each boundary.
def calc_lens_climatology (out_dir='./'):

    out_dir = real_dir(out_dir)
    var_names = ['TEMP', 'SALT', 'UVEL', 'VVEL', 'aice', 'hi', 'hs', 'uvel', 'vvel']
    gtype = ['t', 't', 'u', 'v', 't', 't', 't', 'u', 'v']
    domain = ['oce', 'oce', 'oce', 'oce', 'ice', 'ice', 'ice', 'ice', 'ice']
    num_var = len(var_names)
    start_year = 1998
    end_year = 2017
    num_years = end_year - start_year + 1
    num_ens = 40
    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/AMUND_ini_grid/'
    bdry_loc = ['N', 'W', 'E']
    num_var = len(var_names)
    out_file_head = 'LENS_climatology_'
    out_file_tail = '_'+str(start_year)+'-'+str(end_year)

    # Read/generate grids
    pop_grid_file = find_cesm_file('LENS', var_names[0], 'oce', 'monthly', 1, start_year)[0]
    pop_tlon, pop_tlat, pop_ulon, pop_ulat, pop_z_1d, pop_nx, pop_ny, nz = read_pop_grid(pop_grid_file, return_ugrid=True)
    cice_grid_file = find_cesm_file('LENS', var_names[-1], 'ice', 'monthly', 1, start_year)[0]
    cice_tlon, cice_tlat, cice_ulon, cice_ulat, cice_nx, cice_ny = read_cice_grid(cice_grid_file, return_ugrid=True)
    mit_grid = Grid(mit_grid_dir)

    for v in range(num_var):
        for bdry in bdry_loc:
            print('Processing '+var_names[v]+' on '+bdry+' boundary')
            loc0_centre, loc0_edge = find_obcs_boundary(mit_grid, bdry)
            if (bdry in ['N', 'S'] and gtype[v] == 'v') or (bdry in ['E', 'W'] and gtype[v] == 'u'):
                loc0 = loc0_edge
            else:
                loc0 = loc0_centre
            if domain[v] == 'oce':
                if gtype[v] == 't':
                    lon = pop_tlon
                    lat = pop_tlat
                else:
                    lon = pop_ulon
                    lat = pop_ulat
            else:
                if gtype[v] == 't':
                    lon = cice_tlon
                    lat = cice_tlat
                else:
                    lon = cice_ulon
                    lat = cice_ulat
            if bdry in ['N', 'S']:
                direction = 'lat'
                h_2d = lon
            elif bdry in ['E', 'W']:
                direction = 'lon'
                h_2d = lat
            i1, i2, c1, c2 = interp_slice_helper_nonreg(lon, lat, loc0, direction)
            h_full = extract_slice_nonreg(h_2d, direction, i1, i2, c1, c2)
            h = trim_slice_to_grid(h_full, h_full, mit_grid, direction)[0]
            nh = h.size
            # Set up array for monthly climatology
            if domain[v] == 'oce':
                shape = [months_per_year, nz, nh]
            else:
                shape = [months_per_year, nh]
            lens_clim = np.ma.zeros(shape)
            # Loop over ensemble members and time indices
            for n in range(num_ens):
                print('Processing ensemble member '+str(n+1))
                for year in range(start_year, end_year+1):
                    print('...'+str(year))
                    for month in range(months_per_year):
                        file_path, t0_year, tf_year = find_cesm_file('LENS', var_names[v], domain[v], 'monthly', n+1, year)
                        t0 = t0_year+month
                        data_3d = read_netcdf(file_path, var_names[v], t_start=t0, t_end=t0+1)
                        if var_names[v] in ['UVEL', 'VVEL', 'uvel', 'vvel', 'aice']:
                            # Convert from cm/s to m/s, or percent to fraction
                            data_3d *= 1e-2
                        data_slice = extract_slice_nonreg(data_3d, direction, i1, i2, c1, c2)
                        data_slice = trim_slice_to_grid(data_slice, h_full, mit_grid, direction, warn=False)[0]
                        lens_clim[month,:] += data_slice
            # Convert from integral to average
            lens_clim /= (num_ens*num_years)
            # Save to binary file
            write_binary(lens_clim, out_dir+out_file_head+var_names[v]+'_'+bdry+out_file_tail)


# Calculate bias correction files for each boundary condition, based on the LENS ensemble mean climatology compared to the WOA/SOSE fields we used previously.
def make_lens_bias_correction_files (out_dir='./'):

    base_dir = '/data/oceans_output/shelf/kaight/'
    mit_grid_dir = base_dir + 'mitgcm/PAS_grid/'
    lens_dir = base_dir + 'CESM_bias_correction/obcs/'
    obcs_dir = base_dir + 'ics_obcs/PAS/'
    bdry_loc = ['N', 'W', 'E']
    var_obcs_oce = ['theta_woa_mon', 'salt_woa_mon', 'uvel_sose', 'vvel_sose']
    var_obcs_ice = ['area_sose', 'heff_sose', 'hsnow_sose', 'uice_sose', 'vice_sose']
    obcs_gtype_oce = ['t', 't', 'u', 'v']
    obcs_gtype_ice = ['t', 't', 't', 'u', 'v']
    obcs_file_head = obcs_dir + 'OB'
    obcs_file_tail = '.bin'
    var_lens_oce = ['TEMP', 'SALT', 'UVEL', 'VVEL']
    var_lens_ice = ['aice', 'hi', 'hs', 'uvel', 'vvel']
    num_var_oce = len(var_lens_oce)
    num_var_ice = len(var_lens_ice)
    lens_gtype_oce = ['t', 't', 'u', 'u']
    lens_gtype_ice = ['t', 't', 't', 'u', 'u']
    lens_file_head = lens_dir + 'LENS_climatology_'
    lens_file_tail = '_2013-2017.nc'
    out_dir = real_dir(out_dir)

    # Read the grids
    mit_grid = Grid(mit_grid_dir)
    oce_grid_file = lens_file_head + var_lens_oce[0] + lens_file_tail
    oce_tlat = read_netcdf(oce_grid_file, 'TLAT')
    oce_tlon = fix_lon_range(read_netcdf(oce_grid_file, 'TLONG'))
    oce_ulat = read_netcdf(oce_grid_file, 'ULAT')
    oce_ulon = fix_lon_range(read_netcdf(oce_grid_file, 'ULONG'))
    oce_z = -1*read_netcdf(oce_grid_file, 'z_t')*1e-2
    oce_nx = oce_tlon.shape[1]
    oce_ny = oce_tlat.shape[0]
    oce_nz = oce_z.size
    ice_grid_file = lens_file_head + var_lens_ice[0] + lens_file_tail
    ice_tlat = read_netcdf(ice_grid_file, 'TLAT')
    ice_tlon = fix_lon_range(read_netcdf(ice_grid_file, 'TLON'))
    ice_ulat = read_netcdf(ice_grid_file, 'ULAT')
    ice_ulon = fix_lon_range(read_netcdf(ice_grid_file, 'ULON'))
    ice_nx = ice_tlon.shape[1]
    ice_ny = ice_tlat.shape[0]

    # Loop over boundaries
    for bdry in bdry_loc:
        # Find the location of this boundary (lat/lon)
        loc0_centre, loc0_edge = find_obcs_boundary(mit_grid, bdry)
        # Loop over domains (ocean + ice)
        for var_obcs, obcs_gtype, var_lens, lens_gtype, tlat, tlon, ulat, ulon, nx, ny, num_var, oce in zip([var_obcs_oce, var_obcs_ice], [obcs_gtype_oce, obcs_gtype_ice], [var_lens_oce, var_lens_ice], [lens_gtype_oce, lens_gtype_ice], [oce_tlat, ice_tlat], [oce_tlon, ice_tlon], [oce_ulat, ice_ulat], [oce_ulon, ice_ulon], [oce_nx, ice_nx], [oce_ny, ice_ny], [num_var_oce, num_var_ice], [True, False]):
            # Loop over variables
            for v in range(num_var):
                print('Processing ' + var_lens[v] + ' on ' + bdry + ' boundary')
                
                # Read the data from LENS and OBCS
                lens_file = lens_file_head + var_lens[v] + lens_file_tail
                lens_clim = read_netcdf(lens_file, var_lens[v])
                obcs_file = obcs_file_head + bdry + var_obcs[v] + obcs_file_tail
                if bdry in ['N', 'S']:
                    dimensions = 'x'
                elif bdry in ['E', 'W']:
                    dimensions = 'y'
                if oce:
                    dimensions += 'z'
                dimensions += 't'
                obcs_clim_bdry = read_binary(obcs_file, [mit_grid.nx, mit_grid.ny, mit_grid.nz], dimensions)
                    
                # Select the correct grid type in LENS
                if lens_gtype[v] == 't':
                    lens_lat = tlat
                    lens_lon = tlon
                elif lens_gtype[v] == 'u':
                    lens_lat = ulat
                    lens_lon = ulon
                # Select the correct boundary location in MITgcm
                if bdry in ['N', 'S'] and obcs_gtype[v] == 'v':
                    loc0 = loc0_edge
                elif bdry in ['E', 'W'] and obcs_gtype[v] == 'u':
                    loc0 = loc0_edge
                else:
                    loc0 = loc0_centre
                mit_lon, mit_lat = mit_grid.get_lon_lat(gtype=obcs_gtype[v], dim=1)
                mit_hfac = get_hfac_bdry(mit_grid, bdry, gtype=obcs_gtype[v])
                if not oce:
                    mit_hfac = mit_hfac[0,:]
                if bdry in ['N', 'S']:
                    lens_h_2d = lens_lon
                    mit_n = mit_grid.nx
                    mit_h = mit_lon
                    direction = 'lat'
                elif bdry in ['E', 'W']:
                    lens_h_2d = lens_lat
                    mit_n = mit_grid.ny
                    mit_h = mit_lat
                    direction = 'lon'
                
                # Calculate interpolation coefficients to select this boundary in LENS
                i1, i2, c1, c2 = interp_slice_helper_nonreg(lens_lon, lens_lat, loc0, direction)
                # Now select the boundary in LENS
                lens_clim_bdry = extract_slice_nonreg(lens_clim, direction, i1, i2, c1, c2)
                lens_h = extract_slice_nonreg(lens_h_2d, direction, i1, i2, c1, c2)
                if direction == 'lon' and oce:
                    # Throw away the northern hemisphere because the tripolar? grid causes interpolation issues
                    lens_clim_bdry, lens_h = trim_slice(lens_clim_bdry, lens_h, hmax=0, lon=True)

                # Now interpolate this slice of LENS data to the MITgcm grid on the boundary, one month at a time.
                shape = [months_per_year]
                if oce:
                    shape += [mit_grid.nz]
                shape += [mit_n]
                lens_clim_bdry_interp = np.zeros(shape)
                for t in range(months_per_year):
                    print('...interpolating month '+str(t+1)+' of '+str(months_per_year))
                    lens_clim_bdry_interp[t,:] = interp_bdry(lens_h, oce_z, lens_clim_bdry[t,:], np.invert(lens_clim_bdry[t,:].mask).astype(float), mit_h, mit_grid.z, mit_hfac, lon=(bdry in ['N', 'S']), depth_dependent=oce)
                    
                # Now get the bias correction
                lens_offset = obcs_clim_bdry - lens_clim_bdry_interp
                # Save to file
                out_file = out_dir + 'LENS_offset_' + var_lens[v] + '_' + bdry
                write_binary(lens_offset, out_file)


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
def calc_obcs_trends_lens (var_name, bdry, tmp_file, fig_name=None):

    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/'
    num_ens = 40
    start_year = 2006
    end_year = 2100
    num_years = end_year - start_year + 1
    p0 = 0.05
    if var_name == 'TEMP':
        units = deg_string+'C'
    elif var_name == 'SALT':
        units = 'psu'
    else:
        print('Error (calc_obcs_trends_lens): unknown variable ' + var_name)
        sys.exit()

    # Read POP grid
    grid_file = find_cesm_file('LENS', var_name, 'oce', 'monthly', 1, start_year)[0]
    lon, lat, z, nx, ny, nz = read_pop_grid(grid_file)

    # Read MITgcm grid and get boundary location
    mit_grid = Grid(mit_grid_dir)
    loc0 = find_obcs_boundary(mit_grid, bdry)[0]
    # Find interpolation coefficients to the POP grid
    if bdry in ['N', 'S']:
        direction = 'lat'
        h_2d = lon
    elif bdry in ['E', 'W']:
        direction = 'lon'
        h_2d = lat
    i1, i2, c1, c2 = interp_slice_helper_nonreg(lon, lat, loc0, direction)
    # Interpolate the horizontal axis to this boundary
    h = extract_slice_nonreg(h_2d, direction, i1, i2, c1, c2)
    nh = h.size
    # Trim to MITgcm grid
    h_trim = trim_slice_to_grid(h, h, mit_grid, direction)[0]
    nh_trim = h_trim.size

    if not os.path.isfile(tmp_file):
        # Calculate the trends
        trends = np.ma.zeros([num_ens, nz, nh_trim])
        # Loop over ensemble members
        for n in range(num_ens):
            print('Processing ensemble member ' + str(n+1))
            data_ens = np.ma.empty([num_years, nz, nh])
            for year in range(start_year, end_year+1):
                file_path, t0, tf = find_cesm_file('LENS', var_name, 'oce', 'monthly', n+1, year)
                print('...processing indices '+str(t0)+'-'+str(tf-1)+' from '+file_path)
                # Read just this year
                data_tmp = read_netcdf(file_path, var_name, t_start=t0, t_end=tf)
                # Annually average
                ndays = np.array([days_per_month(month+1, year) for month in range(12)])
                data_tmp = np.average(data_tmp, axis=0, weights=ndays)
                # Interpolate to the boundary
                data_ens[year-start_year,:] = extract_slice_nonreg(data_tmp, direction, i1, i2, c1, c2)
            # Now trim to the other boundaries
            data_ens = trim_slice_to_grid(data_ens, h, mit_grid, direction)[0]
            # Loop over each point and calculate trends
            print('...calculating trends')
            for k in range(nz):
                for j in range(nh_trim):
                    trends[n,k,j] = linregress(np.arange(num_years), data_ens[:,k,j])[0]
        # Save results in temporary binary file
        write_binary(trends, tmp_file)
    else:
        # Trends have been precomputed; read them (hack to assume nx=ny=nh)
        trends = read_binary(tmp_file, [nh_trim, nh_trim, nz], 'yzt')

    # Calculate the mean trend and significance
    mean_trend = np.mean(trends, axis=0)*1e2  # Per century
    p_val = ttest_1samp(trends, 0, axis=0)[1]
    # For any trends which aren't significant, fill with zeros
    mean_trend[p_val > p0] = 0

    # Plot
    fig, ax = plt.subplots()
    cmap, vmin, vmax = set_colours(mean_trend, ctype='plusminus')
    img = ax.pcolormesh(h_trim, z, mean_trend, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img)
    plt.title(var_name+' trend at '+bdry+' boundary ('+units+'/century)', fontsize=14)
    finished_plot(fig, fig_name=fig_name)


# Calculate a monthly climatology of T, S, and z from LENS in normalised potential density space: ensemble mean over 40 members, climatology over 1998-2017 for comparison with WOA at each boundary.
def calc_lens_climatology_density_space (out_dir='./'):

    out_dir = real_dir(out_dir)
    var_names = ['TEMP', 'SALT', 'z']
    start_year = 1998  # for climatology
    end_year = 2017
    num_years = end_year-start_year+1
    num_ens = 40
    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/'
    nrho = 100
    bdry_loc = ['N', 'W', 'E']
    num_bdry = len(bdry_loc)
    num_var = len(var_names)
    out_file_head = 'LENS_climatology_density_space_'
    out_file_tail = '_'+str(start_year)+'-'+str(end_year)

    # Read/generate grids
    grid_file = find_cesm_file('LENS', var_names[0], 'oce', 'monthly', 1, start_year)[0]
    lon, lat, z_1d, nx, ny, nz = read_pop_grid(grid_file)
    mit_grid = Grid(mit_grid_dir)
    loc0 = [find_obcs_boundary(mit_grid, bdry)[0] for bdry in bdry_loc]
    rho_axis = np.linspace(0, 1, num=nrho)

    for b in range(num_bdry):
        print('Processing '+bdry_loc[b]+' boundary')
        if bdry_loc[b] in ['N', 'S']:
            direction = 'lat'
            h_2d = lon
            mit_h = mit_grid.lon_1d
        elif bdry_loc[b] in ['E', 'W']:
            direction = 'lon'
            h_2d = lat
            mit_h = mit_grid.lat_1d
        # Get interpolation coefficients to extract the slice at this boundary
        i1, i2, c1, c2 = interp_slice_helper_nonreg(lon, lat, loc0[b], direction)
        # Interpolate the horizontal axis to this boundary
        h_full = extract_slice_nonreg(h_2d, direction, i1, i2, c1, c2)
        h = trim_slice_to_grid(h_full, h_full, mit_grid, direction)[0]
        nh = h.size
        # Tile h and z over the slice to make them 2D
        h = np.tile(h, (nz, 1))
        z = np.tile(np.expand_dims(z_1d, 1), (1, nh))
        # Set up array for monthly climatology of T, S, z in density space
        lens_clim = np.ma.zeros([num_var, months_per_year, nrho, mit_h.size])
        # Loop over ensemble members and time indices
        for n in range(num_ens):
            print('Processing ensemble member '+str(n+1))
            for year in range(start_year, end_year+1):
                print('...'+str(year))
                for month in range(months_per_year):
                    # Read temperature and salinity and slice to boundary
                    ts_slice = np.ma.empty([num_var-1, nz, nh])
                    for v in range(num_var-1):
                        file_path, t0_year, tf_year = find_cesm_file('LENS', var_names[v], 'oce', 'monthly', n+1, year)
                        t0 = t0_year+month
                        data_3d = read_netcdf(file_path, var_names[v], t_start=t0, t_end=t0+1)
                        data_slice = extract_slice_nonreg(data_3d, direction, i1, i2, c1, c2)
                        data_slice = trim_slice_to_grid(data_slice, h_full, mit_grid, direction, warn=False)[0]
                        ts_slice[v,:] = data_slice
                    # Calculate potential density at this slice
                    rho = potential_density('MDJWF', ts_slice[1,:], ts_slice[0,:])
                    # Apply land mask
                    rho = np.ma.masked_where(ts_slice[1,:].mask, rho)
                    # Normalise to the range 0-1
                    rho_min = np.amin(rho)
                    rho_max = np.amax(rho)
                    rho_norm = (rho - rho_min)/(rho_max - rho_min)
                    # Now fill the land mask with something higher than the highest density
                    rho_norm[ts_slice[1,:].mask] = 1.1
                    # Regrid each variable to the new density axis, at the same time as to the MITgcm horizontal axis, and accumulate climatology
                    for data, v in zip([ts_slice[0,:], ts_slice[1,:], z], np.arange(num_var)):
                        lens_clim[v,month,:] += interp_nonreg_xy(h, rho_norm, data, mit_h, rho_axis, fill_mask=True)
        # Convert from integral to average
        lens_clim /= (num_ens*num_years)
        # Save to binary file
        for v in range(num_var):
            write_binary(lens_clim[v,:], out_dir+out_file_head+var_names[v]+'_'+bdry_loc[b]+out_file_tail)


# Calculate the WOA climatology of T, S, and z in normalised potential density space, as above.
def calc_woa_density_space (out_dir='./'):

    out_dir = real_dir(out_dir)
    base_dir = '/data/oceans_output/shelf/kaight/'
    in_dir = base_dir + 'ics_obcs/PAS/'
    grid_dir = base_dir + 'mitgcm/PAS_grid/'
    bdry_loc = ['N', 'W', 'E']
    var_names_in = ['theta', 'salt']
    var_names_out = ['TEMP', 'SALT', 'z']
    file_head_in = in_dir + 'OB'
    file_tail_in = '_woa_mon.bin'
    file_head_out = out_dir + 'WOA_density_space_'
    num_var = len(var_names_out)
    num_bdry = len(bdry_loc)
    nrho = 100

    grid = Grid(grid_dir)
    rho_axis = np.linspace(0, 1, num=nrho)

    for b in range(num_bdry):
        print('Processing '+bdry_loc[b]+' boundary')
        if bdry_loc[b] in ['N', 'S']:
            h = grid.lon_1d
            nh = grid.nx
            dimensions = 'xzt'
        elif bdry_loc[b] in ['E', 'W']:
            h = grid.lat_1d
            nh = grid.ny
            dimensions = 'yzt'
        h_2d = np.tile(h, (grid.nz, 1))
        z = np.tile(np.expand_dims(grid.z, 1), (1, nh))
        if bdry_loc[b] == 'N':
            hfac = grid.hfac[:,-1,:]
        elif bdry_loc[b] == 'S':
            hfac = grid.hfac[:,0,:]
        elif bdry_loc[b] == 'E':
            hfac = grid.hfac[:,:,-1]
        elif bdry_loc[b] == 'W':
            hfac = grid.hfac[:,:,0]
        # Read temperature and salinity at this boundary
        ts_bdry = np.ma.empty([num_var-1, months_per_year, grid.nz, nh])
        for v in range(num_var-1):
            file_path = file_head_in + bdry_loc[b] + var_names_in[v] + file_tail_in
            ts_bdry[v,:] = read_binary(file_path, [grid.nx, grid.ny, grid.nz], dimensions)
        woa_clim = np.ma.zeros([num_var, months_per_year, nrho, nh])            
        for month in range(months_per_year):
            # Calculate potential density at this boundary for this month
            rho = potential_density('MDJWF', ts_bdry[1,month,:], ts_bdry[0,month,:])
            # Apply land mask
            rho = np.ma.masked_where(hfac==0, rho)
            # Normalise to the range 0-1
            rho_min = np.amin(rho)
            rho_max = np.amax(rho)
            rho_norm = (rho - rho_min)/(rho_max - rho_min)
            # Now fill the land mask with something even higher than the highest density (shouldn't mess up the interpolation that way)
            rho_norm[hfac==0] = 1.1
            # Regrid to the new density axis
            for data, v in zip([ts_bdry[0,month,:], ts_bdry[1,month,:], z], np.arange(num_var)):
                woa_clim[v,month,:] = interp_nonreg_xy(h_2d, rho_norm, data, h, rho_axis, fill_mask=True)
        # Save to binary file
        for v in range(num_var):
            write_binary(woa_clim[v,:], file_head_out+var_names_out[v]+'_'+bdry_loc[b])


# Calculate offsets for LENS with respect to WOA
def calc_lens_offsets_density_space (in_dir='./', out_dir='./'):

    in_dir = real_dir(in_dir)
    out_dir = real_dir(out_dir)    
    var_names = ['TEMP', 'SALT', 'z']
    bdry_loc = ['N', 'W', 'E']
    lens_file_head = in_dir+'LENS_climatology_density_space_'
    lens_file_tail = '_1998-2017'
    woa_file_head = in_dir+'WOA_density_space_'
    out_file_head = out_dir+'LENS_offset_density_space_'
    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/'
    nrho = 100
    
    grid = Grid(mit_grid_dir)

    for bdry in bdry_loc:
        if bdry in ['N', 'S']:
            nh = grid.nx
            dimensions = 'xzt'
        elif bdry in ['E', 'W']:
            nh = grid.ny
            dimensions = 'yzt'
        for var in var_names:
            woa_data = read_binary(woa_file_head+var+'_'+bdry, [grid.nx, grid.ny, nrho], dimensions)
            lens_data = read_binary(lens_file_head+var+'_'+bdry+lens_file_tail, [grid.nx, grid.ny, nrho], dimensions)
            lens_offset = woa_data - lens_data
            write_binary(lens_offset, out_file_head+var+'_'+bdry)


# Helper function to read and correct the LENS temperature and salinity for a given year, month, boundary, and ensemble member. Both month and ens are 1-indexed.
def read_correct_lens_density_space (bdry, ens, year, month, in_dir='/data/oceans_output/shelf/kaight/CESM_bias_correction/obcs/', mit_grid_dir='/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/', return_raw=False):

    in_dir = real_dir(in_dir)
    file_head = in_dir+'LENS_offset_density_space_'
    var_names = ['TEMP', 'SALT', 'z']
    num_var = len(var_names)
    nrho = 100

    # Read the grids and slice to boundary
    grid = Grid(mit_grid_dir)
    lens_grid_file = find_cesm_file('LENS', var_names[0], 'oce', 'monthly', 1, year)[0]
    lens_lon, lens_lat, lens_z, lens_nx, lens_ny, lens_nz = read_pop_grid(lens_grid_file)
    loc0 = find_obcs_boundary(grid, bdry)[0]
    if bdry in ['N', 'S']:
        direction = 'lat'
        dimensions = 'xzt'
        lens_h_2d = lens_lon
        mit_h = grid.lon_1d
    elif bdry in ['E', 'W']:
        direction = 'lon'
        dimensions = 'yzt'
        lens_h_2d = lens_lat
        mit_h = grid.lat_1d
    hfac = get_hfac_bdry(grid, bdry)
    i1, i2, c1, c2 = interp_slice_helper_nonreg(lens_lon, lens_lat, loc0, direction)
    lens_h_full = extract_slice_nonreg(lens_h_2d, direction, i1, i2, c1, c2)
    lens_h = trim_slice_to_grid(lens_h_full, lens_h_full, grid, direction)[0]
    lens_nh = lens_h.size
    lens_h = np.tile(lens_h, (lens_nz, 1))
    lens_z = np.tile(np.expand_dims(lens_z, 1), (1, lens_nh))
    rho_axis = np.linspace(0, 1, num=nrho)
    mit_h_2d = np.tile(mit_h, (nrho, 1))

    # Read and slice temperature and salinity for this month
    lens_ts_z = np.ma.empty([num_var-1, lens_nz, lens_nh])
    for v in range(num_var-1):
        file_path, t0_year, tf_year = find_cesm_file('LENS', var_names[v], 'oce', 'monthly', ens, year)
        t0 = t0_year + month-1
        data_3d = read_netcdf(file_path, var_names[v], t_start=t0, t_end=t0+1)
        data_slice = extract_slice_nonreg(data_3d, direction, i1, i2, c1, c2)
        data_slice = trim_slice_to_grid(data_slice, lens_h_full, grid, direction, warn=False)[0]
        lens_ts_z[v,:] = data_slice
    # Calculate potential density, mask, normalise, and fill as before
    lens_rho_z = potential_density('MDJWF', lens_ts_z[1,:], lens_ts_z[0,:])
    lens_mask = lens_ts_z[1,:].mask
    lens_rho_z = np.ma.masked_where(lens_mask, lens_rho_z)
    rho_min = np.amin(lens_rho_z, axis=0)
    rho_max = np.amax(lens_rho_z, axis=0)
    lens_rho_norm = (lens_rho_z - rho_min[None,:])/(rho_max[None,:] - rho_min[None,:])
    lens_rho_norm[lens_mask] = 1.1
    # Regrid each variable to the new density axis and the MITgcm horizontal axis, then apply corrections
    lens_corrected_density = np.ma.empty([num_var, nrho, mit_h.size])
    for data, v in zip([lens_ts_z[0,:], lens_ts_z[1,:], lens_z], np.arange(num_var)):
        data_interp_density = interp_nonreg_xy(lens_h, lens_rho_norm, data, mit_h, rho_axis, fill_mask=True)
        file_path_corr = file_head + var_names[v] + '_' + bdry
        corr = read_binary(file_path_corr, [mit_h.size, mit_h.size, nrho], dimensions)[month-1,:]
        lens_corrected_density[v,:] = data_interp_density + corr
    # Now regrid back to z-space on the MITgcm grid and apply the land mask
    lens_corrected_z = np.ma.empty([num_var, grid.nz, mit_h.size])
    for v in range(num_var-1):
        data_interp_z = interp_nonreg_xy(mit_h_2d, lens_corrected_density[-1,:], lens_corrected_density[v,:], mit_h, grid.z, fill_mask=True)
        lens_corrected_z[v,:] = np.ma.masked_where(hfac==0, data_interp_z)

    if return_raw:
        return lens_corrected_z[0,:], lens_corrected_z[1,:], lens_ts_z[0,:], lens_ts_z[1,:], lens_h, lens_z
    else:
        return lens_corrected_z[0,:], lens_corrected_z[1,:]
    
        
# Plot the LENS and WOA density space climatologies and the offset, for the given variable, boundary, and month (1-indexed)
def plot_lens_offsets_density_space (var, bdry, month, in_dir='./', fig_name=None):

    in_dir = real_dir(in_dir)
    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/'
    nrho = 100
    lens_file = in_dir+'LENS_climatology_density_space_'+var+'_'+bdry+'_1998-2017'
    woa_file = in_dir+'WOA_density_space_'+var+'_'+bdry
    offset_file = in_dir+'LENS_offset_density_space_'+var+'_'+bdry
    file_paths = [lens_file, woa_file, offset_file]
    titles = ['LENS climatology', 'WOA climatology', 'LENS offset']
    num_panels = len(titles)

    grid = Grid(mit_grid_dir)
    rho_axis = np.linspace(0, 1, num=nrho)
    if bdry in ['N', 'S']:
        h = grid.lon_1d
        nh = grid.nx
        dimensions = 'xzt'
    elif bdry in ['E', 'W']:
        h = grid.lat_1d
        nh = grid.ny
        dimensions = 'yzt'
    hfac = get_hfac_bdry(grid, bdry)
    hfac_sum = np.tile(np.sum(hfac, axis=0), (nrho, 1))

    data = np.ma.empty([num_panels, nrho, nh])
    for n in range(num_panels):
        data_tmp = read_binary(file_paths[n], [grid.nx, grid.ny, nrho], dimensions)[month-1,:]
        data[n,:] = np.ma.masked_where(hfac_sum==0, data_tmp)
    fig, gs, cax1, cax2 = set_panels('1x3C2')
    cax = [cax1, None, cax2]
    for n in range(num_panels):
        ax = plt.subplot(gs[0,n])
        if n < 2:
            cmap, vmin, vmax = set_colours(data[:2,:])
        else:
            cmap, vmin, vmax = set_colours(data[n,:], ctype='plusminus')
        if n > 0:
            ax.set_yticklabels([])
        img = ax.pcolormesh(h, rho_axis, data[n,:], cmap=cmap, vmin=vmin, vmax=vmax)
        if cax[n] is not None:
            plt.colorbar(img, cax=cax[n])
        ax.set_ylim([1, 0])  # Highest density at bottom
        ax.set_title(titles[n], fontsize=14)
    plt.suptitle(var+' at '+bdry+' boundary, month '+str(month), fontsize=18)
    finished_plot(fig, fig_name=fig_name)


def plot_all_offsets_density_space (in_dir='./'):

    for bdry in ['N', 'W', 'E']:
        for var in ['TEMP', 'SALT', 'z']:
            for month in range(12):
                plot_lens_offsets_density_space(var, bdry, month+1, in_dir=in_dir)
                

# Scale temperature and salinity in the LENS climatology for each boundary, so the given min and max annual mean T and S over each boundary become the same as WOA.
def scale_lens_climatology (out_dir='./'):

    out_dir = real_dir(out_dir)
    base_dir = '/data/oceans_output/shelf/kaight/'
    mit_grid_dir = base_dir + 'mitgcm/PAS_grid/'
    lens_dir = base_dir + 'CESM_bias_correction/obcs/'
    woa_dir = base_dir + 'ics_obcs/PAS/'
    var_lens = ['TEMP', 'SALT']
    var_woa = ['theta', 'salt']
    file_head_woa = woa_dir + 'OB'
    file_tail_woa = '_woa_mon.bin'
    file_head_lens = lens_dir + 'LENS_climatology_'
    file_tail_lens = '_1998-2017'
    file_head_lens_out = lens_dir + 'LENS_climatology_scaled_'
    sources = ['WOA', 'LENS']
    num_sources = 2
    num_var = len(var_lens)
    ndays = np.array([days_per_month(t+1, 1998) for t in range(12)])
    bdry_loc = ['N', 'W', 'E']
    num_bdry = len(bdry_loc)

    grid = Grid(mit_grid_dir)

    # Loop over boundaries
    for bdry in bdry_loc:
        print(bdry + ' boundary:')
        hfac = get_hfac_bdry(grid, bdry)
        if bdry in ['N', 'S']:
            nh = grid.nx
            dimensions = 'xzt'
        elif bdry in ['E', 'W']:
            nh = grid.ny
            dimensions = 'yzt'
        if bdry == 'N':
            dV_bdry = grid.dV[:,-1,:]
        elif bdry == 'S':
            dV_bdry = grid.dV[:,0,:]
        elif bdry == 'E':
            dV_bdry = grid.dV[:,:,-1]
        elif bdry == 'W':
            dV_bdry = grid.dV[:,:,0]
        hfac_time = add_time_dim(hfac, months_per_year)
        # Read the data
        ts_data = np.ma.empty([num_sources, num_var, months_per_year, grid.nz, nh])
        for n in range(num_sources):
            for v in range(num_var):
                if n == 0:
                    # WOA
                    file_path = file_head_woa + bdry + var_woa[v] + file_tail_woa
                else:
                    # LENS
                    file_path = file_head_lens + var_lens[v] + '_' + bdry + file_tail_lens
                data_tmp = read_binary(file_path, [grid.nx, grid.ny, grid.nz], dimensions)
                ts_data[n,v,:] = np.ma.masked_where(hfac_time==0, data_tmp)
        # Correct based on annual mean
        ts_annual_mean = np.average(ts_data, axis=2, weights=ndays)
        ts_min = np.amin(ts_annual_mean, axis=(-2,-1))
        ts_max = np.amax(ts_annual_mean, axis=(-2,-1))
        for v in range(num_var):
            # Normalise the full LENS data
            lens_data = ts_data[1,v,:]
            lens_data_norm = (lens_data - ts_min[1,v])/(ts_max[1,v] - ts_min[1,v])
            # Invert the normalisation using the WOA limits
            lens_data_scaled_final = lens_data_norm*(ts_max[0,v] - ts_min[0,v]) + ts_min[0,v]
            # Write to file
            file_path = file_head_lens_out + var_lens[v] + '_' + bdry
            write_binary(lens_data_scaled_final, file_path)
        '''# Correct one month at a time
        for v in range(num_var):                 
            # Correct one month at a time
            lens_data_scaled_final = np.ma.empty(ts_data.shape[2:])
            for t in range(months_per_year):
                # Calculate min, max, and volume-mean for each source
                vmin = np.amin(ts_data[:,v,t,:,:], axis=(-2,-1))
                vmax = np.amax(ts_data[:,v,t,:,:], axis=(-2,-1))
                vmean = np.empty([num_sources])
                for n in range(num_sources):
                    vmean[n] = np.sum(ts_data[n,v,t,:,:]*hfac*dV_bdry)/np.sum(hfac*dV_bdry)
                # Piecewise-normalise the LENS data
                lens_data = ts_data[1,v,t,:]
                lens_data_scaled_full = np.ma.empty(lens_data.shape)
                # Start with everything below the mean
                index = lens_data < vmean[1]
                lens_data_norm = (lens_data - vmin[1])/(vmean[1] - vmin[1])
                lens_data_scaled = lens_data_norm*(vmean[0] - vmin[0]) + vmin[0]
                lens_data_scaled_full[index] = lens_data_scaled[index]
                # Now everything above the mean
                index = lens_data >= vmean[1]
                lens_data_norm = (lens_data - vmean[1])/(vmax[1] - vmean[1])
                lens_data_scaled = lens_data_norm*(vmax[0] - vmean[0]) + vmean[0]
                lens_data_scaled_full[index] = lens_data_scaled[index]
                lens_data_scaled_final[t,:] = lens_data_scaled_full
            # Write to file
            file_path = file_head_lens_out + var_lens[v] + '_' + bdry
            write_binary(lens_data_scaled_final, file_path)'''


# As above but using scaling instead of density correction
def read_correct_lens_scaled (bdry, ens, year, month, in_dir='/data/oceans_output/shelf/kaight/CESM_bias_correction/obcs/', mit_grid_dir='/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/', return_raw=False):

    in_dir = real_dir(in_dir)
    file_head = 'LENS_climatology_'
    file_head_corr = 'LENS_climatology_scaled_'
    file_tail = '_1998-2017'
    var_names = ['TEMP', 'SALT']
    num_var = len(var_names)

    # Read the grids and slice to boundary
    grid = Grid(mit_grid_dir)
    lens_grid_file = find_cesm_file('LENS', var_names[0], 'oce', 'monthly', 1, year)[0]
    lens_lon, lens_lat, lens_z, lens_nx, lens_ny, lens_nz = read_pop_grid(lens_grid_file)
    loc0 = find_obcs_boundary(grid, bdry)[0]
    if bdry in ['N', 'S']:
        direction = 'lat'
        dimensions = 'xzt'
        lens_h_2d = lens_lon
        mit_h = grid.lon_1d
    elif bdry in ['E', 'W']:
        direction = 'lon'
        dimensions = 'yzt'
        lens_h_2d = lens_lat
        mit_h = grid.lat_1d
    hfac = get_hfac_bdry(grid, bdry)
    i1, i2, c1, c2 = interp_slice_helper_nonreg(lens_lon, lens_lat, loc0, direction)
    lens_h_full = extract_slice_nonreg(lens_h_2d, direction, i1, i2, c1, c2)
    lens_h = trim_slice_to_grid(lens_h_full, lens_h_full, grid, direction)[0]
    lens_nh = lens_h.size
    lens_h = np.tile(lens_h, (lens_nz, 1))
    lens_z = np.tile(np.expand_dims(lens_z, 1), (1, lens_nh))

    # Loop over variables
    lens_raw = np.ma.empty([num_var, lens_nz, lens_nh])
    lens_corrected = np.ma.empty([num_var, grid.nz, mit_h.size])
    for v in range(num_var):
        file_path, t0_year, tf_year = find_cesm_file('LENS', var_names[v], 'oce', 'monthly', ens, year)
        t0 = t0_year + month-1
        data_3d = read_netcdf(file_path, var_names[v], t_start=t0, t_end=t0+1)
        data_slice = extract_slice_nonreg(data_3d, direction, i1, i2, c1, c2)
        data_slice = trim_slice_to_grid(data_slice, lens_h_full, grid, direction, warn=False)[0]
        lens_raw[v,:] = data_slice
        # Interpolate to the MITgcm grid
        data_interp = interp_nonreg_xy(lens_h, lens_z, data_slice, mit_h, grid.z, fill_mask=True)
        # Now read baseline climatology and scaled climatology
        lens_clim = read_binary(in_dir + file_head + var_names[v] + '_' + bdry + file_tail, [grid.nx, grid.ny, grid.nz], dimensions)[month-1,:]
        lens_clim_corr = read_binary(in_dir + file_head_corr + var_names[v] + '_' + bdry, [grid.nx, grid.ny, grid.nz], dimensions)[month-1,:]
        lens_corrected[v,:] = np.ma.masked_where(hfac==0, data_interp - lens_clim + lens_clim_corr)
    if return_raw:
        return lens_corrected[0,:], lens_corrected[1,:], lens_raw[0,:], lens_raw[1,:], lens_h, lens_z
    else:
        return lens_corrected[0,:], lens_corrected[1,:]


# Plot T/S diagrams of the WOA and LENS climatologies at the given boundary and month (set month=None for annual mean), and the difference in volumes between the two.
def plot_obcs_ts_lens_woa (bdry, month=None, num_bins=100, fig_name=None, corr=False):

    base_dir = '/data/oceans_output/shelf/kaight/'
    mit_grid_dir = base_dir + 'mitgcm/PAS_grid/'
    lens_dir = base_dir + 'CESM_bias_correction/obcs/'
    woa_dir = base_dir + 'ics_obcs/PAS/'
    var_lens = ['TEMP', 'SALT']
    var_woa = ['theta', 'salt']
    file_head_woa = woa_dir + 'OB'
    file_tail_woa = '_woa_mon.bin'
    if corr:
        file_head_lens = lens_dir + 'LENS_climatology_scaled_'
        file_tail_lens = ''
    else:
        file_head_lens = lens_dir + 'LENS_climatology_'
        file_tail_lens = '_1998-2017'
    num_sources = 2
    num_var = len(var_lens)
    ndays = np.array([days_per_month(t+1, 1998) for t in range(12)])

    # Build the grids
    grid = Grid(mit_grid_dir)
    hfac = get_hfac_bdry(grid, bdry)
    mask = np.invert(hfac==0)
    if bdry == 'N':
        dV_bdry = grid.dV[:,-1,:]
    elif bdry == 'S':
        dV_bdry = grid.dV[:,0,:]
    elif bdry == 'E':
        dV_bdry = grid.dV[:,:,-1]
    elif bdry == 'W':
        dV_bdry = grid.dV[:,:,0]
    if bdry in ['N', 'S']:
        nh = grid.nx
        dimensions = 'xzt'
    elif bdry in ['E', 'W']:
        nh = grid.ny
        dimensions = 'yzt'
    lens_grid_file = find_cesm_file('LENS', var_lens[0], 'oce', 'monthly', 1, 1998)[0]
    lens_lon, lens_lat, lens_z, lens_nx, lens_ny, lens_nz = read_pop_grid(lens_grid_file)
    # Need a few more fields to get the volume integrand
    lens_dA = read_netcdf(lens_grid_file, 'TAREA')*1e-4
    lens_dz = read_netcdf(lens_grid_file, 'dz')*1e-2
    lens_dV = xy_to_xyz(lens_dA, [lens_nx, lens_ny, lens_nz])*z_to_xyz(lens_dz, [lens_nx, lens_ny, lens_z])
    loc0 = find_obcs_boundary(grid, bdry)[0]
    if bdry in ['N', 'S']:
        direction = 'lat'
        dimensions = 'xzt'
        lens_h_2d = lens_lon
    elif bdry in ['E', 'W']:
        direction = 'lon'
        dimensions = 'yzt'
        lens_h_2d = lens_lat
    hfac = get_hfac_bdry(grid, bdry)
    i1, i2, c1, c2 = interp_slice_helper_nonreg(lens_lon, lens_lat, loc0, direction)
    lens_h_full = extract_slice_nonreg(lens_h_2d, direction, i1, i2, c1, c2)
    lens_h = trim_slice_to_grid(lens_h_full, lens_h_full, grid, direction, warn=False)[0]
    lens_nh = lens_h.size
    lens_dV_bdry = extract_slice_nonreg(lens_dV, direction, i1, i2, c1, c2)
    lens_dV_bdry = trim_slice_to_grid(lens_dV_bdry, lens_h_full, grid, direction, warn=False)[0]
        
    # Read the data
    ts_data_woa = np.ma.empty([num_var, grid.nz, nh])
    ts_data_lens = np.ma.empty([num_var, lens_nz, lens_nh])
    for v in range(num_var):
        woa_file = file_head_woa + bdry + var_woa[v] + file_tail_woa
        woa_data = read_binary(woa_file, [grid.nx, grid.ny, grid.nz], dimensions)
        lens_file = file_head_lens + var_lens[v] + '_' + bdry + file_tail_lens
        lens_data = read_binary(lens_file, [lens_nh, lens_nh, lens_nz], dimensions)
        lens_data = np.ma.masked_where(lens_data==0, lens_data)
        if month is None:
            # Annual mean
            ts_data_woa[v,:] = np.average(woa_data, axis=0, weights=ndays)
            ts_data_lens[v,:] = np.average(lens_data, axis=0, weights=ndays)
            month_str = ', annual mean'
        else:
            ts_data_woa[v,:] = woa_data[month-1,:]
            ts_data_lens[v,:] = lens_data[month-1,:]
            month_str = ', month '+str(month)
    mask_lens = np.invert(ts_data_lens[0,:].mask)

    # Bin T and S
    tmin = min(np.amin(ts_data_woa[0,:]), np.amin(ts_data_lens[0,:]))
    tmax = max(np.amax(ts_data_woa[0,:]), np.amax(ts_data_lens[0,:]))
    smin = min(np.amin(ts_data_woa[1,:]), np.amin(ts_data_lens[1,:]))
    smax = max(np.amax(ts_data_woa[1,:]), np.amax(ts_data_lens[1,:]))
    volume_woa, temp_centres, salt_centres, temp_edges, salt_edges = ts_binning(ts_data_woa[0,:], ts_data_woa[1,:], grid, mask, num_bins=num_bins, tmin=tmin, tmax=tmax, smin=smin, smax=smax, bdry=True, dV_bdry=dV_bdry)
    volume_lens = ts_binning(ts_data_lens[0,:], ts_data_lens[1,:], None, mask_lens, num_bins=num_bins, tmin=tmin, tmax=tmax, smin=smin, smax=smax, bdry=True, dV_bdry=lens_dV_bdry)[0]
    volume = [volume_woa, volume_lens]
    # Get difference in volume
    volume_diff = volume[1].data - volume[0].data
    volume.append(np.ma.masked_where(volume_diff==0, volume_diff))
    vmin_abs = min(np.amin(volume[0]), np.amin(volume[1]))
    vmax_abs = max(np.amax(volume[0]), np.amax(volume[1]))
    # Prepare to plot density
    salt_2d, temp_2d = np.meshgrid(np.linspace(smin, smax), np.linspace(tmin, tmax))
    density = potential_density('MDJWF', salt_2d, temp_2d)

    # Plot
    fig, gs, cax1, cax2 = set_panels('1x3C2')
    cax = [cax1, None, cax2]
    cmap_abs = set_colours(volume[0], vmin=vmin_abs, vmax=vmax_abs)[0]
    cmap_diff, vmin_diff, vmax_diff = set_colours(volume[2], ctype='plusminus')
    cmap = [cmap_abs, cmap_abs, cmap_diff]
    vmin = [vmin_abs, vmin_abs, vmin_diff]
    vmax = [vmax_abs, vmax_abs, vmax_diff]
    titles = ['WOA', 'LENS', 'Difference']
    for n in range(num_sources+1):
        ax = plt.subplot(gs[0,n])
        plt.contour(salt_2d, temp_2d, density, colors='DarkGrey', linestyles='dotted')
        img = plt.pcolor(salt_centres, temp_centres, volume[n], vmin=vmin[n], vmax=vmax[n], cmap=cmap[n])
        ax.set_xlim([smin, smax])
        ax.set_ylim([tmin, tmax])
        if n == 0:
            plt.xlabel('Salinity (psu)')
            plt.ylabel('Temperature ('+deg_string+'C)')
        if cax[n] is not None:
            plt.colorbar(img, cax=cax[n])
        ax.set_title(titles[n])
    plt.suptitle('Log of volumes at '+bdry+' boundary' + month_str)
    finished_plot(fig, fig_name=fig_name)


# For a given year, month, variable, boundary, and ensemble member, plot the uncorrected and corrected LENS fields as well as the WOA climatology.
def plot_obcs_corrected (var, bdry, ens, year, month, fig_name=None, option='ts'):

    base_dir = '/data/oceans_output/shelf/kaight/'
    obcs_dir = base_dir + 'ics_obcs/AMUND/'
    grid_dir = base_dir + 'archer2_mitgcm/AMUND_ini_grid/'
    woa_file_mid = '_WOA18.OBCS_'
    if var == 'TEMP':
        woa_var = 'THETA'
        var_title = 'Temperature ('+deg_string+'C)'
    elif var == 'SALT':
        woa_var = 'SALT'
        var_title = 'Salinity (psu)'
    titles = ['WOA', 'LENS', 'LENS corrected']
    num_panels = len(titles)

    grid = Grid(grid_dir)
    if bdry in ['N', 'S']:
        mit_h = grid.lon_1d
        dimensions = 'xzt'
    elif bdry in ['E', 'W']:
        mit_h = grid.lat_1d
        dimensions = 'yzt'
    hfac = get_hfac_bdry(grid, bdry)

    # Read the corrected and uncorrected LENS fields
    if option == 'density':
        lens_temp_corr, lens_salt_corr, lens_temp_raw, lens_salt_raw, lens_h, lens_z = read_correct_lens_density_space(bdry, ens, year, month, return_raw=True)
    elif option == 'scaled':
        lens_temp_corr, lens_salt_corr, lens_temp_raw, lens_salt_raw, lens_h, lens_z = read_correct_lens_scaled(bdry, ens, year, month, return_raw=True)
    elif option == 'ts':
        lens_temp_corr, lens_salt_corr, lens_temp_raw, lens_salt_raw, lens_h, lens_z = read_correct_cesm_ts_space('LENS', bdry, ens, year, month, return_raw=True)
    if var == 'TEMP':
        lens_data_corr = lens_temp_corr
        lens_data_raw = lens_temp_raw
    elif var == 'SALT':
        lens_data_corr = lens_salt_corr
        lens_data_raw = lens_salt_raw
    # Read the WOA fields
    woa_file = obcs_dir + woa_var + woa_file_mid + bdry
    woa_data = read_binary(woa_file, [grid.nx, grid.ny, grid.nz], dimensions)[month-1,:]
    woa_data = np.ma.masked_where(hfac==0, woa_data)

    # Wrap up for plotting
    data = [woa_data, lens_data_raw, lens_data_corr]
    h = [mit_h, lens_h, mit_h]
    z = [grid.z, lens_z, grid.z]
    vmin = min(min(np.amin(woa_data), np.amin(lens_data_raw)), np.amin(lens_data_corr))
    vmax = max(max(np.amax(woa_data), np.amax(lens_data_raw)), np.amax(lens_data_corr))
    cmap = set_colours(data[0], vmin=vmin, vmax=vmax)[0]
    fig, gs, cax = set_panels('1x3C1')
    for n in range(num_panels):
        ax = plt.subplot(gs[0,n])
        img = ax.pcolormesh(h[n], z[n], data[n], cmap=cmap, vmin=vmin, vmax=vmax)
        if n==2:
            plt.colorbar(img, cax=cax, orientation='horizontal')
        if n > 0:
            ax.set_yticklabels([])
        ax.set_title(titles[n], fontsize=14)
    plt.suptitle(var_title+' at '+bdry+' boundary, '+str(year)+'/'+str(month), fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Plot T/S profiles horizontally averaged over the eastern boundary from 70S to the coastline, for a given month (1-indexed) and year. Show the original WOA climatology, the LENS climatology, the uncorrected LENS field from the first ensemble member, and the corrected LENS field.
def plot_obcs_profiles (year, month, fig_name=None):

    base_dir = '/data/oceans_output/shelf/kaight/'
    obcs_dir = base_dir + 'ics_obcs/AMUND/'
    clim_dir = base_dir + 'CESM_bias_correction/AMUND/obcs/'
    grid_dir = base_dir + 'mitgcm/AMUND_ini_grid/'
    woa_file_mid = '_WOA18.OBCS_'
    lens_file_head = clim_dir + 'LENS_climatology_'
    lens_file_tail = '_1998-2017'
    bdry = 'E'
    woa_var = ['THETA', 'SALT']
    lens_var = ['TEMP', 'SALT']
    units = [deg_string+'C', 'psu']
    ymax = -70
    num_var = len(woa_var)
    direction = 'lon'
    ndays = np.array([days_per_month(t+1, year) for t in range(12)])
    titles = ['WOA', 'LENS climatology', 'LENS uncorrected', 'LENS corrected']
    colours = ['blue', 'black', 'green', 'red']
    num_profiles = len(titles)

    # Build the grids
    grid = Grid(grid_dir)    
    lon0 = find_obcs_boundary(grid, bdry)[0]
    hfac_slice = get_hfac_bdry(grid, bdry)
    # Mask out dA north of 70S, tile in the z direction, and select the boundary
    dA = np.ma.masked_where(grid.lat_2d > ymax, grid.dA)
    dA = xy_to_xyz(dA, grid)
    dA_slice = dA[:,:,-1]

    # Read the corrected and uncorrected LENS fields
    lens_temp_corr, lens_salt_corr, lens_temp_raw, lens_salt_raw, lens_h, lens_z = read_correct_cesm_ts_space('LENS', bdry, 1, year, month, return_raw=True) 
    
    profiles = np.ma.empty([num_var, num_profiles, grid.nz])
    # Loop over variables
    for v in range(num_var):
        
        # Read WOA climatology
        woa_data = read_binary(obcs_dir+woa_var[v]+woa_file_mid+bdry, [grid.nx, grid.ny, grid.nz], 'yzt')
        # Extract the right month
        woa_data = woa_data[month-1,:]

        # Choose LENS data
        if lens_var[v] == 'TEMP':
            lens_data_uncorrected = lens_temp_raw
            lens_data_corrected = lens_temp_corr
        elif lens_var[v] == 'SALT':
            lens_data_uncorrected = lens_salt_raw
            lens_data_corrected = lens_salt_corr
        lens_mask = np.invert(lens_data_uncorrected.mask).astype(float)
        # Interpolate the LENS slice to the MITgcm grid
        lens_data_uncorrected = interp_bdry(lens_h, lens_z, lens_data_uncorrected, lens_mask, grid.lat_1d, grid.z, hfac_slice, lon=False, depth_dependent=True)

        # Read LENS climatology
        lens_clim_raw = read_binary(lens_file_head+lens_var[v]+'_'+bdry+lens_file_tail, [lens_h.size, lens_h.size, lens_z.size], 'yzt')
        lens_clim_raw = lens_clim_raw[month-1,:]
        # Interpolate to the MITgcm grid
        lens_clim = interp_bdry(lens_h, lens_z, lens_clim_raw, lens_mask, grid.lat_1d, grid.z, hfac_slice, lon=False, depth_dependent=True)

        # Horizontally average everything south of 70S
        for data_slice, n in zip([woa_data, lens_clim, lens_data_uncorrected, lens_data_corrected], range(num_profiles)):
            profiles[v,n,:] = np.sum(data_slice*hfac_slice*dA_slice, axis=-1)/np.sum(hfac_slice*dA_slice, axis=-1)

    # Plot
    fig, gs = set_panels('1x2C0')
    gs.update(left=0.08, bottom=0.18, top=0.87)
    for v in range(num_var):
        ax = plt.subplot(gs[0,v])
        for n in range(num_profiles):
            ax.plot(profiles[v,n,:], grid.z, color=colours[n], label=titles[n])
        ax.grid(linestyle='dotted')
        ax.set_title(lens_var[v], fontsize=16)
        ax.set_xlabel(units[v])
        if v==0:
            ax.set_ylabel('Depth (m)')
            ax.legend(loc='lower right', bbox_to_anchor=(1.7, -0.2), ncol=num_profiles)
        else:
            ax.set_yticklabels([])
    plt.suptitle(bdry + ' boundary, '+str(year)+'/'+str(month).zfill(2), fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Plot Hovmollers of temp or salt in the given region for all 5 ensemble members, LENS average, and PACE average.
def plot_hovmoller_lens_ensemble (var, region, num_ens=5, base_dir='./', fig_name=None):

    base_dir = real_dir(base_dir)
    grid_dir = base_dir + 'PAS_grid/'
    lens_mean_file = base_dir + 'hovmoller_lens_mean.nc'
    pace_mean_file = base_dir + 'hovmoller_pace_mean.nc'
    file_paths = ['PAS_LENS'+str(n+1).zfill(3)+'/output/hovmoller.nc' for n in range(num_ens)] + [lens_mean_file, pace_mean_file]
    var_name = region + '_' + var
    if var == 'temp':
        var_title = 'Temperature ('+deg_string+'C)'
        vmin = -1.6
        vmax = 1.4
    elif var == 'salt':
        var_title = 'Salinity (psu)'
        vmin = 34
        vmax = 34.8
    suptitle = var_title + ' in ' + region_names[region]
    smooth = 12
    start_year = 1920
    end_year = 2100
    titles = ['LENS '+str(n+1).zfill(3) for n in range(num_ens)] + ['LENS\nmean', 'PACE\nmean']
    
    grid = Grid('PAS_grid/')
    all_data = []
    all_time = []
    for n in range(num_ens+2):
        data = read_netcdf(file_paths[n], var_name)
        time = netcdf_time(file_paths[n], monthly=False)
        t_start = index_year_start(time, start_year)
        data = data[t_start:]
        time = time[t_start:]
        if n == 0:
            mask = data[0,:].mask
        else:
            if not isinstance(data, np.ma.MaskedArray):
                mask_full = add_time_dim(mask, data.shape[0])
                data = np.ma.masked_where(mask_full, data)
        all_data.append(data)
        all_time.append(time)

    fig = plt.figure(figsize=(6,12))
    gs = plt.GridSpec(num_ens+2,1)
    gs.update(left=0.07, right=0.85, bottom=0.04, top=0.95, hspace=0.08)
    cax = fig.add_axes([0.75, 0.96, 0.24, 0.012])
    for n in range(num_ens+2):
        ax = plt.subplot(gs[n,0])
        img = hovmoller_plot(all_data[n], all_time[n], grid, smooth=smooth, ax=ax, make_cbar=False, vmin=vmin, vmax=vmax)
        ax.set_xlim([datetime.date(start_year, 1, 1), datetime.date(end_year, 12, 31)])
        ax.set_xticks([datetime.date(year, 1, 1) for year in np.arange(start_year, end_year, 20)])
        if n == 0:
            ax.set_yticks([0, -500, -1000])
            ax.set_yticklabels(['0', '0.5', '1'])
            ax.set_ylabel('')
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
        if n == 1:
            ax.set_ylabel('Depth (km)', fontsize=10)
        if n != num_ens+1:
            ax.set_xticklabels([])
        ax.set_xlabel('')
        plt.text(1.01, 0.5, titles[n], ha='left', va='center', transform=ax.transAxes, fontsize=11)
    plt.suptitle(suptitle, fontsize=16, x=0.05, ha='left')
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
    cax.xaxis.set_ticks_position('top')
    reduce_cbar_labels(cbar)
    finished_plot(fig, fig_name=fig_name)


# For a given year, month, boundary, and ensemble member, plot the raw anomalies in LENS and the corrected anomalies as applied to WOA, for both temperature and salinity.
def plot_obcs_anomalies (bdry, ens, year, month, fig_name=None, zmin=None):

    base_dir = '/data/oceans_output/shelf/kaight/'
    in_dir = '/data/oceans_output/shelf/kaight/CESM_bias_correction/AMUND/obcs/'
    obcs_dir = base_dir + 'ics_obcs/AMUND/'
    grid_dir = base_dir + 'archer2_mitgcm/AMUND_ini_grid/'
    lens_file_head = in_dir + 'LENS_climatology_'
    lens_file_tail = '_1998-2017'
    lens_var = ['TEMP', 'SALT']
    woa_var = ['THETA', 'SALT']
    woa_file_mid = '_WOA18.OBCS_'
    var_titles = ['Temperature anomaly ('+deg_string+'C)', 'Salinity anomaly (psu)']
    source_titles = ['Uncorrected', 'Corrected']
    num_var = len(lens_var)
    num_sources = len(source_titles)

    grid = Grid(grid_dir)
    if bdry in ['N', 'S']:
        mit_h = grid.lon_1d
        dimensions = 'xzt'
    elif bdry in ['E', 'W']:
        mit_h = grid.lat_1d
        dimensions = 'yzt'
    hfac = get_hfac_bdry(grid, bdry)

    # Read the corrected and uncorrected LENS fields
    temp_corr, salt_corr, lens_temp_raw, lens_salt_raw, lens_h, lens_z = read_correct_cesm_ts_space('LENS', bdry, ens, year, month, return_raw=True)
    lens_nh = lens_h.size
    lens_nz = lens_z.size

    # Read the LENS climatology
    lens_clim = np.ma.empty([num_var, lens_nz, lens_nh])
    for v in range(num_var):
        file_path = lens_file_head + lens_var[v] + '_' + bdry + lens_file_tail
        lens_clim[v,:] = read_binary(file_path, [lens_nh, lens_nh, lens_nz], dimensions)[month-1,:]
    # Calculate uncorrected anomalies
    dtemp_uncorr = lens_temp_raw - lens_clim[0,:]
    dsalt_uncorr = lens_salt_raw - lens_clim[1,:]

    # Read the WOA climatology
    woa_clim = np.ma.empty([num_var, grid.nz, mit_h.size])
    for v in range(num_var):
        file_path = obcs_dir + woa_var[v] + woa_file_mid + bdry
        woa_clim[v,:] = read_binary(file_path, [grid.nx, grid.ny, grid.nz], dimensions)[month-1,:]
    # Calculate corrected anomalies
    dtemp_corr = temp_corr - woa_clim[0,:]
    dsalt_corr = salt_corr - woa_clim[1,:]

    # Wrap up for plotting
    data = [[dtemp_uncorr, dtemp_corr], [dsalt_uncorr, dsalt_corr]]
    h = [lens_h, mit_h]
    z = [lens_z, grid.z]
    vmin = [min(np.amin(dtemp_uncorr), np.amin(dtemp_corr)), min(np.amin(dsalt_uncorr), np.amin(dsalt_corr))]
    vmax = [max(np.amax(dtemp_uncorr), np.amax(dtemp_corr)), max(np.amax(dsalt_uncorr), np.amax(dsalt_corr))]
    cmap = [set_colours(dtemp_uncorr, vmin=vmin[0], vmax=vmax[0], ctype='plusminus')[0], set_colours(dsalt_uncorr, vmin=vmin[1], vmax=vmax[1], ctype='plusminus')[0]]
    fig, gs, cax1, cax2 = set_panels('2x2C2')
    cax = [cax1, cax2]
    for v in range(num_var):
        for n in range(num_sources):
            ax = plt.subplot(gs[v,n])
            img = ax.pcolormesh(h[n], z[n], data[v][n], cmap=cmap[v], vmin=vmin[v], vmax=vmax[v])                
            if n == num_sources-1:
                plt.colorbar(img, cax=cax[v])
                ax.set_yticks([])
            if v == 0:
                ax.set_xticklabels([])
            ax.set_title(source_titles[n], fontsize=14)
            if zmin is not None:
                ax.set_ylim([zmin, 0])
        plt.text(0.45, 0.97-0.49*v, var_titles[v]+' on '+bdry+' boundary, '+str(year)+'/'+str(month), fontsize=16, ha='center', va='center', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_name)
    

# Precompute the trend at every point in every ensemble member, for a bunch of variables. Split it into historical (1920-2005) and each future scenario (2006-2100).
def precompute_ensemble_trends (base_dir='./', num_hist=10, num_LENS=10, num_MENS=10, num_LW2=10, num_LW1=5, out_dir='precomputed_trends/', grid_dir='PAS_grid/'):

    var_names = ['wind_speed'] #['ismr', 'u_bottom100m', 'v_bottom100m', 'vel_bottom100m_speed', 'barotropic_u', 'barotropic_v', 'barotropic_vel_speed', 'THETA', 'temp_btw_200_700m', 'EXFuwind', 'EXFvwind', 'wind_speed', 'oceFWflx'] #['ismr', 'sst', 'sss', 'temp_btw_200_700m', 'salt_btw_200_700m', 'SIfwfrz', 'SIfwmelt', 'EXFatemp', 'EXFpreci', 'EXFuwind', 'EXFvwind', 'wind_speed', 'oceFWflx', 'barotropic_u', 'barotropic_v', 'baroclinic_u_bottom100m', 'baroclinic_v_bottom100m', 'THETA', 'SALT', 'thermocline', 'UVEL', 'VVEL', 'isotherm_0.5C_below_100m', 'isotherm_1.5C_below_100m', 'barotropic_vel_speed', 'baroclinic_vel_bottom100m_speed']
    base_dir = real_dir(base_dir)
    out_dir = real_dir(out_dir)
    periods = ['historical', 'LENS', 'MENS', 'LW2.0', 'LW1.5']
    start_years = [1920, 2006, 2006, 2006, 2006]
    end_years = [2005, 2100, 2080, 2100, 2100]
    num_periods = len(periods)

    for var in var_names:
        if var == 'ismr':
            region = 'ice'
        else:
            region = 'all'
        if var in ['THETA', 'SALT', 'speed', 'ADVx_TH', 'ADVy_TH', 'UVEL', 'VVEL']:
            dim = 3
        else:
            dim = 2
        if var in ['barotropic_u', 'baroclinic_u_bottom100m', 'UVEL', 'ADVx_TH']:
            gtype = 'u'
        elif var in ['barotropic_v', 'baroclinic_v_bottom100m', 'VVEL', 'ADVy_TH']:
            gtype = 'v'
        else:
            gtype = 't'
        for t in range(num_periods):
            print('Calculating '+periods[t]+' trends in '+var)
            out_file = out_dir + var + '_trend_' + periods[t] + '.nc'
            if periods[t] in ['historical', 'LENS']:
                if periods[t] == 'historical':
                    num_ens = num_hist
                else:
                    num_ens = num_LENS
                sim_dir = [base_dir+'PAS_LENS'+str(n+1).zfill(3)+'_O' for n in range(num_ens)]
            else:
                if periods[t] == 'MENS':
                    num_ens = num_MENS
                elif periods[t] == 'LW2.0':
                    num_ens = num_LW2
                elif periods[t] == 'LW1.5':
                    num_ens = num_LW1
                sim_dir = [base_dir+'PAS_'+periods[t]+'_'+str(n+1).zfill(3)+'_O' for n in range(num_ens)]            
            make_trend_file(var, region, sim_dir, grid_dir, out_file, dim=dim, start_year=start_years[t], end_year=end_years[t], gtype=gtype)


# Plot the historical and future trends (in each scenario) for the given variable (precomputed in precompute_ensemble_trends).
def plot_trend_maps (var, trend_dir='precomputed_trends/', grid_dir='PAS_grid/',lon0=-106, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, hmin=None, hmax=None, vmin=None, vmax=None, chunk_x=20, chunk_y=10, fig_name=None):

    if var in ['ismr', 'sst', 'sss', 'temp_btw_200_700m', 'salt_btw_200_700m', 'SIfwfrz', 'SIfwmelt', 'EXFatemp', 'EXFpreci', 'oceFWflx', 'thermocline'] or var.startswith('isotherm'):
        option = 'scalar'
    elif var in ['wind', 'barotropic_vel', 'baroclinic_vel_bottom100m']:
        option = 'vector'
        if var == 'wind':
            threshold = 0.25
            scale = 10
        elif var == 'barotropic_vel':
            threshold = 0.02
            scale = 1
        elif var == 'baroclinic_vel_bottom100m':
            threshold = 0.02
            scale = 0.8
    elif var in ['THETA', 'SALT', 'UVEL', 'VVEL']:
        option = 'slice'
        if var in ['THETA', 'SALT']:
            gtype = 't'
        elif var == 'UVEL':
            gtype = 'u'
        elif var == 'VVEL':
            gtype = 'v'
    else:
        print('Error (plot_trend_maps): unknown variable '+var)
        sys.exit()
    trend_dir = real_dir(trend_dir)
    grid = Grid(grid_dir)
    periods = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    start_years = [1920, 2006, 2006, 2006, 2006]
    end_years = [2005, 2100, 2100, 2080, 2100]
    num_periods = len(periods)
    p0 = 0.05

    # Inner function to read the precomputed trend in each ensemble member, calculate the mean trend, set it to 0 where not significant, and convert to trend per century
    def read_trend (var_in_file):
        file_path = trend_dir + var_in_file + '_trend_' + periods[t] + '.nc'
        trends, long_name, units = read_netcdf(file_path, var_in_file +'_trend', return_info=True)
        mean_trend = np.mean(trends, axis=0)
        t_val, p_val = ttest_1samp(trends, 0, axis=0)
        mean_trend[p_val > p0] = 0
        if var == 'thermocline':
            mean_trend = mask_land_ice(mean_trend, grid)
        return mean_trend*1e2, long_name, units        

    # Calculate the trends for each period/scenario
    if option in ['scalar', 'vector']:
        data_plot = np.ma.empty([num_periods, grid.ny, grid.nx])
    elif option == 'slice':
        data_plot = []
    if option == 'vector':
        data_plot_u = np.ma.empty([num_periods, grid.ny, grid.nx])
        data_plot_v = np.ma.empty([num_periods, grid.ny, grid.nx])
    for t in range(num_periods):
        if option == 'scalar':
            data_plot[t,:], long_name, units = read_trend(var)
        elif option == 'vector':
            if var == 'wind':
                var_u = 'EXFuwind'
                var_v = 'EXFvwind'
            elif var == 'barotropic_vel':
                var_u = 'barotropic_u'
                var_v = 'barotropic_v'
            elif var == 'baroclinic_vel_bottom100m':
                var_u = 'baroclinic_u_bottom100m'
                var_v = 'baroclinic_v_bottom100m'
            trends_u = read_trend(var_u)[0]
            trends_v = read_trend(var_v)[0]
            if var in ['barotropic_vel', 'baroclinic_vel_bottom100m']:
                # Interpolate to tracer grid
                trends_u = interp_grid(trends_u, grid, 'u', 't')
                trends_v = interp_grid(trends_v, grid, 'v', 't')
            # Also read trends in magnitude
            trends, long_name, units = read_trend(var+'_speed')
            index = np.abs(trends) < threshold
            trends_u = np.ma.masked_where(index, trends_u)
            trends_v = np.ma.masked_where(index, trends_v)
            data_plot[t,:] = trends
            data_plot_u[t,:] = trends_u
            data_plot_v[t,:] = trends_v
        elif option == 'slice':
            trend_3d, long_name, units = read_trend(var)
            if t==0:
                patches, values, lon0, hmin, hmax, zmin, zmax, vmin0, vmax0, left, right, below, above = slice_patches(trend_3d, grid, lon0=lon0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, gtype=gtype, return_bdry=True)
            else:
                values, vmin_tmp, vmax_tmp = slice_values(trend_3d, grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, gtype=gtype)
                vmin0 = min(vmin0, vmin_tmp)
                vmax0 = max(vmax0, vmax_tmp)
            data_plot.append(values)            

    if vmin is None:
        if option in ['scalar', 'vector']:
            vmin = np.amin([var_min_max(data_plot[t,:], grid, xmin=xmin, xmax=xmax, ymax=ymax)[0] for t in range(num_periods)])
        elif option == 'slice':
            vmin = vmin0
    if vmax is None:
        if option in ['scalar', 'vector']:
            vmax = np.amax([var_min_max(data_plot[t,:], grid, xmin=xmin, xmax=xmax, ymax=ymax)[1] for t in range(num_periods)])
        elif option == 'slice':
            vmax = vmax0

    # Plot
    fig, gs, cax = set_panels('3x2-1C1')
    for t in range(num_periods):
        ax = plt.subplot(gs[(t+1)//2, (t+1)%2])
        if option in ['scalar', 'vector']:
            img = latlon_plot(data_plot[t,:], grid, ax=ax, make_cbar=False, ctype='plusminus', vmin=vmin, vmax=vmax, ymax=ymax, xmin=xmin, xmax=xmax, title=periods[t], titlesize=14)
            if option == 'vector':
                overlay_vectors(ax, data_plot_u[t,:], data_plot_v[t,:], grid, chunk_x=chunk_x, chunk_y=chunk_y, scale=scale, headwidth=4, headlength=5)
        elif option == 'slice':
            img = make_slice_plot(patches, data_plot[t], lon0, hmin, hmax, zmin, zmax, vmin, vmax, lon0=lon0, ax=ax, make_cbar=False, ctype='plusminus', title=periods[t], titlesize=14)
        if t != 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
    plt.colorbar(img, cax=cax, orientation='horizontal')
    plt.text(0.05, 0.95, long_name+'\n('+units[:-2]+'/century)', fontsize=14, ha='left', va='top', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_name)


# Plot historical and future temperature and salinity trends as a slice through any longitude.
def plot_ts_trend_slice (lon0, ymax=None, tmin=None, tmax=None, smin=None, smax=None, trend_dir='precomputed_trends/', grid_dir='PAS_grid/', fig_name=None):

    trend_dir = real_dir(trend_dir)
    grid = Grid(grid_dir)
    periods = ['historical', 'future']
    num_periods = len(periods)
    p0 = 0.05
    var_names = ['THETA_trend', 'SALT_trend']
    var_titles = [' temperature ('+deg_string+'C)', ' salinity (psu)']
    num_var = len(var_names)

    # Read dat and calculate significant mean trends
    mean_trends = np.ma.empty([num_var, num_periods, grid.nz, grid.ny, grid.nx])
    for v in range(num_var):
        for t in range(num_periods):
            file_path = trend_dir + var_names[v] + '_' + periods[t] + '.nc'
            trends = read_netcdf(file_path, var_names[v])*1e2
            mean_trend_tmp = np.mean(trends, axis=0)
            t_val, p_val = ttest_1samp(trends, 0, axis=0)
            mean_trend_tmp[p_val > p0] = 0
            mean_trends[v,t,:] = mean_trend_tmp
    # Prepare patches and values for slice plots
    values = []
    vmin = [tmin, smin]
    vmax = [tmax, smax]
    for v in range(num_var):
        for t in range(num_periods):
            if v==0 and t==0:
                patches, values_tmp, lon0, ymin, ymax, zmin, zmax, vmin_tmp, vmax_tmp, left, right, below, above = slice_patches(mean_trends[v,t,:], grid, lon0=lon0, hmax=ymax, return_bdry=True)
            else:
                values_tmp, vmin_tmp, vmax_tmp = slice_values(mean_trends[v,t,:], grid, left, right, below, above, ymin, ymax, zmin, zmax, lon0=lon0)
            values.append(values_tmp)
            if vmin[v] is None:
                vmin[v] = vmin_tmp
            elif (v==0 and tmin is None) or (v==1 and smin is None):
                vmin[v] = min(vmin[v], vmin_tmp)
            if vmax[v] is None:
                vmax[v] = vmax_tmp
            elif (v==0 and tmax is None) or (v==1 and smax is None):
                vmax[v] = max(vmax[v], vmax_tmp)

    # Plot
    fig, gs, cax1, cax2 = set_panels('2x2C2')
    cax = [cax1, cax2]
    extend = [get_extend(vmin=tmin, vmax=tmax), get_extend(vmin=smin, vmax=smax)]
    for v in range(num_var):
        for t in range(num_periods):
            ax = plt.subplot(gs[v,t])
            img = make_slice_plot(patches, values[v*num_periods+t], lon0, ymin, ymax, zmin, zmax, vmin[v], vmax[v], lon0=lon0, ax=ax, make_cbar=False, ctype='plusminus', title=None)
            if v==0 and t==0:
                ax.set_ylabel('Depth (m)', fontsize=10)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel('')
                ax.set_ylabel('')
            ax.set_title(periods[t] + var_titles[v], fontsize=14)
        plt.colorbar(img, cax=cax[v], extend=extend[v])
    plt.suptitle('Trends per century through '+lon_label(lon0), fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Helper function to contour shelf break
def contour_shelf_break (grid, ax, colour='black', z_shelf=-1750):

    bathy = grid.bathy
    if z_shelf == -1000:
        bathy[grid.lat_2d < -74.2] = 0
        bathy[(grid.lon_2d > -125)*(grid.lat_2d < -73)] = 0
        bathy[(grid.lon_2d > -110)*(grid.lat_2d < -72)] = 0
    elif z_shelf == -1750:
        bathy[grid.ice_mask] = 0
        bathy[grid.lat_2d > -69] = 2*z_shelf
        bathy[(grid.lon_2d < -110)*(grid.lat_2d > -71)] = 2*z_shelf
    ax.contour(grid.lon_2d, grid.lat_2d, grid.bathy, levels=[z_shelf], colors=(colour), linewidths=1)
        

# Recreate the PACE advection trend map for the historical and future trends in LENS, side by side - now using velocity not advection
def plot_velocity_trend_maps (z0=-400, trend_dir='precomputed_trends/', grid_dir='PAS_grid/', fig_name=None):

    trend_dir = real_dir(trend_dir)
    grid = Grid(grid_dir)
    p0 = 0.05
    threshold = [0.1, 0.25]
    z_shelf = -1000
    periods = ['historical', 'future']
    num_periods = len(periods)
    ymax = -70
    vmax = 1
    
    def read_component (key, period):
        var_name = key+'VEL' 
        trends_3d = read_netcdf(trend_dir+var_name+'_trend_'+period+'.nc', var_name+'_trend')
        trends = interp_to_depth(trends_3d, z0, grid, time_dependent=True)
        mean_trend = np.mean(trends, axis=0)
        t_val, p_val = ttest_1samp(trends, 0, axis=0)
        mean_trend[p_val > p0] = 0
        if key == 'U': 
            gtype = 'u'
            dh = grid.dx_s
        elif key == 'V':
            gtype = 'v'
            dh = grid.dy_w
        trend_interp = interp_grid(mean_trend, grid, gtype, 't')*1e3
        return trend_interp
    magnitude_trend = np.ma.empty([num_periods, grid.ny, grid.nx])
    uvel_trend = np.ma.empty([num_periods, grid.ny, grid.nx])
    vvel_trend = np.ma.empty([num_periods, grid.ny, grid.nx])
    for t in range(num_periods):
        uvel_trend_tmp = read_component('U', periods[t])
        vvel_trend_tmp = read_component('V', periods[t])
        magnitude_trend[t,:] = np.sqrt(uvel_trend_tmp**2 + vvel_trend_tmp**2)
        index = magnitude_trend[t,:] < threshold[t]
        uvel_trend[t,:] = np.ma.masked_where(index, uvel_trend_tmp)
        vvel_trend[t,:] = np.ma.masked_where(index, vvel_trend_tmp)

    fig, gs, cax = set_panels('1x2C1')
    for t in range(num_periods):
        ax = plt.subplot(gs[0,t])
        img = latlon_plot(magnitude_trend[t,:], grid, ax=ax, make_cbar=False, ctype='plusminus', ymax=ymax, title=periods[t], titlesize=14, vmax=vmax)
        contour_shelf_break(grid, ax, colour='blue', z_shelf=z_shelf)
        overlay_vectors(ax, uvel_trend[t,:], vvel_trend[t,:], grid, chunk_x=9, chunk_y=6, scale=8, headwidth=4, headlength=5)
        if t > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.colorbar(img, cax=cax, extend='max', orientation='horizontal')
    plt.suptitle('Trends in ocean velocity at '+str(-z0)+r'm (10$^{-3}$m/s/century)', fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Helper function to read and correct the LENS velocity variables (ocean or sea ice) for OBCS: correction in polar coordinates with respect to SOSE climatology
def read_correct_lens_vel_polar_coordinates (domain, bdry, ens, year, in_dir='/data/oceans_output/shelf/kaight/CESM_bias_correction/obcs/', obcs_dir='/data/oceans_output/shelf/kaight/ics_obcs/PAS/', mit_grid_dir='/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/', return_raw=False, return_clim=False, return_sose_clim=False):

    if domain == 'oce':
        var_names = ['UVEL', 'VVEL']
        var_sose = ['uvel', 'vvel']
    elif domain == 'ice':
        var_names = ['uvel', 'vvel']
        var_sose = ['uice', 'vice']
    num_cmp = len(var_names)
    lens_file_head = real_dir(in_dir) + 'LENS_climatology_'
    lens_file_tail = '_1998-2017'
    sose_file_head = real_dir(obcs_dir) + 'OB' + bdry
    sose_file_tail = '_sose.bin'
    sose_file_tail_alt = '_sose_corr.bin'
    scale_cap = 3

    # Read grids
    # Will do almost everything on the MITgcm tracer grid - introduces negligible errors
    mit_grid = Grid(mit_grid_dir)
    hfac = []  # Get on all 3 grids
    for gtype in ['t', 'u', 'v']:
        hfac_tmp = get_hfac_bdry(mit_grid, bdry)
        if domain == 'ice':
            hfac_tmp = hfac_tmp[0,:]
        hfac_tmp = add_time_dim(hfac_tmp, months_per_year)
        hfac.append(hfac_tmp)
    loc0 = find_obcs_boundary(mit_grid, bdry)[0]
    lens_grid_file = find_cesm_file('LENS', var_names[0], domain, 'monthly', ens, year)[0]
    if domain == 'oce':
        lens_lon, lens_lat, lens_z, lens_nx, lens_ny, lens_nz = read_pop_grid(lens_grid_file, return_ugrid=True)[2:]
    elif domain == 'ice':
        lens_lon, lens_lat, lens_nx, lens_ny = read_cice_grid(lens_grid_file, return_ugrid=True)[2:]
        lens_z = None
        lens_nz = 1
    if bdry in ['N', 'S']:
        direction = 'lat'
        dimensions = 'x'
        lens_h_2d = lens_lon
        mit_h = mit_grid.lon_1d
    elif bdry in ['E', 'W']:
        direction = 'lon'
        dimensions = 'y'
        lens_h_2d = lens_lat
        mit_h = mit_grid.lat_1d
    if domain == 'oce':
        dimensions += 'z'
    dimensions += 't'
    i1, i2, c1, c2 = interp_slice_helper_nonreg(lens_lon, lens_lat, loc0, direction)
    lens_h_full = extract_slice_nonreg(lens_h_2d, direction, i1, i2, c1, c2)
    lens_h = trim_slice_to_grid(lens_h_full, lens_h_full, mit_grid, direction, warn=False)[0]
    lens_nh = lens_h.size

    # Read LENS u and v components and slice to boundary
    data_slice = []
    for v in range(num_cmp):
        file_path, t_start, t_end = find_cesm_file('LENS', var_names[v], domain, 'monthly', ens, year)
        data_full_cmp = read_netcdf(file_path, var_names[v], t_start=t_start, t_end=t_end)*1e-2
        data_slice_cmp = extract_slice_nonreg(data_full_cmp, direction, i1, i2, c1, c2)
        data_slice_cmp = trim_slice_to_grid(data_slice_cmp, lens_h_full, mit_grid, direction, warn=False)[0]
        data_slice.append(data_slice_cmp)
    try:
        lens_mask = np.invert(data_slice[0].mask[0,:])
    except(IndexError):
        lens_mask = np.ones(data_slice[0][0,:].shape)
    # Calculate magnitude and angle
    lens_magnitude = np.sqrt(data_slice[0]**2 + data_slice[1]**2)
    lens_angle = np.arctan2(data_slice[1], data_slice[0])

    # Read LENS climatology and calculate magnitude and angle
    data_clim = []
    for v in range(num_cmp):
        file_path = lens_file_head + var_names[v] + '_' + bdry + lens_file_tail
        data_clim.append(read_binary(file_path, [lens_nh, lens_nh, lens_nz], dimensions))
    lens_clim_magnitude = np.sqrt(data_clim[0]**2 + data_clim[1]**2)
    lens_clim_angle = np.arctan2(data_clim[1], data_clim[0])

    # Calculate scaling factor (subject to cap) and rotation angle (take mod 2pi when necessary)
    scale = np.minimum(lens_magnitude/lens_clim_magnitude, scale_cap)
    rotate = lens_angle - lens_clim_angle
    rotate[rotate < -np.pi] += 2*np.pi
    rotate[rotate > np.pi] -= 2*np.pi

    # Interpolate to MITgcm tracer grid
    shape = [months_per_year]
    if domain == 'oce':
        shape += [mit_grid.nz]
    shape += [mit_h.size]
    scale_interp = np.ma.zeros(shape)
    rotate_interp = np.ma.zeros(shape)
    for month in range(months_per_year):
        for in_data, out_data in zip([scale, rotate], [scale_interp, rotate_interp]):
            out_data[month,:] = interp_bdry(lens_h, lens_z, in_data[month,:], lens_mask, mit_h, mit_grid.z, hfac[0][0,:], lon=(direction=='lat'), depth_dependent=(domain=='oce'))

    # Read SOSE climatology and calculate magnitude and angle
    sose_clim = []
    for v in range(num_cmp):
        file_path = sose_file_head + var_sose[v]
        if (bdry=='N' and var_names[v]=='VVEL') or (bdry in ['E','W'] and var_names[v]=='UVEL'):
            file_path += sose_file_tail_alt
        else:
            file_path += sose_file_tail
        sose_clim_tmp = read_binary(file_path, [mit_grid.nx, mit_grid.ny, mit_grid.nz], dimensions)
        # Fill land mask on correct grid with zeros
        sose_clim_tmp[hfac[v+1]==0] = 0
        sose_clim.append(sose_clim_tmp)
    sose_magnitude = np.sqrt(sose_clim[0]**2 + sose_clim[1]**2)
    sose_angle = np.arctan2(sose_clim[1], sose_clim[0])

    # Now scale magnitude and rotate angle
    new_magnitude = sose_magnitude*scale_interp
    new_angle = sose_angle + rotate_interp
    # Convert back to u and v components
    new_u = new_magnitude*np.cos(new_angle)
    new_v = new_magnitude*np.sin(new_angle)

    return_vars = [new_u, new_v]
    if return_raw:
        return_vars += [data_slice[0], data_slice[1]]
    if return_clim:
        return_vars += [data_clim[0], data_clim[1]]
    if return_raw or return_clim:
        return_vars += [lens_h, lens_z]
    if return_sose_clim:
        return_vars += [sose_clim[0], sose_clim[1]]
    return return_vars


# For the given variable, boundary, ensemble member, year, and month: plot the uncorrected and corrected LENS fields as well as the SOSE climatology.
def plot_obcs_corrected_non_ts (var, bdry, ens, year, month, polar_coordinates=True, fig_name=None):

    if var == 'UVEL':
        var_title = 'Zonal velocity (m/s)'
    elif var == 'VVEL':
        var_title = 'Meridional velocity (m/s)'
    elif var == 'aice':
        var_title = 'Sea ice area (fraction)'
    elif var == 'hi':
        var_title = 'Sea ice thickness (m)'
    elif var == 'hs':
        var_title = 'Snow thickness (m)'
    elif var == 'uvel':
        var_title = 'Sea ice zonal velocity (m/s)'
    elif var == 'vvel':
        var_title = 'Sea ice meridional velocity (m/s)'
    if var in ['UVEL', 'VVEL']:
        domain = 'oce'
    else:
        domain = 'ice'
    if var in ['aice', 'hi', 'hs']:
        ctype = 'basic'
    else:
        ctype = 'plusminus'
    titles = ['SOSE climatology', 'LENS', 'LENS corrected']
    colours = ['blue', 'green', 'red']
    num_sources = len(titles)
    main_title = var_title+' at '+bdry+' boundary, '+str(year)+'/'+str(month)
    
    mit_grid_dir = '/data/oceans_output/shelf/kaight/archer2_mitgcm/AMUND_grid/'
    grid = Grid(mit_grid_dir)

    if polar_coordinates and var in ['UVEL', 'VVEL', 'uvel', 'vvel']:
        lens_corr_u, lens_corr_v, lens_uncorr_u, lens_uncorr_v, lens_h, lens_z, sose_clim_u, sose_clim_v = read_correct_lens_vel_polar_coordinates(domain, bdry, ens, year, return_raw=True, return_sose_clim=True)
        if var in ['UVEL', 'uvel']:
            lens_corr = lens_corr_u
            lens_uncorr = lens_uncorr_u
            sose_clim = sose_clim_u
        elif var in ['VVEL', 'vvel']:
            lens_corr = lens_corr_v
            lens_uncorr = lens_uncorr_v
            sose_clim = sose_clim_v
        if bdry in ['N', 'S']:
            mit_h = grid.lon_1d
        else:
            mit_h = grid.lat_1d
        mit_z = grid.z
    else:
        lens_corr, lens_uncorr, lens_h, lens_z, sose_clim, mit_h, mit_z = read_correct_cesm_non_ts('LENS', var, bdry, ens, year, return_raw=True, return_sose_clim=True)
    data = [sose_clim, lens_uncorr, lens_corr]
    h = [mit_h, lens_h, mit_h]
    if domain == 'oce':               
        z = [mit_z, lens_z, mit_z]
        vmin = min(min(np.amin(sose_clim), np.amin(lens_uncorr)), np.amin(lens_corr))
        vmax = max(max(np.amax(sose_clim), np.amax(lens_uncorr)), np.amax(lens_corr))
        cmap = set_colours(data[0], vmin=vmin, vmax=vmax, ctype=ctype)[0]
        fig, gs, cax = set_panels('1x3C1')
        for n in range(num_sources):
            ax = plt.subplot(gs[0,n])
            img = ax.pcolormesh(h[n], z[n], data[n][month-1,:], cmap=cmap, vmin=vmin, vmax=vmax)
            if n==2:
                plt.colorbar(img, cax=cax, orientation='horizontal')
            if n>0:
                ax.set_yticklabels([])
            ax.set_title(titles[n], fontsize=14)
        plt.suptitle(main_title, fontsize=18)
    elif domain == 'ice':
        fig, ax = plt.subplots()
        for n in range(num_sources):
            ax.plot(h[n], data[n][month-1,:], color=colours[n], label=titles[n])
        ax.set_title(main_title, fontsize=16)
        ax.grid(linestyle='dotted')
        ax.legend()        
    finished_plot(fig, fig_name=fig_name)


def compare_bedmachine_mask (grid_dir, bedmachine_file='/data/oceans_input/raw_input_data/BedMachine/v2.0/BedMachineAntarctica_2020-07-15_v02.nc', fig_name=None):

    grid = Grid(grid_dir)
    x_mit, y_mit = polar_stereo(grid.lon_2d, grid.lat_2d, lat_c=-70)
    x_bm = read_netcdf(bedmachine_file, 'x')
    y_bm = read_netcdf(bedmachine_file, 'y')
    mask_bm = read_netcdf(bedmachine_file, 'mask')

    fig, ax = plt.subplots()
    ax.contourf(x_bm, y_bm, mask_bm)
    ax.set_xlim([-2e6, -1.25e6])
    ax.set_ylim([-7e5, -8e4])
    ax.scatter(x_mit, y_mit, grid.ice_mask, color='red')
    finished_plot(fig, fig_name=fig_name)


# Generate new XC, YC, XG, YG points for use in Ua base mesh generation.
def new_grid_points (nc_file, delY_file):

    import netCDF4 as nc

    lon_g, lat_g = latlon_points(-140, -80, -76, -62.3, 0.1, delY_file)
    lon_c = 0.5*(lon_g[:-1] + lon_g[1:])
    lat_c = 0.5*(lat_g[:-1] + lat_g[1:])
    lon_g = lon_g[:-1]
    lat_g = lat_g[:-1]
    nx = lon_c.size
    ny = lat_c.size

    id = nc.Dataset(nc_file, 'w')
    id.createDimension('YC', ny)
    id.createVariable('YC', 'f8', ('YC'))
    id.variables['YC'].long_name = 'latitude at cell center'
    id.variables['YC'].units = 'degrees_north'
    id.variables['YC'][:] = lat_c
    id.createDimension('YG', ny)
    id.createVariable('YG', 'f8', ('YG'))
    id.variables['YG'].long_name = 'latitude at SW corner'
    id.variables['YG'].units = 'degrees_north'
    id.variables['YG'][:] = lat_g
    id.createDimension('XC', nx)
    id.createVariable('XC', 'f8', ('XC'))
    id.variables['XC'].long_name = 'longitude at cell center'
    id.variables['XC'].units = 'degrees_east'
    id.variables['XC'][:] = lon_c
    id.createDimension('XG', nx)
    id.createVariable('XG', 'f8', ('XG'))
    id.variables['XG'].long_name = 'longitude at SW corner'
    id.variables['XG'].units = 'degrees_east'
    id.variables['XG'][:] = lon_g
    id.close()


# Plot slices of ensemble mean potential density across the shelf break at 120W, averaged over three periods of LENS (1920s, 2000s, 2090s).
def plot_isopycnal_slices (lon0=-120, base_dir='./', fig_name=None):

    base_dir = real_dir(base_dir)
    num_ens = 5
    sim_dir = [base_dir + 'PAS_LENS' + str(n+1).zfill(3) + '_noOBC/output/' for n in range(num_ens)]
    grid_dir = base_dir + 'PAS_grid/'
    start_years = [1920, 2000, 2090]
    num_years = 10
    num_periods = len(start_years)
    vmin = 27
    vmax = 27.85
    zmin = -1200
    hmax = -71
    contours = np.arange(vmin, vmax, 0.1)
    
    grid = Grid(grid_dir)

    # Calculate ensemble mean potential density for each period
    patches = None    
    for n in range(num_ens):
        print('Processing ensemble member '+str(n+1))
        for p in range(num_periods):
            for t in range(num_years):
                year = start_years[p]+t
                print('...'+str(year))
                file_path = sim_dir[n] + str(year)+'01/MITgcm/output.nc'
                temp_full = read_netcdf(file_path, 'THETA')
                salt_full = read_netcdf(file_path, 'SALT')
                for t in range(months_per_year):
                    if patches is None:
                        # This is the first calculation - need all the slice variables
                        patches, temp_values, lon0, hmin, hmax, zmin, zmax, vmin_tmp, vmax_tmp, left, right, below, above, temp_slice, haxis, zaxis = slice_patches(mask_3d(temp_full[t,:],grid), grid, lon0=lon0, hmax=hmax, zmin=zmin, return_bdry=True, return_gridded=True)
                        # Also set up the master array
                        rho_mean = np.ma.zeros([num_periods, temp_slice.shape[0], temp_slice.shape[1]])
                        ndays_total = np.zeros(num_periods)
                    else:
                        temp_slice = slice_values(mask_3d(temp_full[t,:], grid), grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, return_gridded=True)[-1]
                    salt_slice = slice_values(mask_3d(salt_full[t,:], grid), grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, return_gridded=True)[-1]
                    # Calculate potential density
                    rho_slice = potential_density('MDJWF', salt_slice, temp_slice)
                    rho_slice = np.ma.masked_where(temp_slice.mask, rho_slice)
                    ndays = days_per_month(t+1, year, allow_leap=False)
                    # Integrate to master array
                    rho_mean[p,:] += rho_slice*ndays
                    ndays_total[p] += ndays
    # Convert from integral to average
    rho_mean /= ndays_total[:,None,None]
    # Subtract 1000 for readability
    rho_mean -= 1e3

    # Plot
    fig, gs, cax = set_panels('1x3C1')
    for p in range(num_periods):
        ax = plt.subplot(gs[0,p])
        img = make_slice_plot(patches, rho_mean[p,:].ravel(), lon0, hmin, hmax, zmin, zmax, vmin, vmax, lon0=lon0, ax=ax, make_cbar=False, contours=contours, data_grid=rho_mean[p,:], haxis=haxis, zaxis=zaxis)
        ax.set_title(str(start_years[p])+'s', fontsize=16)
        if p > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    plt.suptitle('Potential density at '+lon_label(lon0), fontsize=20)
    finished_plot(fig, fig_name=fig_name)

    
# Precompute the mean and standard deviation of monthly climatology sea level pressure at 40S and 65S in the LENS ensemble. This will be used to calculate the SAM index later.
def precompute_sam_climatologies (out_file):

    expt = 'LENS'
    num_ens = 40
    start_year = 1920
    end_year = 2005
    num_years = end_year-start_year+1
    var = 'PSL'
    lat0 = [-40, -65]
    num_lat = len(lat0)
    cesm_grid = CAMGrid()
    cesm_lat = cesm_grid.get_lon_lat(dim=1)[1]

    # First get zonal mean interpolated to each latitude, for each time index and ensemble member
    psl_points = np.empty([num_lat, months_per_year, num_years, num_ens])
    for e in range(num_ens):
        print('Processing ensemble member '+str(e+1))
        for y in range(num_years):
            print('...'+str(start_year+y))
            file_path, t_start, t_end = find_cesm_file(expt, var, 'atm', 'monthly', e+1, start_year+y)
            psl = read_netcdf(file_path, var, t_start=t_start, t_end=t_end)
            # Take zonal mean - regular grid so can just do a simple mean
            psl_zonal_mean = np.mean(psl, axis=-1)
            # Interpolate to each latitude
            for n in range(num_lat):
                i1, i2, c1, c2 = interp_slice_helper(cesm_lat, lat0[n])
                psl_points[n,:,y,e] = c1*psl_zonal_mean[:,i1] + c2*psl_zonal_mean[:,i2]

    # Now get monthly climatology mean and standard deviation across all years and ensemble members, for each latitude.
    psl_mean = np.mean(psl_points, axis=(2,3))
    psl_std = np.std(psl_points, axis=(2,3))

    # Save to file
    ncfile = NCfile(out_file, None, 't')
    ncfile.add_time(np.arange(months_per_year)+1, units='months')
    for n in range(num_lat):
        ncfile.add_variable('psl_mean_'+str(abs(lat0[n]))+'S', psl_mean[n,:], 't')
        ncfile.add_variable('psl_std_'+str(abs(lat0[n]))+'S', psl_std[n,:], 't')
    ncfile.close()    


# Calculate monthly timeseries of the given variable from the given CESM scenario and ensemble member.
def cesm_timeseries (var, expt, ens, out_file, sam_clim_file='LENS_SAM_climatology.nc'):

    if expt == 'LENS':
        start_year = 1920
    else:
        start_year = 2006
    if expt == 'MENS':
        end_year = 2080
    else:
        end_year = 2100
    num_years = end_year - start_year + 1
    if var == 'TS_global_mean':
        var_in = 'TS'
        long_name = 'global mean surface temperature'
        units = 'K'
        domain = 'atm'
    elif var == 'TS_SH_mean':
        var_in = 'TS'
        long_name = 'Southern Hemisphere mean surface temperature'
        units = 'K'
        domain = 'atm'
    elif var == 'seaice_extent_SH':
        var_in = 'aice'
        long_name = 'Southern Hemisphere sea ice extent'
        units = 'million km^2'
        factor_in = 1e-2
        factor_out = 1e-12
        threshold = 0.15
        domain = 'ice'
    elif var == 'SAM':
        var_in = 'PSL'
        long_name = 'Southern Annular Mode index'
        units = '1'
        domain = 'atm'
        lat0 = [-40, -65]
        num_lat = len(lat0)
    else:
        print('Error (cesm_timeseries): undefined variable '+var)
        sys.exit()
    
    if domain == 'atm':
        cesm_grid = CAMGrid()
        lat = cesm_grid.lat
        dA = cesm_grid.dA
    elif domain == 'ice':
        file_path = find_cesm_file(expt, var_in, domain, 'monthly', ens, start_year)[0]
        lon, lat, nx, ny, dA = read_cice_grid(file_path, return_dA=True)        
    if var in ['TS_global_mean', 'TS_SH_mean', 'seaice_extent_SH']:
        dA = add_time_dim(dA, months_per_year)
    if var in ['TS_SH_mean', 'seaice_extent_SH']:
        SH_mask = add_time_dim((lat < 0).astype(float), months_per_year)
    if var == 'SAM':
        lat_1d = cesm_grid.get_lon_lat(dim=1)[1]        

    for year in range(start_year, end_year+1):
        print('Processing '+str(year))
        file_path, t_start, t_end = find_cesm_file(expt, var_in, domain, 'monthly', ens, year)
        data_full = read_netcdf(file_path, var_in, t_start=t_start, t_end=t_end)
        time_tmp = netcdf_time(file_path, t_start=t_start, t_end=t_end)
        if var == 'TS_global_mean':
            data_tmp = np.sum(data_full*dA, axis=(1,2))/np.sum(dA, axis=(1,2))
        elif var == 'TS_SH_mean':
            data_tmp = np.sum(data_full*dA*SH_mask, axis=(1,2))/np.sum(dA*SH_mask, axis=(1,2))
        elif var == 'seaice_extent_SH':
            data_full *= factor_in
            data_full[data_full < threshold] = 0
            data_tmp = np.sum(data_full*dA*SH_mask, axis=(1,2))*factor_out
        elif var == 'SAM':
            # Take zonal mean
            psl_zonal_mean = np.mean(data_full, axis=-1)
            psl_points_norm = []
            for n in range(num_lat):
                # Interpolate to given latitude
                i1, i2, c1, c2 = interp_slice_helper(lat_1d, lat0[n])
                psl_point = c1*psl_zonal_mean[:,i1] + c2*psl_zonal_mean[:,i2]
                # Read monthly climatology mean and std for this latitude
                psl_mean = read_netcdf(sam_clim_file, 'psl_mean_'+str(abs(lat0[n]))+'S')
                psl_std = read_netcdf(sam_clim_file, 'psl_std_'+str(abs(lat0[n]))+'S')
                psl_points_norm.append((psl_point - psl_mean)/psl_std)
            # Take difference between latitudes to get SAM index
            data_tmp = psl_points_norm[0] - psl_points_norm[1]                
        if year == start_year:
            data = data_tmp
            time = time_tmp
        else:
            data = np.concatenate((data, data_tmp))
            time = np.concatenate((time, time_tmp))

    print('Writing to '+out_file)
    ncfile = NCfile(out_file, None, 't')
    ncfile.add_time(time)
    ncfile.add_variable(var, data, 't', long_name=long_name, units=units)
    ncfile.close()


# Call the above function for 5 members of each experiment
def all_cesm_timeseries (var, out_dir='./'):

    num_ens = 5
    out_dir = real_dir(out_dir)
    for expt in ['LENS', 'MENS', 'LW1.5', 'LW2.0']:
        for ens in range(1, num_ens+1):
            print('Processing '+expt+' '+str(ens).zfill(3))
            cesm_timeseries(var, expt, ens, out_dir+expt+'_'+str(ens).zfill(3)+'_'+var+'.nc')


# Plot timeseries of the given variable across all scenarios, showing the ensemble mean and range of each.            
def plot_scenario_timeseries (var_name, base_dir='./', timeseries_file='timeseries.nc', num_LENS=10, num_noOBCS=0, num_MENS=5, num_LW2=5, num_LW1=5, plot_pace=False, timeseries_file_pace='timeseries_final.nc', fig_name=None):

    if var_name in ['TS_global_mean', 'TS_SH_mean', 'SAM', 'seaice_extent_SH']:
        if num_noOBCS > 0:
            print('Warning: setting num_noOBCS back to 0')
            num_noOBCS = 0
        if plot_pace:
            print('Warning: setting plot_pace back to False')
            plot_pace = False

    base_dir = real_dir(base_dir)
    num_ens = [num_noOBCS, num_LENS, num_MENS, num_LW2, num_LW1]
    num_expt = len(num_ens)
    expt_names = ['LENS', 'LENS', 'MENS', 'LW2.0', 'LW1.5']
    expt_mid = ['', '', '_', '_', '_']
    expt_tails = ['_noOBC'] + ['_O' for n in range(num_expt-1)]
    expt_colours = ['BurlyWood', 'DarkGrey', 'IndianRed', 'MediumSeaGreen', 'DodgerBlue']
    smooth = 24
    start_year = [1920, 1920, 2006, 2006, 2006]
    if plot_pace:
        num_PACE = 20
        num_ens += [num_PACE]
        num_expt += 1
        expt_names += ['PACE']
        expt_colours += ['MediumOrchid']
        start_year += [1920]
        if timeseries_file_pace is None:
            timeseries_file_pace = timeseries_file
    fill_value = 9999
    
    data_mean = []
    data_min = []
    data_max = []
    time = []
    for n in range(num_expt):
        # Read all the data for this experiment
        for e in range(num_ens[n]):
            if var_name in ['TS_global_mean', 'TS_SH_mean', 'SAM', 'seaice_extent_SH']:
                file_path = base_dir + 'cesm_timeseries/' + expt_names[n] + '_' + str(e+1).zfill(3) + '_' + var_name + '.nc'
            else:            
                if expt_names[n] == 'PACE':
                    file_path = base_dir + '../mitgcm/PAS_PACE' + str(e+1).zfill(2) + '/output/' + timeseries_file_pace
                else:
                    file_path = base_dir + 'PAS_' + expt_names[n] + expt_mid[n] + str(e+1).zfill(3) + expt_tails[n] + '/output/' + timeseries_file
            time_tmp = netcdf_time(file_path, monthly=False)
            t0 = index_year_start(time_tmp, start_year[n])
            time_tmp = time_tmp[t0:]
            data_tmp, title, units = read_netcdf(file_path, var_name, return_info=True)
            data_tmp = data_tmp[t0:]
            data_smooth, time_smooth = moving_average(data_tmp, smooth, time=time_tmp)
            if e == 0:
                data_sim = np.empty([num_ens[n], time_smooth.size])
                time.append(time_smooth)
            if time_smooth.size < data_sim.shape[1]:
                num_time = time_smooth.size
                data_sim[e,:num_time] = data_smooth
                data_sim[e,num_time:] = fill_value
            else:
                data_sim[e,:] = data_smooth
        if num_ens[n] > 0:
            data_sim = np.ma.masked_where(data_sim==fill_value, data_sim)
            data_mean.append(np.mean(data_sim, axis=0))
            data_min.append(np.amin(data_sim, axis=0))
            data_max.append(np.amax(data_sim, axis=0))
        else:
            data_mean.append(None)
            data_min.append(None)
            data_max.append(None)
            time.append(None)

    fig, ax = plt.subplots()
    for n in range(num_expt):
        if num_ens[n] > 0:
            ax.fill_between(time[n], data_min[n], data_max[n], color=expt_colours[n], alpha=0.3)
            ax.plot(time[n], data_mean[n], color=expt_colours[n], label=expt_names[n], linewidth=1.5)
    plt.title(title, fontsize=14)
    plt.ylabel(units, fontsize=12)
    ax.grid(linestyle='dotted')
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.85, box.height])
    finished_plot(fig, fig_name=fig_name)


# Helper function to read and calculate trend per century.
def read_calc_trend (var, file_path, start_year=2006, end_year=2080, smooth=24, p0=0.05, percent=False, baseline=None, annual_avg=False):

    data = read_netcdf(file_path, var)
    if percent:
        data = (data-baseline)/baseline*100
    time = netcdf_time(file_path, monthly=False)
    t0, tf = index_period(time, start_year, end_year)
    time = time[t0:tf]
    data = data[t0:tf]
    time_sec = np.array([(t-time[0]).total_seconds() for t in time])
    time = time_sec/(365*sec_per_day*100)
    if annual_avg:
        time, data = calc_annual_averages(time, data)
    else:
        data, time = moving_average(data, smooth, time=time)
    slope, intercept, r_value, p_value, std_err = linregress(time, data)
    sig = p_value < p0
    return slope, sig


# For the given timeseries variable, create a bar graph showing when each combination of 2 scenarios is statistically distinct. Also print out the year at which they diverge for good, as well as the year at which the ensemble means last intersect.
def plot_scenario_divergence (var, num_LENS=10, num_MENS=10, num_LW2=10, num_LW1=5, window=11, timeseries_file='timeseries.nc', base_dir='./', fig_name=None):

    from scipy.stats import norm

    scenarios = ['LENS', 'MENS', 'LW2.0', 'LW1.5']
    num_ens = [num_LENS, num_MENS, num_LW2, num_LW1]
    num_scenarios = len(scenarios)
    start_year = 2006
    end_year_MENS = 2080
    end_year_other = 2100
    p0 = 0.05
    base_dir = real_dir(base_dir)
    if window % 2 != 1:
        print('Error (plot_scenario_divergence): window must be an odd number of years.')
        sys.exit()

    combo_names = []
    combo_distinct = []
    combo_time = []
    # Loop over every combination of scenarios
    for s1 in range(num_scenarios):
        for s2 in range(s1+1, num_scenarios):
            combo_names += [scenarios[s1]+' vs\n'+scenarios[s2]]
            print(combo_names[-1])
            # Read all the data, annually averaged
            if 'MENS' in [scenarios[s1], scenarios[s2]]:
                end_year = end_year_MENS
            else:
                end_year = end_year_other
            num_years = end_year - start_year + 1
            all_data1 = np.empty([num_ens[s1], num_years])
            all_data2 = np.empty([num_ens[s2], num_years])
            for s, all_data in zip([s1, s2], [all_data1, all_data2]):
                for n in range(num_ens[s]):
                    if var in ['TS_global_mean', 'TS_SH_mean', 'SAM', 'seaice_extent_SH']:
                        file_path = base_dir + 'cesm_timeseries/' + scenarios[s] + '_' + str(n+1).zfill(3) + '_' + var + '.nc'
                    else:
                        file_path = base_dir + 'PAS_' + scenarios[s]
                        if scenarios[s] != 'LENS':
                            file_path += '_'
                        file_path += str(n+1).zfill(3) + '_O/output/' + timeseries_file
                    time = netcdf_time(file_path, monthly=False)
                    data = read_netcdf(file_path, var)
                    t_start, t_end = index_period(time, start_year, end_year)
                    time = time[t_start:t_end]
                    data = data[t_start:t_end]
                    data, time = monthly_to_annual(data, time)
                    all_data[n,:] = data
            # Calculate running mean over window to find the last intersection of ensemble means
            data1_smooth, time_smooth = moving_average(np.mean(all_data1,axis=0), window, time=np.copy(time))
            data2_smooth = moving_average(np.mean(all_data2,axis=0), window)
            if (data1_smooth < data2_smooth).all() or (data1_smooth > data2_smooth).all():
                print('Ensemble means never cross')
            else:
                if data1_smooth[-1] < data2_smooth[-1]:
                    t0 = np.where(data1_smooth >= data2_smooth)[0][-1] + 1
                elif data1_smooth[-1] > data2_smooth[-1]:
                    t0 = np.where(data1_smooth <= data2_smooth)[0][-1] + 1
                print('Ensemble means stop crossing at '+str(time_smooth[t0]))
            # Now do a 2-sample t-test over each window
            radius = (window-1)//2
            time = time[radius:-radius]
            combo_time.append(np.array([t.year for t in time]))
            distinct = []
            for t in range(radius, num_years-radius):
                sample1 = all_data1[:,t-radius:t+radius+1].ravel()
                sample2 = all_data2[:,t-radius:t+radius+1].ravel()
                #t_val, p_val = ttest_ind(sample1, sample2, equal_var=False)
                #distinct.append(p_val < p0)
                min1, max1 = norm.interval(1-p0, loc=np.mean(sample1), scale=np.std(sample1))
                min2, max2 = norm.interval(1-p0, loc=np.mean(sample2), scale=np.std(sample2))
                mean1 = np.mean(sample1)
                mean2 = np.mean(sample2)
                distinct.append((mean2 > max1) or (mean2 < min1) or (mean1 > max2) or (mean1 < min2))
            combo_distinct.append(distinct)
            if not distinct[-1]:
                print('Ensembles never diverge for good')
            else:
                t0 = np.where(np.invert(distinct))[0][-1] + 1
                print('Ensembles diverge for good at '+str(time[t0]))
            
    num_combos = len(combo_names)

    # Plot
    fig, ax = plt.subplots()
    for n in range(num_combos):
        for t in range(len(combo_time[n])):
            if combo_distinct[n][t]:
                colour = 'IndianRed'
            else:
                colour = 'DodgerBlue'
            ax.barh(combo_names[n], 1, left=t, color=colour)
    ax.invert_yaxis()
    box = ax.get_position()
    ax.set_position([box.x0*1.25, box.y0, box.width, box.height])
    tick_years = np.arange(2020, 2100+20, 20)
    ax.set_xticks(tick_years-combo_time[0][0])
    ax.set_xticklabels([str(t) for t in tick_years])
    ax.set_title(var+', window='+str(window)+' years')
    finished_plot(fig, fig_name=fig_name)


# Directly interpolate all PAS files (ICs, OBCs, etc) to the AMUND domain for testing purposes.
def interp_PAS_files_to_AMUND ():

    grid_dir_old = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_grid/'
    grid_dir_new = '/data/oceans_output/shelf/kaight/archer2_mitgcm/AMUND_ini_grid_dig/'
    in_dir = '/data/oceans_output/shelf/kaight/ics_obcs/PAS/'
    fnames_3d = ['ICtheta_woa_clim.bin', 'ICsalt_woa_clim.bin', 'addmass_bedmach_merino.bin']
    fnames_2d = ['ICarea_sose.bin', 'ICheff_sose.bin', 'IChsnow_sose.bin', 'katabatic_rotate_PAS_90W', 'katabatic_scale_PAS_90W', 'aqh_offset_PAS', 'atemp_offset_PAS', 'lwdown_offset_PAS', 'precip_offset_PAS', 'swdown_offset_PAS', 'panom.bin']
    in_dir_obcs = '/data/oceans_input/processed_input_data/CESM/PAS_obcs/LENS_obcs/'
    obcs_head = 'LENS_ens001_'
    obcs_var = ['TEMP', 'SALT', 'UVEL', 'VVEL', 'aice', 'hi', 'hs', 'uvel', 'vvel']
    obcs_dim = [3, 3, 3, 3, 2, 2, 2, 2, 2]
    obcs_loc = ['N', 'W', 'E']
    obcs_start_year = 1920
    obcs_end_year = 2005
    out_dir = '/data/oceans_output/shelf/kaight/ics_obcs/AMUND/'
    out_dir_obcs = '/data/oceans_input/processed_input_data/CESM/AMUND_obcs/LENS_obcs/PAS_interp/'

    grid_old = Grid(grid_dir_old)
    grid_new = Grid(grid_dir_new)
    fill = get_fill_mask(grid_old, grid_new, missing_cavities=False)
    mask_PAS = grid_old.hfac==0
    mask_AMUND = grid_new.hfac==0

    # Extend to the north
    lat_old = np.concatenate((grid_old.lat_1d, [grid_new.lat_1d[-1]]))
    fill = np.concatenate((fill, np.expand_dims(fill[...,-1,:],-2)), axis=-2)
    mask_PAS = np.concatenate((mask_PAS, np.expand_dims(mask_PAS[...,-1,:],-2)), axis=-2)

    # 3D files
    for fname in fnames_3d:
        data_PAS = read_binary(in_dir + fname, [grid_old.nx, grid_old.ny, grid_old.nz], 'xyz', prec=64)
        data_PAS = np.concatenate((data_PAS, np.expand_dims(data_PAS[...,-1,:],-2)), axis=-2)
        data_PAS = discard_and_fill(data_PAS, mask_PAS, fill)
        data_AMUND = interp_reg_xyz(grid_old.lon_1d, lat_old, grid_old.z, data_PAS, grid_new.lon_1d, grid_new.lat_1d, grid_new.z)
        data_AMUND[mask_AMUND] = 0
        write_binary(data_AMUND, out_dir+fname+'_PAS_interp', prec=64)

    # 2D files
    for fname in fnames_2d:
        data_PAS = read_binary(in_dir + fname, [grid_old.nx, grid_old.ny, grid_old.nz], 'xy', prec=64)
        data_PAS = np.concatenate((data_PAS, np.expand_dims(data_PAS[...,-1,:],-2)), axis=-2)
        data_PAS = discard_and_fill(data_PAS, mask_PAS[0,:], fill[0,:], use_3d=False)
        data_AMUND = interp_reg_xy(grid_old.lon_1d, lat_old, data_PAS, grid_new.lon_1d, grid_new.lat_1d)
        data_AMUND[mask_AMUND[0,:]] = 0
        write_binary(data_AMUND, out_dir+fname+'_PAS_interp', prec=64)

    # OBCS files
    for bdry in obcs_loc:
        for var, dim in zip(obcs_var, obcs_dim):
            if var in ['UVEL', 'uvel']:
                gtype = 'u'
            elif var in ['VVEL', 'vvel']:
                gtype = 'v'
            else:
                gtype = 't'
            lon_PAS, lat_PAS = grid_old.get_lon_lat(dim=1, gtype=gtype)
            lon_AMUND, lat_AMUND = grid_new.get_lon_lat(dim=1, gtype=gtype)
            hfac_PAS = get_hfac_bdry(grid_old, bdry, gtype=gtype)
            hfac_AMUND = get_hfac_bdry(grid_new, bdry, gtype=gtype)
            if bdry in ['N', 'S']:
                h_PAS = lon_PAS
                h_AMUND = lon_AMUND
                dimensions = 'x'
                shape = [grid_new.nx]                    
            elif bdry in ['E', 'W']:
                h_PAS = lat_PAS
                h_AMUND = lat_AMUND
                dimensions = 'y'
                shape = [grid_new.ny]
            if dim == 3:
                dimensions += 'z'
                shape = [grid_new.nz] + shape
            else:
                hfac_PAS = hfac_PAS[0,:]
                hfac_AMUND = hfac_AMUND[0,:]
            dimensions += 't'
            shape = [months_per_year] + shape
            extend_left = h_AMUND[0] < h_PAS[0]
            extend_right = h_AMUND[-1] > h_PAS[-1]
            if extend_left:
                h_PAS = np.concatenate(([h_AMUND[0]], h_PAS))
                hfac_PAS = np.concatenate((np.expand_dims(hfac_PAS[...,0],-1), hfac_PAS), axis=-1)
            if extend_right:
                h_PAS = np.concatenate((h_PAS, [h_AMUND[-1]]))
                hfac_PAS = np.concatenate((hfac_PAS, np.expand_dims(hfac_PAS[...,-1],-1)), axis=-1)
            for year in range(obcs_start_year, obcs_end_year+1):
                fname = obcs_head + var + '_' + bdry + '_' + str(year)
                file_path = in_dir_obcs + fname
                data_PAS = read_binary(file_path, [grid_old.nx, grid_old.ny, grid_old.nz], dimensions, prec=32)
                if extend_left:                    
                    data_PAS = np.concatenate((np.expand_dims(data_PAS[...,0],-1), data_PAS), axis=-1)
                if extend_right:                    
                    data_PAS = np.concatenate((data_PAS, np.expand_dims(data_PAS[...,-1],-1)), axis=-1)
                data_AMUND = np.empty(shape)
                for t in range(months_per_year):
                    data_AMUND[t,:] = interp_bdry(h_PAS, grid_old.z, data_PAS[t,:], hfac_PAS, h_AMUND, grid_new.z, hfac_AMUND, depth_dependent=(dim==3))
                write_binary(data_AMUND, out_dir_obcs+fname, prec=32)


# Plot Hovmollers of temp or salt in Pine Island Bay for the historical LENS simulation and all four future scenarios. Express as anomalies from the 1920s mean.
def plot_hovmoller_scenarios (var, num_hist=10, num_LENS=5, num_MENS=5, num_LW2=5, num_LW1=5, base_dir='./', fig_name=None, option='mean'):

    base_dir = real_dir(base_dir)
    grid_dir = base_dir + 'PAS_grid/'
    region = 'pine_island_bay'
    hovmoller_file = 'hovmoller.nc'
    if var == 'temp':
        var_title = 'Temperature ('+deg_string+'C)'
        if option == 'anomaly':
            vmin = -1.25
            vmax = 2
        elif option == 'mean':
            vmin = -1.6
            vmax = 1.4
        elif option == 'std':
            vmin = 0
            vmax = 0.75
    elif var == 'salt':
        var_title = 'Salinity (psu)'
        if option == 'anomaly':
            vmin = -0.5
            vmax = 0.25
        elif option == 'mean':
            vmin = 34
            vmax = 34.8
        elif option == 'std':
            vmin = 0
            vmax = 0.1
    if option == 'anomaly':
        var_title += ' anomalies\nfrom historical mean '
        ctype = 'plusminus'
    elif option == 'mean':
        var_title += '\n'
        ctype = 'basic'
    elif option == 'std':
        var_title += ' standard\ndeviation '
        ctype = 'basic'
    suptitle = var_title + 'in ' + region_names[region]
    smooth = 12
    scenarios = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    num_ens = [num_hist, num_LW1, num_LW2, num_MENS, num_LENS]
    start_year = [1920, 2006, 2006, 2006, 2006]
    end_year = [2005, 2100, 2100, 2080, 2100]
    num_scenarios = len(scenarios)

    grid = Grid(grid_dir)

    time_plot = []
    data_plot = []
    for n in range(num_scenarios):
        for e in range(num_ens[n]):
            file_path = base_dir + 'PAS_'
            if scenarios[n] in ['historical', 'LENS']:
                file_path += 'LENS'
            else:
                file_path += scenarios[n] + '_'
            file_path += str(e+1).zfill(3) + '_O/output/' + hovmoller_file
            time = netcdf_time(file_path, monthly=False)
            data = read_netcdf(file_path, region+'_'+var)
            t_start, t_end = index_period(time, start_year[n], end_year[n])
            data = data[t_start:t_end,:]
            time = time[t_start:t_end]
            if e==0:
                time_plot.append(time)
                data_ens = np.ma.empty([num_ens[n], data.shape[0], data.shape[1]])
            data_ens[e,:] = data
        if option in ['mean', 'anomaly']:
            data_save = np.ma.mean(data_ens, axis=0)
        elif option == 'std':
            data_save = np.ma.std(data_ens, axis=0)            
        if option == 'anomaly':
            if n==0:
                data_baseline = np.ma.mean(data_save, axis=0)
            data_save -= data_baseline[None,:]
        data_plot.append(data_save)

    fig = plt.figure(figsize=(6,8))
    gs = plt.GridSpec(num_scenarios-1, 1) #(num_scenarios,1)
    gs.update(left=0.07, right=0.85, bottom=0.04, top=0.9, hspace=0.08)
    cax = fig.add_axes([0.75, 0.94, 0.24, 0.012])
    for n in range(1, num_scenarios): #num_scenarios):
        ax = plt.subplot(gs[n-1,0]) #n,0])
        img = hovmoller_plot(data_plot[n], time_plot[n], grid, smooth=smooth, ax=ax, make_cbar=False, vmin=vmin, vmax=vmax, ctype=ctype)
        ax.set_xlim([datetime.date(start_year[1], 1, 1), datetime.date(end_year[-1], 12, 31)]) #[0], 1, 1), datetime.date(end_year[-1], 12, 31)])
        ax.set_xticks([datetime.date(year, 1, 1) for year in np.arange(2020, end_year[-1], 20)]) #start_year[0], end_year[-1], 20)])
        if n == 1: #0:
            ax.set_yticks([0, -500, -1000])
            ax.set_yticklabels(['0', '0.5', '1'])
            ax.set_ylabel('')
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
        if n == 2: #1:
            ax.set_ylabel('Depth (km)', fontsize=10)
        if n != num_scenarios-1:
            ax.set_xticklabels([])
        ax.set_xlabel('')
        plt.text(1.01, 0.5, scenarios[n], ha='left', va='center', transform=ax.transAxes, fontsize=11)
    plt.suptitle(suptitle, fontsize=16, x=0.05, ha='left')
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
    cax.xaxis.set_ticks_position('top')
    reduce_cbar_labels(cbar)
    finished_plot(fig, fig_name=fig_name)


# Compare the bathymetry, 
def compare_topo (var, grid_dir_old='PAS_grid/', grid_dir_new='AMUND_ini_grid_dig/', xmin=-115, xmax=-98, ymin=-75.5, ymax=-73.5, vmin=0, vmax=None, vmin_diff=None, vmax_diff=None, fig_name=None):

    grid_old = Grid(grid_dir_old)
    grid_new = Grid(grid_dir_new)

    def prep_data (grid):
        if var == 'bathy':
            return abs(mask_land(grid.bathy, grid))
        elif var == 'draft':
            return abs(mask_except_ice(grid.draft, grid))
        elif var == 'wct':
            return abs(mask_land(grid.draft-grid.bathy, grid))

    data_old = prep_data(grid_old)
    data_new = prep_data(grid_new)
    data_new_fill = fill_into_mask(np.ma.copy(data_new), use_3d=False, log=False)
    data_new_interp = interp_reg(grid_new, grid_old, data_new_fill, dim=2)
    data_diff = data_new_interp - data_old
    mask_new = data_new.mask.astype(float)
    mask_new_interp = np.ceil(interp_reg(grid_new, grid_old, mask_new, dim=2)).astype(bool)
    mask_both = (data_old.mask + mask_new_interp).astype(bool)
    data_diff = np.ma.masked_where(mask_both, data_diff)

    fig, gs, cax1, cax2 = set_panels('1x3C2')
    cax = [cax1, None, cax2]
    data = [data_old, data_new, data_diff]
    grid = [grid_old, grid_new, grid_old]
    titles = ['PAS', 'AMUND', 'Difference']
    vmin_old, vmax_old = var_min_max(data_old, grid_old, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    vmin_new, vmax_new = var_min_max(data_new, grid_new, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    vmin_diff_tmp, vmax_diff_tmp = var_min_max(data_diff, grid_old, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    if vmin is None:
        vmin_abs = min(vmin_old, vmin_new)
    else:
        vmin_abs = vmin
    if vmax is None:
        vmax_abs = max(vmax_old, vmax_new)
    else:
        vmax_abs = vmax
    vmin = [vmin_abs, vmin_abs, vmin_diff]
    vmax = [vmax_abs, vmax_abs, vmax_diff]
    ctype = ['basic', 'basic', 'plusminus']
    for n in range(len(data)):
        ax = plt.subplot(gs[0,n])
        img = latlon_plot(data[n], grid[n], ax=ax, vmin=vmin[n], vmax=vmax[n], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ctype=ctype[n], make_cbar=False)
        ax.set_title(titles[n], fontsize=14)
        if n != 0:
            ax.set_xticks([])
            ax.set_yticks([])
        if cax[n] is not None:
            plt.colorbar(img, cax=cax[n])
    plt.suptitle(var, fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Calculate some extra timeseries of surface freshwater flux terms
def calc_sfc_fw_timeseries (base_dir='./', num_LENS=10, num_MENS=10, num_LW2=10, num_LW1=5, timeseries_file='timeseries_oceFWflx_shelf.nc'):

    var_names = ['amundsen_shelf_fwflx']
    base_dir = real_dir(base_dir)
    sim_dir = ['PAS_LENS'+str(n+1).zfill(3)+'_O/' for n in range(num_LENS)] + ['PAS_MENS_'+str(n+1).zfill(3)+'_O/' for n in range(num_MENS)] + ['PAS_LW2.0_'+str(n+1).zfill(3)+'_O/' for n in range(num_LW2)] + ['PAS_LW1.5_'+str(n+1).zfill(3)+'_O/' for n in range(num_LW1)]
    for sd in sim_dir:
        precompute_timeseries_coupled(output_dir=base_dir+sd+'output/', timeseries_types=var_names, hovmoller_loc=[], timeseries_file=timeseries_file)
    


def plot_fw_timeseries (base_dir='./', timeseries_file='timeseries.nc', num_historical=10, num_LENS=5, num_MENS=5, num_LW2=5, num_LW1=5, fig_name=None):

    var_names = ['all_massloss', 'PAS_shelf_seaice_freeze', 'PAS_shelf_seaice_melt', 'PAS_shelf_pmepr']
    var_titles = ['Ice shelf melting', 'Sea ice freezing', 'Sea ice melting', 'Precip minus evap']
    var_factor = [1e6/(rho_ice*sec_per_year), 1e-3, 1e-3, 1e-3]
    var_colours = ['MediumSeaGreen', 'DodgerBlue', 'IndianRed', 'DarkGrey']
    num_var = len(var_names)
    base_dir = real_dir(base_dir)
    num_ens = [num_historical, num_LW1, num_LW2, num_MENS, num_LENS]
    num_expt = len(num_ens)
    expt_names = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    start_year_historical = 1920
    start_year_future = 2006
    start_year_baseline = 1996
    end_year_future = 2100
    end_year_MENS = 2080
    smooth = 24
    units = 'Sv'
    vmin = -0.02
    vmax = 0.025

    all_data = []
    time = []
    for n in range(num_expt):
        if expt_names[n] == 'historical':
            start_year = start_year_historical
            end_year = start_year_future - 1
            baseline = np.zeros(num_var)
        else:
            start_year = start_year_future
            if expt_names[n] == 'MENS':
                end_year = end_year_MENS
            else:
                end_year = end_year_future
        for e in range(num_ens[n]):
            file_path = base_dir + 'PAS_'
            if expt_names[n] in ['historical', 'LENS']:
                file_path += 'LENS'
            else:
                file_path += expt_names[n] + '_'
            file_path += str(e+1).zfill(3) + '_O/output/' + timeseries_file
            time_tmp = netcdf_time(file_path, monthly=False)
            t0, tf = index_period(time_tmp, start_year, end_year)
            if expt_names[n] == 'historical':
                t0_baseline, tf_baseline = index_period(time_tmp, start_year_baseline, start_year_future-1)
            time_tmp = time_tmp[t0:tf]
            for v in range(num_var):
                data_tmp = read_netcdf(file_path, var_names[v])*var_factor[v]
                if expt_names[n] == 'historical':
                    baseline[v] += np.mean(data_tmp[t0_baseline:tf_baseline])/num_ens[n]
                data_tmp = data_tmp[t0:tf]
                data_smooth, time_smooth = moving_average(data_tmp, smooth, time=time_tmp)
                if e==0 and v==0:
                    data_sim = np.empty([num_var, num_ens[n], time_smooth.size])
                data_sim[v,e,:] = data_smooth
        all_data.append(data_sim)
        time.append(time_smooth)

    fig, gs = set_panels('3x2-1C0')
    for n in range(num_expt):
        ax = plt.subplot(gs[(n+1)//2, (n+1)%2])
        ax.axhline(color='black')
        for v in range(num_var):
            data = all_data[n][v,:]
            if expt_names[n] != 'historical':
                data -= baseline[v]
            ax.fill_between(time[n], np.amin(data, axis=0), np.amax(data, axis=0), color=var_colours[v], alpha=0.3)
            ax.plot(time[n], np.mean(data, axis=0), color=var_colours[v], label=var_titles[v], linewidth=1.5)
        title = expt_names[n]
        if expt_names[n] != 'historical':
            title += ' anomalies'
        plt.title(title, fontsize=14)
        if expt_names[n] == 'historical':
            start_year = start_year_historical
            end_year = start_year_future - 1
        else:
            start_year = start_year_future
            end_year = end_year_future
        ax.set_xlim([datetime.date(start_year, 1, 1), datetime.date(end_year, 12, 31)])
        if expt_names[n] != 'historical':
            ax.set_ylim([vmin, vmax])
            if (n+1)%2 == 1:
                ax.set_yticklabels([])
        ax.grid(linestyle='dotted')
        if n == 0:
            ax.legend(loc='center left', bbox_to_anchor=(-1,0.5), fontsize=12)
            plt.text(0.05, 0.95, 'Freshwater sources onto\ncontinental shelf (Sv)', fontsize=14, ha='left', va='top', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_name)


# Calculate and print the ensemble mean trends over the full length of each scenario.
def calc_mean_trends (var, base_dir='./', timeseries_file='timeseries.nc', timeseries_file_2=None, num_LENS=5, num_MENS=5, num_LW2=5, num_LW1=5, num_noOBC=0, massloss_percent=True):

    base_dir = real_dir(base_dir)
    num_ens = [num_LENS, num_MENS, num_LW2, num_LW1, num_noOBC]
    num_expt = len(num_ens)
    expt_names = ['LENS', 'MENS', 'LW2.0', 'LW1.5', 'LENS']
    expt_mid = ['', '_', '_', '_', '']
    expt_tail = ['_O', '_O', '_O', '_O', '_noOBC']
    smooth = 24
    p0 = 0.05
    start_year = 2006
    end_year = [2100, 2080, 2100, 2100, 2100]
    start_year_baseline = 2000
    end_year_baseline = 2010
    percent = 'massloss' in var and massloss_percent

    if percent:
        baseline_ens = np.empty(num_ens[0])
        for e in range(num_ens[0]):
            file_path = base_dir + 'PAS_' + expt_names[0] + expt_mid[0] + str(e+1).zfill(3) + expt_tail[n] + '/output/' + timeseries_file
            time = netcdf_time(file_path, monthly=False)
            data = read_netcdf(file_path, var)
            t_start, t_end = index_period(time, start_year_baseline, end_year_baseline)
            baseline_ens[e] = np.mean(data[t_start:t_end])
        baseline = np.mean(baseline_ens)
    else:
        baseline = None
        
    for n in range(num_expt):
        if num_ens[n] == 0:
            continue
        trends = np.empty(num_ens[n])
        for e in range(num_ens[n]):
            file_path = base_dir + 'PAS_' + expt_names[n] + expt_mid[n] + str(e+1).zfill(3) + expt_tail[n] + '/output/' + timeseries_file
            trends[e] = read_calc_trend(var, file_path, smooth=smooth, p0=p0, start_year=start_year, end_year=end_year[n], percent=percent, baseline=baseline)[0]
        p_val = ttest_1samp(trends, 0)[1]
        if p_val > p0:
            print(expt_names[n]+': no significant trend')
        else:
            print(expt_names[n]+' '+str(np.mean(trends))+' units/century')


def make_obcs_trend_file (var_name, bdry, expt_name, num_ens, obcs_dir, grid_dir, out_file, start_year, end_year):

    obcs_dir = real_dir(obcs_dir)
    grid_dir = real_dir(grid_dir)
    num_years = end_year - start_year + 1
    
    grid = Grid(grid_dir)
    hfac = get_hfac_bdry(grid, bdry)
    mask = hfac != 0
    num_pts = np.count_nonzero(mask)
    if bdry in ['N', 'S']:
        nh = grid.nx
        dimensions = 'xzt'
    elif bdry in ['E', 'W']:
        nh = grid.ny
        dimensions = 'yzt'
    trends = np.zeros([num_ens, grid.nz, nh])

    data_save = np.empty([num_ens, num_years, num_pts])
    for e in range(num_ens):
        print('Processing ensemble member '+str(e+1))
        for t in range(num_years):
            if var_name == 'speed':
                file_path = obcs_dir + expt_name + '_ens' + str(e+1).zfill(3) + '_UVEL_' + bdry + '_' + str(start_year+t)
                file_path_2 = obcs_dir + expt_name + '_ens' + str(e+1).zfill(3) + '_VVEL_' + bdry + '_' + str(start_year+t)
            else:
                file_path = obcs_dir + expt_name + '_ens' + str(e+1).zfill(3) + '_' + var_name + '_' + bdry + '_' + str(start_year+t)
            print('...Reading '+file_path)
            data = np.mean(read_binary(file_path, [grid.nx, grid.ny, grid.nz], dimensions), axis=0)
            if var_name == 'speed':
                print('...Reading '+file_path_2)
                data_2 = np.mean(read_binary(file_path_2, [grid.nx, grid.ny, grid.nz], dimensions), axis=0)
                data = np.sqrt(data**2 + data_2**2)
            data_save[e,t,:] = data[mask]
    print('Calculating trends')
    for e in range(num_ens):
        print('...member '+str(e+1))
        trends_tmp = np.empty(num_pts)
        for p in range(num_pts):
            slope, intercept, r_value, p_value, std_err = linregress(np.arange(num_years), data_save[e,:,p])
            trends_tmp[p] = slope
        trends[e,mask] = trends_tmp
    trends = np.ma.masked_where(trends==0, trends)

    ncfile = NCfile(out_file, grid, dimensions)
    ncfile.add_time(np.arange(num_ens)+1, units='ensemble member')
    ncfile.add_variable(var_name+'_'+bdry+'_trend', trends, dimensions, long_name='trend in '+var_name, units='units/y')
    ncfile.close()


def precompute_obcs_trends (num_hist=10, num_LENS=5, num_MENS=5, num_LW2=5, num_LW1=5, out_dir='precomputed_trends/obcs/', obcs_dir='/data/oceans_input/processed_input_data/CESM/PAS_obcs/', grid_dir='PAS_grid/'):

    var_names = ['TEMP', 'SALT', 'UVEL', 'VVEL', 'speed']
    periods = ['historical', 'LENS', 'MENS', 'LW2.0', 'LW1.5']
    start_years = [1920, 2006, 2006, 2006, 2006]
    end_years = [2005, 2100, 2080, 2100, 2100]
    num_ens = [num_hist, num_LENS, num_MENS, num_LW2, num_LW1]
    num_periods = len(periods)
    bdry = ['E', 'W', 'N']

    for var in var_names:
        for t in [0]: #range(num_periods):
            for loc in bdry:
                print('Calculating '+periods[t]+' trends in '+var+' at '+loc+' bdry')
                out_file = out_dir + var + '_' + loc + '_trend_' + periods[t] + '.nc'
                if periods[t] == 'historical':
                    expt_name = 'LENS'
                else:
                    expt_name = periods[t]
                make_obcs_trend_file (var, loc, expt_name, num_ens[t], obcs_dir+expt_name+'_obcs/', grid_dir, out_file, start_years[t], end_years[t])


def plot_obcs_trend_maps (var, bdry, trend_dir='precomputed_trends/obcs/', grid_dir='PAS_grid/', hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, fig_name=None):

    trend_dir = real_dir(trend_dir)
    grid_dir = real_dir(grid_dir)
    periods = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    num_periods = len(periods)
    p0 = 0.05

    grid = Grid(grid_dir)
    hfac = get_hfac_bdry(grid, bdry)
    if bdry in ['N', 'S']:
        nh = grid.nx
        h = grid.lon_1d
        h_axis = 'lon'
        dimensions = 'xzt'
    elif bdry in ['E', 'W']:
        nh = grid.ny
        h = grid.lat_1d
        h_axis = 'lat'
        dimensions = 'yzt'
    if var == 'TEMP':
        units = deg_string+'C/century'
    elif var == 'SALT':
        units = 'psu/century'
    elif var in ['UVEL', 'VVEL', 'speed']:
        units = 'm/s/century'

    data_plot = np.ma.empty([num_periods, grid.nz, nh])
    for t in range(num_periods):
        file_path = trend_dir + var + '_' + bdry + '_trend_' + periods[t] + '.nc'
        trends = read_netcdf(file_path, var+'_'+bdry+'_trend')
        mean_trend = np.mean(trends, axis=0)
        t_val, p_val = ttest_1samp(trends, 0, axis=0)
        mean_trend[p_val > p0] = 0
        data_plot[t,:] = np.ma.masked_where(hfac==0, mean_trend)*1e2

    if vmin is None:
        vmin = np.amin(data_plot)
    if vmax is None:
        vmax = np.amax(data_plot)
    cmap = set_colours(data_plot, ctype='plusminus', vmin=vmin, vmax=vmax)[0]
    fig, gs, cax = set_panels('3x2-1C1')
    for t in range(num_periods):
        ax = plt.subplot(gs[(t+1)//2, (t+1)%2])
        img = ax.pcolormesh(h, grid.z, data_plot[t,:], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(periods[t], fontsize=14)
        ax.set_xlim([hmin, hmax])
        ax.set_ylim([zmin, zmax])
        slice_axes(ax, h_axis=h_axis)
        if t != 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
    plt.colorbar(img, cax=cax, orientation='horizontal')
    plt.text(0.05, 0.95, var+' ('+units+')', fontsize=14,  ha='left', va='top', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_name)


def plot_warming_trend_profiles (region, fig_name=None):

    # TODO: if use this in publication, recalculate mean baseline temp using proper weighting of different lengths of months - can use average_12_months in utils.py
    baseline_file = 'temp_avg_LENS_last10y.nc'
    # TODO: if use this in publication, consider whether it's more correct to read Hovmollers and then recalculate trends of area-averaged quantities, rather than area-averaging the trends.
    trend_dir = 'precomputed_trends/'
    periods = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    baseline_colour = 'black'
    colours = ['BurlyWood', 'DodgerBlue', 'MediumSeaGreen', 'IndianRed', 'DarkGrey']
    num_periods = len(periods)
    p0 = 0.05
    grid_dir = 'PAS_grid/'

    grid = Grid(grid_dir)
    region_mask = grid.get_region_mask(region)
    # Read baseline temperature and horizontally average over region
    baseline_temp_3d = apply_mask(mask_3d(read_netcdf(baseline_file, 'THETA'), grid), np.invert(region_mask), depth_dependent=True)
    baseline_temp_profile = area_average(baseline_temp_3d, grid)
    # Now get the profiles of mean trends for each period
    temp_trend_profiles = np.ma.empty([num_periods, grid.nz])
    for n in range(num_periods):
        file_path = real_dir(trend_dir) + 'THETA_trend_' + periods[n] + '.nc'
        trends = apply_mask(mask_3d(read_netcdf(file_path, 'THETA_trend'), grid, time_dependent=True), np.invert(region_mask), depth_dependent=True, time_dependent=True)
        trend_profiles = area_average(trends, grid, time_dependent=True)
        mean_trend_profiles = np.mean(trend_profiles, axis=0)*1e2
        p_val = ttest_1samp(trend_profiles, 0, axis=0)[1]
        mean_trend_profiles[p_val > p0] = 0
        mean_trend_profiles = np.ma.masked_where(mean_trend_profiles==0, mean_trend_profiles)
        temp_trend_profiles[n,:] = mean_trend_profiles

    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(1,2)
    gs.update(left=0.1, right=0.97, bottom=0.17, top=0.88, wspace=0.05)
    # Plot baseline temperature on one side
    ax = plt.subplot(gs[0,0])
    ax.plot(baseline_temp_profile, -grid.z, color=baseline_colour)
    ax.invert_yaxis()
    ax.grid(linestyle='dotted')
    ax.set_title('Baseline temperature', fontsize=14)
    ax.set_xlabel(deg_string+'C', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    # Plot trends on other side
    ax = plt.subplot(gs[0,1])
    for n in range(num_periods):
        ax.plot(temp_trend_profiles[n,:], -grid.z, color=colours[n], label=periods[n])
    ax.invert_yaxis()
    ax.grid(linestyle='dotted')
    ax.set_title('Temperature trends', fontsize=14)
    ax.set_xlabel(deg_string+'C/century', fontsize=12)
    ax.set_yticklabels([])
    ax.legend(loc='center left', bbox_to_anchor=(-1,-0.18), fontsize=12, ncol=num_periods)
    plt.suptitle(region_names[region], fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Main text figure, plus print data for table
def trend_box_plot (fig_name=None):

    var_names = ['amundsen_shelf_temp_btw_200_700m', 'dotson_to_cosgrove_massloss']
    var_titles = ['Temperature trend over continental shelf,\n200-700m ', 'Ice shelf basal mass loss trend\nfrom Dotson to Cosgrove ']
    units = [deg_string+'C/century', '%/century']
    expt_names = ['Historical', 'Historical\nFixed BCs', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5', 'RCP 8.5\nFixed BCs']
    num_expt = len(expt_names)
    expt_dir_heads = ['PAS_']*num_expt
    expt_dir_mids = ['LENS', 'LENS', 'LW1.5_', 'LW2.0_', 'MENS_', 'LENS', 'LENS']
    expt_ens_prec = [3]*num_expt
    expt_dir_tails = ['_O', '_noOBC', '_O', '_O', '_O', '_O', '_noOBC']
    timeseries_file = ['/output/timeseries.nc']*num_expt
    num_ens = [10, 5, 5, 10, 10, 10, 5]
    start_years = [1920, 1920, 2006, 2006, 2006, 2006, 2006]
    end_years = [2005, 2005, 2100, 2100, 2080, 2100, 2100]
    baseline_period = 30
    smooth = 24
    p0 = 0.05

    all_trends = []
    for var in var_names:
        var_trends = []
        print('\nTrends in '+var+':')
        for n in range(1, num_expt):
            if var == 'dotson_to_cosgrove_massloss': # and expt_names[n] == 'Historical':
                # Calculate baseline for massloss: historical ensemble mean over 1920-1950
                massloss_baseline = 0
                for e in range(num_ens[n]):
                    file_path = expt_dir_heads[n] + expt_dir_mids[n] + str(e+1).zfill(expt_ens_prec[n]) + expt_dir_tails[n] + timeseries_file[n]
                    data = read_netcdf(file_path, var)
                    time = netcdf_time(file_path, monthly=False)
                    t0, tf = index_period(time, start_years[n], start_years[n]+baseline_period)
                    time = time[t0:tf]
                    data = data[t0:tf]
                    data_annual = monthly_to_annual(data, time)[0]
                    massloss_baseline += np.mean(data_annual)
                massloss_baseline /= num_ens[n]
                print('(using baseline of '+str(massloss_baseline)+' Gt/y) for '+expt_names[n])
            percent = (var == 'dotson_to_cosgrove_massloss')
            if percent:
                baseline = massloss_baseline
            else:
                baseline = None
            # Calculate trends in each ensemble member
            trends = np.zeros(num_ens[n])
            for e in range(num_ens[n]):
                file_path = expt_dir_heads[n] + expt_dir_mids[n] + str(e+1).zfill(expt_ens_prec[n]) + expt_dir_tails[n] + timeseries_file[n]
                if var_names == 'dotson_to_cosgrove_massloss':
                    percent = True
                slope, sig = read_calc_trend(var, file_path, start_year=start_years[n], end_year=end_years[n], smooth=smooth, p0=p0, percent=percent, baseline=baseline)
                if sig:
                    trends[e] = slope
            mean_trend = np.mean(trends)
            std_trend = np.std(trends)
            p_val = ttest_1samp(trends, 0)[1]
            if p_val < p0:
                print(expt_names[n]+': '+str(mean_trend)+' +/- '+str(std_trend))
            else:
                print(expt_names[n]+': no significant trend')
            var_trends.append(trends)
        all_trends.append(var_trends)

    # Make the box plot
    fig = plt.figure(figsize=(8,5))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.105, right=0.9, bottom=0.1, top=0.9)
    ax = plt.subplot(gs[0,0])        
    bplot = ax.boxplot(all_trends[0], positions=np.arange(num_expt)-0.17, widths=0.3, showmeans=True, whis='range', medianprops=dict(color='DarkGreen'), meanprops=dict(markeredgecolor='black', markerfacecolor='white', marker='*', markersize=8, zorder=300), patch_artist=True, zorder=0)
    for patch in bplot['boxes']:
        patch.set_facecolor('LightCoral')
    # Plot individual points
    for n in range(num_expt):
        ax.plot([n-0.17]*num_ens[n], all_trends[0][n], 'o', color='black', markersize=2)
    ax.grid(linestyle='dotted')
    ax.set_ylabel(var_titles[0]+'('+units[0]+')', fontsize=11, color='DarkRed')
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.tick_params(axis='y', colors='DarkRed')
    ax.axhline(color='black', linewidth=1, linestyle='dashed')
    ax.axvline(1.5, color='black', linewidth=1)
    plt.text(-0.4, 1.75, 'Historical\nscenarios', fontsize=14, ha='left', va='top')
    plt.text(1.6, 1.75, 'Future\nscenarios', fontsize=14, ha='left', va='top')
    # Second y-axis for mass loss trends
    ax2 = ax.twinx()
    bplot2 = ax2.boxplot(all_trends[1], positions=np.arange(num_expt)+0.17, widths=0.3, showmeans=True, whis='range', medianprops=dict(color='DarkGreen'), meanprops=dict(markeredgecolor='black', markerfacecolor='white', marker='*', markersize=8, zorder=300), patch_artist=True, zorder=0)
    for patch in bplot2['boxes']:
        patch.set_facecolor('LightBlue')
    for n in range(num_expt):
        ax2.plot([n+0.17]*num_ens[n], all_trends[1][n], 'o', color='black', markersize=2)
    ax2.set_xlim([-0.5, num_expt-0.5])
    ax2.set_xticks(np.arange(num_expt))
    ax2.set_xticklabels(expt_names, fontsize=11)
    # Align zeros at y-axis
    # Following https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
    y1 = ax.transData.transform((0, 0))[1]
    y2 = ax2.transData.transform((0, 0))[1]
    inv = ax2.transData.inverted()
    dy = (inv.transform((0, 0)) - inv.transform((0, y1-y2)))[1]
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim([miny+dy, maxy+dy])
    ax2.set_ylabel(var_titles[1]+'('+units[1]+')\n', fontsize=11, color='DarkBlue', rotation=-90)
    ax2.yaxis.set_label_coords(1.12, 0.5)
    ax2.tick_params(axis='y', colors='DarkBlue')
    plt.title('Warming and melting trends in each scenario', fontsize=16)
    finished_plot(fig, fig_name=fig_name, dpi=300)
    


# Main text figure
def warming_melting_trend_map (fig_name=None):

    import matplotlib.patheffects as pthe
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.colors as cl
    from mpl_toolkits.basemap import Basemap

    expt_name = 'LW2.0'
    expt_title = 'Paris 2'+deg_string+'C'
    trend_dir = 'precomputed_trends/'
    trend_var = ['temp_btw_200_700m', 'ismr']
    grid_dir = 'PAS_grid/'
    region_labels = ['G', 'D', 'Cr', 'T', 'P', 'Co', 'A']
    label_x = [-124, -112.3, -111.5, -106.5, -100.4, -100.5, -95]
    label_y = [-74.5, -74.375, -75, -75, -75.2, -73.6, -72.9]
    num_labels = len(region_labels)
    z_shelf = -1750
    p0 = 0.05
    lat_max = -68
    regions = ['amundsen_shelf'] #, 'pine_island_bay']
    region_colours = ['blue'] #, 'cyan']

    grid = Grid(grid_dir)

    def read_trend (var_name):
        file_path = trend_dir + var_name + '_trend_' + expt_name + '.nc'
        trends = read_netcdf(file_path, var_name + '_trend')
        mean_trend = np.mean(trends, axis=0)
        p_val = ttest_1samp(trends, 0, axis=0)[1]
        mean_trend[p_val > p0] = 0
        return mean_trend*1e2

    temp_trend = mask_land_ice(read_trend(trend_var[0]), grid)
    ismr_trend = mask_except_ice(read_trend(trend_var[1]), grid)

    fig = plt.figure(figsize=(9,5))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.05, right=0.97, bottom=0.05, top=0.9)
    ax = plt.subplot(gs[0,0])
    # Plot warming trends in red
    img1 = latlon_plot(temp_trend, grid, ax=ax, make_cbar=False, ctype='plusminus', ymax=lat_max)
    # Overlay ice shelf melting trends
    x, y, ismr_plot = cell_boundaries(ismr_trend, grid)
    cmap, vmin, vmax = set_colours(ismr_plot, ctype='ismr', change_points=[5,10,30])
    img2 = ax.pcolormesh(x, y, ismr_plot, cmap=cmap)
    # Add colorbars
    cax1 = inset_axes(ax, "22%", "4%", loc='upper right')
    cbar1 = plt.colorbar(img1, cax=cax1, orientation='horizontal', ticks=np.arange(0,2,0.5))
    plt.text(0.885, 0.84, 'Temperature trend\n(200-700m,'+deg_string+'C/century)', fontsize=12, transform=ax.transAxes, ha='center', va='center')
    cax2 = inset_axes(ax, "22%", "4%", loc='lower right')
    cbar2 = plt.colorbar(img2, cax=cax2, orientation='horizontal')
    cbar2.ax.xaxis.set_ticks_position('top')
    plt.text(0.885, 0.16, 'Basal melting trend\n(m/y/century)', fontsize=12, transform=ax.transAxes, ha='center', va='center')
    contour_shelf_break(grid, ax, colour='black', z_shelf=z_shelf)
    # Contour Amundsen Shelf region
    for n in range(len(regions)):
        mask = grid.get_region_mask(regions[n])
        if regions[n] == 'amundsen_shelf':
            all_bounds = [[-106, -104, -73.5, -72.5], [-104, -103, -74.5, -74], [-111, -110, -74.2, -73]]
            for bounds in all_bounds:
                index = (grid.lon_2d >= bounds[0])*(grid.lon_2d <= bounds[1])*(grid.lat_2d >= bounds[2])*(grid.lat_2d <= bounds[3])
                mask[index] = 1
        ax.contour(grid.lon_2d, grid.lat_2d, mask, levels=[0.5], colors=(region_colours[n]), linewidths=1, linestyles='dashed')
    # Add region labels
    for n in range(num_labels):
        txt = plt.text(label_x[n], label_y[n], region_labels[n], fontsize=11, ha='center', va='center', weight='bold', color='black', transform=ax.transData)
        txt.set_path_effects([pthe.withStroke(linewidth=2, foreground='w')])
    # Map inset
    ax3 = inset_axes(ax, "18%", "25%", loc='upper left')
    map = Basemap(projection='spstere', boundinglat=-64, lon_0=180)
    map.drawmapboundary(fill_color='white')
    map.fillcontinents(color=(0.6, 0.6, 0.6), lake_color=(0.6, 0.6, 0.6))
    lon_bdry = np.concatenate((np.ones(50)*grid.lon_1d[0], grid.lon_1d, np.ones(50)*grid.lon_1d[-1], np.flip(grid.lon_1d, axis=0)))
    lat_bdry = np.concatenate((np.linspace(grid.lat_1d[0], lat_max, num=50), np.ones(grid.nx)*lat_max, np.linspace(lat_max, grid.lat_1d[0], num=50), np.ones(grid.nx)*grid.lat_1d[0]))
    map.plot(lon_bdry, lat_bdry, color='red', latlon=True, linewidth=1)    
    ax.set_title('Ocean warming and ice shelf melting trends in '+expt_title+' scenario', fontsize=18)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Calculate year at which RCP 8.5 diverges from rest of scenarios, to print onto timeseries figure 
def calc_rcp85_divergence (window=11, return_year=False, use_ttest=True, test_paris_divergence=False, vs_fixed_BCs=False):

    var_name = 'amundsen_shelf_temp_btw_200_700m'
    if test_paris_divergence:
        file_head = ['PAS_LW1.5_', 'PAS_LW2.0_', 'PAS_MENS_']
        num_ens = [5, 10, 10]
        file_tail = ['_O/output/timeseries.nc']*len(file_head)
        start_year = 2006
        end_year = 2080
    elif vs_fixed_BCs:
        file_head = ['PAS_LENS', 'PAS_LENS']
        num_ens = [10, 5]
        file_tail = ['_O/output/timeseries.nc', '_noOBC/output/timeseries.nc']
        start_year = 1920
        end_year = 2100
    else:
        file_head = ['PAS_LENS', 'PAS_MENS_', 'PAS_LW2.0_', 'PAS_LW1.5_']
        num_ens = [10, 10, 10, 5]
        file_tail = ['_O/output/timeseries.nc']*len(file_head)
        start_year = 2006
        end_year = 2080
    num_expt = len(file_head)
    num_years = end_year - start_year + 1
    p0 = 0.05

    # Read annually averaged data from one scenario
    def read_process_expt (n):
        all_data = np.empty([num_ens[n], num_years])
        for e in range(num_ens[n]):
            file_path = file_head[n] + str(e+1).zfill(3) + file_tail[n]
            time = netcdf_time(file_path, monthly=False)
            data = read_netcdf(file_path, var_name)
            t_start, t_end = index_period(time, start_year, end_year)
            time = time[t_start:t_end]
            data = data[t_start:t_end]
            data, time = monthly_to_annual(data, time)
            all_data[e,:] = data
        return all_data, time

    data_scenario, time = read_process_expt(0)
    data_other = np.empty([np.sum(num_ens[1:]), num_years])
    e0 = 0
    for n in range(1, num_expt):
        data_other[e0:e0+num_ens[n],:] = read_process_expt(n)[0]
        e0 += num_ens[n]
    # Now compare the two samples over each window
    radius = (window-1)//2
    time = time[radius:-radius]
    distinct = []
    min1_vals = []
    max1_vals = []
    mean1_vals = []
    min2_vals = []
    max2_vals = []
    mean2_vals = []
    for t in range(radius, num_years-radius):
        sample1 = data_scenario[:,t-radius:t+radius+1].ravel()
        sample2 = data_other[:,t-radius:t+radius+1].ravel()
        if use_ttest:
            t_val, p_val = ttest_ind(sample1, sample2, equal_var=False)
            distinct.append(p_val < p0)
        else:
            # Compare 5-95% range
            min1, max1 = norm.interval(1-p0, loc=np.mean(sample1), scale=np.std(sample1))
            min2, max2 = norm.interval(1-p0, loc=np.mean(sample2), scale=np.std(sample2))
            mean1 = np.mean(sample1)
            mean2 = np.mean(sample2)
            distinct.append((mean1 > max2) or (mean1 < min2) or (mean2 > max1) or (mean2 < min1))
            min1_vals.append(min1)
            max1_vals.append(max1)
            mean1_vals.append(mean1)
            min2_vals.append(min2)
            max2_vals.append(max2)
            mean2_vals.append(mean2)
    if not distinct[-1]:
        print('Scenario never diverges for good')
    elif np.all(distinct):
        print('Scenarios are always distinct')
    else:
        t0 = np.where(np.invert(distinct))[0][-1] + 1
        print('Scenario diverges for good at '+str(time[t0]))
        if return_year:
            return time[t0].year
    if use_ttest:
        fig, ax = plt.subplots()
        for t in range(time.size):
            if distinct[t]:
                colour = 'IndianRed'
            else:
                colour = 'DodgerBlue'
            ax.barh('Scenario', 1, left=t, color=colour)
        plt.title('Distinct = red, not distinct = blue')
        fig.show()
    else:
        fig, ax = plt.subplots()
        ax.fill_between(time, min1_vals, max1_vals, color='red', alpha=0.3)
        ax.plot(time, mean1_vals, color='red', linewidth=1.5, label='Scenario')
        ax.fill_between(time, min2_vals, max2_vals, color='blue', alpha=0.3)
        ax.plot(time, mean2_vals, color='blue', linewidth=1.5, label='Others')
        ax.legend(loc='upper left')
        ax.grid(linestyle='dotted')
        fig.show()


# Main text figure with option for supplementary figure to show extra simulations
def timeseries_shelf_temp (fig_name=None, supp=False):

    if supp:
        expt_names = ['Transient BCs', 'Fixed BCs']
        num_expt = len(expt_names)
        num_ens = [10, 5]
        start_year = [1920, 1920]
        end_year = [2100, 2100]
        expt_file_head = ['PAS_']*num_expt
        expt_file_mid = ['LENS']*num_expt
        expt_ens_prec = [3]*num_expt
        expt_dir_tail = ['_O', '_noOBC']
        colours = [(0.8,0.47,0.65), 'Teal']
        expt_file_tail = ['/output/timeseries.nc']*num_expt
    else:
        expt_names = ['Historical', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5']
        num_expt = len(expt_names)
        num_ens = [10, 5, 10, 10, 10]
        start_year = [1920, 2006, 2006, 2006, 2006]
        end_year = [2005, 2100, 2100, 2080, 2100]
        expt_file_head = ['PAS_']*num_expt
        expt_file_mid = ['LENS', 'LW1.5_', 'LW2.0_', 'MENS_', 'LENS']
        expt_ens_prec = [3]*num_expt
        expt_dir_tail = ['_O']*num_expt
        colours = [(0.6,0.6,0.6), (0,0.45,0.7), (0,0.62,0.45), (0.9,0.62,0), (0.8,0.47,0.65)]
        expt_file_tail = ['/output/timeseries.nc']*num_expt
    var_name = 'amundsen_shelf_temp_btw_200_700m'
    smooth = 24
    rcp85_div_year = calc_rcp85_divergence(return_year=True, vs_fixed_BCs=supp)

    data_mean = []
    data_min = []
    data_max = []
    time = []
    # Loop over scenarios
    for n in range(num_expt):
        # Loop over ensemble members
        for e in range(num_ens[n]):
            # Read data
            file_path = expt_file_head[n] + expt_file_mid[n] + str(e+1).zfill(expt_ens_prec[n]) + expt_dir_tail[n] + expt_file_tail[n]
            time_raw = netcdf_time(file_path, monthly=False)
            data_raw = read_netcdf(file_path, var_name)
            if expt_names[n] == 'Historical':
                # Save historical data for concatenating to others later
                t_end = index_year_end(time_raw, end_year[n])
                time_hist = np.copy(time_raw[:t_end])
                data_hist = np.copy(data_raw[:t_end])
            elif expt_names[n] == 'Historical fixed BCs':
                t_end = index_year_end(time_raw, end_year[n])
                time_hist_noBC = np.copy(time_raw[:t_end])
                data_hist_noBC = np.copy(data_raw[:t_end])
            elif expt_names[n] != 'PACE' and not supp:
                # Concatenate with historical data for smoothing
                if expt_names[n] == 'RCP 8.5 fixed BCs':
                    time_raw = np.concatenate((time_hist_noBC, time_raw))
                    data_raw = np.concatenate((data_hist_noBC, data_raw))
                else:
                    time_raw = np.concatenate((time_hist, time_raw))
                    data_raw = np.concatenate((data_hist, data_raw))
            # Smooth
            data_smooth, time_smooth = moving_average(data_raw, smooth, time=time_raw)
            # Trim
            t_start = index_year_start(time_smooth, start_year[n])
            if expt_names[n] in ['Historical', 'Historical fixed BCs']:
                t_end = index_year_end(time_smooth, end_year[n])
            else:
                t_end = time_smooth.size
            time_smooth = time_smooth[t_start:t_end]
            data_smooth = data_smooth[t_start:t_end]
            if e == 0:
                data_sim = np.empty([num_ens[n], time_smooth.size])
                time.append(time_smooth)
            # Store the timeseries for this ensemble member
            data_sim[e,:] = data_smooth
        # Now calculate mean, min, max over the ensemble at each time index
        data_mean.append(np.mean(data_sim, axis=0))
        data_min.append(np.amin(data_sim, axis=0))
        data_max.append(np.amax(data_sim, axis=0))

    # Plot
    fig = plt.figure(figsize=(8,5))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.9)
    ax = plt.subplot(gs[0,0])
    for n in range(num_expt):
        # Shade ensemble range
        ax.fill_between(time[n], data_min[n], data_max[n], color=colours[n], alpha=0.3)
        # Plot ensemble mean as solid line
        ax.plot(time[n], data_mean[n], color=colours[n], label=expt_names[n], linewidth=1.5)
        if n == num_expt-1:
            if not supp:
                # Label beginning of future scenarios
                ax.axvline(time[n][0], color=colours[0], linestyle='dashed')
                plt.text(datetime.date(2008,1,1), -0.8, 'Future\nscenarios', fontsize=13, color=colours[0], ha='left', va='bottom', weight='bold')
            # Label year of RCP 8.5 divergence
            if supp:
                div_colour = 'black'
                plt.text(datetime.date(rcp85_div_year+2,1,1), 0.6, str(rcp85_div_year)+':\nSimulations\ndiverge', fontsize=13, color=div_colour, ha='left', va='bottom')
            else:
                div_colour = colours[n]
                plt.text(datetime.date(rcp85_div_year+2,1,1), -0.5, str(rcp85_div_year)+':\nRCP 8.5\ndiverges', fontsize=13, color=div_colour, ha='left', va='bottom', weight='bold')
            ax.axvline(datetime.date(rcp85_div_year,1,1), color=div_colour, linestyle='dashed')
    ax.grid(linestyle='dotted')
    ax.set_xlim([datetime.date(start_year[0],1,1), np.amax(time[-1])])
    ax.set_xticks([datetime.date(year,1,1) for year in np.arange(start_year[0], end_year[-1], 20)])
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature on continental shelf, 200-700m ('+deg_string+'C)', fontsize=12)
    if supp:
        ax.set_title('Ocean temperature (Historical and RCP 8.5)', fontsize=17)
        ax.legend(loc='upper center', fontsize=11)
    else:
        ax.set_title('Evolution of ocean temperature', fontsize=17)
        ax.legend(loc='upper left', fontsize=11)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Main text figure with option for supplementary figure to show extra simulations
def temp_profiles (fig_name=None, supp=False, region='amundsen_shelf', density_space=False, var='temp'):

    if region == 'amundsen_shelf':
        hovmoller_file_main = '/output/hovmoller_shelf.nc'
        hovmoller_file_pace = '/output/hovmoller3.nc'
        if density_space:
            hovmoller_file_main = '/output/hovmoller_density_space.nc'
    elif region == 'pine_island_bay':
        hovmoller_file_main = '/output/hovmoller.nc'
        hovmoller_file_pace = '/output/hovmoller1.nc'
    if supp:
        expt_names = ['Historical', 'Historical fixed BCs', 'PACE', 'RCP 8.5', 'RCP 8.5 fixed BCs']
        num_expt = len(expt_names)
        expt_dir_heads = ['PAS_']*num_expt
        expt_dir_heads[2] = '../mitgcm/PAS_'
        expt_dir_mids = ['LENS', 'LENS', 'PACE', 'LENS', 'LENS']
        expt_ens_prec = [3]*num_expt
        expt_ens_prec[2] = 2
        expt_dir_tails = ['_O', '_noOBC', '', '_O', '_noOBC']
        hovmoller_file = [hovmoller_file_main]*num_expt
        hovmoller_file[2] = hovmoller_file_pace
        num_ens = [10, 5, 20, 10, 5]
        start_years = [1920, 1920, 1920, 2006, 2006]
        end_years = [2005, 2005, 2013, 2100, 2100]
        colours = [(0.6,0.6,0.6), (0,0.45,0.7), (0,0.62,0.45), (0.8,0.47,0.65), (0.9,0.62,0)]
    else:
        expt_names = ['Historical', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5']
        num_expt = len(expt_names)
        expt_dir_heads = ['PAS_']*num_expt
        expt_dir_mids = ['LENS', 'LW1.5_', 'LW2.0_', 'MENS_', 'LENS']
        expt_ens_prec = [3]*num_expt
        expt_dir_tails = ['_O']*num_expt
        hovmoller_file = [hovmoller_file_main]*num_expt
        num_ens = [10, 5, 10, 10, 10]
        start_years = [1920, 2006, 2006, 2006, 2006]
        end_years = [2005, 2100, 2100, 2080, 2100]
        colours = [(0.6,0.6,0.6), (0,0.45,0.7), (0,0.62,0.45), (0.9,0.62,0), (0.8,0.47,0.65)]    
    final_period = 20
    grid_dir = 'PAS_grid/'
    grid = Grid(grid_dir)
    smooth = 24
    p0 = 0.05
    ndays = [days_per_month(t+1, 1979) for t in range(months_per_year)]
    if var == 'temp':
        var_title = 'Temperature'
        var_units = deg_string+'C'
    elif var == 'salt':
        var_title = 'Salinity'
        var_units = 'psu'
    elif var == 'volume':
        var_title = 'Volume'
        var_units = r'm$^3$'

    # Get mask of all ice front points
    #icefront_mask = grid.get_icefront_mask()
    # Mask out everywhere outside given region
    #[xmin, xmax, ymin, ymax] = region_bounds[region]
    #outside_region = np.invert((grid.lon_2d>=xmin)*(grid.lon_2d<=xmax)*(grid.lat_2d>=ymin)*(grid.lat_2d<=ymax))
    #icefront_mask[outside_region] = 0
    #icefront_mask = icefront_mask.astype(bool)
    #draft_icefront = np.ma.masked_where(np.invert(icefront_mask), grid.draft)
    #bathy_icefront = np.ma.masked_where(np.invert(icefront_mask), grid.bathy)
    #mean_draft = area_average(draft_icefront, grid)
    #mean_bathy = area_average(bathy_icefront, grid)

    if density_space:
        rho = read_netcdf(expt_dir_heads[0]+expt_dir_mids[0]+'001'+expt_dir_tails[0]+hovmoller_file[0], 'rho')
        nz = rho.size
    else:
        nz = grid.nz
    mean_profiles = np.ma.empty([num_expt, nz])
    std_profiles = np.ma.empty([num_expt, nz])
    trend_profiles = np.ma.empty([num_expt, nz])
    # Loop over experiments
    for n in range(num_expt):
        num_years = end_years[n] - start_years[n] + 1
        expt_annual_means = np.ma.empty([num_ens[n], num_years, nz])
        expt_trends = np.ma.empty([num_ens[n], nz])
        # Loop over ensemble members
        for e in range(num_ens[n]):
            # Read the data
            file_path = expt_dir_heads[n] + expt_dir_mids[n] + str(e+1).zfill(expt_ens_prec[n]) + expt_dir_tails[n] + hovmoller_file[n]
            time = netcdf_time(file_path, monthly=False)
            temp = read_netcdf(file_path, region+'_'+var)
            # Trim to correct years
            t0, tf = index_period(time, start_years[n], end_years[n])
            time = time[t0:tf]
            temp = temp[t0:tf,:]
            # Calculate annual means
            for t in range(num_years):
                expt_annual_means[e,t,:] = np.ma.average(temp[t*months_per_year:(t+1)*months_per_year,:], axis=0, weights=ndays)
            # Calculate trend per century of smoothed data at each depth
            time_cent = np.array([(t-time[0]).total_seconds() for t in time])/(365*sec_per_day*100)
            temp_smoothed, time_smoothed = moving_average(temp, smooth, time=time_cent)
            for k in range(nz):
                slope, intercept, r_value, p_value, std_err = linregress(time_smoothed, temp_smoothed[:,k])
                if p_value < p0:
                    expt_trends[e,k] = slope
                else:
                    expt_trends[e,k] = 0
        # Now get mean and std over last 20 years and all ensemble members
        mean_profiles[n,:] = np.mean(expt_annual_means[:,-final_period:,:], axis=(0,1))
        std_profiles[n,:] = np.std(expt_annual_means[:,-final_period:,:], axis=(0,1))
        if n==0:
            mean_profile_beg = np.mean(expt_annual_means[:,:final_period,:], axis=(0,1))
            std_profile_beg = np.std(expt_annual_means[:,:final_period,:], axis=(0,1))
        # Calculate ensemble mean trend and test significance
        mean_trend = np.mean(expt_trends, axis=0)
        p_val = ttest_1samp(expt_trends, 0, axis=0)[1]
        mean_trend[p_val > p0] = 0
        trend_profiles[n,:] = mean_trend
    # Mask out depths where trend is not significantly different from 0
    trend_profiles = np.ma.masked_where(trend_profiles==0, trend_profiles)

    # Plot
    fig = plt.figure(figsize=(8,5.5))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.1, right=0.95, bottom=0.18, top=0.82, wspace=0.05)
    data_plot = [mean_profiles, std_profiles, trend_profiles]
    data_plot_beg = [mean_profile_beg, std_profile_beg]
    titles = [r'$\bf{a}$. '+'Ensemble mean\n(last 20y)', r'$\bf{b}$. '+'Standard deviation\n(last 20y)', r'$\bf{c}$. '+'Trend']
    units = [var_units]*2 + [var_units+'/century']
    if density_space:
        depth = rho
        ylabel = r'potential density (kg/m$^3$)'
    else:
        depth = -grid.z
        ylabel = 'depth (m)'
    for v in range(3):
        ax = plt.subplot(gs[0,v])
        for n in range(num_expt): #[0, 2, 5, 3, 1, 4]:
            ax.plot(data_plot[v][n,:], depth, color=colours[n], linewidth=1.5, label=expt_names[n])
            #if n==0 and v < 2:
                #ax.plot(data_plot_beg[v], depth, color=colours[n], linewidth=1.5, linestyle='dotted')
        ax.tick_params(direction='in')
        ax.grid(linestyle='dotted')
        if density_space:
            ax.set_ylim([27, 28])
        else:
            ax.set_ylim([0, None])
        ax.invert_yaxis()
        ax.set_title(titles[v], fontsize=14)
        ax.set_xlabel(units[v], fontsize=12)
        if v==0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.axhline(200, color='black', linestyle='dashed', linewidth=1)
        ax.axhline(700, color='black', linestyle='dashed', linewidth=1)
    if supp:
        legend_fontsize = 10
    else:
        legend_fontsize = 12
    ax.legend(loc='lower center', bbox_to_anchor=(-0.65,-0.25), fontsize=legend_fontsize, ncol=num_expt)
    plt.suptitle(var_title+' profiles over '+region_names[region], fontsize=18)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Main text figure
def velocity_trends (fig_name=None):

    import matplotlib.patheffects as pthe

    expt_name = 'LW2.0'
    expt_title = 'Paris 2'+deg_string+'C'
    trend_dir = 'precomputed_trends/'
    trend_var = ['u_bottom100m', 'UVEL', 'VVEL']
    lon0 = [None, -118, None]
    lat0 = [None, None, -73]
    h_bounds = [None, [-72.3, -71.5], [-107, -105]]
    var_titles = [r'$\bf{a}$. '+'Bottom 100m', r'$\bf{b}$. '+'Undercurrent ('+lon_label(lon0[1])+'), zonal', r'$\bf{c}$. '+'PITE Trough ('+lat_label(lat0[2])+'), meridional']
    direction = ['magnitude', 'east', 'north']
    gtypes = [None, 'u', 'v']
    num_var = len(trend_var)
    title = 'Velocity trends in Paris 2'+deg_string+'C scenario'
    grid_dir = 'PAS_grid/'
    xmin = -122
    xmax = -98
    ymax = -70
    p0 = 0.05
    threshold = 0.02
    z_shelf = -1000
    region_labels = ['DG', 'PITE']
    label_x = [-116, -107]
    label_y = [-71.8, -71.34]

    grid = Grid(grid_dir)

    def read_single_trend (var_name, dim=2, gtype='t'):
        file_path = trend_dir + var_name + '_trend_' + expt_name + '.nc'
        trends = read_netcdf(file_path, var_name + '_trend')
        mean_trend = np.mean(trends, axis=0)
        p_val = ttest_1samp(trends, 0, axis=0)[1]
        mean_trend[p_val > p0] = 0
        if dim == 2:
            mean_trend = mask_land_ice(mean_trend, grid, gtype=gtype)
        elif dim == 3:
            mean_trend = mask_3d(mean_trend, grid, gtype=gtype)
        return mean_trend*1e2
    
    def read_trend_vector(u_var, dim=2):
        u_trend = read_single_trend(u_var, dim=dim, gtype='u')
        v_trend = read_single_trend(u_var.replace('u', 'v'), dim=dim, gtype='v')
        vel_trend = read_single_trend(u_var.replace('u', 'vel')+'_speed', dim=dim)
        return u_trend, v_trend, vel_trend

    fig = plt.figure(figsize=(6,12))
    gs = plt.GridSpec(3,1)
    gs.update(left=0.12, right=0.8, bottom=0.05, top=0.92, hspace=0.25)
    for n in range(num_var):
        ax = plt.subplot(gs[n,0])
        if n == 0:
            # First panel: bottom 100m velocity map
            u_trend, v_trend, vel_trend = read_trend_vector(trend_var[n], dim=2)
            u_trend = interp_grid(u_trend, grid, 'u', 't')
            v_trend = interp_grid(v_trend, grid, 'v', 't')
            magnitude_trend = np.sqrt(u_trend**2 + v_trend**2)
            index = magnitude_trend < threshold
            u_trend = np.ma.masked_where(index, u_trend)
            v_trend = np.ma.masked_where(index, v_trend)
            img = latlon_plot(magnitude_trend, grid, ax=ax, make_cbar=False, ctype='plusminus', xmin=xmin, xmax=xmax, ymax=ymax)
            contour_shelf_break(grid, ax, colour='magenta', z_shelf=z_shelf)
            overlay_vectors(ax, u_trend, v_trend, grid, chunk_x=5, chunk_y=8, scale=0.6, headwidth=6, headlength=7)
            ax.plot([lon0[1], lon0[1]], h_bounds[1], color='blue', linewidth=1.5)
            plt.text(lon0[1], h_bounds[1][1]+0.1, 'b', color='blue', weight='bold', ha='center', va='bottom')
            ax.plot(h_bounds[2], [lat0[2], lat0[2]], color='blue', linewidth=1.5)
            plt.text(h_bounds[2][0]-0.2, lat0[2], 'c', color='blue', weight='bold', ha='right', va='center')
            for m in range(len(region_labels)):
                txt = plt.text(label_x[m], label_y[m], region_labels[m], fontsize=10, ha='center', va='center', weight='bold', color='blue')
                txt.set_path_effects([pthe.withStroke(linewidth=2, foreground='w')])
        else:
            # Second and third panels: velocity slices
            trend = read_single_trend(trend_var[n], dim=3, gtype=gtypes[n])
            img = slice_plot(trend, grid, gtype=gtypes[n], lon0=lon0[n], lat0=lat0[n], hmin=h_bounds[n][0], hmax=h_bounds[n][1], ctype='plusminus', ax=ax, make_cbar=False)
        ax.set_title(var_titles[n], fontsize=14)
        if n > 0:
            ax.set_ylabel('Depth (m)', fontsize=12)
            labels = ax.xaxis.get_ticklabels()
            for label in labels[1::2]:
                label.set_visible(False)
        cax = fig.add_axes([0.83, 0.7-0.315*n, 0.03, 0.2])
        cbar = plt.colorbar(img, cax=cax)
        plt.text(0.99, 0.8-0.315*n, 'm/s/century ('+direction[n]+')', rotation=-90, ha='right', va='center', transform=fig.transFigure, fontsize=12)
        labels = ax.xaxis.get_ticklabels()
    plt.suptitle(title, fontsize=18)
    finished_plot(fig, fig_name=fig_name, dpi=300)    


# Main text figure
def melt_trend_buttressing (fig_name=None, shelf='all', supp=False, depth_classes=False, num_bins=40, group1=None, group2=None, plot_distinct=False, use_ttest=True, out_file=None):

    from scipy.io import loadmat
    import matplotlib.colors as cl
    #import matplotlib.patheffects as pthe

    grid_dir = 'PAS_grid/'
    trend_dir='precomputed_trends/'
    buttressing_file='/data/oceans_output/shelf/kaight/BFRN/AMUND_BFRN_Bedmachinev2_mainGL_withLatLon.mat'
    if supp:
        periods = ['historical', 'historical_noOBCs', 'PACE', 'LENS', 'LENS_noOBCs']
        expt_names = ['Historical', 'Historical Fixed BCs', 'PACE Fixed BCs', 'RCP 8.5', 'RCP 8.5 Fixed BCs']
        colours = [(0.6,0.6,0.6), (0,0.45,0.7), (0,0.62,0.45), (0.8,0.47,0.65), (0.9,0.62,0)]
    else:
        periods = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
        expt_names = ['Historical', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5']
        colours = [(0.6,0.6,0.6), (0,0.45,0.7), (0,0.62,0.45), (0.9,0.62,0), (0.8,0.47,0.65)]
    num_periods = len(periods)
    p0 = 0.05
    vmin = 1e-2
    vmax = 40
    ymax = -71.5
    n_subgrid = 10
    test_distinct = group1 is not None and group2 is not None
    shelf_labels = ['G', 'D', 'Cr', 'T', 'P', 'Co', 'A']
    shelf_x = [-124, -112.4, -111.5, -105, -98, -97.5, -94]
    shelf_y = [-75.1, -74, -75.5, -75.5, -75.15, -73.7, -72.1]
    
    grid = Grid(grid_dir)
    ice_mask = grid.get_ice_mask(shelf=shelf)
    x_mit, y_mit = polar_stereo(grid.lon_2d, grid.lat_2d)

    if depth_classes:
        bin_quantity = np.abs(grid.draft)
    else:
        f = loadmat(buttressing_file)
        buttressing = np.transpose(f['BFRN'])
        x_ua = f['x'][:,0]
        y_ua = f['y'][0,:]
        buttressing_mask = (np.isnan(buttressing) + np.isinf(buttressing)).astype(bool)
        buttressing[buttressing_mask]=0
        # Extend into mask so that we can interpolate to MITgcm ice shelf points without having missing values
        fill = np.ceil(np.minimum(interp_nonreg_xy(x_mit, y_mit, grid.ice_mask.astype(float), x_ua, y_ua, fill_value=0),1))
        fill = extend_into_mask(fill, missing_val=0, num_iters=3)
        buttressing_fill = discard_and_fill(buttressing, buttressing==0, fill, missing_val=0, use_3d=False, log=False) 
        # Now interpolate to MITgcm grid
        bin_quantity = interp_reg_xy(x_ua, y_ua, buttressing_fill, x_mit, y_mit, fill_value=0)
        bin_quantity = np.ma.masked_where(np.invert(ice_mask), bin_quantity)
        if out_file is not None:
            ncfile = NCfile(out_file, grid, 'xy')
            ncfile.add_variable('BFRN', bin_quantity, 'xy')
            ncfile.close()

    if depth_classes:
        bin_edges = np.linspace(np.amin(bin_quantity), np.amax(bin_quantity), num=num_bins+1)
        bin_centres = 0.5*(bin_edges[:-1] + bin_edges[1:])
    else:
        bin_centres = np.logspace(np.log10(vmin), np.log10(vmax), num=num_bins)
        bin_edges = np.concatenate(([np.amin(bin_quantity)], 0.5*(bin_centres[:-1] + bin_centres[1:]), [np.amax(bin_quantity)]))

    # Inner function to read trends, for a single member or the mean for all members in the given period.
    def read_trends (period, ens='all'):
        trend_file = real_dir(trend_dir) + 'ismr_trend_' + period + '.nc'
        trends = read_netcdf(trend_file, 'ismr_trend')
        if ens == 'all':
            mean_trend = np.mean(trends, axis=0)*1e2
            p_val = ttest_1samp(trends, 0, axis=0)[1]
            mean_trend[p_val > p0] = 0
            return mean_trend
        else:
            return trends[ens,:]

    # Inner function to bin the given mean trends into classes
    def bin_trends (mean_trend):
        bin_trends_mean = np.zeros(num_bins)
        bin_area = np.zeros(num_bins)
        for trend_val_mean, bin_val, dA_val, in zip(mean_trend[ice_mask], bin_quantity[ice_mask], grid.dA[ice_mask]):
            bin_index = np.nonzero(bin_edges >= bin_val)[0][0]-1
            bin_trends_mean[bin_index] += trend_val_mean*dA_val
            bin_area[bin_index] += dA_val
        bin_trends_mean /= bin_area
        bin_trends_mean = np.ma.masked_where(bin_area==0, bin_trends_mean)
        bin_trends_mean = np.ma.masked_where(bin_trends_mean==0, bin_trends_mean)
        return bin_trends_mean

    # Read and bin the mean trends for each scenario
    bin_trends_mean_all = []
    for n in range(num_periods):
        mean_trend = read_trends(periods[n])
        bin_trends_mean_all.append(bin_trends(mean_trend))

    # Inner functions to get the ensemble size for the given period, and total for the group of periods.
    def read_ens_size (period):
        file_path = real_dir(trend_dir) + 'ismr_trend_' + period + '.nc'
        return read_netcdf(file_path, 'ismr_trend').shape[0]
    def read_total_ens_size (group):
        num_ens = 0
        for period in group:
            num_ens += read_ens_size(period)
        return num_ens 

    # Inner function to read individual trends for each member of each scenario in the group, and then test significance for the whole group at each point. If the group-ensemble mean trend is not significant, set all individual trends to zero at that point.
    def read_group_trends (group):
        num_ens = read_total_ens_size(group)
        trends = np.ma.empty([num_ens, grid.ny, grid.nx])
        posn = 0
        for period in group:
            num_ens_tmp = read_ens_size(period)
            for e in range(num_ens_tmp):
                trends[posn+e,:] = read_trends(period, ens=e)
            posn += num_ens_tmp
        p_val = ttest_1samp(trends, 0, axis=0)[1]
        p_val = np.tile(p_val, (num_ens,1,1))
        trends[p_val > p0] = 0
        return trends

    # Inner function to read and bin the trends for each member of the group.
    def read_bin_group_trends (group):
        trends = read_group_trends(group)
        num_ens = trends.shape[0]
        bin_trends_group = np.ma.empty([num_ens, num_bins])
        for e in range(num_ens):
            bin_trends_group[e,:] = bin_trends(trends[e,:])
        return bin_trends_group    

    if test_distinct:
        # Need to re-read and bin the individual trends for each member.
        bin_trends_group1 = read_bin_group_trends(group1)
        bin_trends_group2 = read_bin_group_trends(group2)
        # Now test if they're distinct in each bin.
        distinct = np.zeros(num_bins)
        for n in range(num_bins):
            sample1 = bin_trends_group1[:,n]
            sample2 = bin_trends_group2[:,n]
            if use_ttest:
                t_val, p_val = ttest_ind(sample1, sample2, equal_var=False)
                distinct[n] = p_val < p0
            else:
                min1, max1 = norm.interval(1-p0, loc=np.mean(sample1), scale=np.std(sample1))
                min2, max2 = norm.interval(1-p0, loc=np.mean(sample2), scale=np.std(sample2))
                mean1 = np.mean(sample1)
                mean2 = np.mean(sample2)
                distinct[n] = ((mean1 > max2) or (mean1 < min2) or (mean2 > max1) or (mean2 < min1)).astype(float)
        # Find first bin where they are not distinct.
        try:
            n0 = np.where(distinct==0)[0][0]
            print('Groups are distinct for all bins less than '+str(bin_edges[n0]))
            if np.count_nonzero(distinct[n0:]) > 0:
                print('and for '+str(np.count_nonzero(distinct[n0:]))+' bins after that')
                for m in range(n0, distinct.size):
                    if distinct[m]:
                        print(str(bin_edges[m])+'-'+str(bin_edges[m+1]))
                
        except(IndexError):
            print('Groups are always distinct')
        if plot_distinct:
            # Now get the inverse of distinct on the bin edges. If a bin centre is true, want both edges to be true.
            not_distinct = 1-distinct
            not_distinct_edges = np.concatenate(([not_distinct[0]], np.maximum(not_distinct[:-1],not_distinct[1:]), [not_distinct[-1]])).astype(bool)
            if group1 == ['LW1.5', 'LW2.0', 'MENS'] and group2 == ['LENS']:
                max_distinct = bin_edges[n0]
                distinct_label_posn = bin_edges[n0//2]
        
    if supp or depth_classes:
        # Just plot bottom panel - melt as function of BFRN or depth classes
        fig = plt.figure(figsize=(6.5,4.5))
        gs = plt.GridSpec(1,1)
        gs.update(left=0.1, right=0.97, bottom=0.25, top=0.9)
        ax = plt.subplot(gs[0,0])
        if supp:
            # Hack the order of plotting so the legend looks logical
            n_vals = [0, 3, 1, 4, 2]
        else:
            n_vals = range(num_periods)
        for n in n_vals:
            ax.plot(bin_centres, bin_trends_mean_all[n], color=colours[n], marker='o', markersize=5, label=expt_names[n])
        if test_distinct and plot_distinct:
            ax.fill_between(bin_edges, 0, 1, where=not_distinct_edges, color='black', alpha=0.1, transform=ax.get_xaxis_transform())
            plt.text(bin_centres[n0], 14, 'RCP 8.5\nnot distinct', fontsize=12, color=colours[-1], ha='left', va='center', weight='bold')
        ax.grid(linestyle='dotted')
        if depth_classes:
            ax.set_xlabel('Ice draft (m)', fontsize=10)
            ax.set_title('Melting trends as function of depth', fontsize=14)
        else:
            ax.set_xscale('log')
            xticks = ax.get_xticks()
            xtick_labels = []
            for x in xticks:
                if x > 1:
                    xtick_labels.append(str(int(x))+'%')
                else:
                    xtick_labels.append(str(x)+'%')
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel('Buttressing flux response number', fontsize=10)
            ax.set_title('Melting trends as function of buttressing', fontsize=14)
        ax.set_ylabel('Mean basal melting trend (m/y/century)', fontsize=10)
        ax.set_xlim([0, bin_edges[-1]])
        if supp:
            ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.36))
        else:
            ax.legend(ncol=num_periods, loc='lower center', bbox_to_anchor=(0.46, -0.3))
        finished_plot(fig, fig_name=fig_name, dpi=300)
    else:
        # 2 panel plot
        fig = plt.figure(figsize=(7,6))
        gs = plt.GridSpec(7,1)
        gs.update(left=0.1, right=0.88, bottom=0.15, top=0.95, hspace=3)
        # Plot BFRN
        ax = plt.subplot(gs[0:3,0])
        img = latlon_plot(bin_quantity, grid, norm=cl.LogNorm(), vmin=vmin, vmax=vmax, ymax=ymax, ctype='buttressing', ax=ax, make_cbar=False, title=r'$\bf{a}$. '+'Buttressing flux response number', titlesize=14)
        # Shade negative values in light blue
        neg_mask = bin_quantity < 0
        shade_mask(ax, neg_mask, grid, colour='PowderBlue')
        # Label ice shelves
        for m in range(len(shelf_labels)):
            txt = plt.text(shelf_x[m], shelf_y[m], shelf_labels[m], fontsize=8, ha='center', va='center', weight='bold', color='black')
            #txt.set_path_effects([pthe.withStroke(linewidth=1, foreground='w')])
        ytick_labels = ax.get_yticklabels()
        for label in ytick_labels[1::2]:
            label.set_visible(False)
        cax = fig.add_axes([0.9, 0.665, 0.02, 0.28])
        cbar = plt.colorbar(img, cax=cax, extend='both')
        ctick_labels = cax.get_yticklabels()
        for label in ctick_labels:
            label_text = label.get_text()
            if len(label_text) > 0:
                exp_start = label_text.index('10^{')+4
                exp_end = label_text.index('}')
                exp = float(label_text[exp_start:exp_end])
                y = 10**exp
                if y > 1:
                    label.set_text(str(int(y))+'%')
                else:
                    label.set_text(str(y)+'%')
        cax.set_yticklabels(ctick_labels)
        # Plot melt as function of BFRN
        ax = plt.subplot(gs[3:,0])
        for n in range(num_periods):
            ax.plot(bin_centres, bin_trends_mean_all[n], color=colours[n], marker='o', markersize=5, label=expt_names[n])
        if test_distinct and plot_distinct:
            if group1 == ['LW1.5', 'LW2.0', 'MENS'] and group2 == ['LENS']:
                ax.axvline(max_distinct, color='black', linestyle='dashed', linewidth=1)
                plt.text(distinct_label_posn, 9, 'RCP 8.5 distinct', fontsize=12, color=colours[-1], ha='center', va='center', weight='bold')
            else:
                ax.fill_between(bin_edges, 0, 1, where=distinct_edges, color='black', alpha=0.1, transform=ax.get_xaxis_transform())    
        ax.set_xscale('log')
        xticks = ax.get_xticks()
        xtick_labels = []
        for x in xticks:
            if x > 1:
                xtick_labels.append(str(int(x))+'%')
            else:
                xtick_labels.append(str(x)+'%')
        ax.set_xticklabels(xtick_labels)
        ax.grid(linestyle='dotted')
        ax.set_xlabel('Buttressing flux response number', fontsize=10)
        ax.set_ylabel('Mean basal melting trend (m/y/century)', fontsize=10)
        ax.set_title(r'$\bf{b}$. '+'Melting trends as function of buttressing', fontsize=14)
        ax.legend(ncol=num_periods, loc='lower center', bbox_to_anchor=(0.5, -0.33))
        finished_plot(fig, fig_name=fig_name, dpi=300)


# Supplementary figures (2)
def sfc_forcing_trends (var, fig_name=None):

    grid_dir = 'PAS_grid/'
    trend_dir = 'precomputed_trends/'
    title = 'Trends in '
    if var == 'wind':
        var_name_u = 'EXFuwind'
        var_name_v = 'EXFvwind'
        var_name = 'wind_speed'
        threshold = 0 #0.25
        scale = 10
        chunk_x = 24
        chunk_y = 12
        title += 'surface winds (m/s/century)'
        extend = 'neither'
        ymax = None
    elif var == 'fwflux':
        var_name = 'oceFWflx'
        title += r'surface freshwater flux (m/y/century)'
        vmin = -4
        vmax = 4
        extend = 'both'
        ymax = -70
    periods = ['historical', 'PACE', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    expt_names = ['Historical', 'PACE', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5']
    num_periods = len(periods)
    p0 = 0.05
    xmin = None # -138
    xmax = None #-82
    ymin = None
    z_shelf = -1750

    def read_single_trend (var_in_file, period):
        file_path = trend_dir + var_in_file + '_trend_' + period + '.nc'
        trends = read_netcdf(file_path, var_in_file+'_trend')
        mean_trend = np.mean(trends, axis=0)
        p_val = ttest_1samp(trends, 0, axis=0)[1]
        mean_trend[p_val > p0] = 0
        if var_in_file == 'oceFWflx':
            mean_trend *= sec_per_year/rho_fw
        return mask_land_ice(mean_trend, grid)*1e2

    data_plot = np.ma.empty([num_periods, grid.ny, grid.nx])
    if var == 'wind':
        data_plot_u = np.ma.empty([num_periods, grid.ny, grid.nx])
        data_plot_v = np.ma.empty([num_periods, grid.ny, grid.nx])
    for t in range(num_periods):
        data_plot[t,:] = read_single_trend(var_name, periods[t])
        if var == 'wind':
            trends_u = read_single_trend(var_name_u, periods[t])
            trends_v = read_single_trend(var_name_v, periods[t])
            index = ((np.abs(data_plot[t,:]) < threshold) + (trends_u==0)*(trends_v==0)).astype(bool)
            trends_u = np.ma.masked_where(index, trends_u)
            trends_v = np.ma.masked_where(index, trends_v)
            data_plot_u[t,:] = trends_u
            data_plot_v[t,:] = trends_v

    if var == 'wind':
        vmin = np.amin([var_min_max(data_plot[t,:], grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)[0] for t in range(num_periods)])
        vmax = np.amax([var_min_max(data_plot[t,:], grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)[1] for t in range(num_periods)])

    fig = plt.figure(figsize=(8,9))
    gs = plt.GridSpec(3,2)
    gs.update(left=0.07, right=0.93, bottom=0.07, top=0.92, wspace=0.05, hspace=0.2)
    cax = fig.add_axes([0.22, 0.03, 0.56, 0.02])
    for t in range(num_periods):
        ax = plt.subplot(gs[t//2, t%2])
        img = latlon_plot(data_plot[t,:], grid, ax=ax, make_cbar=False, vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title=expt_names[t], titlesize=14, ctype='plusminus')
        if var == 'wind':
            overlay_vectors(ax, data_plot_u[t,:], data_plot_v[t,:], grid, chunk_x=chunk_x, chunk_y=chunk_y, scale=scale, headwidth=4, headlength=5)
        ax.tick_params(direction='in')
        contour_shelf_break(grid, ax, colour='magenta', z_shelf=z_shelf)
        if t != 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.colorbar(img, cax=cax, orientation='horizontal', extend=extend)
    plt.suptitle(title, fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Test if the trends in the given variable are distinct between the given two scenarios over their common time period.
def trend_scenarios_distinct (var_name, expt_name_1, expt_name_2, timeseries_file='timeseries.nc', timeseries_file_pace='timeseries_final.nc', start_year=None, end_year=None):

    p0 = 0.05
    smooth = 24
    # Note no need to calculate mass loss as percent of baseline, as linear transformation will not change significance of trends

    # For the given experiment name, return a list of paths to timeseries files, and years to consider.
    def expt_name_setup (expt_name, start_year, end_year):
        # Defaults
        expt_head = 'PAS_LENS'
        expt_tail = '_O'
        prec = 3
        num_ens = 10
        if start_year is None:
            start_year0 = 2006
        else:
            start_year0 = start_year
        if end_year is None:
            end_year0 = 2100
        else:
            end_year0 = end_year
        ts_file = timeseries_file
        if expt_name == 'Historical':
            if start_year is None:
                start_year0 = 1920
            if end_year is None:
                end_year0 = 2005
        elif expt_name == 'Historical fixed BCs':
            expt_tail = '_noOBC'
            num_ens = 5
            if start_year is None:
                start_year0 = 1920
            if end_year is None:
                end_year0 = 2005
        elif expt_name == 'PACE':
            expt_head = '../mitgcm/PAS_PACE'
            expt_tail = ''
            prec = 2
            num_ens = 20
            if start_year is None:
                start_year0 = 1920
            if end_year is None:
                end_year0 = 2013
            ts_file = timeseries_file_pace
        elif expt_name == 'Paris 1.5C':
            expt_head = 'PAS_LW1.5_'
            num_ens = 5
        elif expt_name == 'Paris 2C':
            expt_head = 'PAS_LW2.0_'
        elif expt_name == 'RCP 4.5':
            expt_head = 'PAS_MENS_'
            if end_year is None:
                end_year0 = 2080
        elif expt_name == 'RCP 8.5':
            pass
        elif expt_name == 'RCP 8.5 fixed BCs':
            expt_tail = '_noOBC'
            num_ens = 5
        file_paths = [expt_head+str(n+1).zfill(prec)+expt_tail+'/output/'+ts_file for n in range(num_ens)]
        return file_paths, start_year0, end_year0

    file_paths_1, start_year_1, end_year_1 = expt_name_setup(expt_name_1, start_year, end_year)
    file_paths_2, start_year_2, end_year_2 = expt_name_setup(expt_name_2, start_year, end_year)
    start_year = max(start_year_1, start_year_2)
    end_year = min(end_year_1, end_year_2)

    # Calculate the trend in all the given files
    def calc_expt_trends (file_paths):
        num_ens = len(file_paths)
        trends = np.zeros(num_ens)
        for n in range(num_ens):
            slope, sig = read_calc_trend(var_name, file_paths[n], start_year=start_year, end_year=end_year, smooth=smooth, p0=p0)
            if sig:
                trends[n] = slope
        return trends

    trends1 = calc_expt_trends(file_paths_1)
    trends2 = calc_expt_trends(file_paths_2)
    p_val = ttest_ind(trends1, trends2, equal_var=False)[1]
    distinct = p_val < p0
    # For reviewer, also test 1-sample t-test on member-wise diffferences
    num_ens = min(trends1.size, trends2.size)
    trends_diff = trends2[:num_ens] - trends1[:num_ens]
    p_val2 = ttest_1samp(trends_diff, 0)[1]
    distinct2 = p_val2 < p0    
    print('Trends in '+var_name+' over '+str(start_year)+'-'+str(end_year)+':')
    for expt_name, trends in zip([expt_name_1, expt_name_2], [trends1, trends2]):
        print(expt_name+': mean trend '+str(np.mean(trends)))
    print('p_vals: '+str(p_val)+', '+str(p_val2))
    if distinct and distinct2:
        print('They are distinct under both tests\n')
    elif (not distinct) and (not distinct2):
        print('They are not distinct under both tests\n')
    elif distinct and (not distinct2):
        print('They are distinct under 2-sample t-test only\n')
    elif (not distinct) and distinct2:
        print('They are distinct under 1-sample t-test only\n')


# Call the above function for all the sensible combinations of trends in table
def all_trends_distinct ():

    var_names = ['amundsen_shelf_temp_btw_200_700m', 'dotson_to_cosgrove_massloss']
    expt_names = ['Historical', 'Historical fixed BCs', 'Paris 1.5C', 'Paris 2C', 'RCP 4.5', 'RCP 8.5', 'RCP 8.5 fixed BCs', 'PACE']
    for var in var_names:
        # All combinations of historical scenarios
        # Historical with and without transient BCs
        trend_scenarios_distinct(var, expt_names[0], expt_names[1])
        # Historical vs PACE
        trend_scenarios_distinct(var, expt_names[0], expt_names[-1])
        # Historical with fixed BCs vs PACE
        trend_scenarios_distinct(var, expt_names[1], expt_names[-1])
        
        # All combinations of future scenarios with transient BCs
        for n1 in range(2, 5+1):
            for n2 in range(n1+1, 5+1):
                if n1 == n2:
                    continue
                trend_scenarios_distinct(var, expt_names[n1], expt_names[n2])
        # RCP 8.5 with and without transient BCs
        trend_scenarios_distinct(var, expt_names[5], expt_names[6])


# Plot a scatterplot of the trends in any 2 variables across all ensemble members and future scenarios
def trend_scatterplots (var1, var2, base_dir='./', timeseries_file='timeseries.nc', timeseries_file_2=None, num_LENS=10, num_MENS=10, num_LW2=10, num_LW1=5, num_PACE=20, num_noOBC=5, fig_name=None, timeseries_file_PACE='timeseries_final.nc', option='future'):

    base_dir = real_dir(base_dir)
    if option == 'future':
        num_ens = [num_LENS, num_MENS, num_LW2, num_LW1]
        expt_names = ['LENS', 'MENS', 'LW2.0', 'LW1.5']
        expt_mid = ['', '_', '_', '_']
        expt_tail = ['_O', '_O', '_O', '_O']
        expt_titles = ['RCP 8.5', 'RCP 4.5', 'Paris 2C', 'Paris 1.5C']
        expt_colours = ['DarkGrey', 'IndianRed', 'MediumSeaGreen', 'DodgerBlue']
        start_year = 2006
        if num_MENS == 0:
            end_year = 2100
        else:
            end_year = 2080
    elif option == 'historical':
        num_ens = [num_LENS, num_noOBC, num_PACE]
        expt_names = ['LENS', 'LENS', 'PACE']
        expt_mid = ['', '', '']
        expt_tail = ['_O', '_noOBC', '']
        expt_titles = ['Historical', 'Historical fixed BCs', 'PACE fixed BCs']
        expt_colours = ['DarkGrey', 'DodgerBlue', 'MediumSeaGreen']
        start_year = 1920
        end_year = 2005        
    
    num_expt = len(num_ens)
    smooth = 24
    p0 = 0.05
    if timeseries_file_2 is None:
        timeseries_file_2 = timeseries_file
    timeseries_files = [timeseries_file, timeseries_file_2]
    timeseries_files_PACE = [timeseries_file_PACE, timeseries_file_PACE]

    trend1 = []
    trend2 = []
    labels = []
    colours = []
    for n in range(num_expt):
        if expt_names[n] == 'PACE':
            expt_prec = 2
            head_dir = base_dir + '../mitgcm/'
            timeseries_files_expt = timeseries_files_PACE
        else:
            expt_prec = 3
            head_dir = base_dir
            timeseries_files_expt = timeseries_files
        for e in range(num_ens[n]):
            both_trends = []
            for var, k in zip([var1, var2], range(2)):
                if var == 'TS_global_mean':
                    file_path = base_dir + 'cesm_sat_timeseries/' + expt_names[n] + '_' + str(e+1).zfill(expt_prec) + '_TS_global_mean.nc'
                else:
                    file_path = head_dir + 'PAS_' + expt_names[n] + expt_mid[n] + str(e+1).zfill(expt_prec) + expt_tail[n] + '/output/' + timeseries_files_expt[k]
                trend_tmp = read_calc_trend(var, file_path, p0=p0, start_year=start_year, end_year=end_year, smooth=smooth)[0]
                both_trends.append(trend_tmp)
            trend1.append(both_trends[0])
            trend2.append(both_trends[1])
            if expt_titles[n] not in labels:
                labels.append(expt_titles[n])
            else:
                labels.append(None)
            colours.append(expt_colours[n])

    fig, ax = plt.subplots(figsize=(10,6))
    ax.axhline()
    ax.axvline()
    for m in range(len(trend1)):
        ax.plot(trend1[m], trend2[m], 'o', color=colours[m], label=labels[m])
    ax.grid(linestyle='dotted')
    slope, intercept, r_value, p_value, std_err = linregress(np.array(trend1), np.array(trend2))
    if p_value < p0:
        [x0, x1] = ax.get_xlim()
        [y0, y1] = slope*np.array([x0, x1]) + intercept
        ax.plot([x0, x1], [y0, y1], '-', color='black', linewidth=1, zorder=0)
        trend_title = 'r$^2$='+str(round_to_decimals(r_value**2, 3))
        print('r^2='+str(r_value**2))
    else:
        trend_title = 'no significant relationship'
        print('p='+str(p_value)+', r^2='+str(r_value**2))
    ax.text(0.05, 0.95, trend_title, ha='left', va='top', fontsize=12, transform=ax.transAxes)
    ax.set_xlabel(var1, fontsize=14)
    ax.set_ylabel(var2, fontsize=14)
    ax.set_title('Trends per century, '+str(start_year)+'-'+str(end_year), fontsize=18)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    finished_plot(fig, fig_name=fig_name)


# Supplementary figure
def hovmoller_anomaly_std (fig_name=None):

    var = 'temp'
    region = 'amundsen_shelf'
    var_name = 'Temperature'
    units = deg_string + 'C'
    hovmoller_file = 'hovmoller_shelf.nc'
    expt_names = [r'$\bf{a}$. Historical', r'$\bf{b}$. Paris 2'+deg_string+'C', r'$\bf{c}$. RCP 8.5']
    expt_dir_head = 'PAS_'
    expt_dir_mids = ['LENS', 'LW2.0_', 'LENS']
    expt_dir_tail = '_O/output/'
    num_ens = 10
    start_years = [1920, 2006, 2006]
    end_years = [2005, 2100, 2100]
    num_expt = len(expt_names)
    grid_dir = 'PAS_grid/'
    vmin_anom = -1
    vmax_anom = 1.75
    vmin_std = 0
    vmax_std = 0.5
    contours_std = 0.2
    smooth = 12
    grid = Grid(grid_dir)
    timeseries_iso_file = 'timeseries_isotherm.nc'
    iso_var = region+'_isotherm_0.5C_below_100m'
    z0 = -440

    time_plot = []
    anom_plot = []
    std_plot = []
    time_conv_plot = []
    conv_plot = []
    for n in range(num_expt):
        for e in range(num_ens):
            output_dir = expt_dir_head + expt_dir_mids[n] + str(e+1).zfill(3) + expt_dir_tail
            file_path = output_dir + hovmoller_file
            iso_file_path = output_dir + timeseries_iso_file
            time = netcdf_time(file_path, monthly=False)
            data = read_netcdf(file_path, region+'_'+var)
            iso_depth = read_netcdf(iso_file_path, iso_var)
            if time.size != iso_depth.size:
                print('Error: isotherm timeseries file does not match Hovmollers time axis')
                sys.exit()
            t_start, t_end = index_period(time, start_years[n], end_years[n])
            data = data[t_start:t_end,:]
            time = time[t_start:t_end]
            iso_depth = iso_depth[t_start:t_end]
            iso_depth_smooth, time_smooth = moving_average(iso_depth, smooth, time=time)
            conv = (iso_depth_smooth < z0).astype(float)
            if e==0:
                time_plot.append(time)
                data_ens = np.ma.empty([num_ens, time.size, grid.nz])
                time_conv_plot.append(time_smooth)
                conv_ens = np.ma.zeros([time_smooth.size])
            data_ens[e,:] = data
            conv_ens += conv
        if n==0:
            data_baseline = np.ma.mean(data_ens, axis=(0,1))
        anom_plot.append(np.ma.mean(data_ens, axis=0) - data_baseline[None,:])
        std_plot.append(np.ma.std(data_ens, axis=0))
        conv_plot.append(conv_ens/num_ens*1e2)

    fig = plt.figure(figsize=(8,8))
    gs = plt.GridSpec(num_expt,2)
    gs.update(left=0.07, right=0.97, bottom=0.12, top=0.9, hspace=0.4, wspace=0.05)
    cax1 = fig.add_axes([0.095, 0.06, 0.4, 0.02])
    cax2 = fig.add_axes([0.555, 0.06, 0.4, 0.02])
    data_plot = [anom_plot, std_plot]
    vmin = [vmin_anom, vmin_std]
    vmax = [vmax_anom, vmax_std]
    var_title = ['Anomaly', 'Standard deviation']
    cax = [cax1, cax2]
    ctype = ['plusminus', 'Purples']
    contours = [None, contours_std]
    for n in range(num_expt):
        for m in range(2):
            ax = plt.subplot(gs[n,m])
            img = hovmoller_plot(data_plot[m][n], time_plot[n], grid, smooth=smooth, ax=ax, make_cbar=False, vmin=vmin[m], vmax=vmax[m], ctype=ctype[m], contours=contours[m])
            '''if m==1:
                # Plot percentage of members convecting
                ax2 = ax.twinx()
                ax2.plot_date(time_conv_plot[n], conv_plot[n], '-', color='green', linewidth=2)
                ax2.set_ylim([0, 100])
                ax2.set_ylabel('% ensemble convecting', fontsize=12)'''
            ax.set_xlim([datetime.date(start_years[n], 1, 1), datetime.date(end_years[n]-1, 12, 31)])
            if n==0 and m==1:
                ax.set_xticks([datetime.date(year, 1, 1) for year in np.arange(start_years[n]+20, end_years[n], 20)])
            else:
                ax.set_xticks([datetime.date(year, 1, 1) for year in np.arange(int(np.ceil(start_years[n]/10)*10), end_years[n], 20)])
            ax.set_yticks([0, -500, -1000])
            if m==0:
                ax.set_yticklabels(['0', '0.5', '1'])                    
            else:
                ax.set_yticklabels([])
            if m==0 and n==0:
                ax.set_ylabel('Depth (km)', fontsize=12)
            else:
                ax.set_ylabel('')
            ax.tick_params(direction='in')
            ax.set_xlabel('')
            if n==num_expt-1:
                cbar = plt.colorbar(img, cax=cax[m], orientation='horizontal', extend='both')
                plt.text(0.3+0.46*m, 0.015, var_title[m], ha='center', va='center', transform=fig.transFigure, fontsize=14)
        plt.text(0.5, 0.92-0.285*n, expt_names[n], ha='center', va='center', transform=fig.transFigure, fontsize=16)
    plt.suptitle(var_name+' ('+units+') on '+region_names[region], fontsize=18)
    finished_plot(fig, fig_name=fig_name)


def plot_barotropic_strf_mit (expt_name, num_ens, start_year, end_year, fig_name=None, vmin=None, vmax=None, ymax=None):

    grid_dir = 'PAS_grid/'
    if expt_name == 'LENS':
        expt_dir_head = 'PAS_'+expt_name
    elif expt_name in ['MENS', 'LW1.5', 'LW2.0']:
        expt_dir_head = 'PAS_'+expt_name+'_'
    else:
        print('Error (plot_barotropic_transport_mit): invalid expt_name '+expt_name)
        sys.exit()
    expt_dir = [expt_dir_head+str(n+1).zfill(3)+'_O/output/' for n in range(num_ens)]
    
    grid = Grid(grid_dir)
    wct = abs(mask_land(grid.draft-grid.bathy, grid))
    dy = grid.dy_w
    strf_accum = np.ma.zeros([grid.ny, grid.nx])

    for n in range(num_ens):
        print('Processing member '+str(n+1))
        for year in range(start_year, end_year+1):
            print('...'+str(year))
            file_path = expt_dir[n] + str(year) + '01/MITgcm/output.nc'
            # Calculate based on annual means of u
            u = mask_3d(read_netcdf(file_path, 'UVEL', time_average=True), grid, gtype='u')
            ubar = vertical_average(u, grid, gtype='u')
            strf_accum += np.ma.cumsum(ubar*wct*dy, axis=0)*1e-6
    strf = strf_accum/(num_ens*(end_year-start_year+1))

    latlon_plot(strf, grid, gtype='u', ctype='plusminus', title='Barotropic streamfunction (Sv), '+expt_name+' ('+str(start_year)+'-'+str(end_year)+')', fig_name=fig_name, vmin=vmin, vmax=vmax, ymax=ymax)


def plot_correlation_map (var_2d='EXFuwind', var_1d='amundsen_shelf_temp_btw_200_700m', fig_name=None):

    periods = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    num_expt = len(periods)
    expt_names = ['Historical', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5']
    num_ens = [10, 5, 10, 10, 10]
    start_years = [1920, 2006, 2006, 2006, 2006]
    end_years = [2005, 2100, 2100, 2080, 2100]
    expt_dir_head = 'PAS_'
    expt_dir_mids = ['LENS', 'LW1.5_', 'LW2.0_', 'MENS_', 'LENS']
    expt_dir_tail = '_O/output/'
    timeseries_file = 'timeseries.nc'
    trend_file_head = 'precomputed_trends/'+var_2d+'_trend_'
    trend_file_tail = '.nc'
    p0_weak = 0.25
    p0_strong = 0.05
    grid_dir = 'PAS_grid/'
    grid = Grid(grid_dir)
    if var_2d == 'EXFuwind':
        var_title_2d = 'zonal wind'
    elif var_2d == 'EXFvwind':
        var_title_2d = 'meridional wind'
    elif var_2d == 'oceFWflx':
        var_title_2d = 'freshwater flux'
    else:
        var_title_2d = var_2d
    if var_1d == 'amundsen_shelf_temp_btw_200_700m':
        var_title_1d = 'shelf temperature'
    elif var_1d == 'dotson_to_cosgrove_massloss':
        var_title_1d = 'ice shelf basal melting'
    elif var_1d == 'PITE_trans':
        var_title_1d = 'PITE transport'
    else:
        var_title_1d = var_1d

    # Read the timeseries and calculate trends for every simulation
    trends_1d = []
    for n in range(num_expt):
        expt_trends = np.ma.empty(num_ens[n])
        for e in range(num_ens[n]):
            file_path = expt_dir_head + expt_dir_mids[n] + str(e+1).zfill(3) + expt_dir_tail + timeseries_file
            slope, sig = read_calc_trend(var_1d, file_path, start_year=start_years[n], end_year=end_years[n], annual_avg=True)
            expt_trends[e] = slope
        trends_1d.append(expt_trends)

    # Read the 2D trend fields for every simulation
    trends_2d = []
    for n in range(num_expt):
        file_path = trend_file_head + periods[n] + trend_file_tail
        expt_trends = read_netcdf(file_path, var_2d+'_trend')
        trends_2d.append(expt_trends)

    # Now calculate the correlation of trends at each point, considering different sets of simulations
    correlations = np.ma.zeros([num_expt+2, grid.ny, grid.nx])
    p_values = np.ma.zeros([num_expt+2, grid.ny, grid.nx])
    mask = grid.get_open_ocean_mask().astype(bool)
    for j in range(grid.ny):
        for i in range(grid.nx):
            if not mask[j,i]:
                # Land or ice shelf point
                continue
            for n in range(num_expt+2):
                if n < num_expt:
                    # First calculate intra-ensemble correlations for each scenario (num_ens[n] number of points)
                    trends1 = trends_1d[n]
                    trends2 = trends_2d[n][:,j,i]
                elif n == num_expt:
                    # Then calculate inter-ensemble correlations considering all members of all scenarios (sum(num_ens) number of points)
                    for m in range(num_expt):
                        trends1_tmp = trends_1d[m]
                        trends2_tmp = trends_2d[m][:,j,i]
                        if m==0:
                            trends1 = trends1_tmp
                            trends2 = trends2_tmp
                        else:
                            trends1 = np.concatenate((trends1, trends1_tmp))
                            trends2 = np.concatenate((trends2, trends2_tmp))
                elif n == num_expt+1:
                    # Then calculate inter-ensemble correlations considering the ensemble mean of each scenario (num_expt number of points)
                    trends1 = np.ma.empty([num_expt])
                    trends2 = np.ma.empty([num_expt])
                    for m in range(num_expt):
                        trends1[m] = np.mean(trends_1d[m])
                        trends2[m] = np.mean(trends_2d[m][:,j,i])
                # Now get the correlation between these two sets of trends
                slope, intercept, r_value, p_value, std_err = linregress(trends1, trends2)
                correlations[n,j,i] = r_value
                p_values[n,j,i] = p_value
    correlations = np.ma.masked_where(p_values > p0_weak, correlations)

    titles = expt_names + ['All', 'Ensemble Means']
    vmin = np.amin(correlations)
    vmax = np.amax(correlations)
    fig = plt.figure(figsize=(7,10))
    gs = plt.GridSpec(4,2)
    gs.update(left=0.05, right=0.95, bottom=0.02, top=0.95)
    cax = fig.add_axes([0.05, 0.78, 0.4, 0.02])
    for n in range(len(titles)):
        ax = plt.subplot(gs[(n+1)//2, (n+1)%2])
        img = latlon_plot(mask_land_ice(correlations[n,:], grid), grid, ax=ax, make_cbar=False, vmin=vmin, vmax=vmax, title=titles[n], titlesize=14, ctype='plusminus')
        ax.contour(grid.lon_2d, grid.lat_2d, p_values[n,:], levels=[p0_strong], colors=('black'), linewidths=1, linestyles='solid')
        ax.tick_params(direction='in')
        if n != 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.colorbar(img, cax=cax, orientation='horizontal')
    plt.text(0.02, 0.98, 'Correlation between trends in\n'+var_title_2d+' (2D)\n and trends in\n'+var_title_1d+' (1D)', fontsize=16, ha='left', va='top', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_name)


# Supplementary figure
def forcing_temp_correlation_maps (base_dir='./', fig_name=None):

    forcing_var_names = ['EXFuwind', 'EXFvwind', 'EXFatemp', 'EXFpreci']
    forcing_var_titles = [r'$\bf{a}$. Winds', None, r'$\bf{b}$. Surface air temperature', r'$\bf{c}$. Precipitation']
    num_forcing = len(forcing_var_names)
    periods = ['historical', 'LW1.5', 'LW2.0', 'MENS', 'LENS']
    num_expt = len(periods)
    num_ens = [10, 5, 10, 10, 10]
    start_years = [1920, 2006, 2006, 2006, 2006]
    end_years = [2005, 2100, 2100, 2080, 2100]
    expt_dir_head = base_dir+'PAS_'
    expt_dir_mids = ['LENS', 'LW1.5_', 'LW2.0_', 'MENS_', 'LENS']
    expt_dir_tail = '_O/output/'
    timeseries_file = 'timeseries.nc'
    trend_dir = base_dir+'precomputed_trends/'
    temp_var_name = 'amundsen_shelf_temp_btw_200_700m'
    p0_weak = 0.25
    p0 = 0.05
    z_shelf = -1750
    main_title = 'Correlation between trends in atmospheric forcing (2D)\nand trends in continental shelf temperature (1D)'
    grid_dir = 'PAS_grid/'
    grid = Grid(grid_dir)

    # Read the timeseries and calculate trends for every simulation
    trends_1d = []
    for n in range(num_expt):
        expt_trends = np.ma.empty(num_ens[n])
        for e in range(num_ens[n]):
            file_path = expt_dir_head + expt_dir_mids[n] + str(e+1).zfill(3) + expt_dir_tail + timeseries_file
            slope, sig = read_calc_trend(temp_var_name, file_path, start_year=start_years[n], end_year=end_years[n], annual_avg=True)
            expt_trends[e] = slope
        trends_1d.append(expt_trends)

    # Read the 2D trend fields for every simulation and forcing variable
    trends_2d = []
    for n in range(num_expt):
        expt_trends = np.ma.empty([num_forcing, num_ens[n], grid.ny, grid.nx])
        for v in range(num_forcing):
            file_path = trend_dir + forcing_var_names[v] + '_trend_' + periods[n] + '.nc'
            expt_trends[v,:] = read_netcdf(file_path, forcing_var_names[v]+'_trend')
        trends_2d.append(expt_trends)

    # Calculate the correlations of trends at each point, considering all simulations together
    correlations = np.ma.zeros([num_forcing, grid.ny, grid.nx])
    p_values = np.ma.zeros([num_forcing, grid.ny, grid.nx])
    mask = grid.get_open_ocean_mask().astype(bool)
    for j in range(grid.ny):
        print('...latitude index '+str(j+1)+' of '+str(grid.ny))
        for i in range(grid.nx):
            if not mask[j,i]:
                # Land or ice shelf point
                continue
            for v in range(num_forcing):
                for m in range(num_expt):
                    trends1_tmp = trends_1d[m]
                    trends2_tmp = trends_2d[m][v,:,j,i]
                    if m==0:
                        trends1 = trends1_tmp
                        trends2 = trends2_tmp
                    else:
                        trends1 = np.concatenate((trends1, trends1_tmp))
                        trends2 = np.concatenate((trends2, trends2_tmp))
                # Now get the correlation between these two sets of trends
                slope, intercept, r_value, p_value, std_err = linregress(trends1, trends2)
                correlations[v,j,i] = r_value
                p_values[v,j,i] = p_value
    correlations_strong = np.ma.masked_where(p_values > p0, correlations)
    
    # Set up figure
    fig = plt.figure(figsize=(10,4))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.05, right=0.98, bottom=0.18, top=0.75, wspace=0.05)
    cax = fig.add_axes([0.475, 0.07, 0.4, 0.04])
    for v in range(num_forcing):
        if v==0:
            ax = plt.subplot(gs[0,v])
            # Vector plot of winds
            # Start with an empty plot
            data_empty = np.ma.masked_where(correlations[v,:]<2, correlations[v,:])
            latlon_plot(data_empty, grid, ax=ax, make_cbar=False, include_shelf=False)
            contour_shelf_break(grid, ax, colour='magenta', z_shelf=z_shelf)
            # Average blocks before checking significance
            chunk_x = 24
            chunk_y = 20
            headwidth = 6
            headlength = 7
            lon_plot, lat_plot, correlations_u, correlations_v = average_blocks(grid.lon_2d, grid.lat_2d, mask_land_ice(correlations[v,:],grid), mask_land_ice(correlations[v+1,:],grid), chunk_x, chunk_y, 'avg')
            p_values_u, p_values_v = average_blocks(grid.lon_2d, grid.lat_2d, p_values[v,:], p_values[v+1,:], chunk_x, chunk_y, 'avg')[2:]
            # Overlay vectors in grey regardless of significance
            ax.quiver(lon_plot, lat_plot, correlations_u, correlations_v, scale=10, color=(0.6,0.6,0.6), headwidth=headwidth, headlength=headlength)
            # Overlay vectors in black where at least one component is significant at 95% level
            neither_sig = (p_values_u > p0)*(p_values_v > p0)
            correlations_u_strong = np.ma.masked_where(neither_sig, correlations_u)
            correlations_v_strong = np.ma.masked_where(neither_sig, correlations_v)
            q = ax.quiver(lon_plot, lat_plot, correlations_u_strong, correlations_v_strong, scale=10, color='black', headwidth=headwidth, headlength=headlength)
            ax.quiverkey(q, X=0.5, Y=-0.2, U=1, label='1', labelpos='E')
            labels = ax.xaxis.get_ticklabels()
            for label in labels[1::2]:
                label.set_visible(False)
        elif v==1:
            # Already dealt with the second wind component
            continue
        else:
            ax = plt.subplot(gs[0,v-1])
            # Contour plot of thermodynamics
            img = latlon_plot(mask_land_ice(correlations_strong[v,:], grid), grid, ax=ax, make_cbar=False, vmin=-1, vmax=1, ctype='plusminus', include_shelf=False)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(forcing_var_titles[v], fontsize=15)
    plt.colorbar(img, cax=cax, orientation='horizontal')
    plt.suptitle(main_title, fontsize=18)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# 2 supplementary figures showing example of OBCS correction
def plot_obcs_correction (fig_name_physical_space=None, fig_name_ts_space=None):

    expt = 'LENS'
    bdry = 'E'
    year = 2100
    month = 7
    ens = 1
    case_label = 'RCP 8.5 (member 1), July 2100, eastern boundary'
    main_title_physical_space = 'Bias correction of salinity (psu):\n'+case_label
    main_title_ts_space = 'Water masses in normalised T/S space:\n'+case_label
    hmin = -73.25
    zmin = -1500
    grid_dir = 'PAS_grid/'
    grid = Grid(grid_dir)

    mit_h, mit_z, woa_clim, cesm_h, cesm_z, cesm_data, cesm_clim, cesm_anom, bin_edges, woa_volume_perbin, cesm_volume_perbin, cesm_anom_ts, cesm_anom_ts_filled, final_anom, data_corrected = read_correct_cesm_ts_space(expt, bdry, ens, year, month, return_all_for_plotting=True)
    final_anom = np.ma.masked_where(woa_clim.mask, final_anom)

    # Inner function to set colour bounds to the min/max of a number of arrays
    def set_colour_bounds (arrays, index, ctype):
        vmin = np.amin([np.amin(A[index,:]) for A in arrays])
        vmax = np.amax([np.amax(A[index,:]) for A in arrays])
        cmap = set_colours(arrays[0][index,:], ctype=ctype, vmin=vmin, vmax=vmax)[0]
        return vmin, vmax, cmap

    # Overview figure in physical space
    fig = plt.figure(figsize=(8,8))
    gs = plt.GridSpec(3,2)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.87, hspace=0.25, wspace=0.05)
    data_plot = [woa_clim[1,:], cesm_clim[1,:], cesm_data[1,:], data_corrected[1,:], cesm_anom[1,:], final_anom[1,:]]
    h_axis = [mit_h, cesm_h, cesm_h, mit_h, cesm_h, mit_h]
    z_axis = [mit_z, cesm_z, cesm_z, mit_z, cesm_z, mit_z]
    titles = [r'$\bf{a}$. WOA climatology', r'$\bf{b}$. CESM1 climatology', r'$\bf{c}$. CESM1 raw fields', r'$\bf{d}$. Corrected fields', r'$\bf{e}$. CESM1 raw anomalies', r'$\bf{f}$. Corrected anomalies']
    vmin_abs, vmax_abs, cmap_abs = set_colour_bounds([woa_clim, cesm_clim, cesm_data, data_corrected], 1, 'parula')
    vmin_anom, vmax_anom, cmap_anom = set_colour_bounds([cesm_anom, final_anom], 1, 'plusminus')
    colour_set = ['abs', 'abs', 'abs', 'abs', 'anom', 'anom']
    for n in range(6):
        if colour_set[n] == 'abs':
            vmin = vmin_abs
            vmax = vmax_abs
            cmap = cmap_abs
        elif colour_set[n] == 'anom':
            vmin = vmin_anom
            vmax = vmax_anom
            cmap = cmap_anom
        ax = plt.subplot(gs[n//2, n%2])
        shade_background(ax)
        img = ax.pcolormesh(h_axis[n], z_axis[n], data_plot[n], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(titles[n], fontsize=16)
        ax.set_xlim([hmin, None])
        ax.set_ylim([zmin, 0])
        ax.tick_params(direction='in')
        if n==0:
            slice_axes(ax, h_axis='lat')
            ax.set_ylabel('depth (m)', fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if n==1 or n==5:
            cax = fig.add_axes([0.92, 0.655-0.295*(n//2), 0.02, 0.2])
            plt.colorbar(img, cax=cax)
    plt.suptitle(main_title_physical_space, fontsize=18)
    finished_plot(fig, fig_name=fig_name_physical_space, dpi=300)

    # Water mass figure in normalised T/S space
    fig = plt.figure(figsize=(7,8))
    gs = plt.GridSpec(3,2)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.84, hspace=0.4, wspace=0.1)
    num_bins = bin_edges.size-1
    log_volume = np.ma.empty([2, num_bins, num_bins])
    log_volume[0,:] = np.log(cesm_volume_perbin)
    log_volume[1,:] = np.log(woa_volume_perbin)
    data_plot = [log_volume, cesm_anom_ts, cesm_anom_ts_filled]
    labels = [[r'$\bf{a}$. CESM1', r'$\bf{b}$. WOA'], [r'$\bf{c}$. Temperature', r'$\bf{d}$. Salinity'], [r'$\bf{e}$. Temperature', r'$\bf{f}$. Salinity']]
    titles = ['Volume of each class at climatology', 'CESM1 anomalies from climatology', 'Filled surface to look up anomalies']
    vmin_temp, vmax_temp, cmap_temp = set_colour_bounds([cesm_anom_ts, cesm_anom_ts_filled], 0, 'plusminus')
    vmin_salt, vmax_salt, cmap_salt = set_colour_bounds([cesm_anom_ts, cesm_anom_ts_filled], 1, 'plusminus')
    make_cbar = [[True,  True], [True, True], [False, False]]
    cbar_label = [[r'log(m$^3$)', r'log(m$^3$)'], [deg_string+'C', 'psu'], [None, None]]
    for n in range(3):     
        for v in range(2):
            if n==0:
                cmap, vmin, vmax = set_colours(data_plot[n][v,:], ctype='parula')
            elif v==0:
                vmin = vmin_temp
                vmax = vmax_temp
                cmap = cmap_temp
            elif v==1:
                vmin = vmin_salt
                vmax = vmax_salt
                cmap = cmap_salt
            ax = plt.subplot(gs[n,v])
            img = ax.pcolormesh(bin_edges, bin_edges, data_plot[n][v,:], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(labels[n][v], fontsize=13)
            ax.tick_params(direction='in')
            if n==2 and v==0:
                ax.set_xlabel('Normalised salinity')
                ax.set_ylabel('Normalised temperature')
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            if make_cbar[n][v]:
                cax = fig.add_axes([0.02+0.89*v, 0.645-0.29*n, 0.02, 0.18])
                plt.colorbar(img, cax=cax)
                plt.text(0.02+0.89*v, 0.825-0.29*n, cbar_label[n][v], fontsize=10, ha='left', va='bottom', transform=fig.transFigure)
        plt.text(0.5, 0.885-0.2925*n, titles[n], fontsize=16, transform=fig.transFigure, ha='center', va='center')
    plt.suptitle(main_title_ts_space, fontsize=18)
    finished_plot(fig, fig_name=fig_name_ts_space, dpi=300)


# Quick plot for reviewers
def shelf_bathy_histogram (grid_path='PAS_grid/', fig_name=None):

    num_bins = 50
    eps = 0.1
    grid = Grid(real_dir(grid_path))
    mask = grid.get_region_mask('amundsen_shelf')
    bathy = np.abs(np.ma.masked_where(np.invert(mask), grid.bathy))
    bin_edges = np.linspace(np.amin(bathy)-eps, np.amax(bathy)+eps, num=num_bins+1)
    bin_centres = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    area = np.zeros(num_bins)

    for bathy_val, dA_val in zip(bathy[mask], grid.dA[mask]):
        bin_index = np.nonzero(bin_edges > bathy_val)[0][0]-1
        area[bin_index] += dA_val*1e-9

    fig, ax = plt.subplots()
    ax.bar(bin_centres, area, width=bin_width)
    ax.grid(linestyle='dotted')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel(r'Area of shelf (10$^3$ km$^2$)')
    plt.title('Bathymetry on Amundsen Sea continental shelf')
    finished_plot(fig, fig_name=fig_name)


# Quick plot for reviewer
def coastline_compare_mit_forcing (grid_path='PAS_grid/', fig_name=None, option='CESM'):

    grid = Grid('PAS_grid/')
    mit_mask = (grid.land_mask + grid.ice_mask).astype(float)
    cesm_file = find_cesm_file('LENS', 'TEMP', 'oce', 'monthly', 1, 2006)[0]
    cesm_mask = (read_netcdf(cesm_file, 'TEMP', time_index=0)[0,:].mask).astype(float)
    cesm_lon, cesm_lat = read_pop_grid(cesm_file)[:2]
    era5_file = '/data/oceans_input/raw_input_data/ERA5_monthly/era5_monthly_sst_1979_2021.nc'
    era5_mask = (read_netcdf(era5_file, 'sst', time_index=0).mask).astype(float)
    era5_lon = fix_lon_range(read_netcdf(era5_file, 'longitude'))
    era5_lat = read_netcdf(era5_file, 'latitude')
    era5_lon, era5_lat = np.meshgrid(era5_lon, era5_lat)

    fig, ax = plt.subplots(figsize=(8,4))
    # Plot land points in red and ocean points in blue
    if option == 'CESM':
        ax.plot(cesm_lon[cesm_mask==1], cesm_lat[cesm_mask==1], 'o', color='red', markersize=2, label='CESM land points')
        ax.plot(cesm_lon[cesm_mask==0], cesm_lat[cesm_mask==0], 'o', color='blue', markersize=2, label='CESM ocean points')
    elif option == 'ERA5':
        ax.plot(era5_lon[era5_mask==1], era5_lat[era5_mask==1], 'o', color='red', markersize=2, label='ERA5 land points')
        ax.plot(era5_lon[era5_mask==0], era5_lat[era5_mask==0], 'o', color='blue', markersize=2, label='ERA5 ocean points')
    # Contour MITgcm coastline in black
    cs = ax.contour(grid.lon_2d, grid.lat_2d, mit_mask, levels=[0.5], colors=('black'), linestyles='solid')
    cs.collections[0].set_label('MITgcm coastline')
    latlon_axes(ax, grid.lon_1d, grid.lat_1d, ymax=-71)
    plt.legend(loc='upper left', framealpha=1)
    plt.tight_layout()
    finished_plot(fig, fig_name=fig_name)


# Quick plot for reviewer
def convection_map (base_dir='./', grid_dir='PAS_grid/', fig_name=None):

    file_head = real_dir(base_dir) + 'PAS_LENS'
    num_ens = 10
    start_year = 2006 #1920
    end_year = 2100 #2005
    num_years = end_year-start_year+1
    file_mid = '_O/output/'
    file_tail = '01/MITgcm/output.nc'
    grid = Grid(grid_dir)
    bwtemp = np.ma.empty([num_years*num_ens, grid.ny, grid.nx])
    temp0 = 0.5

    # Get annually averaged bottom water temperature for each year and ensemble member
    for n in range(num_ens):
        for year in range(start_year, end_year+1):
            print ('Reading '+str(year)+' from member '+str(n+1))
            file_path = file_head + str(n+1).zfill(3) + file_mid + str(year) + file_tail
            temp = read_netcdf(file_path, 'THETA')
            temp = average_12_months(temp, calendar='noleap')
            temp = mask_3d(temp, grid)
            bwtemp[n*num_years+(year-start_year),:] = select_bottom(temp)
    # Standard deviation of bottom water temperature
    #data_plot = np.std(bwtemp, axis=0)
    # Percentage of years below threshold value
    data_plot = np.sum(bwtemp < temp0, axis=0)/np.size(bwtemp, 0)*1e2
    data_plot = mask_land_ice(data_plot, grid)
    data_plot = np.ma.masked_where(grid.bathy > -500, data_plot)
    data_plot = np.ma.masked_where(grid.bathy < -1750, data_plot)
    
    fig, ax = latlon_plot(data_plot, grid, ymax=-71, figsize=(9,4), return_fig=True, vmax=100)    
    #plt.title('Standard deviation of bottom water temperature\nin historical ensemble ('+deg_string+'C)', fontsize=15)
    plt.title('% years with bottom water temperature below '+str(temp0)+deg_string+'C (RCP 8.5)', fontsize=15)
    plt.tight_layout()
    finished_plot(fig, fig_name=fig_name)


# Calculate change in RCP 8.5 mass loss using conventions of Jourdain 2022
def compare_jourdain_massloss (base_dir='./'):

    file_head = real_dir(base_dir) + 'PAS_LENS'
    num_ens = 10
    file_tail = '_O/output/timeseries.nc'
    var_name = 'all_massloss'
    period1 = [1989, 2009]
    period2 = [2080, 2100]

    # Calculate time-mean mass loss over the given time period
    def calc_mean_massloss (period):
        massloss_annual_means = np.zeros([num_ens, period[1]-period[0]+1])
        for n in range(num_ens):
            file_path = file_head + str(n+1).zfill(3) + file_tail
            time = netcdf_time(file_path, monthly=False)
            massloss = read_netcdf(file_path, var_name)
            for year in range(period[0], period[1]+1):
                t_start, t_end = index_period(time, year, year)
                massloss_annual_means[n, year-period[0]] = average_12_months(massloss[t_start:t_end], calendar='noleap')
        return np.mean(massloss_annual_means, axis=1)

    massloss_present = calc_mean_massloss(period1)
    massloss_future = calc_mean_massloss(period2)
    massloss_percent = (massloss_future-massloss_present)/massloss_present*1e2
    print('Mass loss increases by '+str(np.amin(massloss_percent))+' to '+str(np.amax(massloss_percent))+' depending on ensemble member, mean '+str(np.mean(massloss_percent)))


# Supplementary plot
def bathy_draft_map (grid_dir='PAS_grid/', fig_name=None):

    grid = Grid(grid_dir)
    bathy = np.abs(mask_land(grid.bathy, grid))
    draft = np.abs(mask_except_ice(grid.draft, grid))
    data_plot = [bathy, draft]
    titles = [r'$\bf{a}$. Bathymetry (m)', r'$\bf{b}$. Ice shelf draft (m)']
    vmax = [1500, 1000]
    ymax = [-69, -71.5]
    suptitle = 'Topography of Amundsen Sea continental shelf'
    ctype = 'plusminus_r'

    fig = plt.figure(figsize=(7,6))
    gs = plt.GridSpec(5,1)
    gs.update(left=0.07, right=0.88, bottom=0.05, top=0.88, hspace=1)
    handles = [gs[:3,0], gs[3:,0]]
    cbar_bottom = [0.45, 0.06]
    cbar_height = [0.4, 0.25]
    for n in range(2):
        ax = plt.subplot(handles[n])
        img = latlon_plot(data_plot[n], grid, ax=ax, ctype=ctype, ymax=ymax[n], title=titles[n], titlesize=14, make_cbar=False, vmax=vmax[n])            
        cax = fig.add_axes([0.9, cbar_bottom[n], 0.02, cbar_height[n]])
        plt.colorbar(img, cax=cax, extend='max')
    plt.suptitle(suptitle, fontsize=16)
    finished_plot(fig, fig_name=fig_name, dpi=300)

    
# Quick analysis/plot for reviewer
def test_member_trend_correlation (base_dir='./', fig_name=None):

    var_name = 'amundsen_shelf_temp_btw_200_700m'
    var_title = 'continental shelf temperature'
    var_units = deg_string+'C'
    expt_names = ['LENS', 'MENS', 'LW2.0', 'LW1.5']
    expt_mid = ['', '_', '_', '_']
    expt_titles = ['RCP 8.5', 'RCP 4.5', 'Paris 2C', 'Paris 1.5C']
    num_ens = [10, 10, 10, 5]
    start_year = 2006
    end_years = [2100, 2080, 2100, 2100]
    ens_colours = choose_n_colours(10, base_cmap='tab10')
    timeseries_file = 'timeseries.nc'
    num_expt = len(expt_names)
    smooth = 24
    p0 = 0.05
    base_dir = real_dir(base_dir)

    # Calculate all the trends for a given ensemble over the given timespan
    def get_ensemble_trends (n, ens_size, end_year):
        trends = np.empty([ens_size])
        for e in range(ens_size):
            file_path = base_dir + 'PAS_' + expt_names[n] + expt_mid[n] + str(e+1).zfill(3) + '_O/output/' + timeseries_file
            trends[e] = read_calc_trend(var_name, file_path, start_year=start_year, end_year=end_year, smooth=smooth)[0]
        return trends        

    fig = plt.figure(figsize=(9,6))
    gs = plt.GridSpec(2,3)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.87, wspace=0.2, hspace=0.3)
    posn = 0
    for n1 in range(num_expt):
        for n2 in range(n1+1, num_expt):
            # Choose the correct time range and ensemble size to calculate the trend over
            end_year = min(end_years[n1], end_years[n2])
            ens_size = min(num_ens[n1], num_ens[n2])
            trends1 = get_ensemble_trends(n1, ens_size, end_year)
            trends2 = get_ensemble_trends(n2, ens_size, end_year)
            ax = plt.subplot(gs[posn//3, posn%3])
            ax.grid(linestyle='dotted')
            for e in range(ens_size):
                ax.plot(trends1[e], trends2[e], 'o', color=ens_colours[e])            
            slope, intercept, r_value, p_value, std_err = linregress(trends1, trends2)
            if p_value < p0:
                [x0, x1] = ax.get_xlim()
                [y0, y1] = slope*np.array([x0, x1]) + intercept
                ax.plot([x0, x1], [y0, y1], '-', color='black', linewidth=1, zorder=0)
            ax.text(0.02, 0.98, 'p='+str(round_to_decimals(p_value, 3))+', r$^2$='+str(round_to_decimals(r_value**2, 3)), ha='left', va='top', fontsize=10, transform=ax.transAxes)
            ax.set_title(expt_titles[n1]+' vs '+expt_titles[n2])
            posn +=1
    plt.suptitle('Correlation in trends of '+var_title+' ('+var_units+'/century)', fontsize=16)
    finished_plot(fig, fig_name=fig_name)


def trends_ex_convection ():

    var_name_temp = 'amundsen_shelf_temp_btw_200_700m'
    timeseries_file_temp = 'timeseries.nc'
    var_name_iso = 'amundsen_shelf_isotherm_0.5C_below_100m'
    timeseries_file_iso = 'timeseries_isotherm.nc'
    z0 = -440
    expt_names = ['Historical', 'Historical\nFixed BCs', 'Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5', 'RCP 8.5\nFixed BCs']
    num_expt = len(expt_names)
    expt_dir_heads = ['PAS_']*num_expt
    expt_dir_mids = ['LENS', 'LENS', 'LW1.5_', 'LW2.0_', 'MENS_', 'LENS', 'LENS']
    expt_dir_tails = ['_O', '_noOBC', '_O', '_O', '_O', '_O', '_noOBC']
    num_ens = [10, 5, 5, 10, 10, 10, 5]
    start_years = [1920, 1920, 2006, 2006, 2006, 2006, 2006]
    end_years = [2005, 2005, 2100, 2100, 2080, 2100, 2100]
    smooth = 24
    p0 = 0.05

    for n in range(num_expt):
        trends = np.zeros(num_ens[n])
        trends_noconv = np.zeros(num_ens[n])
        conv_t = 0
        total_t = 0
        for e in range(num_ens[n]):
            output_dir = expt_dir_heads[n] + expt_dir_mids[n] + str(e+1).zfill(3) + expt_dir_tails[n] + '/output/'
            file_path_temp = output_dir + timeseries_file_temp
            file_path_iso = output_dir + timeseries_file_iso
            time = netcdf_time(file_path_temp, monthly=False)            
            temp = read_netcdf(file_path_temp, var_name_temp)
            iso_depth = read_netcdf(file_path_iso, var_name_iso)
            if iso_depth.size != temp.size:
                print('Error (trends_ex_convection): isotherm timeseries does not match temperature timeseries')
                sys.exit()
            t0, tf = index_period(time, start_years[n], end_years[n])
            time = time[t0:tf]
            temp = temp[t0:tf]
            iso_depth = iso_depth[t0:tf]
            time_sec = np.array([(t-time[0]).total_seconds() for t in time])
            time = time_sec/(365*sec_per_day*100)
            temp_smooth, time_smooth = moving_average(temp, smooth, time=time)
            iso_depth_smooth = moving_average(iso_depth, smooth)
            # Calculate baseline trend
            trends[e] = linregress(time_smooth, temp_smooth)[0]
            # Calculate trend with convective periods excluded
            index = iso_depth_smooth >= z0  # Indices which are not convecting
            trends_noconv[e] = linregress(time_smooth[index], temp_smooth[index])[0]
            # Integrate number of time indices convecting
            conv_t += np.sum(np.invert(index).astype(float))
            total_t += iso_depth_smooth.size
        mean_trend = np.mean(trends)
        p_val = ttest_1samp(trends, 0)[1]
        mean_trend_noconv = np.mean(trends_noconv)
        p_val_noconv = ttest_1samp(trends_noconv, 0)[1]
        percent_change = (mean_trend_noconv - mean_trend)/mean_trend*1e2
        percent_conv = conv_t/total_t*1e2
        print('\n'+expt_names[n])
        print('Convects '+str(percent_conv)+'% of the time')
        if p_val < p0:
            print('Baseline trend '+str(mean_trend))
            if p_val_noconv < p0:
                print('Trend without convection '+str(mean_trend_noconv))
                print('Changes trend by '+str(percent_change)+'%')
            else:
                print('Without convection, trend is no longer significant')
        else:
            print('No significant trend')
    

# Precompute volumes in T/S space (over bins) for the given experiment
def precompute_ts_volumes (output_dir, fname_out='TS_volume.nc', region='amundsen_shelf', start_year=2006, end_year=2100, tmin=-1.95, tmax=2.1, smin=33, smax=34.75, num_bins=1000, grid_path='PAS_grid/'):

    import netCDF4 as nc

    grid = Grid(grid_path)
    mask = grid.get_region_mask(region, is_3d=True)
    num_years = end_year-start_year+1
    volume = np.zeros([num_years, num_bins, num_bins])

    def read_bound_data (file_path, var_name, vmin, vmax):
        data = read_netcdf(file_path, var_name, time_average=True)
        data = apply_mask(data, np.invert(mask), depth_dependent=True)
        if np.amin(data) < vmin:
            print('Warning: minimum is '+str(np.amin(data))+'; '+str(np.count_nonzero(data < vmin))+' instances below previous min')
            data[data < vmin] = vmin
        if np.amax(data) > vmax:
            print('Warning: maximum is '+str(np.amax(data))+'; '+str(np.count_nonzero(data > vmax))+' instances above previous max')
            data[data > vmax] = vmax
        return data
    
    for year in range(start_year, end_year+1):
        print('Processing '+str(year))
        file_path = output_dir + str(year) + '01/MITgcm/output.nc'
        temp = read_bound_data(file_path, 'THETA', tmin, tmax)
        salt = read_bound_data(file_path, 'SALT', smin, smax)
        volume_tmp, temp_centres, salt_centres, temp_edges, salt_edges = ts_binning(temp, salt, grid, mask, num_bins=num_bins, tmin=tmin, tmax=tmax, smin=smin, smax=smax)
        volume_tmp[volume_tmp.mask] = 0
        volume[year-start_year,:] = volume_tmp

    print('Writing to file')
    id = nc.Dataset(output_dir+fname_out, 'w')
    id.createDimension('time', None)
    id.createVariable('time', 'f8', ('time'))
    id.variables['time'].units = 'year'
    id.variables['time'][:] = np.arange(start_year, end_year+1)
    def add_dimension (var_name, data, units):
        id.createDimension(var_name, data.size)
        id.createVariable(var_name, 'f8', (var_name))
        id.variables[var_name].units = units
        id.variables[var_name][:] = data
    add_dimension('temp_centres', temp_centres, 'degC')
    add_dimension('temp_edges', temp_edges, 'degC')
    add_dimension('salt_centres', salt_centres, 'psu')
    add_dimension('salt_edges', salt_edges, 'psu')
    id.createVariable('volume', 'f8', ('time', 'temp_centres', 'salt_centres'))
    id.variables['volume'][:] = volume
    id.close()
                      

# Supplementary figure
def volume_ts_space (fig_name=None):

    import matplotlib.colors as cl

    file_head = 'PAS_LW2.0_'
    num_ens = 10
    file_tail = '_O/output/TS_volume.nc'
    p0 = 0.05
    num_years = 2100-2006+1
    salt_bounds = [33, 34.8]
    temp_bounds = [-1.95, 2.2]
    wm_labels = ['WW', 'CDW']
    wm_x = [34.25, 34.5]
    wm_y = [-1.25, 1.5]

    for n in range(num_ens):
        print('Processing ensemble member '+str(n+1))
        file_path = file_head + str(n+1).zfill(3) + file_tail
        if n==0:
            temp_centres = read_netcdf(file_path, 'temp_centres')
            salt_centres = read_netcdf(file_path, 'salt_centres')
            num_bins = temp_centres.size
            volume_all = np.zeros([num_years, num_ens, num_bins, num_bins])
            data_plot = np.zeros([2, num_bins, num_bins])
        volume_all[:,n,:] = read_netcdf(file_path, 'volume')
    # Get ensemble mean over first and last 10 years
    data_plot[0,:] = np.mean(volume_all[:10,:,:,:], axis=(0,1))
    data_plot[1,:] = np.mean(volume_all[-10:,:,:,:], axis=(0,1))
    # Get on a log scale
    data_plot_log = np.log(data_plot)
    data_plot_log[data_plot==0] = 0
    # Mask bins with zero volume
    data_plot_log = np.ma.masked_where(data_plot_log==0, data_plot_log)
    # Calculate potential density to overlay contours
    salt_2d, temp_2d = np.meshgrid(np.linspace(salt_bounds[0], salt_bounds[1]), np.linspace(temp_bounds[0], temp_bounds[1]))
    density = potential_density('MDJWF', salt_2d, temp_2d)-1e3

    def fmt(x):
        s = f"{x:.2f}"
        if s.endswith("0"):
            s = f"{x:.1f}"
        return f"{s}"

    # Plot
    fig = plt.figure(figsize=(8,5))
    gs = plt.GridSpec(1,2)
    gs.update(left=0.1, right=0.88, bottom=0.1, top=0.87, wspace=0.01)
    cax = fig.add_axes([0.9, 0.18, 0.03, 0.6])
    cmap, vmin, vmax = set_colours(data_plot_log, ctype='parula')
    titles = [r'$\bf{a}$. 2006-2015', r'$\bf{b}$. 2091-2100']
    for n in range(2):
        ax = plt.subplot(gs[0,n])
        cs = ax.contour(salt_2d, temp_2d, density, 12, colors='black', linestyles='dotted', linewidths=1)
        if n==1:
            ax.clabel(cs, cs.levels, inline=True, fontsize=8, fmt=fmt)
        img = ax.pcolor(salt_centres, temp_centres, data_plot_log[n,:], cmap=cmap, vmin=vmin, vmax=vmax)
        # Shade in grey the bins that exist in the other panel but not this one
        missing_bins = ((data_plot[1-n,:] != 0)*(data_plot[n,:] == 0)).astype(float)
        missing_bins = np.ma.masked_where(missing_bins==0, missing_bins)
        ax.pcolor(salt_centres, temp_centres, missing_bins, cmap=cl.ListedColormap(['DarkGrey']))
        ax.tick_params(direction='in')
        ax.set_xlim(salt_bounds)
        ax.set_ylim(temp_bounds)
        if n == 0:            
            ax.set_xlabel('Salinity (psu)', fontsize=12)
            ax.set_ylabel('Temperature ('+deg_string+'C)', fontsize=12)
            for i in range(len(wm_labels)):
                plt.text(wm_x[i], wm_y[i], wm_labels[i], ha='center', va='center', fontsize=12, weight='bold', color='black')            
        elif n==1:
            ax.set_yticklabels([])
            plt.colorbar(img, cax=cax)
            plt.text(.98, .6, r'log of volume (m$^3$)', ha='center', rotation=-90, transform=fig.transFigure)
        ax.set_title(titles[n], fontsize=14)
    plt.suptitle('Water masses on continental shelf (Paris 2'+deg_string+'C)', fontsize=16)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Plot annual means of undercurrent strength vs continental shelf temp, and a correlation.
def undercurrent_temp_scatterplot (base_dir='./', fig_name=None):

    var_names = ['amundsen_shelf_temp_btw_200_700m', 'PITE_trans']
    expt_dir_head = base_dir+'PAS_'
    num_ens = [5, 10, 10, 10]
    expt_dir_mids = ['LW1.5_', 'LW2.0_', 'MENS_', 'LENS']
    expt_dir_tail = '_O/output/'
    timeseries_file = 'timeseries.nc'
    start_year = 2006
    end_years = [2100, 2100, 2080, 2100]
    expt_titles = ['Paris 1.5'+deg_string+'C', 'Paris 2'+deg_string+'C', 'RCP 4.5', 'RCP 8.5']
    expt_colours = [(0,0.45,0.7), (0,0.62,0.45), (0.9,0.62,0), (0.8,0.47,0.65)]
    num_expt = len(expt_titles)
    ndays = [days_per_month(t+1, 1979) for t in range(months_per_year)]
    p0 = 0.05

    all_temp = []
    all_trans = []
    fig, ax = plt.subplots()
    for n in range(num_expt):
        for e in range(num_ens[n]):
            file_path = expt_dir_head + expt_dir_mids[n] + str(e+1).zfill(3) + expt_dir_tail + timeseries_file
            time = netcdf_time(file_path, monthly=False)
            t0, tf = index_period(time, start_year, end_years[n])
            time = time[t0:tf]
            temp = read_netcdf(file_path, var_names[0])[t0:tf]
            trans = read_netcdf(file_path, var_names[1])[t0:tf]
            # Calculate annual means
            num_years = end_years[n] - start_year + 1
            for t in range(num_years):
                temp0 = np.ma.average(temp[t*months_per_year:(t+1)*months_per_year], weights=ndays)
                trans0 = np.ma.average(trans[t*months_per_year:(t+1)*months_per_year], weights=ndays)
                # Plot each point as we go
                ax.plot(trans0, temp0, 'o', color=expt_colours[n], markersize=3, label=expt_titles[n] if e==0 and t==0 else None)
                # Save to master array for correlating later
                all_temp.append(temp0)
                all_trans.append(trans0)
    ax.grid(linestyle='dotted')
    slope, intercept, r_value, p_value, std_err = linregress(np.array(all_trans), np.array(all_temp))
    if p_value < p0:
        trend_title = 'r$^2$='+str(round_to_decimals(r_value**2, 3))
        ax.text(0.05, 0.95, trend_title, ha='left', va='top', fontsize=11, transform=ax.transAxes)
        [x0, x1] = ax.get_xlim()
        [y0, y1] = slope*np.array([x0, x1]) + intercept
        ax.plot([x0, x1], [y0, y1], '-', color='black', linewidth=1)
        ax.set_xlim([x0, x1])
    ax.set_ylabel('Temperature on continental shelf, 200-700m ('+deg_string+'C)', fontsize=12)
    ax.set_xlabel('Transport through PITE Trough (Sv)', fontsize=12)
    ax.set_title('Temperature on shelf vs. undercurrent strength', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    finished_plot(fig, fig_name=fig_name, dpi=300)
    
    
    
    
                    
                    
                    
                
                

    

            
        
                
            

    
        

    
           
    
        
    
    
    

    

    
    

    
        
            
            

    
        
            
                    

    
        
        
            
            
        

    
    
    

    
    
    
    
    
    

    

    

    

    
    
    
            
            
        
    
    
            
            
            
        
        
    
    

    

    

    

    
    
    

            
        

    
    

    
