##################################################################
# JSPS Amundsen Sea simulations forced with LENS
##################################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..plot_1d import read_plot_timeseries_ensemble
from ..utils import real_dir, fix_lon_range
from ..grid import Grid
from ..ics_obcs import find_obcs_boundary
from ..file_io import read_netcdf, read_binary
from ..constants import deg_string
from ..plot_utils.windows import set_panels, finished_plot


# Plot a bunch of precomputed timeseries from ongoing LENS-forced test simulations (ensemble of 5 to start), compared to the PACE-forced ensemble mean.
def check_lens_timeseries (num_ens=5, base_dir='./'):

    var_names = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m']
    base_dir = real_dir(base_dir)
    pace_file = base_dir+'timeseries_pace_mean.nc'
    file_paths = ['PAS_LENS'+str(n+1).zfill(3)+'/output/timeseries.nc' for n in range(num_ens)] + [pace_file]
    sim_names = ['LENS ensemble'] + [None for n in range(num_ens-1)] + ['PACE mean']
    colours = ['DarkGrey' for n in range(num_ens)] + ['blue']
    smooth = 24
    start_year = 1920

    for var in var_names:
        read_plot_timeseries_ensemble(var, file_paths, sim_names=sim_names, precomputed=True, colours=colours, smooth=smooth, vline=start_year, time_use=None)
        

# Compare time-averaged temperature and salinity boundary conditions from PACE over 2006-2013 (equivalent to the first 20 ensemble members of LENS actually!) to the WOA boundary conditions used for the original simulations.
def compare_bcs_ts_mean ():

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
    
    # Build the grids
    mit_grid = Grid(grid_dir)
    pace_grid_file = pace_dir + pace_var[0] + pace_file_head + '01' + pace_file_mid + pace_var[0] + pace_file_tail
    pace_lon = fix_lon_range(read_netcdf(pace_grid_file, 'TLONG'))
    pace_lat = read_netcdf(pace_grid_file, 'TLAT')
    pace_z = read_netcdf(pace_grid_file, 'z_t')
    pace_nz = pace_z.size
    pace_ny = pace_lat.size[0]
    pace_nx = pace_lat.size[1]

    # Read the PACE output to create a time-mean, ensemble-mean (no seasonal cycle for now)
    pace_data_mean_3d = np.ma.zeros([num_var, pace_nz, pace_ny, pace_nx])
    for v in range(num_var):
        for n in range(num_ens):
            pace_file = pace_dir + pace_var[v] + pace_file_head + str(n+1).zfill(2) + pace_file_mid + pace_var[v] + pace_file_tail
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
            mit_h = grid.lon_1d
        elif bdry in ['E', 'W']:
            pace_n = pace_ny
            mit_h = grid.lat_1d
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
            # Apply land mask (where identically 0)
            obcs_data_mean = np.ma.masked_where(obcs_data_mean==0, obcs_data_mean)
            # Interpolate to the PACE grid
            pace_data_mean = np.ma.empty([pace_nz, pace_n])
            pace_h = np.ma.empty([pace_n])
            for j in range(pace_n):
                if bdry in ['N', 'S']:
                    pace_data_mean[:,j] = c1[j]*pace_data_mean_3d[:,int(i1[j]),:] + c2[j]*pace_data_mean_3d[:,int(i2[j]),:]
                    pace_h[j] = c1[j]*pace_lon[int(i1[j]),:] + c2[j]*pace_lon[int(i2[j]),:]
                elif bdry in ['E', 'W']:
                    pace_data_mean[:,j] = c1[j]*pace_data_mean_3d[:,:,int(i1[j])] + c2[j]*pace_data_mean_3d[:,:,int(i2[j])]
                    pace_h[j] = c1[j]*pace_lat[:,int(i1[j])] + c2[j]*pace_lat[:,int(i2[j])]
            # Trim
            if bdry in ['N', 'S']:
                # Get limits on longitude in MITgcm
                hmin = find_obcs_bdry(mit_grid, 'W')[0]
                hmax = find_obcs_bdry(mit_grid, 'E')[0]
            elif bdry in ['E', 'W']:
                hmin = find_obcs_bdry(mit_grid, 'S')[0]
                hmax = find_obcs_bdry(mit_grid, 'N')[0]
            j1 = np.nonzero(pace_h < hmin)[0][-1]
            j2 = np.nonzero(pace_h > hmax)[0][0]
            pace_data_mean = pace_data_mean[:,j1:j2+1]
            pace_h = pace_h[j1:j2+1]
            # Find bounds for colour scale
            vmin = min(np.amin(obcs_data_mean), np.amin(pace_data_mean))
            vmax = max(np.amax(obcs_data_mean), np.amax(pace_data_mean))
            # Plot
            fig, gs, cax = set_panels('1x2C1')
            # WOA OBCS
            ax = plt.subplot(gs[0,0])
            ax.pcolormesh(mit_h, grid.z, obcs_data_mean, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('WOA 2018', fontsize=14)
            # PACE
            ax = plt.subplot(gs[0,1])
            img = plt.subplot(pace_h, pace_z, pace_data_mean, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('LENS 20-member mean', fontsize=14)
            plt.colorbar(img, cax=cax, orientation='horizontal')
            plt.suptitle(var_title[v], fontsize=16)
            finished_plot(fig)
            
            
            
    
