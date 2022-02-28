##################################################################
# JSPS Amundsen Sea simulations forced with LENS
##################################################################

from ..plot_1d import read_plot_timeseries_ensemble
from ..utils import real_dir


def check_lens_timeseries (num_ens=5, base_dir='./'):

    var_names = ['amundsen_shelf_break_uwind_avg', 'all_massloss', 'amundsen_shelf_temp_btw_200_700m', 'amundsen_shelf_salt_btw_200_700m', 'amundsen_shelf_sst_avg', 'amundsen_shelf_sss_avg', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_isotherm_0.5C_below_100m']
    base_dir = real_dir(base_dir)
    pace_file = base_dir+'timeseries_pace_mean.nc'
    file_paths = ['PAS_LENS'+str(n+1).zfill(3)+'/output/timeseries.nc' for n in range(num_ens)] + [pace_file]
    sim_names = ['LENS ensemble'] + [None for n in range(num_ens-1)] + ['PACE mean']
    colours = ['DimGrey' for n in range(num_ens)] + ['blue']
    smooth = 24
    start_year = 1920

    for var in var_names:
        read_plot_timeseries_ensemble(var, file_paths, sim_names=sim_names, precomputed=True, colours=colours, smooth=smooth, vline=start_year, time_use=None)
        
