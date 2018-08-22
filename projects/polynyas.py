##################################################################
# Weddell Sea polynya project
##################################################################

from ..grid import Grid
from ..plot_1d import read_plot_timeseries, read_plot_timeseries_diff
from ..plot_latlon import read_plot_latlon, read_plot_latlon_diff
from ..plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff
from ..postprocess import build_file_list, select_common_time, precompute_timeseries
from ..utils import real_dir
from ..plot_utils.labels import parse_date


# Get longitude and latitude at the centre of the polynya
def get_polynya_loc (polynya):
    
    if polynya == 'maud_rise':
        lon0 = 0
        lat0 = -65
    elif polynya == 'near_shelf':
        lon0 = -30
        lat0 = -70
    else:
        print 'Error (get_polynya_loc): please specify a valid polynya.'
        sys.exit()
    return lon0, lat0


# Precompute timeseries for temperature and salinity, depth-averaged in the centre of the given polynya.
def precompute_polynya_timeseries (mit_file, timeseries_file, polynya=None):

    lon0, lat0 = get_polynya_loc(polynya)
    precompute_timeseries(mit_file, timeseries_file, polynya=True, lon0=lon0, lat0=lat0)

    

# A whole bunch of basic preliminary plots to analyse things.
# First must run precompute_polynya_timeseries.
def prelim_plots (polynya_dir='./', baseline_dir=None, polynya=None, timeseries_file=None, grid_path='../grid/', fig_dir='./', option='last_year', unravelled=False):

    if baseline_dir is None:
        print 'Error (prelim_plots): must specify baseline_dir.'
        sys.exit()

    # Make sure proper directories
    polynya_dir = real_dir(polynya_dir)
    baseline_dir = real_dir(baseline_dir)
    fig_dir = real_dir(fig_dir)

    lon0, lat0 = get_polynya_loc(polynya)
    if timeseries_file is None:
        timeseries_file = 'timeseries_polynya_'+polynya+'.nc'

    # Build the grid
    grid = Grid(grid_path)

    # Build the list of output files in each directory
    output_files = build_file_list(polynya_dir, unravelled=unravelled)
    baseline_files = build_file_list(baseline_dir, unravelled=unravelled)
    # Select files and time indices etc. corresponding to last common period of simulation
    file_path, file_path_baseline, time_index, time_index_baseline, t_start, t_start_baseline, t_end, t_end_baseline, time_average = select_common_time(output_files, baseline_files, option=option)
    # Set date string
    if option == 'last_year':
        date_string = 'year beginning ' + parse_date(file_path=file_path, time_index=t_start)
    elif option == 'last_month':
        date_string = parse_date(file_path=file_path, time_index=time_index)

    # Timeseries of depth-averaged temperature and salinity through the centre of the polynya, as well as FRIS basal mass balance
    var_names = ['temp_polynya', 'salt_polynya', 'fris_melt']
    for var in var_names:
        read_plot_timeseries(var, polynya_dir+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_'+var+'.png')
        # Repeat for anomalies from baseline
        read_plot_timeseries_diff(var, baseline_dir+timeseries_file, polynya_dir+timeseries_file, precomputed=True, fig_name=fig_dir+'timeseries_'+var+'_diff.png')

    # Lat-lon plots over the last year/month
    var_names = ['aice', 'bwtemp', 'bwsalt', 'vel', 'ismr']
    for var in var_names:
        # Want to zoom both in and out
        for zoom_fris in [False, True]:
            # Set figure size
            if zoom_fris:
                figsize = (8,6)
            else:
                figsize = (10,6)
            # Get zooming in the figure name
            zoom_key = ''
            if zoom_fris:
                zoom_key = '_zoom'
            # Don't need a zoomed-in sea ice plot
            if var == 'aice' and zoom_fris:
                continue
            # Set variable bounds
            vmin = None
            vmax = None
            if var == 'bwsalt':
                vmin = 34.2
                vmax = 34.8
            elif var == 'bwtemp' and zoom_fris=True:
                vmax = -1
            # Now make the plot
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, zoom_fris=zoom_fris, vmin=vmin, vmax=vmax, date_string=date_string, fig_name=fig_dir+var+zoom_key+'.png', figsize=figsize)
            # Repeat for anomalies from baseline
            read_plot_latlon_diff(var, file_path_baseline, file_path, grid=grid, time_index=time_index_baseline, t_start=t_start_baseline, t_end=t_end_baseline, time_average=time_average, time_index_2=time_index, t_start_2=t_start, t_end_2=t_end, zoom_fris=zoom_fris, date_string=date_string, fig_name=fig_dir+var+zoom_key+'_diff.png', figsize=figsize)

    # Meridional slices through centre of polynya
    # Full water column as well as upper 1000 m
    for zmin in [None, -1000]:
        zoom_key = ''
        if zmin is not None:
            zoom_key = '_zoom'
        read_plot_ts_slice(file_path, grid=grid, lon0=lon0, zmin=zmin, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, date_string=date_string, fig_name='ts_slice_polynya'+zoom_key+'.png')
        # Repeat for anomalies from baseline
        read_plot_ts_slice_diff(file_path_baseline, file_path, grid=grid, lon0=lon0, zmin=zmin, time_index=time_index_baseline, t_start=t_start_baseline, t_end=t_end_baseline, time_average=time_average, time_index_2=time_index, t_start_2=t_start, t_end_2=t_end, date_string=date_string, fig_name='ts_slice_polynya'+zoom_key+'_diff.png')
