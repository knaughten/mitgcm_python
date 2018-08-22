##################################################################
# Weddell Sea polynya project
##################################################################

from ..timeseries import calc_timeseries, calc_timeseries_diff
from ..plot_1d import make_timeseries_plot
from ..plot_latlon import read_plot_latlon, read_plot_latlon_diff
from ..plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff
from ..postprocess import build_file_list, select_common_time
from ..utils import real_dir
from ..constants import deg_string
from ..plot_utils.labels import lon_label, lat_label

# A whole bunch of basic preliminary plots to analyse things.
def prelim_plots (polynya_dir, baseline_dir, grid_path='../grid/', fig_dir='./', option='last_year', unravelled=False, polynya=None):

    if polynya == 'maud_rise':
        lon0 = 0
        lat0 = -65
    elif polynya == 'near_shelf':
        lon0 = -30
        lat0 = -70
    else:
        print 'Error (prelim_plots): please specify a valid polynya.'
        sys.exit()    
    point_string = lon_label(lon0, 0) + ',' + lat_label(lat0, 0)

    # Make sure proper directories
    polynya_dir = real_dir(polynya_dir)
    baseline_dir = real_dir(baseline_dir)
    fig_dir = real_dir(fig_dir)

    # Build the grid
    grid = Grid(grid_path)

    # Build the list of output files in each directory
    output_files = build_file_list(polynya_dir, unravelled=unravelled)
    baseline_files = build_file_list(baseline_dir, unravelled=unravelled)
    # Select files and time indices etc. corresponding to last common period of simulation
    file_path, file_path_baseline, time_index, time_index_baseline, t_start, t_start_baseline, t_end, t_end_baseline, time_average = select_common_time(output_files, baseline_files, option=option)
    # Set date string
    if option == 'last_year':
        date_string = 'year beginning ' + parse_date(file_path=file_path_1, time_index=t_start_1)
    elif option == 'last_month':
        date_string = parse_date(file_path=file_path_1, time_index=time_index_1)

    # Timeseries of depth-averaged temperature and salinity through the centre of the polynya
    var_names = ['THETA', 'SALT']
    long_names = ['temperature', 'salinity']
    short_names = ['temp', 'salt']
    units = [deg_string+'C', 'psu']
    for i in range(2):
        time, var = calc_timeseries(output_files, option='point_vavg', grid=grid, var_name=var_names[i], lon0=lon0, lat0=lat0)
        make_timeseries_plot(time, var, title='Depth-averaged '+long_names[i]+' at '+point_string, units=units[i], fig_name=fig_dir+'timeseries_polynya_'+short_names[i]+'.png')
        # Repeat for anomalies from baseline
        time, var_diff = calc_timeseries(baseline_files, output_files, option='point_vavg', grid=grid, var_name=var_names[i], lon0=lon0, lat0=lat0)
        make_timeseries_plot(time, var, title='Change in depth-averaged '+long_names[i]+' at '+point_string, units=units[i], fig_name=fig_dir+'timeseries_polynya_'+short_names[i]+'_diff.png')

    # Lat-lon plots over the last year/month
    var_names = ['bwtemp', 'bwsalt', 'vel', 'ismr']
    figsize = (10,6)
    for var in var_names:
        # Want to zoom both in and out
        for zoom_fris in [True, False]:
            zoom_key = ''
            if zoom_fris:
                zoom_key = '_zoom'
            read_plot_latlon(var, file_path, grid=grid, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, zoom_fris=zoom_fris, date_string=date_string, fig_name=fig_dir+var+zoom_key+'.png', figsize=figsize)
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
