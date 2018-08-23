##################################################################
# Weddell Sea polynya project
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..file_io import read_netcdf
from ..plot_1d import read_plot_timeseries, read_plot_timeseries_diff
from ..plot_latlon import read_plot_latlon, read_plot_latlon_diff, latlon_plot
from ..plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff
from ..postprocess import build_file_list, select_common_time, precompute_timeseries
from ..utils import real_dir, mask_land_ice, mask_3d, mask_except_ice, select_bottom, convert_ismr, var_min_max
from ..constants import deg_string
from ..plot_utils.labels import parse_date
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.latlon import prepare_vel


# Get longitude and latitude at the centre of the polynya
def get_polynya_loc (polynya):
    
    if polynya == 'maud_rise':
        lon0 = 0
        lat0 = -65
    elif polynya == 'near_shelf':
        lon0 = -30
        lat0 = -70
    elif polynya == 'free':
        lon0 = -25
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
                vmin = 34.3
                vmax = 34.8
            elif var == 'bwtemp' and zoom_fris:
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


def combined_plots (base_dir='./', fig_dir='./'):

    # File paths
    grid_path = 'WSB_001/grid/'
    output_dir = ['WSB_001/output/', 'WSB_007/output/', 'WSB_002/output/', 'WSB_003/output/']
    expt_names = ['Baseline', 'Free polynya', 'Polynya at Maud Rise', 'Polynya near shelf']
    mit_file = 'output_001.nc'
    timeseries_files = ['timeseries.nc', 'timeseries_polynya_free.nc', 'timeseries_polynya_maud_rise.nc', 'timeseries_polynya_near_shelf.nc']
    restoring_file = 'sss_restoring.nc'

    # Smaller boundaries on surface plots (where ice shelves are ignored)
    xmin_sfc = -67
    ymin_sfc = -80

    # Make sure real directories
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_path)

    '''print 'Plotting restoring masks'
    # 3x1 plot of restoring masks in the simulations where they exist
    fig, gs, cax = set_panels('1x3C1')
    for i in [0, 2, 3]:
        # Read the restoring mask at the surface
        restoring = read_netcdf(base_dir+output_dir[i]+restoring_file, 'restoring_mask')[0,:]
        # Mask land and ice shelves
        restoring = mask_land_ice(restoring, grid)
        # Make plot
        ax = plt.subplot(gs[0,max(i-1,0)])
        img = latlon_plot(restoring, grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, xmin=xmin_sfc, ymin=ymin_sfc, title=expt_names[i])
        if i > 0:
            # Remove latitude labels
            ax.set_yticklabels([])
    # Colourbar
    plt.colorbar(img, cax=cax, orientation='horizontal')
    # Main title
    plt.suptitle('Restoring mask for sea surface salinity', fontsize=22)
    finished_plot(fig, fig_name=fig_dir+'restoring_mask.png')
        
    print 'Plotting aice'
    # 2x2 plot of sea ice
    fig, gs, cax = set_panels('2x2C1')
    for i in range(4):
        # Read and mask data
        aice = read_netcdf(base_dir+output_dir[i]+mit_file, 'SIarea', time_index=-1)
        aice = mask_land_ice(aice, grid)
        # Make plot
        ax = plt.subplot(gs[i/2,i%2])
        img = latlon_plot(aice, grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, xmin=xmin_sfc, ymin=ymin_sfc, title=expt_names[i])
        if i%2==1:
            # Remove latitude labels
            ax.set_yticklabels([])
        if i/2==0:
            # Remove longitude labels
            ax.set_xticklabels([])
    # Colourbar
    plt.colorbar(img, cax=cax, orientation='horizontal')
    # Main title
    plt.suptitle('Sea ice concentration (add date later)', fontsize=22)
    finished_plot(fig, fig_name=fig_dir+'aice.png')'''

    # 3x1 difference plots of polynya simulations minus baseline
    var_names = ['bwtemp', 'bwsalt', 'ismr', 'vel']
    titles = ['Bottom water temperature anomaly ('+deg_string+'C)', 'Bottom water salinity anomaly (psu)', 'Ice shelf melt rate anomaly (m/y)', 'Absolute barotropic velocity anomaly (m/s)']
    # Inner function to read variable from a file and process appropriately
    def read_and_process (var, file_path):
        if var == 'bwtemp':
            data = select_bottom(mask_3d(read_netcdf(file_path, 'THETA', time_index=-1), grid))
        elif var == 'bwsalt':
            data = select_bottom(mask_3d(read_netcdf(file_path, 'SALT', time_index=-1), grid))
        elif var == 'ismr':
            data = convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx', time_index=-1), grid))
        elif var == 'vel':
            u = mask_3d(read_netcdf(file_path, 'UVEL', time_index=-1), grid, gtype='u')
            v = mask_3d(read_netcdf(file_path, 'VVEL', time_index=-1), grid, gtype='v')
            data = prepare_vel(u, v, grid)[0]
        return data
    # Now make the plots, zoomed both in and out
    for zoom_fris in [False, True]:
        if zoom_fris:
            zoom_string = '_zoom'
        else:
            zoom_string = ''
        for j in range(len(var_names)):
            print 'Plotting ' + var_names[j] + zoom_string
            figsize = None
            if zoom_fris:
                figsize = (12, 5)
            fig, gs, cax = set_panels('1x3C1', figsize=figsize)
            # Read baseline data
            baseline = read_and_process(var_names[j], base_dir+output_dir[0]+mit_file)
            vmin = 0
            vmax = 0
            data = []
            for i in range(1,4):
                # Read data for this simulation and get the anomaly
                data.append(read_and_process(var_names[j], base_dir+output_dir[i]+mit_file) - baseline)
                # Get min and max values and update global min/max as needed
                vmin_tmp, vmax_tmp = var_min_max(data[i-1], grid, zoom_fris=zoom_fris)
                vmin = min(vmin, vmin_tmp)
                vmax = max(vmax, vmax_tmp)
            # Now we can plot
            for i in range(1,4):
                ax = plt.subplot(gs[0,i-1])
                img = latlon_plot(data[i-1], grid, ax=ax, make_cbar=False, ctype='plusminus', zoom_fris=zoom_fris, vmin=vmin, vmax=vmax, title=expt_names[i])
                if i > 0:
                    # Remove latitude labels
                    ax.set_yticklabels([])
            # Colourbar
            plt.colorbar(img, cax=cax, orientation='horizontal')
            # Main title
            plt.suptitle(titles[j]+' (add date later)', fontsize=22)
            finished_plot(fig) #, fig_name=fig_dir+var_names[j]+zoom_string+'_diff.png')
            
            
            
        


    # Zoom difference plots in
    # 2x2 plot of velocity (zoomed in and out)
    # Combined timeseries (4 lines) for FRIS net melting, Brunt & Riiser-Larsen net melting, Fimbul net melting
