##################################################################
# Weddell Sea polynya project
##################################################################

import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..postprocess import precompute_timeseries
from ..utils import real_dir, mask_land_ice, var_min_max, mask_3d
from ..grid import Grid
from ..plot_1d import timeseries_multi_plot
from ..file_io import netcdf_time, read_netcdf, read_binary
from ..constants import deg_string
from ..timeseries import trim_and_diff, monthly_to_annual
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.labels import round_to_decimals
from ..plot_utils.latlon import prepare_vel, overlay_vectors
from ..plot_latlon import latlon_plot
from ..averaging import area_integral


# Get longitude and latitude at the centre of the polynya
def get_polynya_loc (polynya):
    
    if polynya.startswith('maud_rise'):
        lon0 = 0
        lat0 = -65
    elif polynya == 'near_shelf':
        lon0 = -30
        lat0 = -70
    else:
        print 'Error (get_polynya_loc): please specify a valid polynya.'
        sys.exit()
    return lon0, lat0


# Precompute timeseries for analysis. Wrapper for precompute_timeseries in postprocess.py. 
def precompute_polynya_timeseries (mit_file, timeseries_file, polynya=None):

    timeseries_types = ['conv_area', 'fris_ismr', 'ewed_ismr', 'wed_gyre_trans', 'fris_temp', 'fris_salt']
    if polynya is None:
        # Baseline simulation; skip temp_polynya and salt_polynya options
        lon0 = None
        lat0 = None
    else:
        lon0, lat0 = get_polynya_loc(polynya)
        timeseries_types += ['temp_polynya', 'salt_polynya']
    precompute_timeseries(mit_file, timeseries_file, timeseries_types=timeseries_types, lon0=lon0, lat0=lat0)


# Make a bunch of tiled/combined plots showing all polynya simulations at once.
def prelim_plots (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    # File paths
    case_dir = ['polynya_baseline/', 'polynya_maud_rise/', 'polynya_near_shelf/', 'polynya_maud_rise_big/', 'polynya_maud_rise_small/', 'polynya_maud_rise_5y/']
    grid_dir = case_dir[0] + 'grid/'
    timeseries_file = 'output/timeseries_polynya.nc'
    avg_file = 'output/1979_2016_avg.nc'
    forcing_dir = '/work/n02/n02/shared/baspog/MITgcm/WS/WSK/'
    polynya_file = [None, 'polynya_mask_maud_rise', 'polynya_mask_near_shelf', 'polynya_mask_maud_rise_big', 'polynya_mask_maud_rise_small', None]
    # Titles etc. for plotting
    expt_names = ['Baseline', 'Maud Rise', 'Near Shelf', 'Maud Rise Big', 'Maud Rise Small', 'Maud Rise 5y']
    expt_colours = ['black', 'blue', 'green', 'red', 'cyan', 'magenta']
    num_expts = len(case_dir)
    # Smaller boundaries on surface plots not including ice shelves
    xmin_sfc = -67
    ymin_sfc = -80

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    # Inner function to plot timeseries on the same axes, plus potentially a difference plot and/or a percent difference plot.
    def plot_polynya_timeseries (var_name, title, units, use_baseline=True, diff=None, percent_diff=None, annual=True):
        
        if use_baseline:
            i0 = 0
        else:
            i0 = 1
        if diff is None:
            diff = use_baseline
        if percent_diff is None:
            percent_diff = diff
        if percent_diff and not diff:
            print "Error (plot_polynya_timeseries): can't make percent difference plot without a difference plot"
            sys.exit()
        if annual:
            monthly_str = ''
        else:
            monthly_str = '_monthly'

        # Read data
        data = []
        for i in range(i0, num_expts):
            file_path = base_dir + case_dir[i] + timeseries_file
            if i==i0:
                # Read the time axis
                time = netcdf_time(file_path)
            # Read the variable
            data_tmp = read_netcdf(file_path, var_name)
            if annual:
                # Annually average
                if i == num_expts-1:
                    # This is the last one
                    # Overwrite data_tmp and time
                    data_tmp, time = monthly_to_annual(data_tmp, time)
                else:
                    # Don't overwrite time yet or it will screw up later iterations
                    data_tmp = monthly_to_annual(data_tmp, time)[0]
            # Parcel into array
            data.append(data_tmp)
            
        # Make the plot
        timeseries_multi_plot(time, data, expt_names[i0:], expt_colours[i0:], title=title, units=units, fig_name=fig_dir+'timeseries_'+var_name+monthly_str+'.png')
        
        if diff:
            # Also make a difference plot
            # Calculate the difference
            data_diff = []
            for i in range(1, num_expts):
                data_diff_tmp = trim_and_diff(time, time, data[0], data[i])[1]
                data_diff.append(data_diff_tmp)
            # Make the plot
            timeseries_multi_plot(time, data_diff, expt_names[1:], expt_colours[1:], title=title+' anomaly', units=units, fig_name=fig_dir+'timeseries_'+var_name+monthly_str+'_diff.png')

            if percent_diff:
                # Also make a percent difference plot
                data_diff_percent = []
                for i in range(num_expts-1):
                    data_diff_percent.append(data_diff[i]/data[0]*100)
                timeseries_multi_plot(time, data_diff_percent, expt_names[1:], expt_colours[1:], title=title+' % anomaly', fig_name=fig_dir+'timeseries_'+var_name+monthly_str+'_percent_diff.png')

    # end inner function

    # Now make the timeseries plots
    plot_polynya_timeseries('conv_area', 'Convective area', r'million km$^2$', use_baseline=False)
    plot_polynya_timeseries('fris_ismr', 'FRIS basal mass loss', 'Gt/y')
    plot_polynya_timeseries('fris_ismr', 'FRIS basal mass loss', 'Gt/y', annual=False)
    plot_polynya_timeseries('ewed_ismr', 'EWIS basal mass loss', 'Gt/y')
    plot_polynya_timeseries('wed_gyre_trans', 'Weddell Gyre transport', 'Sv')
    plot_polynya_timeseries('fris_temp', 'FRIS cavity temperature', deg_string+'C', percent_diff=False)
    plot_polynya_timeseries('fris_salt', 'FRIS cavity salinity', 'psu', percent_diff=False)
    plot_polynya_timeseries('temp_polynya', 'Temperature in polynya', deg_string+'C', use_baseline=False)
    plot_polynya_timeseries('salt_polynya', 'Salinity in polynya', 'psu', use_baseline=False)

    # 2x2 lat-lon plot of polynya masks
    fig, gs = set_panels('2x2C0')
    for i in range(1, num_expts-1):
        # Read polynya mask from binary
        data = read_binary(forcing_dir+polynya_file[i], [grid.nx, grid.ny], 'xy', prec=64)
        # Calculate its area in 10^5 km^2
        area = round_to_decimals(area_integral(data, grid)*1e-11,1)
        title = expt_names[i] + ' ('+str(area)+r'$\times$10$^5$ km$^2$)'
        # Mask out land and ice shelves
        data = mask_land_ice(data, grid)
        # Plot
        ax = plt.subplot(gs[(i-1)/2, (i-1)%2])
        img = latlon_plot(data, grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, title=title, xmin=xmin_sfc, ymin=ymin_sfc)
        if (i-1)%2==1:
            # Remove latitude labels
            ax.set_yticklabels([])
        if (i-1)/2==0:
            # Remove longitude labels
            ax.set_xticklabels([])
    # Main title
    plt.suptitle('Imposed convective regions (red)', fontsize=22)
    finished_plot(fig, fig_name=fig_dir+'polynya_masks.png')

    # Inner function to read a lat-lon variable from a file and process appropriately
    def read_and_process (var, file_path, return_vel_components=False):
        if var == 'aice':
            return mask_land_ice(read_netcdf(file_path, 'SIarea', time_index=0), grid)
        elif var == 'bwtemp':
            return select_bottom(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid))
        elif var == 'bwsalt':
            return select_bottom(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid))
        elif var == 'ismr':
            return convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx', time_index=0), grid))
        elif var == 'vel':
            u_tmp = mask_3d(read_netcdf(file_path, 'UVEL', time_index=0), grid, gtype='u')
            v_tmp = mask_3d(read_netcdf(file_path, 'VVEL', time_index=0), grid, gtype='v')
            speed, u, v = prepare_vel(u_tmp, v_tmp, grid)
            if return_vel_components:
                return speed, u, v
            else:
                return speed
        elif var == 'mld':
            return mask_land_ice(read_netcdf(file_path, 'MXLDEPTH', time_index=0), grid)

    # Lat-lon plots of absolute variables
    # No need to plot the 5-year polynya
    var_names = ['aice', 'mld', 'vel']
    titles = ['Sea ice concentration', 'Mixed layer depth (m)', 'Barotropic velocity (m/s)']
    # Colour bounds to impose
    vmin_impose = [0, 0, None]
    vmax_impose = [1, None, None]
    ctype = ['basic', 'basic', 'vel']
    include_shelf = [False, False, True]
    for j in range(len(var_names)):
        print 'Plotting ' + var_names[j]
        # Special cases for velocity so save as a boolean
        is_vel = var_names[j] == 'vel'
        data = []
        if is_vel:
            u = []
            v = []
        vmin = 999
        vmax = -999
        for i in range(num_expts-1):
            # Read data
            if is_vel:
                data_tmp, u_tmp, v_tmp = read_and_process(var_names[j], base_dir+case_dir[i]+avg_file, return_vel_components=True)
                data.append(data_tmp)
                u.append(u_tmp)
                v.append(v_tmp)
            else:
                data.append(read_and_process(var_names[j], base_dir+case_dir[i]+avg_file))
            # Get min and max values and update global min/max as needed
            vmin_tmp, vmax_tmp = var_min_max(data[i], grid)
            vmin = min(vmin, vmin_tmp)
            vmax = max(vmax, vmax_tmp)
        # Overwrite with predetermined bounds if needed
        if vmin_impose[j] is not None:
            vmin = vmin_impose[j]
        if vmax_impose[j] is not None:
            vmax = vmax_impose[j]
        # Now make the plot
        fig, gs, cax = set_panels('5C1')
        for i in range(num_expts-1):
            # Leave the bottom left plot empty for colourbars
            if i < 3:
                ax = plt.subplot(gs[i/3,i%3])
            else:
                ax = plt.subplot(gs[i/3,i%3+1])
            if include_shelf:
                xmin = None
                ymin = None
            else:
                xmin = xmin_sfc
                ymin = ymin_sfc
            img = latlon_plot(data[i], grid, ax=ax, include_shelf=include_shelf[j], make_cbar=False, ctype=ctype[j], vmin=vmin, vmax=vmax, xmin=xmin, ymin=ymin, title=expt_names[i])
            if is_vel:
                # Add velocity vectors
                overlay_vectors(ax, u[i], v[i], grid, chunk=10, scale=0.8)
            if i in [1,2,4]:
                # Remove latitude labels
                ax.set_yticklabels([])
            if i in [1,2]:
                # Remove longitude labels
                ax.set_xticklabels([])
        # Colourbar, hiding every second label so they're not squished
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        for label in cbar.ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        # Main title
        plt.suptitle(titles[j] + ', 1979-2016', fontsize=22)
        finished_plot(fig, fig_name=fig_dir+var_names[j]+'.png')
                    
    
        

#   Lat-lon plots, averaged over entire period (all except 5-year)
#     Baseline absolute and others differenced, zoomed in and out:
#       BW temp and salt
#       ismr
#       Barotropic velocity (only zoomed in)

