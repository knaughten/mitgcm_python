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
from ..plot_utils.labels import round_to_decimals, reduce_cbar_labels
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

        
    # Inner function to make a 5-panelled plot with data from the baseline simulation (absolute) and each polynya simulation except the 5-year polynya (absolute or anomaly from baseline).
    def plot_latlon_5panel (var, title, option='absolute', ctype='basic', include_shelf=True, zoom_fris=False, vmin=None, vmax=None, vmin_diff=None, vmax_diff=None, extend='neither', extend_diff='neither'):

        if option not in ['absolute', 'difference']:
            print 'Error (plot_latlon_5panel): invalid option ' + option
            sys.exit()

        # Read data from each simulation, parcelled into a 5-item list
        data = []
        if var == 'vel':
            # Will also need to read velocity components
            u = []
            v = []
        vmin0 = 999
        vmax0 = -999
        if option == 'anomaly':
            vmin0_diff = 0            
            vmax0_diff = 0
        for i in range(num_expts-1):
            if var=='vel' and (option=='absolute' or i==0):
                # Read velocity components too
                data_tmp, u_tmp, v_tmp = read_and_process(var, base_dir+case_dir[i]+avg_file, return_vel_components=True)
                # Either way, will be saving absolute variable
                data.append(data_tmp)
                u.append(u_tmp)
                v.append(v_tmp)
            else:
                data_tmp = read_and_process(var, base_dir+case_dir[i]+avg_file)
                if option=='absolute' or i==0:
                    # Save absolute variable
                    data.append(data_tmp)
                else:
                    # Save anomaly from baseline
                    data.append(data_tmp-data[0])
            # Get min and max values and update global min/max as needed
            vmin0_tmp, vmax0_tmp = var_min_max(data[i], grid)
            if option=='absolute' or i==0:
                vmin0 = min(vmin0, vmin0_tmp)
                vmax0 = max(vmax0, vmax0_tmp)
            else:
                vmin0_diff = min(vmin0_diff, vmin0_tmp)
                vmax0_diff = max(vmax0_diff, vmax0_tmp)
        # Now consider preset bounds
        if vmin is None:
            vmin = vmin0
        if vmax is None:
            vmax = vmax0
        if vmin_diff is None:
            vmin_diff = vmin0_diff
        if vmax_diff is None:
            vmax_diff = vmax0_diff

        # Prepare some parameters for the plot
        if zoom_fris:
            figsize = (13, 5)
            zoom_string = '_zoom'
            chunk = 10
        else:
            figsize = (16, 5)
            zoom_string = ''
            chunk = 6
        if include_shelf or zoom_fris:
            xmin = None
            ymin = None
        else:
            xmin = xmin_sfc
            ymin = ymin_sfc

        # Make the plot
        if option == 'absolute':
            fig, gs, cax = set_panels('5C1', figsize=figsize)
        elif option == 'anomaly':
            fig, gs, cax1, cax2 = set_panels('5C2', figsize=figsize)
        for i in range(num_expts-1):
            # Leave the bottom left plot empty for colourbars
            if i < 3:
                ax = plt.subplot(gs[i/3,i%3])
            else:
                ax = plt.subplot(gs[i/3,i%3+1])
            if option=='absolute' or i==0:
                ctype_curr = ctype
                vmin_curr = vmin
                vmax_curr = vmax
            else:
                ctype_curr = 'plusminus'
                vmin_curr = vmin_diff
                vmax_curr = vmax_diff
            img = latlon_plot(data[i], grid, ax=ax, include_shelf=include_shelf, make_cbar=False, ctype=ctype_curr, vmin=vmin_curr, vmax=vmax_curr, xmin=xmin, ymin=ymin, zoom_fris=zoom_fris, title=expt_names[i])
            if option=='anomaly' and i==0:
                # First colourbar
                cbar1 = plt.colorbar(img, cax=cax1, orientation='horizontal', extend=extend)
                reduce_cbar_labels(cbar1)
            if var=='vel' and (option=='absolute' or i==0):
                # Add velocity vectors
                overlay_vectors(ax, u[i], v[i], grid, chunk=chunk, scale=0.8)
            if i in [1,2,4]:
                # Remove latitude labels
                ax.set_yticklabels([])
            if i in [1,2]:
                # Remove longitude labels
                ax.set_xticklabels([])
        if option=='anomaly':
            # Get ready for second colourbar
            cax = cax2
            extend = extend_diff
            # Text below labelling anomalies
            plt.text(0.25, 0.1, 'anomalies from baseline', fontsize=12, transform=fig.transFigure)
        # Colourbar
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend=extend)
        reduce_cbar_labels(cbar)
        # Main title
        plt.suptitle(title, fontsize=22)
        finished_plot(fig, fig_name=fig_dir+var+zoom_string+'.png')

    # end inner function

    # Now make 5-panel plots of absolute variables
    plot_latlon_5panel('aice', 'Sea ice concentration, 1979-2016', include_shelf=False, vmin=0, vmax=1)
    plot_latlon_5panel('mld', 'Mixed layer depth (m), 1979-2016', include_shelf=False, vmin=0)
    plot_latlon_5panel('vel', 'Barotropic velocity (m/s), 1979-2016', ctype='vel', include_shelf=False, vmin=0)
    # 5-panel plots of baseline absolute values, and anomalies for other simulations, zoomed both in and out
    plot_latlon_5panel('bwtemp', 'Bottom water temperature ('+deg_string+'), 1979-2016', option='anomaly')
    plot_latlon_5panel('bwtemp', 'Bottom water temperature ('+deg_string+'), 1979-2016', option='anomaly', zoom_fris=True)
    plot_latlon_5panel('bwsalt', 'Bottom water salinity (psu), 1979-2016', option='anomaly', vmin=34.3, extend='min')
    plot_latlon_5panel('bwsalt', 'Bottom water salinity (psu), 1979-2016', option='anomaly', zoom_fris=True, vmin=34.3, extend='min')
    plot_latlon_5panel('ismr', 'Ice shelf melt rate (m/y), 1979-2016', option='anomaly', ctype='ismr')
    plot_latlon_5panel('ismr', 'Ice shelf melt rate (m/y), 1979-2016', option='anomaly', ctype='ismr', zoom_fris=True)
    plot_latlon_5panel('vel', 'Barotropic velocity (m/s), 1979-2016', option='anomaly', ctype='vel', zoom_fris=True, vmin=0)
    
            

    '''# Lat-lon plots of absolute variables
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

    # Lat-lon plots of baseline absolute values, and anomalies for other simulations (except 5-year polynya)
    var_names = ['bwtemp', 'bwsalt', 'ismr', 'vel']
    titles = ['Bottom water temperature ('+deg_string+')', 'Bottom water salinity (psu)', 'Ice shelf melt rate (m/y)', 'Barotropic velocity (m/s)']
    ctype = ['basic', 'basic', 'ismr', 'vel']
    for j in range(len(var_names)):
        print 'Plotting ' + var_names[j]
        is_vel = var_names[j] == 'vel'
        # Repeat for both zoomed-in and zoomed-out
        for zoom_fris in [True, False]:
            if zoom_fris:
                zoom_string = '_zoom'
            else:
                zoom_string = ''
            if is_vel and not zoom_fris:
                # Don't want zoomed-out velocity plot
                continue
            data = []
            if is_vel:
                u = []
                v = []
            vmin = 999
            vmax = -999
            for i in range(num_expts-1):
                # Read data
                if is_vel and i==0:
                    # Save velocity components too
                    data_tmp, u, v = read_and_process(var_names[j], base_dir+case_dir[i]+avg_file, return_vel_components=True)
                else:
                    data_tmp = read_and_process(var_names[j], base_dir+case_dir[i]+avg_file)
                if i==0:
                    # Save absolute values for baseline
                    data.append(data_tmp)
                else:
                    # Save anomalies for other simulations
                    data.append(data_tmp-data[0])
                vmin_tmp, vmax_tmp = var_min_max(data[i], grid)
                vmin = min(vmin, vmin_tmp)
                vmax = max(vmax, vmax_tmp)
            if zoom_fris:
                figsize = (13, 5)
            else:
                figsize = None
            fig, gs, cax1, cax2 = set_panels('5C2', figsize=figsize)
            for i in range(num_expts-1):
                if i < 3:
                    ax = plt.subplot(gs[i/3,i%3])
                else:
                    ax = plt.subplot(gs[i/3,i%3+1])
                if i==0:
                    ctype_tmp = ctype[j]
                else:
                    ctype_tmp = 'plusminus'
                img = latlon_plot(data[i], grid, ax=ax, make_cbar=False, ctype=ctype_tmp, vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, title=expt_names[i])
                if i==0:
                    # First colourbar
                    cbar1 = plt.colorbar(img, cax=cax1, orientation='horizontal')
                    for label in cbar1.ax.xaxis.get_ticklabels()[1::2]:
                        label.set_visible(False)
                if is_vel and i==0:
                    # Add velocity vectors to baseline
                    overlay_vectors(ax, u, v, grid, chunk=6, scale=0.8)
                if i in [1,2,4]:
                    # Remove latitude labels
                    ax.set_yticklabels([])
                if i in [1,2]:
                    # Remove longitude labels
                    ax.set_xticklabels([])
            # Second colourbar
            cbar2 = plt.colorbar(img, cax=cax2, orientation='horizontal')
            for label in cbar.ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
            # Main title
            plt.suptitle(titles[j] + ', 1979-2016', fontsize=22)
            finished_plot(fig, fig_name=fig_dir+var_names[j]+zoom_string+'.png')
                
                
                    
    
        

#   Lat-lon plots, averaged over entire period (all except 5-year)
#     Baseline absolute and others differenced, zoomed in and out:
#       BW temp and salt
#       ismr
#       Barotropic velocity (only zoomed in)'''

