##################################################################
# Weddell Sea polynya project
##################################################################

import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from ..postprocess import precompute_timeseries
from ..utils import real_dir, mask_land_ice, var_min_max, mask_3d, select_bottom, convert_ismr, mask_except_ice, mask_land
from ..grid import Grid
from ..plot_1d import timeseries_multi_plot
from ..file_io import netcdf_time, read_netcdf, read_binary
from ..constants import deg_string
from ..timeseries import trim_and_diff, monthly_to_annual
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.labels import round_to_decimals, reduce_cbar_labels
from ..plot_utils.latlon import prepare_vel, overlay_vectors
from ..plot_latlon import latlon_plot
from ..plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff, read_plot_slice
from ..calculus import area_integral, vertical_average, lat_derivative
from ..diagnostics import potential_density, heat_content_freezing

# Global parameters

# File paths
case_dir = ['polynya_baseline/', 'polynya_maud_rise/', 'polynya_near_shelf/', 'polynya_maud_rise_big/', 'polynya_maud_rise_small/', 'polynya_maud_rise_5y/']
grid_dir = case_dir[0] + 'grid/'
timeseries_file = 'output/timeseries_polynya.nc'
avg_file = 'output/1979_2016_avg.nc'
start_year = 1979
end_year = 2016
num_years = end_year-start_year+1
file_head = 'output/annual_averages/'
file_tail = '_avg.nc'
forcing_dir = '/work/n02/n02/shared/baspog/MITgcm/WS/WSK/'
polynya_file = [None, 'polynya_mask_maud_rise', 'polynya_mask_near_shelf', 'polynya_mask_maud_rise_big', 'polynya_mask_maud_rise_small', None]
# Titles etc. for plotting
expt_names = ['Baseline', 'Maud Rise', 'Near Shelf', 'Maud Rise Big', 'Maud Rise Small', 'Maud Rise 5y']
expt_colours = ['black', 'blue', 'green', 'red', 'cyan', 'magenta']
polynya_types = [None, 'maud_rise', 'near_shelf', 'maud_rise_big', 'maud_rise_small', 'maud_rise_5y']
num_expts = len(case_dir)
# Smaller boundaries on surface plots not including ice shelves
xmin_sfc = -67
ymin_sfc = -80


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

    timeseries_types = ['conv_area', 'fris_ismr', 'ewed_ismr', 'wed_gyre_trans', 'fris_temp', 'fris_salt', 'sws_shelf_temp', 'sws_shelf_salt', 'isw_vol', 'hssw_vol', 'wdw_vol', 'mwdw_vol']
    if polynya is None:
        # Baseline simulation; skip temp_polynya and salt_polynya options
        lon0 = None
        lat0 = None
    else:
        lon0, lat0 = get_polynya_loc(polynya)
        timeseries_types += ['temp_polynya', 'salt_polynya']
    precompute_timeseries(mit_file, timeseries_file, timeseries_types=timeseries_types, lon0=lon0, lat0=lat0)


# Make a bunch of preliminary timeseries plots.
def prelim_timeseries (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

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
                # Read the time axis; don't need to back up one month as that was already done during precomputation
                time = netcdf_time(file_path, monthly=False)
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
    plot_polynya_timeseries('sws_shelf_temp', 'FRIS continental shelf temperature', deg_string+'C', percent_diff=False)
    plot_polynya_timeseries('sws_shelf_salt', 'FRIS continental shelf salinity', 'psu', percent_diff=False)
    plot_polynya_timeseries('isw_vol', 'Volume of ISW', '% of domain')
    plot_polynya_timeseries('hssw_vol', 'Volume of HSSW', '% of domain')
    plot_polynya_timeseries('wdw_vol', 'Volume of WDW', '% of domain')
    plot_polynya_timeseries('mwdw_vol', 'Volume of MWDW', '% of domain')


# Make a bunch of preliminary lat-lon plots.
def prelim_latlon (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

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
        elif var == 'sst':
            return mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid)[0,:]
        elif var == 'sss':
            return mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid)[0,:]
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
        elif var in ['rho', 'drho_dlat', 'HfC']:
            # Will need both temp and salt
            temp = mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid)
            salt = mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid)
            if var in ['rho', 'drho_dlat']:
                # First calculate potential density of each point
                rho_3d = potential_density('MDJWF', salt, temp)
                if var == 'rho':
                    # Vertically average
                    return mask_land(vertical_average(rho_3d, grid), grid)
                elif var == 'drho_dlat':
                    # Take derivative and then vertically average
                    drho_dlat_3d = lat_derivative(rho_3d, grid)
                    return mask_land(vertical_average(drho_dlat_3d, grid), grid)
            elif var == 'HfC':
                # Calculate HfC at each point
                hfc_3d = heat_content_freezing(temp, salt, grid)
                # Vertical sum
                return np.sum(hfc_3d*grid.hfac, axis=0)
        elif var == 'temp_avg':
            # Vertically averaged temperature
            return mask_land(vertical_average(read_netcdf(file_path, 'THETA', time_index=0), grid), grid)
            

    # Inner function to make a 5-panelled plot with data from the baseline simulation (absolute) and each polynya simulation except the 5-year polynya (absolute or anomaly from baseline).
    def plot_latlon_5panel (var, title, option='absolute', ctype='basic', include_shelf=True, zoom_fris=False, vmin=None, vmax=None, vmin_diff=None, vmax_diff=None, extend='neither', extend_diff='neither', zoom_shelf_break=False, zoom_ewed=False):

        if option not in ['absolute', 'anomaly']:
            print 'Error (plot_latlon_5panel): invalid option ' + option
            sys.exit()

        # Get bounds
        if zoom_shelf_break:
            xmin = xmin_sfc
            xmax = -20
            ymin = ymin_sfc
            ymax = -72
        elif zoom_ewed:
            xmin = -30
            xmax = 5
            ymin = -77
            ymax = -68
        elif include_shelf or zoom_fris:
            xmin = None
            xmax = None
            ymin = None
            ymax = None
        else:
            xmin = xmin_sfc
            xmax = None
            ymin = ymin_sfc
            ymax = None

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
            vmin0_tmp, vmax0_tmp = var_min_max(data[i], grid, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
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
        if option == 'anomaly':
            if vmin_diff is None:
                vmin_diff = vmin0_diff
            if vmax_diff is None:
                vmax_diff = vmax0_diff

        # Prepare some parameters for the plot
        if zoom_fris or zoom_shelf_break:
            figsize = (12, 7)
            zoom_string = '_zoom'
            chunk = 6
        elif zoom_ewed:
            figsize = (16, 7)
            zoom_string = '_ewed'
            chunk = 6
        else:
            if include_shelf:
                figsize = (14, 7)
            else:
                figsize = (16, 7)
            zoom_string = ''
            chunk = 10

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
            img = latlon_plot(data[i], grid, ax=ax, include_shelf=include_shelf, make_cbar=False, ctype=ctype_curr, vmin=vmin_curr, vmax=vmax_curr, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zoom_fris=zoom_fris, title=expt_names[i])
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
            plt.text(0.2, 0.1, 'anomalies from baseline', fontsize=12, va='center', ha='center', transform=fig.transFigure)
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
    plot_latlon_5panel('vel', 'Barotropic velocity (m/s), 1979-2016', ctype='vel', vmin=0)
    # 5-panel plots of baseline absolute values, and anomalies for other simulations, zoomed both in and out
    plot_latlon_5panel('bwtemp', 'Bottom water temperature ('+deg_string+'C), 1979-2016', option='anomaly', vmin_diff=-0.75, extend_diff='min')
    plot_latlon_5panel('bwtemp', 'Bottom water temperature ('+deg_string+'C), 1979-2016', option='anomaly', zoom_fris=True, vmin=-2.5, vmax=-1.5, extend='both', vmin_diff=-0.2, vmax_diff=0.25, extend_diff='both')
    plot_latlon_5panel('bwsalt', 'Bottom water salinity (psu), 1979-2016', option='anomaly', vmin=34.1, extend='min', vmax_diff=0.1, extend_diff='max')
    plot_latlon_5panel('bwsalt', 'Bottom water salinity (psu), 1979-2016', option='anomaly', zoom_fris=True, vmin=34.3, extend='min')
    plot_latlon_5panel('ismr', 'Ice shelf melt rate (m/y), 1979-2016', option='anomaly', ctype='ismr', vmax_diff=2, extend_diff='max')
    plot_latlon_5panel('ismr', 'Ice shelf melt rate (m/y), 1979-2016', option='anomaly', ctype='ismr', zoom_fris=True, vmax_diff=1.5, extend_diff='max')
    plot_latlon_5panel('vel', 'Barotropic velocity (m/s), 1979-2016', option='anomaly', ctype='vel', zoom_fris=True, vmin=0)
    plot_latlon_5panel('sst', 'Sea surface temperature ('+deg_string+'C), 1979-2016', option='anomaly', include_shelf=False)
    plot_latlon_5panel('sss', 'Sea surface salinity ('+deg_string+'C), 1979-2016', option='anomaly', include_shelf=False)
    plot_latlon_5panel('mld', 'Mixed layer depth (m), 1979-2016', option='anomaly', include_shelf=False, zoom_shelf_break=True, vmax_diff=100, extend_diff='max')
    plot_latlon_5panel('rho', r'Potential density (kg/m$^3$, vertical average), 1979-2016', option='anomaly', vmin=1027.5, extend='min', vmin_diff=-0.05, vmax_diff=0.05, extend_diff='both')
    plot_latlon_5panel('drho_dlat', r'Density gradient (kg/m$^3$/$^{\circ}$lat, vertical average), 1979-2016', option='anomaly', vmin=-25, vmax=25, extend='min', vmin_diff=-0.1, vmax_diff=0.1, extend_diff='both')
    plot_latlon_5panel('HfC', 'Heat content relative to in-situ freezing point (J), 1979-2016', option='anomaly', zoom_fris=True, vmax=8e16, extend='max', vmin_diff=-5e15, vmax_diff=5e15, extend_diff='both')
    # Eastern Weddell plots
    plot_latlon_5panel('ismr', 'Ice shelf melt rate (m/y), 1979-2016', option='anomaly', ctype='ismr', zoom_ewed=True, vmax=6, extend='max', vmax_diff=3, extend_diff='max')
    plot_latlon_5panel('temp_avg', 'Vertically averaged temperature ('+deg_string+'C), 1979-2016', option='anomaly', zoom_ewed=True, vmin_diff=-0.3, vmax_diff=0.3, extend_diff='both')


# Plot vertically averaged temp and salt anomalies, as well as ice shelf melt rate anomalies, for each year.
def prelim_peryear (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    # Inner function to read vertically averaged temp and salt, and ice shelf melt rate, for each year of the given simulation
    def read_data (directory):
        temp = []
        salt = []
        ismr = []
        # Loop over years
        for year in range(start_year, end_year+1):
            file_path = directory + file_head + str(year) + file_tail
            print 'Reading ' + file_path
            temp.append(mask_land(vertical_average(read_netcdf(file_path, 'THETA', time_index=0), grid),  grid))
            salt.append(mask_land(vertical_average(read_netcdf(file_path, 'SALT', time_index=0), grid), grid))
            ismr.append(convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx', time_index=0), grid)))
        return temp, salt, ismr
    
    # Inner function to find the min and max anomalies over all years of the given data (baseline and perturbation, each a list of length num_years, each element is a lat-lon array)
    def find_min_max_diff (data0, data):
        vmin_diff = 0
        vmax_diff = 0
        for t in range(num_years):
            vmin_diff_tmp, vmax_diff_tmp = var_min_max(data[t]-data0[t], grid, zoom_fris=True)
            vmin_diff = min(vmin_diff, vmin_diff_tmp)
            vmax_diff = max(vmax_diff, vmax_diff_tmp)
        return vmin_diff, vmax_diff            

    # Read the baseline data
    temp0, salt0, ismr0 = read_data(base_dir+case_dir[0])

    # Loop over other simulations
    for i in range(1, num_expts):
        # Read data
        temp, salt, ismr = read_data(base_dir+case_dir[i])
        # Find min/max anomalies across all years
        tmin_diff, tmax_diff = find_min_max_diff(temp0, temp)
        smin_diff, smax_diff = find_min_max_diff(salt0, salt)
        imin_diff, imax_diff = find_min_max_diff(ismr0, ismr)
        # Loop over years
        for year in range(start_year, end_year+1):
            t = year-start_year
            fig, gs, cax1, cax2, cax3 = set_panels('1x3C3')
            # Wrap some things up into lists of length 3 for easier iteration
            data = [temp[t]-temp0[t], salt[t]-salt0[t], ismr[t]-ismr0[t]]
            vmin = [tmin_diff, smin_diff, imin_diff]
            vmax = [tmax_diff, smax_diff, imax_diff]
            cax = [cax1, cax2, cax3]
            title = ['Vertically averaged temperature ('+deg_string+'C)', 'Vertically averaged salinity (psu)', 'Ice shelf basal melt rate (m/y)']
            # Now loop over variables
            for j in range(3):
                ax = plt.subplot(gs[0,j])
                img = latlon_plot(data[j], grid, ax=ax, make_cbar=False, ctype='plusminus', vmin=vmin[j], vmax=vmax[j], zoom_fris=True, title=title[j])
                if j>0:
                    # Remove axes labels
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                cbar = plt.colorbar(img, cax=cax[j], orientation='horizontal')
                reduce_cbar_labels(cbar)
            # Main title
            plt.suptitle(str(year) + ' anomalies', fontsize=24)
            finished_plot(fig, fig_name=fig_dir+polynya_types[i]+'_'+str(year)+'_anom.png')
        

# Make a bunch of preliminary slice plots.
def prelim_slices (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    baseline_file = base_dir+case_dir[0]+avg_file

    # Baseline T-S slices through each polynya, in each direction    
    for i in range(1,2+1):
        ptype = polynya_types[i]
        lon0, lat0 = get_polynya_loc(ptype)
        read_plot_ts_slice(baseline_file, grid=grid, lon0=lon0, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_polynya_'+ptype+'_lon.png')
        read_plot_ts_slice(baseline_file, grid=grid, lat0=lat0, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_polynya_'+ptype+'_lat.png')
    # T-S difference slices for each polynya minus baseline, in each direction
    for i in range(1, num_expts-1):
        ptype = polynya_types[i]
        lon0, lat0 = get_polynya_loc(ptype)
        curr_file = base_dir+case_dir[i]+avg_file
        read_plot_ts_slice_diff(baseline_file, curr_file, grid=grid, lon0=lon0, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_polynya_'+ptype+'_lon_diff.png')
        read_plot_ts_slice_diff(baseline_file, curr_file, grid=grid, lat0=lat0, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_polynya_'+ptype+'_lat_diff.png')

    # Inner function for baseline and polynya T-S slices (absolute or anomaly) through a given longitude, with given bounds
    def make_slices_lon (lon0, string, hmin=None, hmax=None, zmin=None, tmin=None, tmax=None, smin=None, smax=None, tmin_diff=None, tmax_diff=None, smin_diff=None, smax_diff=None, option='anomaly'):
        # Baseline
        read_plot_ts_slice(baseline_file, grid=grid, lon0=lon0, hmin=hmin, hmax=hmax, zmin=zmin, tmin=tmin, tmax=tmax, smin=smin, smax=smax, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_'+string+'.png')
        # Each polynya
        for i in range(1, num_expts-1):
            ptype = polynya_types[i]
            curr_file = base_dir+case_dir[i]+avg_file
            if option == 'absolute':
                read_plot_ts_slice(curr_file, grid=grid, lon0=lon0, hmin=hmin, hmax=hmax, zmin=zmin, tmin=tmin, tmax=tmax, smin=smin, smax=smax, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_'+string+'_'+ptype+'.png')
            elif option == 'anomaly':
                read_plot_ts_slice_diff(baseline_file, curr_file, grid=grid, lon0=lon0, hmin=hmin, hmax=hmax, zmin=zmin, tmin=tmin_diff, tmax=tmax_diff, smin=smin_diff, smax=smax_diff, time_index=0, date_string='1979-2016', fig_name=fig_dir+'ts_slice_'+string+'_'+ptype+'_diff.png')
            else:
                print 'Error (make_slices_lon): invalid option ' + option
                sys.exit()

    # 50W, where WDW comes onto shelf
    make_slices_lon(-50, '50W', hmin=-79, hmax=-65, zmin=-1500)
    # Zoomed in, absolute
    make_slices_lon(-50, '50W_zoom', option='absolute', hmin=-77, hmax=-70, zmin=-1000, tmin=-1.9, tmax=0.5, smin=34, smax=34.65)
    # Fimbul
    make_slices_lon(-1, 'fimbul', hmax=-69, zmin=-1500, option='absolute', tmin=-1.9, tmax=0.5, smin=34, smax=34.7)
    # Riiser-Larsen
    make_slices_lon(-19, 'rl', hmax=-72, zmin=-1000)

    # Inner function for density slices, with isopycnals contoured at 0.025 kg/m^3 intervals, and given colour bounds. Repeat for each simulation with absolute values.
    def density_slices (lon0, string, hmin=None, hmax=None, zmin=None, vmin=None, vmax=None, contour_step=0.025):
        for i in range(num_expts-1):
            ptype = polynya_types[i]
            if ptype is None:
                ptype = 'baseline'
            curr_file = base_dir+case_dir[i]+avg_file
            read_plot_slice('rho', curr_file, grid=grid, lon0=lon0, time_index=0, hmin=hmin, hmax=hmax, zmin=zmin, vmin=vmin, vmax=vmax, contours=np.arange(vmin+contour_step,vmax,contour_step), fig_name=fig_dir+'rho_slice_'+string+'_'+ptype+'.png')

    density_slices(0, '0E', hmax=-65, zmin=-2000, vmin=1027.3, vmax=1027.85)
    density_slices(-20, '20W', hmax=-70, zmin=-1000, vmin=1027.3, vmax=1027.8)
    density_slices(-30, '30W', hmin=-78, hmax=-70, zmin=-1500, vmin=1027.5, vmax=1027.82)
    density_slices(-39, '39W', hmax=-65, zmin=-2000, vmin=1027.6, vmax=1027.85)
    density_slices(-45, '45W', hmin=-78.5, hmax=-65, zmin=-2000, vmin=1027.6, vmax=1027.85)
    density_slices(-48, '48W', hmin=-79, hmax=-65, zmin=-1500, vmin=1027.6, vmax=1027.85)
    density_slices(-50, '50W', hmin=-79, hmax=-65, zmin=-1500, vmin=1027.6, vmax=1027.85)
    density_slices(-52, '52W', hmin=-80.5, hmax=-65, zmin=-2000, vmin=1027.6, vmax=1027.85)
    density_slices(0, 'polynya_0E', vmin=1027.7, vmax=1027.87, contour_step=0.01)
    density_slices(-30, 'polynya_30W', vmin=1027.6, vmax=1027.86, contour_step=0.01)

    # Density slices through Maud Rise polynya each year
    vmin = 1027.7
    vmax = 1027.87
    contour_step = 0.01
    for year in range(start_year, end_year+1):
        file_path = base_dir + case_dir[1] + file_head + str(year) + file_tail
        read_plot_slice('rho', file_path, grid=grid, lon0=0, time_index=0, vmin=vmin, vmax=vmax, contours=np.arange(vmin+contour_step, vmax, contour_step), fig_name=fig_dir+'rho_slice_maud_rise_'+str(year)+'.png')

    

# Make all the preliminary plots.
def prelim_all_plots (base_dir='./', fig_dir='./'):

    print '\nPlotting timeseries'
    prelim_timeseries(base_dir=base_dir, fig_dir=fig_dir)
    print '\nPlotting lat-lon plots'
    prelim_latlon(base_dir=base_dir, fig_dir=fig_dir)
    print '\nPlotting per-year plots'
    prelim_peryear(base_dir=base_dir, fig_dir=fig_dir)
    print '\nPlotting slices'
    prelim_slices(base_dir=base_dir, fig_dir=fig_dir)


# Plot 5 lat-lon panels showing the baseline mean state in the FRIS cavity: bottom water age, barotropic circulation, bottom water temperature and salinity, ice shelf melt rate.
def baseline_panels (base_dir='./', fig_dir='./', input_file=None):

    if input_file is None:
        input_file = base_dir + case_dir[0] + avg_file

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    print 'Processing fields'
    bwage = select_bottom(mask_3d(read_netcdf(input_file, 'TRAC01', time_index=0), grid))
    u_tmp = mask_3d(read_netcdf(input_file, 'UVEL', time_index=0), grid, gtype='u')
    v_tmp = mask_3d(read_netcdf(input_file, 'VVEL', time_index=0), grid, gtype='v')
    speed, u, v = prepare_vel(u_tmp, v_tmp, grid)
    bwtemp = select_bottom(mask_3d(read_netcdf(input_file, 'THETA', time_index=0), grid))
    bwsalt = select_bottom(mask_3d(read_netcdf(input_file, 'SALT', time_index=0), grid))
    ismr = convert_ismr(mask_except_ice(read_netcdf(input_file, 'SHIfwFlx', time_index=0), grid))

    print 'Plotting'
    # Wrap some things up into lists for easier iteration
    data = [bwage, speed, bwtemp, bwsalt, ismr]
    ctype = ['basic', 'vel', 'basic', 'basic', 'ismr']
    vmin = [0, 0, -2.5, 34.3, None]
    vmax = [12, None, -1.5, None, None]
    extend = ['max', 'neither', 'both', 'min', 'neither']
    title = ['a) Bottom water age (years)', 'b) Barotropic velocity (m/s)', 'c) Bottom water temperature ('+deg_string+'C)', 'd) Bottom water salinity (psu)', 'e) Ice shelf melt rate (m/y)']    
    fig, gs = set_panels('5C0')
    for i in range(len(data)):
        # Leave the top left plot empty for title
        ax = plt.subplot(gs[(i+1)/3, (i+1)%3])
        img = latlon_plot(data[i], grid, ax=ax, ctype=ctype[i], vmin=vmin[i], vmax=vmax[i], extend=extend[i], zoom_fris=True, title=title[i])
        if ctype[i] == 'vel':
            # Overlay velocity vectors
            overlay_vectors(ax, u, v, grid, chunk=7, scale=0.6)
        if ctype[i] == 'ismr':
            # Overlay location labels
            lon = [-58, -47, -47, -38]
            lat = [-74.5, -77, -79, -77.5]
            label = ['RD', 'BB', 'BI', 'FT']
            for j in range(len(label)):
                plt.text(lon[j], lat[j], label[j], fontsize=14, va='center', ha='center')
        if i in [1,3,4]:
            # Remove latitude labels
            ax.set_yticklabels([])
        if i in [0,1]:
            # Remove longitude labels
            ax.set_xticklabels([])
    # Main title in top left space
    plt.text(0.18, 0.78, 'Baseline conditions\nbeneath FRIS\n(1979-2016 mean)', fontsize=24, va='center', ha='center', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_dir+'baseline_panels.png')


# Plot 5 lat-lon panels showing sea ice concentration averaged over each simulation.
def aice_simulations (base_dir='./', fig_dir='./'):

    # For now overwrite files with temporary files
    in_files = ['WSK_002/output/1979_2016_avg.nc', 'WSK_003/output/1979_2016_avg.nc', 'WSK_004/output/avg_to_jan06.nc', 'WSK_005/output/avg_to_mar07.nc', 'WSK_006/output/avg_to_feb98.nc']

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    print 'Reading data'
    data = []
    for i in range(num_expts-1):
        data.append(mask_land_ice(read_netcdf(base_dir+in_files[i], 'SIarea', time_index=0), grid))

    print 'Plotting'
    fig, gs, cax = set_panels('5C1', figsize=(11,5))
    for i in range(num_expts-1):
        # Leave the bottom left plot empty for colourbar
        if i < 3:
            ax = plt.subplot(gs[i/3,i%3])
        else:
            ax = plt.subplot(gs[i/3,i%3+1])
        img = latlon_plot(data[i], grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, xmin=xmin_sfc, ymin=ymin_sfc, title=expt_names[i])
        if i in [1,2,4]:
            ax.set_yticklabels([])
        if i in [1,2]:
            ax.set_xticklabels([])
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    reduce_cbar_labels(cbar)
    plt.suptitle('Sea ice concentration, 1979-2016', fontsize=22)
    finished_plot(fig) #, fig_name=fig_dir+'aice_simulations.png')


# Plot a 2-part timeseries showing (a) convective area and (b) Weddell Gyre strength in each simulation.
def deep_ocean_timeseries (base_dir='./', fig_dir='./'):

    # For now overwrite case directories
    tmp_cases = ['WSK_002/', 'WSK_003/', 'WSK_004/', 'WSK_005/', 'WSK_006/', 'WSK_007/']
    
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)    

    print 'Reading data'
    time = []
    conv_area = []
    wed_gyre = []
    for i in range(num_expts):
        file_path = base_dir + tmp_cases[i] + timeseries_file
        time.append(netcdf_time(file_path, monthly=False))
        # Convert convective area to 10^5 km^2
        conv_area.append(read_netcdf(file_path, 'conv_area')*10)
        wed_gyre.append(read_netcdf(file_path, 'wed_gyre_trans'))
        
    # Wrap things up in lists for easier iteration
    data = [conv_area, wed_gyre]
    title = ['Convective area', 'Weddell Gyre transport']
    units = ['10$^5$ km$^2$', 'Sv']

    print 'Plotting'
    fig, gs = set_panels('2TS')
    for j in range(2):
        ax = plt.subplot(gs[0,j])
        for i in range(num_expts):
            ax.plot_date(time[i], data[j][i], '-', color=expt_colours[i], label=expt_names[i])
        ax.grid(True)
        plt.title(title[j])
        plt.ylabel(units[j])
    # Make horizontal legend
    ax.legend(bbox_to_anchor=(0.5,-0.2), ncol=num_expts)
    finished_plot(fig, fig_name=fig_dir+'deep_ocean_timeseries.png')
    

    
    
        
    
