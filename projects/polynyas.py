##################################################################
# Weddell Sea polynya project
##################################################################

import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from ..postprocess import precompute_timeseries
from ..utils import real_dir, mask_land_ice, var_min_max, mask_3d, select_bottom, convert_ismr, mask_except_ice, mask_land, polar_stereo
from ..grid import Grid
from ..plot_1d import timeseries_multi_plot, make_timeseries_plot
from ..file_io import netcdf_time, read_netcdf, read_binary
from ..constants import deg_string, sec_per_year
from ..timeseries import trim_and_diff, monthly_to_annual
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.labels import round_to_decimals, reduce_cbar_labels, lon_label, slice_axes, lon_label, lat_label, latlon_axes
from ..plot_utils.latlon import prepare_vel, overlay_vectors, shade_background, clear_ocean, contour_iceshelf_front
from ..plot_utils.colours import set_colours
from ..plot_latlon import latlon_plot
from ..plot_slices import read_plot_ts_slice, read_plot_ts_slice_diff, read_plot_slice, get_loc
from ..calculus import area_integral, vertical_average, lat_derivative
from ..diagnostics import potential_density, heat_content_freezing, density
from ..plot_utils.slices import transect_patches, transect_values, plot_slice_patches

# Global parameters

# File paths
case_dir = ['polynya_baseline/', 'polynya_maud_rise/', 'polynya_near_shelf/', 'polynya_maud_rise_big/', 'polynya_maud_rise_small/', 'polynya_maud_rise_5y/']
grid_dir = case_dir[0] + 'grid/'
timeseries_file = 'output/timeseries_polynya.nc'
timeseries_age_file = 'output/timeseries_age.nc'
avg_file = 'output/1979_2016_avg.nc'
ice_prod_file = 'output/ice_prod_1979_2016_avg.nc'
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

    timeseries_types = ['conv_area', 'fris_ismr', 'ewed_ismr', 'wed_gyre_trans', 'fris_temp', 'fris_salt', 'fris_age', 'sws_shelf_temp', 'sws_shelf_salt', 'isw_vol', 'hssw_vol', 'wdw_vol', 'mwdw_vol']
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

        if var_name == 'fris_age':
            ts_file = timeseries_age_file
        else:
            ts_file = timeseries_file
        
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
            file_path = base_dir + case_dir[i] + ts_file
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
    plot_polynya_timeseries('fris_age', 'FRIS cavity age', 'years')


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
        elif var == 'hice':
            return mask_land_ice(read_netcdf(file_path, 'SIheff', time_index=0), grid)
        elif var == 'hsnow':
            return mask_land_ice(read_netcdf(file_path, 'SIhsnow', time_index=0), grid)
        elif var == 'saltflx':
            return mask_land_ice(read_netcdf(file_path, 'SIempmr', time_index=0), grid)*1e6
        elif var == 'dh_atm_ice':
            return mask_land_ice(read_netcdf(file_path, 'SIdHbATC', time_index=0), grid)*sec_per_year
        elif var == 'dh_atm_ocn':
            return mask_land_ice(read_netcdf(file_path, 'SIdHbATO', time_index=0), grid)*sec_per_year
        elif var == 'dh_ocn':
            return mask_land_ice(read_netcdf(file_path, 'SIdHbOCN', time_index=0), grid)*sec_per_year
        elif var == 'dh_flo':
            return mask_land_ice(read_netcdf(file_path, 'SIdHbFLO', time_index=0), grid)*sec_per_year
        elif var == 'ice_strength':
            return mask_land_ice(read_netcdf(file_path, 'SIpress', time_index=0), grid)
        elif var == 'ice_stress':
            tx_tmp = mask_land_ice(read_netcdf(file_path, 'SItaux', time_index=0), grid, gtype='u')
            ty_tmp = mask_land_ice(read_netcdf(file_path, 'SItauy', time_index=0), grid, gtype='v')
            return prepare_vel(tx_tmp, ty_tmp, grid, vel_option='ice')[0]
        elif var == 'ice_vel':
            uice_tmp = mask_land_ice(read_netcdf(file_path, 'SIuice', time_index=0), grid, gtype='u')
            vice_tmp = mask_land_ice(read_netcdf(file_path, 'SIvice', time_index=0), grid, gtype='v')
            return prepare_vel(uice_tmp, vice_tmp, grid, vel_option='ice')[0]
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
    # Sea ice plots
    plot_latlon_5panel('aice', 'Sea ice concentration, 1979-2016', option='anomaly', zoom_fris=True)
    plot_latlon_5panel('hice', 'Sea ice effective thickness (m), 1979-2016', option='anomaly', zoom_fris=True, vmax=3, vmax_diff=1, extend='max', extend_diff='max')
    plot_latlon_5panel('hsnow', 'Snow effective thickness (m), 1979-2016', option='anomaly', zoom_fris=True)
    plot_latlon_5panel('saltflx', r'Surface salt flux (10$^{-6}$ kg/m$^2$/s), 1979-2016', option='anomaly', zoom_fris=True, ctype='plusminus', vmax=200, vmin_diff=-50, vmax_diff=50, extend='max', extend_diff='both')
    plot_latlon_5panel('dh_atm_ice', 'Net sea ice production from atmosphere flux over ice (m/y), 1979-2016', option='anomaly', zoom_fris=True)
    plot_latlon_5panel('dh_atm_ocn', 'Net sea ice production from atmosphere flux over ocean (m/y), 1979-2016', option='anomaly', zoom_fris=True)
    plot_latlon_5panel('dh_ocn', 'Net sea ice production from ocean flux (m/y), 1979-2016', option='anomaly', zoom_fris=True, vmin=-3, vmax=3, vmin_diff=-2, vmax_diff=2, ctype='plusminus', extend='both', extend_diff='both')
    plot_latlon_5panel('dh_flo', 'Net sea ice production from snow flooding (m/y), 1979-2016', option='anomaly', zoom_fris=True)
    plot_latlon_5panel('ice_strength', 'Sea ice strength (N/m), 1979-2016', option='anomaly', zoom_fris=True, vmax=30000, vmax_diff=10000, extend='max', extend_diff='max')
    plot_latlon_5panel('ice_vel', 'Sea ice velocity (m/s), 1979-2016', option='anomaly', zoom_fris=True, vmax_diff=0.02, extend_diff='max')


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


# Plot 5 polar stereographic panels showing the baseline mean state in the FRIS cavity: bottom water age, barotropic circulation, bottom water temperature and salinity, ice shelf melt rate.
def baseline_panels (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    input_file = base_dir + case_dir[0] + avg_file

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    print 'Processing fields'
    bwage = select_bottom(mask_3d(read_netcdf(input_file, 'TRAC01', time_index=0), grid))
    # Vertically integrate streamfunction and convert to Sv
    psi = np.sum(mask_3d(read_netcdf(input_file, 'PsiVEL', time_index=0), grid), axis=0)*1e-6
    bwtemp = select_bottom(mask_3d(read_netcdf(input_file, 'THETA', time_index=0), grid))
    bwsalt = select_bottom(mask_3d(read_netcdf(input_file, 'SALT', time_index=0), grid))
    ismr = convert_ismr(mask_except_ice(read_netcdf(input_file, 'SHIfwFlx', time_index=0), grid))

    print 'Plotting'
    # Wrap some things up into lists for easier iteration
    data = [bwage, psi, bwtemp, bwsalt, ismr]
    ctype = ['basic', 'psi', 'basic', 'basic', 'ismr']
    vmin = [0, -0.6, -2.5, 34.3, None]
    vmax = [12, 6, -1.5, None, None]
    extend = ['max', None, 'both', 'min', 'neither']
    title = ['a) Bottom water age (years)', 'b) Velocity streamfunction (Sv)', 'c) Bottom water temperature ('+deg_string+'C)', 'd) Bottom water salinity (psu)', 'e) Ice shelf melt rate (m/y)']    
    fig, gs = set_panels('5C0')
    for i in range(len(data)):
        # Leave the top left plot empty for title
        ax = plt.subplot(gs[(i+1)/3, (i+1)%3])
        # Just overlay lat/lon lines in one plot
        lon_lines = None
        lat_lines = None
        if ctype[i] == 'ismr':
            lon_lines = [-40, -60, -80]
            lat_lines = [-75, -80]
        if ctype[i] == 'psi':
            # Special procedure for streamfunction
            x, y = polar_stereo(grid.lon_corners_2d, grid.lat_corners_2d)
            cmap = set_colours(data[i], ctype='psi', vmin=vmin[i], vmax=vmax[i], change_points=[-0.1, -0.025, 0.025, 0.1, 0.5])[0]
            # First make a dummy plot so we have the image to give to the colourbar
            # Otherwise the saved figure has horizontal lines
            img = ax.contourf(x, y, data[i], levels=np.concatenate((np.arange(vmin[i], 0, 0.025), np.arange(0.025, vmax[i], 0.025))), cmap=cmap, linestyles='solid')
            # Now get rid of it
            plt.cla()
            # Now make the real plot
            shade_background(ax)
            clear_ocean(ax, grid, pster=True)
            # Careful contour levels so we don't contour 0
            ax.contour(x, y, data[i], levels=np.concatenate((np.arange(vmin[i], 0, 0.025), np.arange(0.025, vmax[i], 0.025))), cmap=cmap, linestyles='solid')
            cbar = plt.colorbar(img, ticks=np.arange(-0.5, 5.5+1, 1))
            contour_iceshelf_front(ax, grid, pster=True)    
            latlon_axes(ax, x, y, zoom_fris=True, pster=True)
            plt.title(title[i], fontsize=18)
        else:
            # Plot as normal
            img = latlon_plot(data[i], grid, ax=ax, pster=True, lon_lines=lon_lines, lat_lines=lat_lines, ctype=ctype[i], vmin=vmin[i], vmax=vmax[i], extend=extend[i], zoom_fris=True, title=title[i], change_points=[0.5, 1.5, 4])
        if ctype[i] == 'ismr':
            # Overlay location labels
            lon = [-60, -39, -58, -47, -47, -38, -83, -63, -33, -86]
            lat = [-77, -80, -74.5, -77, -79, -77.5, -84, -84.15, -75.5, -80]
            label = ['RIS', 'FIS','RD', 'BB', 'BI', 'FT', lon_label(-80), lon_label(-60), lat_label(-75), lat_label(-80)]
            fs = [14, 14, 14, 14, 14, 14, 10, 10, 10, 10]
            x, y = polar_stereo(lon, lat)            
            for j in range(len(label)):
                plt.text(x[j], y[j], label[j], fontsize=fs[j], va='center', ha='center')
        if i==0:
            # Overlay transect from (56W, 79S) to (42W, 65S)
            lat = np.linspace(-79, -65)
            lon = lat + 23        
            x, y = polar_stereo(lon, lat)
            ax.plot(x, y, color='white', linestyle='dashed', linewidth=1.5)
    # Main title in top left space
    plt.text(0.18, 0.78, 'Baseline conditions\nbeneath FRIS\n(1979-2016 mean)', fontsize=24, va='center', ha='center', transform=fig.transFigure)
    fig.savefig(fig_dir+'baseline_panels.png', dpi=300)
    #finished_plot(fig, fig_name=fig_dir+'baseline_panels.png')


# Plot 5 lat-lon panels showing sea ice concentration averaged over each simulation.
def aice_simulations (base_dir='./', fig_dir='./'):

    title_prefix = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ']

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    print 'Reading data'
    data = []
    mask = [None]
    for i in range(num_expts-1):
        data.append(mask_land_ice(read_netcdf(base_dir+case_dir[i]+avg_file, 'SIarea', time_index=0), grid))
        if i > 0:
            mask.append(read_binary(forcing_dir+polynya_file[i], [grid.nx, grid.ny], 'xy', prec=64))

    print 'Plotting'
    fig, gs, cax = set_panels('5C1', figsize=(13,6.5))
    for i in range(num_expts-1):
        # Leave the bottom left plot empty for colourbar
        if i < 3:
            ax = plt.subplot(gs[i/3,i%3])
        else:
            ax = plt.subplot(gs[i/3,i%3+1])
        img = latlon_plot(data[i], grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, xmin=xmin_sfc, ymin=ymin_sfc, title=title_prefix[i]+expt_names[i])
        if mask[i] is not None:
            # Contour imposed polynya in white
            ax.contour(grid.lon_2d, grid.lat_2d, mask[i], levels=[0.99], colors=('white'), linestyles='solid')
        if i in [1,2,4]:
            ax.set_yticklabels([])
        if i in [1,2]:
            ax.set_xticklabels([])
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    reduce_cbar_labels(cbar)
    plt.suptitle('Sea ice concentration, 1979-2016', fontsize=22)
    finished_plot(fig, fig_name=fig_dir+'aice_simulations.png')


# Plot a 2-part timeseries showing (a) convective area and (b) Weddell Gyre strength in each simulation.
def deep_ocean_timeseries (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)    

    print 'Reading data'
    time = []
    conv_area = []
    wed_gyre = []
    for i in range(num_expts):
        file_path = base_dir + case_dir[i] + timeseries_file
        time.append(netcdf_time(file_path, monthly=False))
        # Convert convective area to 10^5 km^2
        conv_area.append(read_netcdf(file_path, 'conv_area')*10)
        wed_gyre.append(read_netcdf(file_path, 'wed_gyre_trans'))

    # Wrap things up in lists for easier iteration
    data = [conv_area, wed_gyre]
    title = ['a) Convective area', 'b) Weddell Gyre transport']
    units = ['10$^5$ km$^2$', 'Sv']

    print 'Plotting'
    fig, gs = set_panels('2TS')
    for j in range(2):
        ax = plt.subplot(gs[0,j])
        for i in range(num_expts):
            # Annually average
            data_tmp, time_tmp = monthly_to_annual(data[j][i], time[i])
            ax.plot_date(time_tmp, data_tmp, '-', color=expt_colours[i], label=expt_names[i], linewidth=1.25)
        ax.grid(True)
        plt.title(title[j], fontsize=18)
        plt.ylabel(units[j], fontsize=14)
    # Make horizontal legend
    ax.legend(bbox_to_anchor=(0.99,-0.07), ncol=num_expts, fontsize=12)
    finished_plot(fig, fig_name=fig_dir+'deep_ocean_timeseries.png')


# Plot a transect on the continental shelf for the baseline and Maud Rise simulations as well as the anomaly, showing (a) temperature, (b) salinity, and (c) density.
def mwdw_slices (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    # Slice parameters
    # Location
    point0 = (-56, -79)
    point1 = (-42, -65)
    zmin = -1500
    # Colour bounds
    tmin = -2
    tmax = 0.6
    smin = 34.3
    smax = 34.7
    rmin = 32.4
    rmax = 32.6
    # Difference bounds
    tmin_diff = -0.75
    tmax_diff = 0.75
    smin_diff = -0.02
    smax_diff = 0.1
    rmin_diff = -0.005
    rmax_diff = 0.05
    # Contours
    t_contours = [-1.5]
    s_contours = [34.42]
    r_contours = np.arange(32.45, 32.57, 0.02)

    loc_label = 'from ' + get_loc(None, point0=point0, point1=point1)[-1]

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    print 'Reading data'
    temp = []
    salt = []
    rho = []
    for i in range(2):
        file_path = base_dir + case_dir[i] + avg_file
        temp.append(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid))
        salt.append(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid))
        rho.append(mask_3d(density('MDJWF', salt[i], temp[i], 1000), grid)-1000)

    print 'Building patches'
    temp_values = []
    salt_values = []
    rho_values = []
    temp_gridded = []
    salt_gridded = []
    rho_gridded = []
    for i in range(2):
        if i == 0:
            # The first time, build the patches
            patches, temp_values_tmp, hmin, hmax, zmin, zmax, tmp1, tmp2, left, right, below, above, temp_gridded_tmp, haxis, zaxis = transect_patches(temp[i], grid, point0, point1, zmin=zmin, return_bdry=True, return_gridded=True)
            
        else:
            temp_values_tmp, tmp1, tmp2, temp_gridded_tmp = transect_values(temp[i], grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
        temp_values.append(temp_values_tmp)
        temp_gridded.append(temp_gridded_tmp)
        salt_values_tmp, tmp1, tmp2, salt_gridded_tmp = transect_values(salt[i], grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
        salt_values.append(salt_values_tmp)
        salt_gridded.append(salt_gridded_tmp)
        rho_values_tmp, tmp1, tmp2, rho_gridded_tmp = transect_values(rho[i], grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
        rho_values.append(rho_values_tmp)
        rho_gridded.append(rho_gridded_tmp)

    print 'Plotting'
    fig, gs, cax_t, cax_t_diff, cax_s, cax_s_diff, cax_r, cax_r_diff, titles_y = set_panels('3x3C6+T3')
    # Wrap some things up for easier iteration
    values = [temp_values, salt_values, rho_values]
    gridded = [temp_gridded, salt_gridded, rho_gridded]
    contours = [t_contours, s_contours, r_contours]
    cax = [cax_t, cax_s, cax_r]
    cax_diff = [cax_t_diff, cax_s_diff, cax_r_diff]
    vmin = [tmin, smin, rmin]
    vmax = [tmax, smax, rmax]
    vmin_diff = [tmin_diff, smin_diff, rmin_diff]
    vmax_diff = [tmax_diff, smax_diff, rmax_diff]
    var_names = ['a) Temperature ('+deg_string+'C) '+loc_label, 'b) Salinity (psu)', r'c) Density (kg/m$^3$-1000)']
    # Loop over variables (rows)
    for j in range(3):
        # Loop over experiments (columns)
        for i in range(2):
            ax = plt.subplot(gs[j,i])
            # Make the plot
            img = plot_slice_patches(ax, patches, values[j][i], hmin, hmax, zmin, zmax, vmin[j], vmax[j])
            # Add contours
            plt.contour(haxis, zaxis, gridded[j][i], levels=contours[j], colors='black', linestyles='solid')
            # Make nice axes
            slice_axes(ax, h_axis='trans')
            # Remove depth values from right plots
            if i == 1:
                ax.set_yticklabels([])
            # Remove axes labels on all but top left plot
            if i!=0 or j!=0:
                ax.set_ylabel('')
                ax.set_xlabel('')
            # Add experiment title
            plt.title(expt_names[i], fontsize=18)
        # Add a colourbar on the left and hide every second label
        cbar = plt.colorbar(img, extend='both', cax=cax[j])
        reduce_cbar_labels(cbar)
        # Now plot the anomaly
        values_diff = values[j][1] - values[j][0]
        cmap = set_colours(values_diff, ctype='plusminus', vmin=vmin_diff[j], vmax=vmax_diff[j])[0]    
        ax = plt.subplot(gs[j,2])
        img = plot_slice_patches(ax, patches, values_diff, hmin, hmax, zmin, zmax, vmin_diff[j], vmax_diff[j], cmap=cmap)
        slice_axes(ax, h_axis='trans')
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.title('Anomaly', fontsize=18)
        cbar = plt.colorbar(img, extend='both', cax=cax_diff[j])
        reduce_cbar_labels(cbar)
        # Add variable title
        plt.text(0.5, titles_y[j], var_names[j], fontsize=24, va='center', ha='center', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_dir+'mwdw_slices.png')


# Plot 6 polar stereographic panels showing the anomalies for Maud Rise with respect to the baseline, in the FRIS cavity: bottom water temperature, surface salt flux, bottom water salinity, bottom water age, barotropic velocity, and ice shelf melt rate.
def anomaly_panels (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Building grid'
    grid = Grid(base_dir+grid_dir)

    # Inner function to read and process field from a single simulation
    def read_field (var, file_path):
        if var == 'bwtemp':
            return select_bottom(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid))
        elif var == 'iceprod':
            # Convert from m/s to m/y
            return mask_land_ice(read_netcdf(file_path, 'SIdHbOCN', time_index=0) + read_netcdf(file_path, 'SIdHbATC', time_index=0) + read_netcdf(file_path, 'SIdHbATO', time_index=0) + read_netcdf(file_path, 'SIdHbFLO', time_index=0), grid)*sec_per_year
        elif var == 'saltflx':
            # Multiply by 1e6 for nicer colourbar
            return mask_land_ice(read_netcdf(file_path, 'SIempmr', time_index=0), grid)*1e6
        elif var == 'bwsalt':
            return select_bottom(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid))
        elif var == 'bwage':
            return select_bottom(mask_3d(read_netcdf(file_path, 'TRAC01', time_index=0), grid))
        elif var == 'speed':
            u_tmp = mask_3d(read_netcdf(file_path, 'UVEL', time_index=0), grid, gtype='u')
            v_tmp = mask_3d(read_netcdf(file_path, 'VVEL', time_index=0), grid, gtype='v')
            speed, u, v = prepare_vel(u_tmp, v_tmp, grid)#
            return speed
        elif var == 'ismr':
            return convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx', time_index=0), grid))

    # Inner function to read and calculate anomalies
    def read_anomaly (var):
        return read_field(var, base_dir+case_dir[1]+avg_file) - read_field(var, base_dir+case_dir[0]+avg_file)

    # Now call the functions for each variable
    print 'Processing fields'
    bwtemp_diff = read_anomaly('bwtemp')
    iceprod_diff = read_anomaly('iceprod')
    bwsalt_diff = read_anomaly('bwsalt')
    bwage_diff = read_anomaly('bwage')
    speed_diff = read_anomaly('speed')
    ismr_diff = read_anomaly('ismr')

    print 'Plotting'
    # Wrap things into lists
    data = [bwtemp_diff, iceprod_diff, bwsalt_diff, bwage_diff, speed_diff, ismr_diff]
    vmin_diff = [-0.2, -0.6, -0.04, -2, -0.005, -0.2]
    vmax_diff = [0.2, 0.7, 0.04, 1, 0.01, 0.5]
    include_shelf = [True, False, True, True, True, True]
    title = ['a) Bottom water temperature ('+deg_string+'C)', 'b) Net sea ice production (m/y)', 'c) Bottom water salinity (psu)', 'd) Bottom water age (years)', 'e) Barotropic velocity (m/s)', 'f) Ice shelf melt rate (m/y)']
    fig, gs = set_panels('2x3C0')
    for i in range(len(data)):
        ax = plt.subplot(gs[i/3,i%3])
        img = latlon_plot(data[i], grid, ax=ax, pster=True, ctype='plusminus', vmin=vmin_diff[i], vmax=vmax_diff[i], extend='both', zoom_fris=True, title=title[i])
    # Main title
    plt.suptitle('Maud Rise minus baseline (1979-2016 mean)', fontsize=24)
    finished_plot(fig, fig_name=fig_dir+'anomaly_panels.png')


# Plot a 2-part timeseries showing percent changes in basal mass loss for (a) FRIS and (b) EWIS.
def massloss_timeseries (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print 'Reading data'
    times = []
    fris_ismr = []
    ewed_ismr = []
    for i in range(num_expts):
        file_path = base_dir + case_dir[i] + timeseries_file
        times.append(netcdf_time(file_path, monthly=False))
        fris_ismr.append(read_netcdf(file_path, 'fris_ismr'))
        ewed_ismr.append(read_netcdf(file_path, 'ewed_ismr'))

    # Now calculate percent differences
    # Dummy value in spot 0 so indices line up
    times_new = [None]
    fris_diff_percent = [None]
    ewed_diff_percent = [None]
    for i in range(1, num_expts):
        time, fris_diff = trim_and_diff(times[0], times[i], fris_ismr[0], fris_ismr[i])
        fris_diff_percent.append(fris_diff/fris_ismr[0][:len(fris_diff)]*100)
        time, ewed_diff = trim_and_diff(times[0], times[i], ewed_ismr[0], ewed_ismr[i])
        ewed_diff_percent.append(ewed_diff/ewed_ismr[0][:len(ewed_diff)]*100)
        times_new.append(time)
    times = times_new
    
    # Wrap things up in lists
    data = [fris_diff_percent, ewed_diff_percent]
    title = ['a) FRIS basal mass loss % anomaly', 'b) EWIS basal mass loss % anomaly']
    vmin = [-2, -4]
    vmax = [42, 95]

    print 'Plotting'
    fig, gs = set_panels('2TS')
    for j in range(2):
        ax = plt.subplot(gs[0,j])
        for i in range(1, num_expts):
            # Annually average
            data_tmp, time_tmp = monthly_to_annual(data[j][i], times[i])
            # Print the maximum changes
            print expt_names[i] + ' increases by up to ' + str(np.amax(data_tmp)) + '%'
            ax.plot_date(time_tmp, data_tmp, '-', color=expt_colours[i], label=expt_names[i], linewidth=1.25)
        ax.grid(True)
        ax.set_ylim([vmin[j], vmax[j]])
        ax.axhline(color='black')
        plt.title(title[j], fontsize=18)
        plt.ylabel('%', fontsize=14)
    # Make horizontal legend
    ax.legend(bbox_to_anchor=(0.85,-0.07), ncol=num_expts-1, fontsize=12)
    finished_plot(fig, fig_name=fig_dir+'massloss_timeseries.png')
    
        
# Calculate the change in temperature and salinity depth-averaged through the centre of the Maud Rise polynya (last year minus first year).
def calc_polynya_ts_anom (base_dir='./'):

    base_dir = real_dir(base_dir)
    file_path = base_dir + case_dir[1] + timeseries_file

    # Read the timeseries
    time = netcdf_time(file_path, monthly=False)
    temp_polynya = read_netcdf(file_path, 'temp_polynya')
    salt_polynya = read_netcdf(file_path, 'salt_polynya')

    # Annually average
    temp_polynya, time_tmp = monthly_to_annual(temp_polynya, time)
    salt_polynya, time_tmp = monthly_to_annual(salt_polynya, time)

    print 'Change in temperature: ' + str(temp_polynya[-1]-temp_polynya[0]) + ' degC'
    print 'Change in salinity: ' + str(salt_polynya[-1]-salt_polynya[0]) + ' psu'

# Calculate the Weddell Gyre transport in the given simulation, either averaged over the entire simulation or just the last year.
def calc_wed_gyre_trans (timeseries_path, last_year=False):

    time = netcdf_time(timeseries_path, monthly=False)
    wed_gyre_trans = read_netcdf(timeseries_path, 'wed_gyre_trans')
    wed_gyre_trans, time = monthly_to_annual(wed_gyre_trans, time)
    if last_year:
        print 'Weddell Gyre transport over last year: ' + str(wed_gyre_trans[-1]) + ' Sv'
    else:
        print 'Weddell Gyre transport over entire simulation: ' + str(np.mean(wed_gyre_trans))


# Use signal-to-noise analysis to determine when the Maud Rise 5y experiment has "recovered", based on various timeseries.
def calc_recovery_time (base_dir='./', fig_dir='./'):

    perturb_years = 5
    window = 5  # Must be odd
    n = (window-1)/2 # Must be < perturb_years
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    for var in ['fris_ismr', 'ewed_ismr', 'wed_gyre_trans', 'fris_temp', 'fris_salt', 'fris_age', 'sws_shelf_temp', 'sws_shelf_salt']:

        # Paths to timeseries files for the baseline and Maud Rise 5y simulations
        if var == 'fris_age':
            ts_file = timeseries_age_file
        else:
            ts_file = timeseries_file
        file1 = base_dir + case_dir[0] + ts_file
        file2 = base_dir + case_dir[-1] + ts_file

        # Set up plot
        fig, gs = set_panels('2TS')

        # Read baseline timeseries
        time1 = netcdf_time(file1, 'time', monthly=False)
        data1 = read_netcdf(file1, var)
        # Read Maud Rise 5y timeseries
        time2 = netcdf_time(file2, 'time', monthly=False)
        data2 = read_netcdf(file2, var)
        # Get difference
        time, data = trim_and_diff(time1, time2, data1, data2)
        # Annually average
        data, time = monthly_to_annual(data, time)

        # Plot difference
        ax = plt.subplot(gs[0,0])
        ax.plot_date(time, data, '-', linewidth=1.25)
        ax.grid(True)
        ax.axhline(color='black')
        plt.title(var+ ' anomaly', fontsize=18)

        # Now calculate signal-to-noise in chunks of <window> years
        data_s2n = []
        time_s2n = []
        for t in range(perturb_years, len(data)-n):
            data_chunk = data[t-n:t+n+1]
            data_s2n.append(np.mean(data_chunk)/np.std(data_chunk))
            time_s2n.append(time[t])
        data_s2n = np.array(data_s2n)

        # Find the first index where it falls below 1
        if np.all(np.abs(data_s2n) > 1):
            # This never happens
            year0 = '-'
        else:
            t0 = np.where(np.abs(data_s2n) < 1)[0][0]
            year0 = str(time_s2n[t0].year)

        # Plot signal to noise
        ax = plt.subplot(gs[0,1])
        ax.plot_date(time_s2n, data_s2n, '-', linewidth=1.5)
        if np.amax(data_s2n) > 1:
            ax.axhline(y=1, color='black')
        if np.amin(data_s2n) < -1:
            ax.axhline(y=-1, color='black')
        ax.grid(True)
        plt.title(var + ' signal/noise', fontsize=18)
        plt.text(0.5, 0.05, 'Recovers in '+str(year0), fontsize=18, ha='center', va='center', transform=fig.transFigure)
        finished_plot(fig, fig_name=fig_dir+'s2n_'+var+'.png')


        
        

        
    
    

    

    

    
    
    

    
    
        
    
