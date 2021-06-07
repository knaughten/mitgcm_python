##################################################################
# Weddell Sea polynya project
##################################################################

import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np

from ..postprocess import precompute_timeseries
from ..utils import real_dir, mask_land_ice, var_min_max, mask_3d, select_bottom, convert_ismr, mask_except_ice, mask_land, polar_stereo
from ..grid import Grid
from ..plot_1d import timeseries_multi_plot, make_timeseries_plot
from ..file_io import netcdf_time, read_netcdf, read_binary
from ..constants import deg_string, sec_per_year, deg2rad, region_bounds, rho_fw
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
from ..interpolation import interp_bilinear

# Global parameters

# File paths
case_dir = ['polynya_baseline/', 'polynya_maud_rise/', 'polynya_near_shelf/', 'polynya_maud_rise_big/', 'polynya_maud_rise_small/', 'polynya_maud_rise_5y/']
grid_dir = case_dir[0] + 'grid/'
timeseries_file = 'output/timeseries_polynya.nc'
timeseries_age_file = 'output/timeseries_age.nc'
timeseries_shelf_file = 'output/timeseries_shelf.nc'
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
        print('Error (get_polynya_loc): please specify a valid polynya.')
        sys.exit()
    return lon0, lat0


# Precompute timeseries for analysis. Wrapper for precompute_timeseries in postprocess.py. 
def precompute_polynya_timeseries (mit_file, timeseries_file, polynya=None):

    timeseries_types = ['conv_area', 'fris_massloss', 'ewed_massloss', 'wed_gyre_trans'] #['sws_shelf_temp', 'sws_shelf_salt'] 
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

    print('Building grid')
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
            print("Error (plot_polynya_timeseries): can't make percent difference plot without a difference plot")
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
    plot_polynya_timeseries('fris_massloss', 'FRIS basal mass loss', 'Gt/y')
    plot_polynya_timeseries('fris_massloss', 'FRIS basal mass loss', 'Gt/y', annual=False)
    plot_polynya_timeseries('ewed_massloss', 'EWIS basal mass loss', 'Gt/y')
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

    print('Building grid')
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
        ax = plt.subplot(gs[(i-1)//2, (i-1)%2])
        img = latlon_plot(data, grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, title=title, xmin=xmin_sfc, ymin=ymin_sfc)
        if (i-1)%2==1:
            # Remove latitude labels
            ax.set_yticklabels([])
        if (i-1)//2==0:
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
            print(('Error (plot_latlon_5panel): invalid option ' + option))
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
                ax = plt.subplot(gs[i//3,i%3])
            else:
                ax = plt.subplot(gs[i//3,i%3+1])
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

    print('Building grid')
    grid = Grid(base_dir+grid_dir)

    # Inner function to read vertically averaged temp and salt, and ice shelf melt rate, for each year of the given simulation
    def read_data (directory):
        temp = []
        salt = []
        ismr = []
        # Loop over years
        for year in range(start_year, end_year+1):
            file_path = directory + file_head + str(year) + file_tail
            print(('Reading ' + file_path))
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

    print('Building grid')
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
                print(('Error (make_slices_lon): invalid option ' + option))
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

    print('\nPlotting timeseries')
    prelim_timeseries(base_dir=base_dir, fig_dir=fig_dir)
    print('\nPlotting lat-lon plots')
    prelim_latlon(base_dir=base_dir, fig_dir=fig_dir)
    print('\nPlotting per-year plots')
    prelim_peryear(base_dir=base_dir, fig_dir=fig_dir)
    print('\nPlotting slices')
    prelim_slices(base_dir=base_dir, fig_dir=fig_dir)


# Plot 5 polar stereographic panels showing the baseline mean state in the FRIS cavity: bottom water age, barotropic circulation, bottom water temperature and salinity, ice shelf melt rate.
def baseline_panels (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    input_file = base_dir + case_dir[0] + avg_file

    print('Building grid')
    grid = Grid(base_dir+grid_dir)

    print('Processing fields')
    bwage = select_bottom(mask_3d(read_netcdf(input_file, 'TRAC01', time_index=0), grid))
    # Vertically integrate streamfunction and convert to Sv
    psi = np.sum(mask_3d(read_netcdf(input_file, 'PsiVEL', time_index=0), grid), axis=0)*1e-6
    bwtemp = select_bottom(mask_3d(read_netcdf(input_file, 'THETA', time_index=0), grid))
    bwsalt = select_bottom(mask_3d(read_netcdf(input_file, 'SALT', time_index=0), grid))
    ismr = convert_ismr(mask_except_ice(read_netcdf(input_file, 'SHIfwFlx', time_index=0), grid))

    print('Plotting')
    # Wrap some things up into lists for easier iteration
    data = [bwage, psi, bwtemp, bwsalt, ismr]
    ctype = ['basic', 'psi', 'basic', 'basic', 'ismr']
    vmin = [0, -0.6, -2.5, 34.3, None]
    vmax = [10, 6, -1.5, None, None]
    extend = ['max', None, 'both', 'min', 'neither']
    title = ['a) Bottom water age (years)', 'b) Barotropic streamfunction (Sv)', 'c) Bottom water temperature ('+deg_string+'C)', 'd) Bottom water salinity (psu)', 'e) Ice shelf melt rate (m/y)']    
    fig, gs = set_panels('5C0')
    for i in range(len(data)):
        # Leave the top left plot empty for title
        ax = plt.subplot(gs[(i+1)//3, (i+1)%3])
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
            # Add arrows (hand-positioned in another program)
            x_arrow = np.array([0.84913177+0.003, 0.78426966+0.002, 0.83350358, 0.80326864, 0.86159346+0.01, 0.84484168+0.01, 0.86475996+0.005, 0.78590398+0.005, 0.77997957+0.015, 0.81562819, 0.78702758-0.01, 0.83350358, 0.80531154+0.02, 0.86639428, 0.87671093+0.002, 0.71348315+0.005, 0.71348315+0.01, 0.73135853, 0.746476, 0.73452503, 0.76475996])
            y_arrow = np.array([0.94435946-0.025, 0.91395793, 0.87456979, 0.85124283-0.02, 0.84110899-0.03, 0.81281071-0.03, 0.79063098-0.005, 0.81376673, 0.7581262, 0.76826004, 0.71969407, 0.7126195, 0.68833652, 0.71720841-0.01, 0.67801147+0.01, 0.81797323-0.02, 0.76328872-0.025, 0.74913958, 0.70650096-0.02, 0.68030593, 0.64072658])
            angle_arrow = np.array([100, -100, -85, 100, 110, 120, -55, -105, 170, -70, 15, -60, 200, 75, -105, 110, 120, -60, -80, 100, -70])*deg2rad
            arrow_length = 0.025
            arrow_dx = arrow_length*np.cos(angle_arrow)
            arrow_dy = arrow_length*np.sin(angle_arrow)
            for j in range(len(x_arrow)):
                plt.arrow(x_arrow[j], y_arrow[j], arrow_dx[j], arrow_dy[j], transform=fig.transFigure, color='black', width=1e-3, head_width=0.012, overhang=0.5, length_includes_head=True, zorder=10000+j)
        else:
            # Plot as normal
            img = latlon_plot(data[i], grid, ax=ax, pster=True, lon_lines=lon_lines, lat_lines=lat_lines, ctype=ctype[i], vmin=vmin[i], vmax=vmax[i], extend=extend[i], zoom_fris=True, title=title[i], change_points=[0.5, 1.5, 4])
        if ctype[i] == 'ismr':
            # Overlay location labels
            lon = [-83, -63, -33, -86] #[-60, -39, -58, -47, -47, -38, -83, -63, -33, -86]
            lat = [-84, -84.15, -75.5, -80] #[-77, -80, -74.5, -77, -79, -77.5, -84, -84.15, -75.5, -80]
            label = [ lon_label(-80), lon_label(-60), lat_label(-75), lat_label(-80)] #['RIS', 'FIS','RD', 'BB', 'BI', 'FT', lon_label(-80), lon_label(-60), lat_label(-75), lat_label(-80)]
            fs = [10, 10, 10, 10] #[14, 14, 14, 14, 14, 14, 10, 10, 10, 10]
            x, y = polar_stereo(lon, lat)            
            for j in range(len(label)):
                plt.text(x[j], y[j], label[j], fontsize=fs[j], va='center', ha='center')
        '''if i==0:
            # Overlay transect shown in mwdw_slices
            [lon0, lat0] = (-56, -79)
            [lon1, lat1] = (-40, -65)
            lon = np.linspace(lon0, lon1)
            lat = (lat1-lat0)/float(lon1-lon0)*(lon-lon0) + lat0
            x, y = polar_stereo(lon, lat)
            ax.plot(x, y, color='white', linestyle='dashed', linewidth=1.5)'''
    # Main title in top left space
    plt.text(0.18, 0.78, 'Baseline conditions\nbeneath FRIS\n(1979-2016 mean)', fontsize=24, va='center', ha='center', transform=fig.transFigure)
    finished_plot(fig, fig_name=fig_dir+'baseline_panels.png')


# Plot 5 lat-lon panels showing sea ice concentration averaged over each simulation.
def aice_simulations (base_dir='./', fig_dir='./'):

    title_prefix = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ']

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print('Building grid')
    grid = Grid(base_dir+grid_dir)

    print('Reading data')
    data = []
    mask = [None]
    for i in range(num_expts-1):
        data.append(mask_land_ice(read_netcdf(base_dir+case_dir[i]+avg_file, 'SIarea', time_index=0), grid))
        if i > 0:
            mask.append(read_binary(forcing_dir+polynya_file[i], [grid.nx, grid.ny], 'xy', prec=64))

    print('Plotting')
    fig, gs, cax = set_panels('5C1', figsize=(13,6.5))
    for i in range(num_expts-1):
        # Leave the bottom left plot empty for colourbar
        if i < 3:
            ax = plt.subplot(gs[i//3,i%3])
        else:
            ax = plt.subplot(gs[i//3,i%3+1])
        img = latlon_plot(data[i], grid, ax=ax, include_shelf=False, make_cbar=False, vmin=0, vmax=1, xmin=xmin_sfc, ymin=ymin_sfc, title=title_prefix[i]+expt_names[i])
        if mask[i] is not None:
            # Contour imposed polynya in white
            ax.contour(grid.lon_2d, grid.lat_2d, mask[i], levels=[0.99], colors=('white'), linestyles='solid')
        if i in [1,2,4]:
            ax.set_yticklabels([])
        if i in [1,2]:
            ax.set_xticklabels([])
        '''if i==0:
            # Overlay transect shown in mwdw_slices
            [lon0, lat0] = (-56, -79)
            [lon1, lat1] = (-40, -65)
            lon = np.linspace(lon0, lon1)
            lat = (lat1-lat0)/float(lon1-lon0)*(lon-lon0) + lat0
            ax.plot(lon, lat, color='black', linestyle='dashed')'''
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    reduce_cbar_labels(cbar)
    plt.suptitle('Sea ice concentration, 1979-2016', fontsize=22)
    finished_plot(fig, fig_name=fig_dir+'aice_simulations.png')


# Plot a 2-part timeseries showing (a) convective area and (b) Weddell Gyre strength in each simulation.
def deep_ocean_timeseries (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)    

    print('Reading data')
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

    print('Plotting')
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
    point1 = (-40, -65)
    zmin = -1500
    # Colour bounds
    tmin = -2
    tmax = 0.6
    smin = 34.3
    smax = 34.7
    rmin = 32.4
    rmax = 32.65
    # Difference bounds
    tmin_diff = -0.75
    tmax_diff = 0.75
    smin_diff = -0.06
    smax_diff = 0.1
    rmin_diff = -0.003
    rmax_diff = 0.06
    # Contours
    t_contours = [-1.3]
    s_contours = [34.45]
    r_contours = np.arange(32.47, 32.65, 0.02)
    # Reference depth for density
    ref_depth = 1000

    loc_label = 'from ' + get_loc(None, point0=point0, point1=point1)[-1]

    print('Building grid')
    grid = Grid(base_dir+grid_dir)

    print('Reading data')
    temp = []
    salt = []
    rho = []
    for i in range(2):
        file_path = base_dir + case_dir[i] + avg_file
        temp.append(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid))
        salt.append(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid))
        rho.append(mask_3d(density('MDJWF', salt[i], temp[i], ref_depth), grid)-1000)

    print('Building patches')
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

    print('Plotting')
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


# Plot 6 polar stereographic panels showing the anomalies for Maud Rise with respect to the baseline, in the FRIS cavity: vertically averaged temperature, surface salt flux, vertically averaged salinity, vertically averaged age, barotropic velocity, and ice shelf melt rate.
def anomaly_panels (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print('Building grid')
    grid = Grid(base_dir+grid_dir)

    # Inner function to read and process field from a single simulation
    def read_field (var, file_path):
        if var == 'tempavg':
            return vertical_average(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid), grid)
        elif var == 'iceprod':
            # Convert from m/s to m/y
            return mask_land_ice(read_netcdf(file_path, 'SIdHbOCN', time_index=0) + read_netcdf(file_path, 'SIdHbATC', time_index=0) + read_netcdf(file_path, 'SIdHbATO', time_index=0) + read_netcdf(file_path, 'SIdHbFLO', time_index=0), grid)*sec_per_year
        elif var == 'saltflx':
            # Multiply by 1e6 for nicer colourbar
            return mask_land_ice(read_netcdf(file_path, 'SIempmr', time_index=0), grid)*1e6
        elif var == 'saltavg':
            return vertical_average(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid), grid)
        elif var == 'ageavg':
            return vertical_average(mask_3d(read_netcdf(file_path, 'TRAC01', time_index=0), grid), grid)
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
    print('Processing fields')
    temp_diff = read_anomaly('tempavg')
    iceprod_diff = read_anomaly('iceprod')
    salt_diff = read_anomaly('saltavg')
    age_diff = read_anomaly('ageavg')
    speed_diff = read_anomaly('speed')
    ismr_diff = read_anomaly('ismr')

    print('Plotting')
    # Wrap things into lists
    data = [temp_diff, iceprod_diff, salt_diff, age_diff, speed_diff, ismr_diff]
    vmin_diff = [-0.06, -0.55, -0.004, -1.5, -0.006, -0.3]
    vmax_diff = [0.16, 1, 0.045, 0.75, 0.012, 0.3]
    include_shelf = [True, False, True, True, True, True]
    title = ['a) Temperature ('+deg_string+'C), depth-average', 'b) Net sea ice production (m/y)', 'c) Salinity (psu), depth-average', 'd) Age (years), depth-average', 'e) Barotropic velocity (m/s)', 'f) Ice shelf melt rate (m/y)']
    fig, gs = set_panels('2x3C0')
    for i in range(len(data)):
        ax = plt.subplot(gs[i//3,i%3])
        img = latlon_plot(data[i], grid, ax=ax, pster=True, ctype='plusminus', vmin=vmin_diff[i], vmax=vmax_diff[i], extend='both', zoom_fris=True, title=title[i])
    # Main title
    plt.suptitle('Maud Rise minus baseline (1979-2016 mean)', fontsize=24)
    finished_plot(fig, fig_name=fig_dir+'anomaly_panels.png')


# Plot a 2-part timeseries showing percent changes in basal mass loss for (a) FRIS and (b) EWIS.
def massloss_timeseries (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print('Reading data')
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
    vmax = [35, 85]

    print('Plotting')
    fig, gs = set_panels('2TS')
    for j in range(2):
        ax = plt.subplot(gs[0,j])
        for i in range(1, num_expts):
            # Annually average
            data_tmp, time_tmp = monthly_to_annual(data[j][i], times[i])
            # Print the maximum changes
            print((expt_names[i] + ' increases by up to ' + str(np.amax(data_tmp)) + '%'))
            ax.plot_date(time_tmp, data_tmp, '-', color=expt_colours[i], label=expt_names[i], linewidth=1.25)
        ax.grid(True)
        ax.set_ylim([vmin[j], vmax[j]])
        ax.axhline(color='black')
        plt.title(title[j], fontsize=18)
        plt.ylabel('%', fontsize=14)
    # Make horizontal legend
    ax.legend(bbox_to_anchor=(0.85,-0.07), ncol=num_expts-1, fontsize=12)
    finished_plot(fig, fig_name=fig_dir+'massloss_timeseries.png')


# Create a map of the model domain, including bathymetry and a number of features/transects/etc. labelled.
def domain_map (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    # Build the grid
    grid = Grid(base_dir+grid_dir)
    # Prepare the bathymetry (km)
    bathy = mask_land(grid.bathy, grid)*1e-3
    # Get bounds for nonlinear colour scale, so continental shelf bathymetry is more visible
    bounds = np.array([-6, -5, -4, -3, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0])
    norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
    # Make the plot
    fig, gs, cax = set_panels('smallC1')
    ax = plt.subplot(gs[0,0])
    img = latlon_plot(bathy, grid, ax=ax, ctype='plusminus', norm=norm, make_cbar=False)
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal', ticks=np.arange(-5,1))
    plt.text(.75, .03, 'Bathymetry (km)', fontsize=14, ha='left', va='bottom', transform=fig.transFigure)
    # Trace outline of sws_shelf_mask
    # First add the grounded iceberg to the mask so it doesn't get outlined
    mask = grid.get_region_mask('sws_shelf').astype(float)
    [xmin, xmax, ymin, ymax] = region_bounds['a23a']
    index = (grid.lon_2d >= xmin)*(grid.lon_2d <= xmax)*(grid.lat_2d >= ymin)*(grid.lat_2d <= ymax)*(grid.land_mask)
    mask[index] = 1
    ax.contour(grid.lon_2d, grid.lat_2d, mask, levels=[0.5], colors=('magenta'), linestyles='dashed', linewidth=1.5)
    # Overlay transect shown in mwdw_slices
    [lon0, lat0] = (-56, -79)
    [lon1, lat1] = (-40, -65)
    lon = np.linspace(lon0, lon1)
    lat = (lat1-lat0)/float(lon1-lon0)*(lon-lon0) + lat0
    # Endpoints
    ax.plot([lon0, lon1], [lat0, lat1], 'ro', markeredgewidth=0)
    # Dashed line between
    ax.plot(lon, lat, color='red', linestyle='dashed', linewidth=1.5)
    # Location labels
    lon = [-60, -39, -58, -47, -47, -38, -22, -1, 21, -63.5, -41]
    lat = [-77, -80, -74.5, -77, -79, -77.5, -75, -70.9, -70.7, -67.5, -75.2]
    label = ['RIS', 'FIS', 'RD', 'BB', 'BI', 'FT', 'BrIS', 'FmIS', 'BoIS', 'LrIS', 'A']
    for i in range(len(label)):
        plt.text(lon[i], lat[i], label[i], fontsize=13, va='center', ha='center', color='black')
    plt.text(3, -65, 'MR', fontsize=13, va='center', ha='center', color='white')
    finished_plot(fig, fig_name=fig_dir+'map.png')

        
# Calculate the change in temperature and salinity depth-averaged through the centre of the Maud Rise polynya (default last year minus first year).
def calc_polynya_ts_anom (base_dir='./', year=-1):

    base_dir = real_dir(base_dir)
    file_path = base_dir + case_dir[1] + timeseries_file

    # Read the timeseries
    time = netcdf_time(file_path, monthly=False)
    temp_polynya = read_netcdf(file_path, 'temp_polynya')
    salt_polynya = read_netcdf(file_path, 'salt_polynya')

    # Annually average
    temp_polynya, time_tmp = monthly_to_annual(temp_polynya, time)
    salt_polynya, time_tmp = monthly_to_annual(salt_polynya, time)

    print(('Change in temperature: ' + str(temp_polynya[year]-temp_polynya[0]) + ' degC'))
    print(('Change in salinity: ' + str(salt_polynya[year]-salt_polynya[0]) + ' psu'))

    
# Calculate the percentage change in convective area by the end of the given simulation, compare d to the given imposed value.
def calc_conv_area_anom (timeseries_path, orig_area, year=-1):

    # Read the timeseries and convert to 10^5 km^2
    time = netcdf_time(timeseries_path, monthly=False)
    conv_area = read_netcdf(timeseries_path, 'conv_area')*10

    # Annually average
    conv_area, time_tmp = monthly_to_annual(conv_area, time)

    print(('Change in convective area: ' + str((conv_area[year]-orig_area)/orig_area*100) + '%'))

    
# Calculate the Weddell Gyre transport in the given simulation, either averaged over the entire simulation or just the last year.
def calc_wed_gyre_trans (timeseries_path, last_year=False):

    time = netcdf_time(timeseries_path, monthly=False)
    wed_gyre_trans = read_netcdf(timeseries_path, 'wed_gyre_trans')
    wed_gyre_trans, time = monthly_to_annual(wed_gyre_trans, time)
    if last_year:
        print(('Weddell Gyre transport over last year: ' + str(wed_gyre_trans[-1]) + ' Sv'))
    else:
        print(('Weddell Gyre transport over entire simulation: ' + str(np.mean(wed_gyre_trans))))


# Use signal-to-noise analysis to determine when the Maud Rise 5y experiment has "recovered", based on various timeseries.
def calc_recovery_time (base_dir='./', fig_dir='./'):

    perturb_years = 5
    window = 5  # Must be odd
    n = (window-1)//2 # Must be < perturb_years
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    for var in ['fris_ismr', 'ewed_ismr', 'sws_shelf_temp', 'sws_shelf_salt', 'fris_salt', 'fris_temp', 'fris_age', 'shelf_minus_fris_salt']:

        if var == 'shelf_minus_fris_salt':
            # We already have the timeseries of the component parts
            data1 = shelf_salt_1 - fris_salt_1
            data2 = shelf_salt_2 - fris_salt_2
        else:
            # Paths to timeseries files for the baseline and Maud Rise 5y simulations
            if var in ['sws_shelf_temp', 'sws_shelf_salt']:
                ts_file = timeseries_shelf_file
            elif var == 'fris_age':
                ts_file = timeseries_age_file
            elif var in ['fris_temp','fris_salt']:
                ts_file = 'output/timeseries.nc'
            else:
                ts_file = timeseries_file
            file1 = base_dir + case_dir[0] + ts_file
            file2 = base_dir + case_dir[-1] + ts_file
            # Read baseline timeseries
            time1 = netcdf_time(file1, 'time', monthly=False)
            data1 = read_netcdf(file1, var)
            # Read Maud Rise 5y timeseries
            time2 = netcdf_time(file2, 'time', monthly=False)
            data2 = read_netcdf(file2, var)
            if var == 'sws_shelf_salt':
                # Save for later
                shelf_salt_1 = data1
                shelf_salt_2 = data2
            elif var == 'fris_salt':
                fris_salt_1 = data1
                fris_salt_2 = data2

        # Get difference
        time, data = trim_and_diff(time1, time2, data1, data2)
        # Annually average
        data, time = monthly_to_annual(data, time)

        # Set up plot
        fig, gs = set_panels('2TS')
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


def rho_range (base_dir='./', fig_dir='./', option='shelf_break'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    ref_depth = 1000
    if option == 'shelf_break':
        point0 = (-56, -79)
        point1 = (-42, -65)
        ymin = 31.8
        ymax = 32.7
    elif option == 'ronne':
        point0 = (-72, -78)
        point1 = (-55, -74)
    elif option == 'filchner':
        point0 = (-42, -81)
        point1 = (-35, -75)
    elif option == 'berkner':
        point0 = (-55, -80)
        point1 = (-47, -76)
    else:
        print(('Error (rho_range): invalid option ' + option))
        sys.exit()

    print('Building grid')
    grid = Grid(base_dir+grid_dir)

    # Extract range of densities in each water column along the transect
    rho_min = []
    rho_max = []
    for expt in range(num_expts-1):
        print(('Reading ' + expt_names[expt]))
        # Read temperature and salinity
        file_path = base_dir + case_dir[expt] + avg_file
        temp = mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid)
        salt = mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid)
        # Calculate density
        rho = mask_3d(density('MDJWF', salt, temp, ref_depth), grid)-1000
        # Extract transect
        rho_trans, haxis = transect_patches(rho, grid, point0, point1, return_gridded=True)[8:10]
        # Extract min and max
        rho_min.append(np.amin(rho_trans, axis=0))
        rho_max.append(np.amax(rho_trans, axis=0))

    hmin = haxis[0]
    hmax = haxis[-1]

    for expt in range(1, num_expts-1):
        print(('Plotting ' + expt_names[expt]))
        fig, ax = plt.subplots(figsize=(11,6))
        ax.fill_between(haxis, rho_min[0], rho_max[0], color='blue', alpha=0.5)
        ax.fill_between(haxis, rho_min[expt], rho_max[expt], color='red', alpha=0.5)
        ax.grid(True)
        ax.set_xlim([hmin, hmax])
        if option == 'shelf_break':            
            ax.set_ylim([ymin, ymax])
        plt.xlabel('Distance along transect (km)', fontsize=16)
        plt.ylabel(r'kg/m$^3$', fontsize=16)
        plt.title('Density range in water column: '+expt_names[expt]+' (red) vs Baseline (blue)', fontsize=18)
        ax2 = ax.twinx()
        ax2.set_xlim([hmin, hmax])
        ax2.plot(haxis, rho_max[0]-rho_min[0], color='blue')
        ax2.plot(haxis, rho_max[expt]-rho_min[expt], color='red')
        finished_plot(fig, fig_name=fig_dir+'rho_range_'+case_dir[expt][:-1]+'.png')


# Plot timeseries of volume-averaged salinity anomalies (Maud Rise minus baseline) in the cavity and over the continental shelf (split into inner and outer shelf).
def salinity_timeseries (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    def get_timeseries_diff (var, fname):
        time = netcdf_time(base_dir+case_dir[0]+fname, monthly=False)
        data_tmp = read_netcdf(base_dir+case_dir[1]+fname, var)-read_netcdf(base_dir+case_dir[0]+fname, var)
        data, time = monthly_to_annual(data_tmp, time)
        return time, data

    time, outer_shelf_salt = get_timeseries_diff('sws_shelf_salt_outer', timeseries_shelf_file)
    inner_shelf_salt = get_timeseries_diff('sws_shelf_salt_inner', timeseries_shelf_file)[1]
    fris_salt = get_timeseries_diff('fris_salt', timeseries_file)[1]
    shelf_salt = get_timeseries_diff('sws_shelf_salt', timeseries_file)[1]

    timeseries_multi_plot(time, [shelf_salt, fris_salt], ['Continental shelf', 'FRIS cavity'], ['black', 'blue'], title='Volume-averaged salinity anomalies (Maud Rise minus baseline)', units='psu', fig_name=fig_dir+'timeseries_salt_anomalies.png')

    timeseries_multi_plot(time, [outer_shelf_salt, inner_shelf_salt, fris_salt], ['Outer shelf', 'Inner shelf', 'FRIS cavity'], ['black', 'green', 'blue'], title='Volume-averaged salinity anomalies (Maud Rise minus baseline)', units='psu', fig_name=fig_dir+'timeseries_salt_anomalies_inner_outer.png')


# Plot baseline velocity (see options), and Maud Rise anomalies with vectors.
# Options are:
# 'vice': sea ice velocity
# 'v350': 350 m ocean velocity
def anomaly_vectors (base_dir='./', fig_dir='./', option='vice'):

    if option == 'vice':
        var_name = 'sea ice'
        vel_option = 'ice'
        z0 = None
        chunk = 8
        scale_abs = 2
        scale_diff = 0.2
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        vmax_diff = None
        figsize = (15, 9)
    elif option == 'v350':
        var_name = '350m'
        vel_option = 'interp'
        z0 = -350
        chunk = 3
        scale_abs = 1
        scale_diff = 0.1
        xmin = -70
        xmax = -15
        ymin = -77
        ymax = -73
        vmax_diff = 0.015
        figsize = (12, 5)
    else:
        print(('Error (anomaly_vectors): invalid option ' + option))
        sys.exit()

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    grid = Grid(base_dir+grid_dir)

    def read_mask_uv (file_path):
        if option == 'vice':
            u = mask_land_ice(read_netcdf(file_path, 'SIuice', time_index=0), grid,  gtype='u')
            v = mask_land_ice(read_netcdf(file_path, 'SIvice', time_index=0), grid,  gtype='v')
        elif option == 'v350':
            u = mask_3d(read_netcdf(file_path, 'UVEL', time_index=0), grid, gtype='u')
            v = mask_3d(read_netcdf(file_path, 'VVEL', time_index=0), grid, gtype='v')
        return u, v

    u0, v0 = read_mask_uv(base_dir+case_dir[0]+avg_file)
    u1, v1 = read_mask_uv(base_dir+case_dir[1]+avg_file)
    speed0, u0_plot, v0_plot = prepare_vel(u0, v0, grid, vel_option=vel_option, z0=z0)
    speed1 = prepare_vel(u1, v1, grid, vel_option=vel_option, z0=z0)[0]
    speed_diff = speed1-speed0
    udiff_plot, vdiff_plot = prepare_vel(u1-u0, v1-v0, grid, vel_option=vel_option, z0=z0)[1:]

    fig, ax = latlon_plot(speed0, grid, ctype='vel', include_shelf=False, title='Baseline '+var_name+' velocity (m/s)', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, return_fig=True, figsize=figsize)
    overlay_vectors(ax, u0_plot, v0_plot, grid, chunk=chunk, scale=scale_abs)
    finished_plot(fig, fig_name=fig_dir+option+'_baseline.png')

    fig, ax = latlon_plot(speed_diff, grid, ctype='plusminus', include_shelf=False, title='Anomalies in '+var_name+' velocity (m/s)', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmax=vmax_diff, return_fig=True, figsize=figsize)
    overlay_vectors(ax, udiff_plot, vdiff_plot, grid, chunk=chunk, scale=scale_diff)
    finished_plot(fig, fig_name=fig_dir+option+'_anomalies.png')


# Calculate the mean and standard deviation of the baseline annually averaged FRIS melt rates.
def calc_ismr_stats (base_dir='./'):

    base_dir = real_dir(base_dir)
    file_path = base_dir + case_dir[0] + timeseries_file

    # Read the timeseries
    time = netcdf_time(file_path, monthly=False)
    melt = read_netcdf(file_path, 'fris_ismr')
    # Annually average
    melt, time = monthly_to_annual(melt, time)

    print(('Mean melt rate: ' + str(np.mean(melt)) + ' Gt/y'))
    print(('Standard deviation in annually averaged melt rate: ' + str(np.std(melt)) + ' Gt/y'))


# Calculate the baseline and Maud Rise salt fluxes from ice shelf refreezing beneath FRIS, and from sea ice formation on the continental shelf.
def calc_salt_fluxes (base_dir='./'):

    base_dir = real_dir(base_dir)
    grid = Grid(base_dir+grid_dir)

    for expt in range(2):
        print((expt_names[expt]))
        file_path = base_dir + case_dir[expt] + avg_file
        # Read salt flux from basal melting in kg/m^2/s
        shelf_flux = read_netcdf(file_path, 'SHIfwFlx', time_index=0)
        # Read net sea ice formation in m/s, and convert to kg/m^2/s
        seaice_flux = (read_netcdf(file_path, 'SIdHbOCN', time_index=0) + read_netcdf(file_path, 'SIdHbATC', time_index=0) + read_netcdf(file_path, 'SIdHbATO', time_index=0) + read_netcdf(file_path, 'SIdHbFLO', time_index=0))*rho_fw
        # Mask (FRIS and continental shelf respectively)
        shelf_flux = mask_except_ice(shelf_flux, grid)
        seaice_flux = np.ma.masked_where(np.invert(grid.get_region_mask('sws_shelf')), seaice_flux)
        # Select only positive values
        shelf_flux = np.maximum(shelf_flux, 0)
        seaice_flux = np.maximum(seaice_flux, 0)
        # Integrate over area
        shelf_flux = area_integral(shelf_flux, grid)
        seaice_flux = area_integral(seaice_flux, grid)
        # Print results
        print(('Total salt flux from ice shelf refreezing: ' + str(shelf_flux) + ' kg/s'))
        print(('Total salt flux from sea ice formation: ' + str(seaice_flux) + ' kg/s'))


# Plot profiles of temperature and salinity in the upper 1500 m, in the centre of the Maud Rise polynya, on the edge, and outside.
def plot_ts_profiles (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    zmin = -1500

    points = [(0, -65), (-8, -65), (-15, -65)]
    point_names = ['Centre of polynya', 'Edge of polynya', 'Outside polynya']
    num_points = len(point_names)
    point_colours = ['red', 'green', 'blue']

    grid = Grid(base_dir+grid_dir)
    file_path = base_dir + case_dir[1] + avg_file

    # Inner function to read profiles and make the plot
    def read_and_plot (var_name, title, units, fig_name):
        data = mask_3d(read_netcdf(file_path, var_name, time_index=0), grid)
        profiles = []
        for n in range(num_points):
            profiles.append(interp_bilinear(data, points[n][0], points[n][1], grid))
        fig, ax = plt.subplots(figsize=(11,6))
        for n in range(num_points):
            ax.plot(profiles[n], grid.z, '-', color=point_colours[n], label=point_names[n], linewidth=1.5)
        ax.grid(True)
        plt.title(title, fontsize=18)
        plt.xlabel(units, fontsize=16)
        plt.ylabel('Depth (m)', fontsize=16)
        ax.set_ylim([zmin, 0])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        finished_plot(fig, fig_name=fig_name)

    # Call for temperature and salinity
    read_and_plot('THETA', 'Temperature profiles', deg_string+'C', fig_dir+'temp_profiles.png')
    read_and_plot('SALT', 'Salinity profiles', 'psu', fig_dir+'salt_profiles.png')


# Make a two-panelled figure showing the evolution of salinity profiles each year of the Maud Rise simulation. The first shows salinity profiles within the polynya, and the second shows salinity profiles just outside.
def salinity_profile_evolution (base_dir='./', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)
    zmin = -1500
    points = [(0, -65), (-10, -65)]
    point_names = ['Centre of polynya', 'Outside polynya']
    num_points = len(point_names)

    grid = Grid(base_dir+grid_dir)

    profiles = np.ma.empty([num_points, num_years, grid.nz])
    for year in range(start_year, end_year+1):
        print(year)
        file_path = base_dir + case_dir[1] + 'output/annual_averages/' + str(year) + '_avg.nc'
        salt = mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid)
        for n in range(num_points):
            profiles[n, year-start_year, :] = interp_bilinear(salt, points[n][0], points[n][1], grid)

    year_colours = plt.get_cmap('jet')(np.linspace(0,1,num_years))
    fig, gs = set_panels('2TS')
    for n in range(num_points):
        ax = plt.subplot(gs[0,n])
        for t in range(num_years):
            ax.plot(profiles[n, t, :], grid.z, '-', color=year_colours[t])
        ax.grid(True)
        plt.title(point_names[n], fontsize=18)
        plt.xlabel('psu', fontsize=16)
        if n==0:
            plt.ylabel('Depth (m)', fontsize=16)
        ax.set_xlim([34.6, 34.7])
        ax.set_ylim([zmin, 0])
    finished_plot(fig, fig_dir+'salt_profiles_evolution_zoom.png')
