##################################################################
# ERA-Interim vs ERA5 comparison
##################################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..file_io import read_netcdf, netcdf_time
from ..constants import deg2rad, deg_string
from ..utils import fix_lon_range, split_longitude, real_dir, mask_land_ice, select_bottom, mask_3d, mask_except_ice, convert_ismr, var_min_max
from ..plot_latlon import latlon_plot, prepare_vel, overlay_vectors
from ..plot_1d import timeseries_multi_plot
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.labels import latlon_axes
from ..plot_utils.colours import set_colours

# Figure out whether katabatic winds are stronger in ERA5 than in ERA-Interim. Here we have 3 years of wind data (2008-2010) for each reanalysis, both interpolated to the same regular grid upon downloading, plus a land mask file.
def compare_katabatics (erai_file, era5_file, land_mask_file, fig_dir='./'):

    fig_dir = real_dir(fig_dir)

    # Longitude and latitude ranges we care about
    xmin = -85
    xmax = 30
    ymin = -84
    ymax = -61

    print('Reading grid')
    # Should be the same in both products because we downloaded it that way
    lon = fix_lon_range(read_netcdf(erai_file, 'longitude'))
    # Latitude starts at the North Pole; flip it
    lat = np.flipud(read_netcdf(erai_file, 'latitude'))
    # Split the domain at 180E=180W and rearrange the two halves
    i_split = np.nonzero(lon < 0)[0][0]
    lon = split_longitude(lon, i_split)

    # Select Weddell Sea region
    # i_end and j_end are the first indices not used.
    i_beg = np.nonzero(lon > xmin)[0][0] - 1
    i_end = np.nonzero(lon > xmax)[0][0] + 1
    j_beg = np.nonzero(lat > ymin)[0][0] - 1
    j_end = np.nonzero(lat > ymax)[0][0] + 1
    lon = lon[i_beg:i_end]
    lat = lat[j_beg:j_end]

    # Inner function to read a field, (possibly) time-average, flip along latitude, split and rearrange, trim, (possibly) mask land
    def process_field (file_path, var_name, time_dependent=True, land=None):
        data = read_netcdf(file_path, var_name)
        if time_dependent:
            data = np.mean(data, axis=0)
        data = np.flipud(data)
        data = split_longitude(data, i_split)
        data = data[j_beg:j_end, i_beg:i_end]
        if land is not None:
            data = np.ma.masked_where(land==1, data)
        return data

    # First do this for the land mask
    land = process_field(land_mask_file, 'lsm', time_dependent=False)
    # Now for all the wind fields
    print('Reading data')
    erai_uwind = process_field(erai_file, 'u10', land=land)
    erai_vwind = process_field(erai_file, 'v10', land=land)
    era5_uwind = process_field(era5_file, 'u10', land=land)
    era5_vwind = process_field(era5_file, 'v10', land=land)

    print('Calculating derived variables')
    # Magnitude and direction of wind vectors
    erai_speed = np.sqrt(erai_uwind**2 + erai_vwind**2)
    era5_speed = np.sqrt(era5_uwind**2 + era5_vwind**2)
    erai_angle = np.arctan2(erai_vwind, erai_uwind)/deg2rad
    era5_angle = np.arctan2(era5_vwind, era5_uwind)/deg2rad

    # For each variable (u, v, speed, angle), plot ERA-Interim values and ERA5 anomaly
    # Inner function to make this easier
    def plot_field (erai_data, era5_data, suptitle, fig_name):
        fig, gs, cax1, cax2 = set_panels('1x2C2', figsize=(12,6))
        # Wrap some things up in lists for easy iteration over the 2 subplots
        data = [erai_data, era5_data-erai_data]
        ctype = ['basic', 'plusminus']
        cax = [cax1, cax2]
        title = ['ERA-Interim', 'ERA5 anomaly']
        for t in range(2):
            cmap, vmin, vmax = set_colours(data[t], ctype=ctype[t])
            ax = plt.subplot(gs[0,t])
            img = ax.contourf(lon, lat, data[t], 50, cmap=cmap, vmin=vmin, vmax=vmax)
            latlon_axes(ax, lon, lat)
            if t == 1:
                ax.set_yticklabels([])
            plt.colorbar(img, cax=cax[t], orientation='horizontal')
            plt.title(title[t], fontsize=18)
        plt.suptitle(suptitle, fontsize=22)
        finished_plot(fig, fig_name=fig_name)
        fig.show()

    # Now call it for each variable
    print('Plotting')
    plot_field(erai_uwind, era5_uwind, 'Zonal wind (m/s)', fig_dir+'uwind.png')
    plot_field(erai_vwind, era5_vwind, 'Meridional wind (m/s)', fig_dir+'vwind.png')
    plot_field(erai_speed, era5_speed, 'Wind speed (m/s)', fig_dir+'speed.png')
    plot_field(erai_angle, era5_angle, 'Wind angle (degrees)', fig_dir+'angle.png')


# Make a bunch of tiled plots showing the the three simulations (ERA-Interim, ERA5 6-hourly, ERA5 1-hourly) at once.
def combined_plots(base_dir='./', fig_dir='./'):

    # File paths
    grid_path = 'era_interim/grid/'
    output_dir = ['era_interim/output/', 'era5_6h/output/', 'era5_1h/output/']
    mit_file = '2008_2017.nc'
    timeseries_file = 'timeseries.nc'
    # Titles etc. for plotting
    expt_names = ['ERA-Interim', 'ERA5 (6-hourly)', 'ERA5 (1-hourly)']
    expt_colours = ['black', 'blue', 'green']

    # Smaller boundaries on surface plots (where ice shelves are ignored)
    xmin_sfc = -67
    ymin_sfc = -80

    # Make sure real directories
    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    print('Building grid')
    grid = Grid(base_dir+grid_path)

    # Inner function to read a lat-lon variable from a file and process appropriately
    def read_and_process (var, file_path, return_vel_components=False):        
        if var == 'aice':
            return mask_land_ice(read_netcdf(file_path, 'SIarea', time_index=-1), grid)
        elif var == 'hice':
            return mask_land_ice(read_netcdf(file_path, 'SIheff', time_index=-1), grid)
        elif var == 'hsnow':
            return mask_land_ice(read_netcdf(file_path, 'SIhsnow', time_index=-1), grid)
        elif var == 'bwtemp':
            return select_bottom(mask_3d(read_netcdf(file_path, 'THETA', time_index=-1), grid))
        elif var == 'bwsalt':
            return select_bottom(mask_3d(read_netcdf(file_path, 'SALT', time_index=-1), grid))
        elif var == 'sst':
            return mask_3d(read_netcdf(file_path, 'THETA', time_index=-1), grid)[0,:]
        elif var == 'sss':
            return mask_3d(read_netcdf(file_path, 'SALT', time_index=-1), grid)[0,:]
        elif var == 'ismr':
            return convert_ismr(mask_except_ice(read_netcdf(file_path, 'SHIfwFlx', time_index=-1), grid))
        elif var == 'mld':
            return mask_land_ice(read_netcdf(file_path, 'MXLDEPTH', time_index=-1), grid)
        elif var == 'saltflx':
            return mask_land_ice(read_netcdf(file_path, 'SIempmr', time_index=-1), grid)
        elif var == 'vel':
            u_tmp = mask_3d(read_netcdf(file_path, 'UVEL', time_index=-1), grid, gtype='u')
            v_tmp = mask_3d(read_netcdf(file_path, 'VVEL', time_index=-1), grid, gtype='v')
            speed, u, v = prepare_vel(u_tmp, v_tmp, grid)
            if return_vel_components:
                return speed, u, v
            else:
                return speed
        elif var == 'velice':
            uice_tmp = mask_land_ice(read_netcdf(file_path, 'SIuice', time_index=-1), grid)
            vice_tmp = mask_land_ice(read_netcdf(file_path, 'SIvice', time_index=-1), grid)
            speed, uice, vice = prepare_vel(uice_tmp, vice_tmp, grid, vel_option='ice')
            if return_vel_components:
                return speed, uice, vice
            else:
                return speed

    # 3x1 plots of absolute variables
    var_names = ['aice', 'hice', 'hsnow', 'bwtemp', 'bwsalt', 'sst', 'sss', 'ismr', 'mld', 'saltflx', 'vel', 'velice']
    titles = ['Sea ice concentration', 'Sea ice effective thickness (m)', 'Snow effective thickness (m)', 'Bottom water temperature ('+deg_string+'C)', 'Bottom water salinity (psu)', 'Sea surface temperature ('+deg_string+'C)', 'Sea surface salinity (psu)', 'Ice shelf melt rate (m/y)', 'Mixed layer depth (m)', r'Surface salt flux (kg/m$^2$/s)', 'Barotropic velocity (m/s)', 'Sea ice velocity (m/s)']
    # Colour bounds to impose
    vmin_impose = [0, 0, 0, None, 34.2, None, None, None, 0, -0.001, 0, 0]
    vmax_impose = [1, 5, None, -0.5, None, None, None, None, None, 0.001, None, None]
    ctype = ['basic', 'basic', 'basic', 'basic', 'basic', 'basic', 'basic', 'ismr', 'basic', 'plusminus', 'vel', 'vel']
    extend = ['neither', 'max', 'neither', 'max', 'min', 'neither', 'neither', 'neither', 'neither', 'both', 'neither', 'neither']
    include_shelf = [False, False, False, True, True, False, False, True, False, False, True, False]
    for j in range(len(var_names)):
        print(('Plotting ' + var_names[j]))
        is_vel = var_names[j] in ['vel', 'velice']
        for zoom_fris in [False, True]:
            if zoom_fris and not include_shelf[j]:
                continue
            if not zoom_fris and var_names[j] in ['bwtemp', 'bwsalt']:
                continue
            data = []
            if is_vel:
                u = []
                v = []
            vmin = 999
            vmax = -999
            for i in range(3):
                # Read data
                if is_vel:
                    data_tmp, u_tmp, v_tmp = read_and_process(var_names[j], base_dir+output_dir[i]+mit_file, return_vel_components=True)
                    data.append(data_tmp)
                    u.append(u_tmp)
                    v.append(v_tmp)
                else:
                    data.append(read_and_process(var_names[j], base_dir+output_dir[i]+mit_file))
                # Get min and max values and update global min/max as needed
                vmin_tmp, vmax_tmp = var_min_max(data[i], grid, zoom_fris=zoom_fris)
                vmin = min(vmin, vmin_tmp)
                vmax = max(vmax, vmax_tmp)
            # Overwrite with predetermined bounds if needed
            if vmin_impose[j] is not None:
                vmin = vmin_impose[j]
            if vmax_impose[j] is not None:
                vmax = vmax_impose[j]
            # Now make the plot
            figsize = None
            chunk = 10
            zoom_string = ''
            if zoom_fris:
                figsize = (12, 5)
                chunk = 6
                zoom_string = '_zoom'
            fig, gs, cax = set_panels('1x3C1', figsize=figsize)
            for i in range(3):
                ax = plt.subplot(gs[0,i])
                img = latlon_plot(data[i], grid, ax=ax, include_shelf=include_shelf[j], make_cbar=False, ctype=ctype[j], vmin=vmin, vmax=vmax, zoom_fris=zoom_fris, title=expt_names[i])
                if is_vel:
                    # Add velocity vectors
                    if var_names[j] == 'vel':
                        scale = 0.8
                    elif var_names[j] == 'velice':
                        scale = 4
                    overlay_vectors(ax, u[i], v[i], grid, chunk=chunk, scale=scale)
                if i > 0:
                    # Remove latitude labels
                    ax.set_yticklabels([])
            # Colourbar
            cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend=extend[j])
            # Main title
            plt.suptitle(titles[j] + ', 2008-2017', fontsize=22)
            finished_plot(fig, fig_name=fig_dir+var_names[j]+zoom_string+'.png')

    print('Plotting FRIS melt timeseries')
    times = []
    datas = []
    for i in range(3):
        # Read the timeseries file, cutting off the first 2 years
        file_path = base_dir + output_dir[i] + timeseries_file
        t_start = 2*10
        time = netcdf_time(file_path)[t_start:]
        times.append(time)
        melt = read_netcdf(file_path, 'fris_total_melt')[t_start:]
        freeze = read_netcdf(file_path, 'fris_total_freeze')[t_start:]
        datas.append(melt+freeze)
    # Make the plot
    timeseries_multi_plot(times, datas, expt_names, expt_colours, title='FRIS basal mass loss', units='Gt/y', fig_name=fig_dir+'timeseries_fris_melt.png')
