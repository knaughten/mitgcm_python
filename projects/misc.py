##################################################################
# Analysis for random projects that don't fit into specific papers
##################################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..file_io import read_netcdf
from ..constants import deg2rad
from ..utils import fix_lon_range, split_longitude, real_dir
from ..plot_utils.windows import set_panels
from ..plot_utils.labels import latlon_axes
from ..plot_utils.colours import set_colours

# Figure out whether katabatic winds are stronger in ERA5 than in ERA-Interim, roughly following the methodology of Mathiot et al. 2010. Here we have 3 years of wind data (2008-2010) for each reanalysis, both interpolated to the same regular grid upon downloading, plus a land mask file.
def compare_katabatics (erai_file, era5_file, land_mask_file, fig_dir='./'):

    fig_dir = real_dir(fig_dir)

    # Longitude and latitude ranges we care about
    xmin = -85
    xmax = 30
    ymin = -84
    ymax = -61

    print 'Reading grid'
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
        data = np.flip(data, axis=0)
        data = split_longitude(data, i_split)
        data = data[j_beg:j_end, i_beg:i_end]
        if land is not None:
            data = np.ma.masked_where(land==1, data)
        return data

    # First do this for the land mask
    land = process_field(land_mask_file, 'lsm', time_dependent=False)
    # Now for all the wind fields
    print 'Reading data'
    erai_uwind = process_field(erai_file, 'u10', land=land)
    erai_vwind = process_field(erai_file, 'v10', land=land)
    era5_uwind = process_field(era5_file, 'u10', land=land)
    era5_vwind = process_field(era5_file, 'v10', land=land)

    print 'Calculating derived variables'
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
            cmap = set_colours(data[t], ctype=ctype[t])
            ax = plt.subplot(gs[0,t])
            img = ax.contourf(lon, lat, data[t], 50, cmap=cmap)
            latlon_axes(ax, lon, lat)
            plt.colorbar(img, cax=cax[t])
            plt.title(title[t], fontsize=18)
        plt.suptitle(suptitle, fontsize=22)
        finished_plot(fig, fig_name=fig_name)

    # Now call it for each variable
    print 'Plotting'
    plot_field(erai_uwind, era5_uwind, 'Zonal wind (m/s)', fig_dir+'uwind.png')
    plot_field(erai_vwind, era5_vwind, 'Meridional wind (m/s)', fig_dir+'vwind.png')
    plot_field(erai_speed, era5_speed, 'Wind speed (m/s)', fig_dir+'speed.png')
    plot_field(erai_angle, era5_angle, 'Wind angle (degrees)', fig_dir+'angle.png')    
    
    # Scatterplots: ERA-Interim vs ERA5 u and v at each coastal point (yearly or monthly means?), check slope of regression.
