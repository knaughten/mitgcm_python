# Figures for IRF 2020 application

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..file_io import read_netcdf, netcdf_time
from ..timeseries import calc_annual_averages
from ..utils import moving_average, polar_stereo, select_bottom
from ..constants import deg_string
from ..plot_utils.windows import finished_plot, set_panels

def extract_geomip_westerlies ():

    directories = ['member1/', 'member4/', 'member8/']
    in_files = ['ssp245.nc', 'ssp585.nc', 'g6sulfur.nc', 'g6solar.nc']
    labels = ['low emissions', 'high emissions', 'high emissions + aerosol SRM', 'high emissions + space-based SRM']
    colours = ['black', 'red', 'blue', 'green']
    num_ens = len(directories)
    num_sim = len(in_files)

    times = []
    jet_lat_min = []
    jet_lat_max = []
    jet_lat_mean = []
    for fname in in_files:
        jet_lat_range = None
        for ens in range(num_ens):
            file_path = directories[ens] + fname
            time = netcdf_time(file_path)
            lat = read_netcdf(file_path, 'lat')        
            uas = np.mean(read_netcdf(file_path, 'uas'), axis=2)
            jet_jmax = np.argmax(uas, axis=1)
            jet_lat = np.empty(jet_jmax.shape)  
            for t in range(time.size):
                jet_lat[t] = lat[jet_jmax[t]]
            time, jet_lat = calc_annual_averages(time, jet_lat)
            jet_lat, time = moving_average(jet_lat, 5, time=time)
            if jet_lat_range is None:
                jet_lat_range = np.empty([num_ens, time.size])
            jet_lat_range[ens,:] = jet_lat
        times.append(np.array([t.year for t in time]))
        jet_lat_min.append(np.amin(jet_lat_range, axis=0))
        jet_lat_max.append(np.amax(jet_lat_range, axis=0))
        jet_lat_mean.append(np.mean(jet_lat_range, axis=0))

    fig, ax = plt.subplots()
    for n in range(num_sim):
        ax.fill_between(times[n], jet_lat_min[n], jet_lat_max[n], color=colours[n], alpha=0.15)
        ax.plot(times[n], jet_lat_mean[n], color=colours[n], label=labels[n], linewidth=1.5)
    plt.title('Impact of solar radiation management\non Southern Hemisphere westerly winds', fontsize=18)
    plt.xlabel('year', fontsize=14)
    plt.ylabel('jet latitude', fontsize=14)
    yticks = np.arange(-53, -50, 1)
    yticklabels = [np.str(-y)+deg_string+'S' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim([times[-1][0], times[-1][-1]])
    ax.set_xticks(np.arange(2030, 2100, 20))
    #ax.legend()
    finished_plot(fig, fig_name='geo_winds.png', dpi=300)


# Plot Paul's SO bottom temps versus observations
def bottom_temp_vs_obs (model_file='stateBtemp_avg.nc', obs_file='woa18_decav_t00_04.nc'):

    # Set spatial bounds (60S at opposite corners)
    corner_lon = np.array([-45, 135])
    corner_lat = np.array([-60, -60])
    corner_x, corner_y = polar_stereo(corner_lon, corner_lat)
    [xmin, xmax, ymin, ymax] = [corner_x[0], corner_y[0], corner_x[1], corner_y[1]]
    # Set colour bounds
    vmin = -3
    vmax = 3
    lev = np.linspace(vmin, vmax, num=30)

    # Read model data
    model_lon = read_netcdf(model_file, 'LONGITUDE')
    model_lat = read_netcdf(model_file, 'LATITUDE')
    model_temp = read_netcdf(model_file, 'BTEMP')
    # Apply land mask (filled with zeros)
    model_temp = np.ma.masked_where(model_temp==0, model_temp)
    # Convert coordinates to polar stereographic
    model_lon, model_lat = np.meshgrid(model_lon, model_lat)
    model_x, model_y = polar_stereo(model_lon, model_lat)

    # Read WOA data
    obs_lon = read_netcdf(obs_file, 'lon')
    obs_lat = read_netcdf(obs_file, 'lat')
    obs_temp_3d = np.squeeze(read_netcdf(obs_file, 't_an'))
    # Extract bottom temperatures: deepest unmasked values
    obs_temp = select_bottom(obs_temp_3d)
    # Convert coordinates to polar stereographic
    obs_lon, obs_lat = np.meshgrid(obs_lon, obs_lat)
    obs_x, obs_y = polar_stereo(obs_lon, obs_lat)

    # Plot
    fig, gs, cax = set_panels('1x2C1')
    ax = plt.subplot(gs[0,0])
    img = ax.contourf(model_x, model_y, model_temp, lev)
    ax.set_title('Existing model', fontsize=18)
    ax = plt.subplot(gs[0,1])
    img = ax.contourf(obs_x, obs_y, obs_temp, lev)
    ax.set_title('Observations', fontsize=18)
    cbar = plt.colorbar(img, cax=cax, extend='both', orientation='horizontal')
    plt.suptitle('Bottom temperatures ('+deg_string+'C)', fontsize=24)
    fig.show()
