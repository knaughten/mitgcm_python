# Figures for IRF 2020 application

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..file_io import read_netcdf, netcdf_time
from ..timeseries import calc_annual_averages
from ..utils import moving_average
from ..constants import deg_string
from ..plot_utils.windows import finished_plot

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
