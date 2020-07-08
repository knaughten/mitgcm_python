# Figures for IRF 2020 application

import numpy as np

from ..file_io import read_netcdf, netcdf_time
from ..plot_1d import timeseries_multi_plot
from ..timeseries import calc_annual_averages
from ..utils import moving_average

def extract_geomip_westerlies ():

    in_files = [ 'ssp245.nc', 'ssp585.nc', 'g6sulfur.nc', 'g6solar.nc'] #, 'g7cirrus.nc']
    labels = ['SSP2-45', 'SSP5-85', 'G6sulfur', 'G6solar'] #, 'G7cirrus']
    colours = ['blue', 'red', 'green', 'magenta'] #, 'cyan']

    jet_speed_all = []
    jet_lat_all = []
    time_all = []
    for fname in in_files:
        time = netcdf_time(fname)
        lat = read_netcdf(fname, 'lat')
        uas = np.mean(read_netcdf(fname, 'uas'), axis=2)
        jet_speed = np.amax(uas, axis=1)
        jet_jmax = np.argmax(uas, axis=1)
        jet_lat = np.empty(jet_jmax.shape)  
        for t in range(time.size):
            jet_lat[t] = lat[jet_jmax[t]]
        time, [jet_speed, jet_lat] = calc_annual_averages(time, [jet_speed, jet_lat])
        jet_speed, time = moving_average(jet_speed, 5, time=time)
        jet_lat = moving_average(jet_lat, 5)
        time_all.append(time)
        jet_speed_all.append(jet_speed)
        jet_lat_all.append(jet_lat)
    #timeseries_multi_plot(time_all, jet_speed_all, labels, colours, title='Westerly jet speed', units='m/s')
    timeseries_multi_plot(time_all, jet_lat_all, labels, colours, title='Westerly jet latitude', units='degrees', fig_name='westerlies.png')
