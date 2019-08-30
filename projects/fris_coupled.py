# encoding: utf-8
##################################################################
# Plots for the coupled FRIS simulations
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import sys

from ..plot_ua import read_ua_mesh
from ..postprocess import get_segment_dir
from ..utils import real_dir, polar_stereo, choose_range
from ..grid import Grid
from ..plot_utils.latlon import cell_boundaries
from ..plot_utils.labels import latlon_axes
from ..plot_utils.windows import finished_plot
from ..plot_ua import gl_final
from ..plot_1d import read_plot_timeseries, make_timeseries_plot_2sided, timeseries_multi_plot
from ..file_io import netcdf_time, read_netcdf
from ..constants import deg_string


# Make a plot of the overlapping MITgcm grid and Ua mesh, at the beginning of the simulation.
def plot_domain_mesh (ua_mesh_file='ua_run/NewMeshFile.mat', output_dir='output/', fig_name=None):

    output_dir = real_dir(output_dir)

    # Read Ua mesh
    x_ua, y_ua, connectivity = read_ua_mesh(ua_mesh_file)

    # Read MIT grid
    segment_dir = get_segment_dir(output_dir)
    mit_file = output_dir+segment_dir[0]+'/MITgcm/output.nc'
    grid = Grid(mit_file)
    x_mit, y_mit = polar_stereo(grid.lon_2d, grid.lat_2d)
    # Get ocean mask to plot
    ocean_mask = np.ma.masked_where(grid.land_mask, np.invert(grid.land_mask))

    # Find bounds
    xmin, xmax = choose_range(x_ua, x2=x_mit, pad=0)
    ymin, ymax = choose_range(y_ua, x2=y_mit, pad=0)

    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('white')
    # Plot ocean cell boundaries
    x_plot, y_plot, mask_plot = cell_boundaries(ocean_mask, grid, pster=True)
    ax.pcolor(x_plot, y_plot, mask_plot, facecolor='none', edgecolor='blue', alpha=0.5)
    # Shade the ice sheet in red
    ax.triplot(x_ua, y_ua, connectivity, color='red', alpha=0.5)
    # Set axes limits
    latlon_axes(ax, x_ua, y_ua, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, pster=True)
    # Turn off box
    ax.axis('off')
    # Title
    plt.title(u'Initial MITgcm grid (blue) and Úa mesh (red)', fontsize=24)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Wrapper to gl_final: plot initial and final grounding line positions.
def plot_gl_change (ua_nc_file='output/ua_postprocessed.nc', fig_name=None):
    gl_final(ua_nc_file, fig_name=fig_name, dpi=300)


# Make timeseries plot of FRIS basal mass balance.
def plot_fris_mass_balance (timeseries_file='output/timeseries.nc', fig_name=None):
    read_plot_timeseries('fris_mass_balance', timeseries_file, precomputed=True, fig_name=fig_name, dpi=300)


# Make 2-sided timeseries plot of FRIS temperature and salinity.
def plot_fris_temp_salt (timeseries_file='output/timeseries.nc', fig_name=None):

    time = netcdf_time(timeseries_file)
    temp = read_netcdf(timeseries_file, 'fris_temp')
    salt = read_netcdf(timeseries_file, 'fris_salt')
    make_timeseries_plot_2sided(time, temp, salt, 'Volume-averaged conditions in FRIS cavity', 'Temperature ('+deg_string+')', 'Salinity (psu)', fig_name=fig_name, dpi=300)


# Plot timeseries of integrated ice sheet variables, as percentage anomalies from their initial values.
def plot_ice_changes (timeseries_file='output/timeseries.nc', ua_file='output/ua_postprocessed.nc', fig_name=None):

    # Get the dates from MITgcm timeseries
    time = netcdf_time(timeseries_file)
    # Now read ice sheet timeseries
    groundedArea = read_netcdf(ua_file, 'groundedArea')
    iceVolume = read_netcdf(ua_file, 'iceVolume')
    iceVAF = read_netcdf(ua_file, 'iceVAF')    
    # Trim the time array if needed (for simulation in progress)
    time = time[:iceVAF.size]

    # Convert to percent anomalies
    groundedArea = (groundedArea-groundedArea[0])/groundedArea[0]*100
    iceVolume = (iceVolume-iceVolume[0])/iceVolume[0]*100
    iceVAF = (iceVAF-iceVAF[0])/iceVAF[0]*100

    # Make the plot
    timeseries_multi_plot(time, [groundedArea, iceVolume, iceVAF], ['Grounded\nice area', 'Ice volume', 'Ice volume\nabove flotation'], ['green', 'blue', 'magenta'], title='Drift in integrated ice sheet variables', units='% change from initial value', fig_name=fig_name, dpi=300)
                                   
