# Postprocess CESM scenario data for Sebastian's ice sheet simulations
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..utils import real_dir, average_12_months
from ..calculus import area_average, derivative


# Global variables
shelves = ['pig', 'thwaites', 'dotson_crosson']
grid_dir = 'PAS_grid/'


# Given an experiment title and ensemble member, return the path to the directory.
def simulation_path (expt, ens, base_dir='./'):

    path = real_dir(base_dir) + 'PAS_'
    if expt == 'Paris 1.5C':
        path += 'LW1.5_'
    elif expt == 'Paris 2C':
        path += 'LW2.0_'
    elif expt == 'RCP 4.5':
        path += 'MENS_'
    elif expt == 'RCP 8.5':
        path += 'LENS'
    path += '_O/'
    return path


# Given an experiment title, ensemble member, and year, return the path to the NetCDF output file.
def output_year_path (expt, ens, year, base_dir='./'):

    return simulation_path(expt, ens, base_dir=base_dir) + '/output/' + str(year) + '01/MITgcm/output.nc'


# Given an ice shelf, year, scenario, and ensemble member, plot the annual mean temperature profile averaged over the ice front and its first and second derivatives with respect to depth.
def plot_sample_profiles (shelf, year, expt, ens, fig_name=None, base_dir='./', grid=None):

    # Select the output file
    file_path = output_year_path(expt, ens, year, base_dir=base_dir)
    if grid is None:
        grid = Grid(file_path)
    depth = -grid.z
        
    # Get a 3D mask of ocean cells at this ice front
    icefront_mask = grid.get_icefront_mask(shelf=shelf, is_3d=True)
    # Read temperature data for this year and annually average
    temp = average_12_months(read_netcdf(file_path, 'THETA'), calendar='noleap')
    # Mask everything except the ice front
    temp = np.ma.masked_where(np.invert(icefront_mask), temp)
    # Average over ice front to get profile
    temp = area_average(temp, grid)

    # Take first and second derivatives
    dtemp_dz = derivative(temp, depth)
    d2temp_dz2 = derivative(dtemp_dz, depth)

    # Plot
    fig = plt.figure(figsize=(8,5.5))
    gs = plt.GridSpec(3,1)
    gs.update(left=0.1, right=0.95, bottom=0.05, top=0.8, wspace=0.05)
    data_plot = [temp, dtemp_dz, d2temp_dz2]
    base_units = deg_string+'C'
    units = [base_units, base_units+'/m', base_units+'/m$^2$']
    titles = [r'$T$', r'$\partial T / \partial z$', r'$\partial^2 T / \partial z^2$']
    for n in range(3):
        ax = plt.subplot(gs[0,n])
        ax.plot(data_plot[n], depth, color='blue', linewidth=1.5)
        ax.grid(linestyle='dotted')
        ax.invert_yaxis()
        ax.set_title(titles[n], fontsize=14)
        ax.set_xlabel(units[n], fontsize=12)
        if n==0:
            ax.set_ylabel('depth (m)', fontsize=12)
        else:
            ax.set_yticklabels([])
    plt.suptitle(region_names[shelf] + ' front, '+expt+', '+str(year), fontsize=18)
    finished_plot(fig, fig_name=fig_name)
    
             
    

    

    
    
