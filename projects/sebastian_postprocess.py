# Postprocess CESM scenario data for Sebastian's ice sheet simulations
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..utils import real_dir, average_12_months
from ..calculus import area_average, derivative
from ..file_io import read_netcdf, NCfile
from ..constants import deg_string, region_names
from ..plot_utils.windows import finished_plot


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
    elif expt in ['historical', 'RCP 8.5']:
        path += 'LENS'
    path += str(ens).zfill(3) + '_O/'
    return path


# Given an experiment title, ensemble member, and year, return the path to the NetCDF output file.
def output_year_path (expt, ens, year, base_dir='./'):

    return simulation_path(expt, ens, base_dir=base_dir) + 'output/' + str(year) + '01/MITgcm/output.nc'


# Given an ice shelf, year, scenario, and ensemble member, calculate the annual mean profile averaged over the ice front (default temperature but can also pass var_name='SALT').
def select_profile (shelf, year, expt, ens, grid, base_dir='./', var_name='THETA'):

    # Select the output file
    file_path = output_year_path(expt, ens, year, base_dir=base_dir)        
    # Get a 3D mask of ocean cells at this ice front
    icefront_mask = grid.get_icefront_mask(shelf=shelf, is_3d=True, side='ocean')
    # Read data for this year and annually average
    data = average_12_months(read_netcdf(file_path, var_name), calendar='noleap')
    # Mask everything except the ice front
    data = np.ma.masked_where(np.invert(icefront_mask), data)
    # Average over ice front to get profile
    return area_average(data, grid)


# Given a temperature profile and a corresponding salinity profile, extract the base of the thermocline and return temperature and salinity at that depth.
def extract_thermocline_base (temp, salt, grid, threshold=3e-3):

    depth = -grid.z
    dtemp_dz = derivative(temp, depth)
    # Select deepest depth at which temperature gradient exceeds threshold
    k0 = np.ma.where(np.abs(dtemp_dz) > threshold)[0][-1]
    return depth[k0], temp[k0], salt[k0]


# Similarly, extract the depth of the Winter Water core.
def extract_winter_water_core (temp, salt, grid):

    depth = -grid.z
    k0 = np.ma.argmin(temp)
    return depth[k0], temp[k0], salt[k0]


# Plot the profile and its first and second derivatives with respect to depth.
def plot_sample_profiles (shelf, year, expt, ens, fig_name=None, base_dir='./', grid=None):
    
    if grid is None:
        grid = Grid(base_dir + grid_dir)
    depth = -grid.z

    temp = select_profile(shelf, year, expt, ens, grid=grid, base_dir=base_dir)
    # Take first and second derivatives
    dtemp_dz = derivative(temp, depth)
    d2temp_dz2 = derivative(dtemp_dz, depth)
    salt = select_profile(shelf, year, expt, ens, grid=grid, base_dir=base_dir, var_name='SALT')
    # Extract base of thermocline and Winter Water core
    depth_tcb, temp_tcb, salt_tcb = extract_thermocline_base(temp, salt, grid)
    depth_ww, temp_ww, salt_ww = extract_winter_water_core(temp, salt, grid)

    # Plot
    fig = plt.figure(figsize=(8,5.5))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.8, wspace=0.05)
    data_plot = [temp, dtemp_dz, d2temp_dz2]
    base_units = deg_string+'C'
    units = [base_units, base_units+'/m', base_units+'/m$^2$']
    titles = [r'$T$', r'$\partial T / \partial z$', r'$\partial^2 T / \partial z^2$']
    for n in range(3):
        ax = plt.subplot(gs[0,n])
        ax.plot(data_plot[n], depth, color='blue', linewidth=1.5)
        if n > 0:
            ax.axvline(0, color='black', linewidth=1)
        ax.grid(linestyle='dotted')
        ax.axhline(depth_tcb, color='red', linewidth=1)
        ax.axhline(depth_ww, color='red', linewidth=1)
        if n == 0:
            ax.set_ylim([0, None])
            zlim_deep = ax.get_ylim()[-1]
            print('Thermocline base: '+str(depth_tcb)+'m with temp='+str(temp_tcb)+', salt='+str(salt_tcb))
            print('Winter Water core: '+str(depth_ww)+'m with temp='+str(temp_ww)+', salt='+str(salt_ww))
        else:
            ax.set_ylim([0, zlim_deep])
        ax.invert_yaxis()
        ax.set_title(titles[n], fontsize=14)
        ax.set_xlabel(units[n], fontsize=12)
        if n==0:
            ax.set_ylabel('depth (m)', fontsize=12)
        else:
            ax.set_yticklabels([])
    plt.suptitle(region_names[shelf] + ' front, '+expt+', '+str(year), fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Save a selection of profiles to a NetCDF file to give to Sebastian for thermocline definition testing.
def save_profile_collection (out_file, base_dir='./'):

    # Combination of each of the parameters to select a profile
    # shelves defined above
    # Always use ensemble member 1
    years = np.arange(1990, 2100+1, 20)
    expts = ['Paris 1.5C', 'Paris 2C', 'RCP 4.5', 'RCP 8.5']
    ens = 1

    grid = Grid(base_dir+grid_dir)
    ncfile = NCfile(out_file, grid, 'zt')

    n = 0
    for shelf in shelves:
        for year in years:
            for expt in expts:
                if year < 2006 and expt != 'RCP 8.5':
                    continue
                if year > 2080 and expt == 'RCP 4.5':
                    continue
                print('Profile '+str(n+1))
                temp = select_profile(shelf, year, expt, ens, grid, base_dir=base_dir)
                if n == 0:
                    ncfile.add_time([n+1], units='profile_number')
                    ncfile.add_variable('temperature', temp[None,:], 'zt', units='degC')
                else:
                    ncfile.id.variables['time'][n:] = [n+1]
                    ncfile.id.variables['temperature'][n:] = temp[None,:]
                n += 1
    ncfile.close()


# Calculate all timeseries for a given simulation and save to a NetCDF file.
def process_timeseries (expt, ens, out_file, base_dir='./'):

    if expt == 'historical':
        start_year = 1920
        end_year = 2005
    else:
        start_year = 2006
        if expt == 'RCP 4.5':
            end_year = 2080
        else:
            end_year = 2100
    num_years = end_year-start_year+1
    num_shelves = len(shelves)
    depth_tcb = np.ma.empty([num_shelves, num_years])
    temp_tcb = np.ma.empty([num_shelves, num_years])
    salt_tcb = np.ma.empty([num_shelves, num_years])
    depth_ww = np.ma.empty([num_shelves, num_years])
    temp_ww = np.ma.empty([num_shelves, num_years])
    salt_ww = np.ma.empty([num_shelves, num_years])
    grid = Grid(base_dir+grid_dir)
    icefront_masks = []
    for shelf in shelves:
        icefront_masks.append(grid.get_icefront_mask(shelf=shelf, is_3d=True, side='ocean'))

    # Loop over years
    for t in range(num_years):
        print('...'+str(start_year+t))
        # Read and annually average this year's data
        file_path = output_year_path(expt, ens, start_year+t, base_dir=base_dir)
        temp = average_12_months(read_netcdf(file_path, 'THETA'), calendar='noleap')
        salt = average_12_months(read_netcdf(file_path, 'SALT'), calendar='noleap')
        # Loop over ice shelves
        for n in range(num_shelves):
            # Mask everything except the ice front and area-average to get profile
            temp_profile = area_average(np.ma.masked_where(np.invert(icefront_masks[n]), temp), grid)
            salt_profile = area_average(np.ma.masked_where(np.invert(icefront_masks[n]), salt), grid)
            depth_tcb[n,t], temp_tcb[n,t], salt_tcb[n,t] = extract_thermocline_base(temp_profile, salt_profile, grid)
            depth_ww[n,t], temp_ww[n,t], salt_ww[n,t] = extract_winter_water_core(temp_profile, salt_profile, grid)

    # Now save all the data
    ncfile = NCfile(out_file, grid, 't')
    ncfile.add_time(np.arange(start_year,end_year+1), units='year')
    for n in range(num_shelves):
        ncfile.add_variable('depth_thermocline_base_'+shelves[n], depth_tcb[n,:], 't', units='m')
        ncfile.add_variable('temp_thermocline_base_'+shelves[n], temp_tcb[n,:], 't', units='degC')
        ncfile.add_variable('salt_thermocline_base_'+shelves[n], salt_tcb[n,:], 't', units='psu')
        ncfile.add_variable('depth_winter_water_'+shelves[n], depth_ww[n,:], 't', units='m')
        ncfile.add_variable('temp_winter_water_'+shelves[n], temp_ww[n,:], 't', units='degC')
        ncfile.add_variable('salt_winter_water_'+shelves[n], salt_ww[n,:], 't', units='psu')
    ncfile.close()


# Calculate all timeseries for all simulations.
def process_all_timeseries (base_dir='./'):

    expt_names = ['historical', 'Paris 1.5C', 'Paris 2C', 'RCP 4.5', 'RCP 8.5']
    expt_codes = ['historical', 'paris1.5C', 'paris2C', 'rcp45', 'rcp85']
    num_ens = [10, 5, 10, 10, 10]

    for n in range(len(expt_names)):
        for e in range(1, num_ens[n]+1):
            out_file = base_dir + expt_codes[n] + '_ens' + str(e).zfill(2) + '.nc'
            print('Processing '+out_file)
            process_timeseries(expt_names[n], e, out_file, base_dir=base_dir)
            
        
        

    

    

    


    
    

    

    
    
