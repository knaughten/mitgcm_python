# Postprocess CESM scenario data for Sebastian's ice sheet simulations
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ..grid import Grid
from ..utils import real_dir, average_12_months, convert_ismr, mask_except_ice, polar_stereo
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


# Interpolate onto a finer vertical grid
def refine_dz (data, grid, dz=10, fill_value=999):

    depth_old = -grid.z
    depth_new = np.arange(depth_old[0], depth_old[-1], dz)
    valid_mask = np.invert(data.mask)
    interpolant = interp1d(depth_old[valid_mask], data[valid_mask], kind='slinear', bounds_error=False, fill_value=fill_value)
    data_new = interpolant(depth_new)
    data_new = np.ma.masked_where(data_new==fill_value, data_new)
    return depth_new, data_new


# Extract the depth of the Winter Water core.
def extract_winter_water_core (temp, salt, grid, depth=None):

    if depth is None:
        depth, temp = refine_dz(temp, grid)
        salt = refine_dz(salt, grid)[1]
    # Mask out the bottom third of the water column - removes cases where there's a cold blob at the bottom
    bottom_depth = depth[~temp.mask][-1]
    temp = np.ma.masked_where(depth > bottom_depth*2/3., temp)
    k0 = np.ma.argmin(temp)
    return depth[k0], temp[k0], salt[k0]


# Given a temperature profile and a corresponding salinity profile, extract the base of the thermocline and return temperature and salinity at that depth.
def extract_thermocline_base (temp, salt, grid, depth=None, threshold=3e-3):

    if depth is None:
        depth, temp = refine_dz(temp, grid)
        salt = refine_dz(salt, grid)[1]
    dtemp_dz = derivative(temp, depth)
    # Mask everything above the Winter Water core - removes weird temperature inversions near surface which happen occasionally.
    depth_ww = extract_winter_water_core(temp, salt, grid, depth=depth)[0]
    dtemp_dz[depth <= depth_ww] = 0
    # Also mask the bottom fifth of the water column - removes cases where there's a warm blob at the bottom
    bottom_depth = depth[~temp.mask][-1]
    dtemp_dz[depth > bottom_depth*4/5.] = 0
    # Select deepest depth at which temperature gradient exceeds threshold - this will select for (slow) warming with depth and disregard the case of temperature inversion at seafloor.
    try:
        k0 = np.ma.where(dtemp_dz > threshold)[0][-1]
    except(IndexError):
        print('Warning: trying smaller threshold of '+str(threshold/2))
        return extract_thermocline_base(temp, salt, grid, depth=depth, threshold=threshold/2)
    return depth[k0], temp[k0], salt[k0]


# Plot the profile and its first and second derivatives with respect to depth.
def plot_sample_profiles (shelf, year, expt, ens, fig_name=None, base_dir='./', grid=None):
    
    if grid is None:
        grid = Grid(base_dir + grid_dir)

    temp = select_profile(shelf, year, expt, ens, grid=grid, base_dir=base_dir)
    depth, temp = refine_dz(temp, grid)
    # Take first and second derivatives
    dtemp_dz = derivative(temp, depth)
    d2temp_dz2 = derivative(dtemp_dz, depth)
    salt = select_profile(shelf, year, expt, ens, grid=grid, base_dir=base_dir, var_name='SALT')
    salt = refine_dz(salt, grid)[1]
    # Extract base of thermocline and Winter Water core
    try:
        depth_tcb, temp_tcb, salt_tcb = extract_thermocline_base(temp, salt, grid, depth=depth)
    except(IndexError, RecursionError):
        depth_tcb = None
        temp_tcb = None
        salt_tcb = None
    depth_ww, temp_ww, salt_ww = extract_winter_water_core(temp, salt, grid, depth=depth)

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
        if depth_tcb is not None:
            ax.axhline(depth_tcb, color='red', linewidth=1)
        ax.axhline(depth_ww, color='red', linewidth=1)
        if n == 0:
            ax.set_ylim([0, None])
            zlim_deep = ax.get_ylim()[-1]
            if depth_tcb is not None:
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
        start_year = 1995 #1920
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
            depth, temp_profile = refine_dz(temp_profile, grid)
            salt_profile = refine_dz(salt_profile, grid)[1]
            depth_tcb[n,t], temp_tcb[n,t], salt_tcb[n,t] = extract_thermocline_base(temp_profile, salt_profile, grid, depth=depth)
            depth_ww[n,t], temp_ww[n,t], salt_ww[n,t] = extract_winter_water_core(temp_profile, salt_profile, grid, depth=depth)

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
def process_all_timeseries (base_dir='./', out_dir='data_for_sebastian/'):

    expt_names = ['historical', 'Paris 1.5C', 'Paris 2C', 'RCP 4.5', 'RCP 8.5']
    expt_codes = ['historical', 'paris1.5C', 'paris2C', 'rcp45', 'rcp85']
    num_ens = [10, 5, 10, 10, 10]

    for n in range(len(expt_names)):
        for e in range(1, num_ens[n]+1):
            out_file = out_dir + 'timeseries_' + expt_codes[n] + '_ens' + str(e).zfill(2) + '.nc'
            print('Processing '+out_file)
            process_timeseries(expt_names[n], e, out_file, base_dir=base_dir)


# Extract annual-mean ice shelf melt rates fields for a given simulation and save to a NetCDF file.
def extract_ismr (expt, ens, out_file, base_dir='./'):

    if expt == 'historical':
        start_year = 1995
        end_year = 2005
    else:
        # Just do 20 years
        start_year = 2006
        end_year = 2025
    num_years = end_year-start_year+1
    grid = Grid(base_dir+grid_dir)
    ismr = np.ma.empty([num_years, grid.ny, grid.nx])

    for t in range(num_years):
        print('...'+str(start_year+t))
        file_path = output_year_path(expt, ens, start_year+t, base_dir=base_dir)
        ismr[t,:] = mask_except_ice(convert_ismr(average_12_months(read_netcdf(file_path, 'SHIfwFlx'), calendar='noleap')), grid)
    ncfile = NCfile(out_file, grid, 'xyt')
    ncfile.add_time(np.arange(start_year, end_year+1), units='year')
    ncfile.add_variable('basal_melt_rate', ismr, 'xyt', units='m/y')
    ncfile.close()


# Do this for all simulations
def extract_all_ismr (base_dir='./', out_dir='data_for_sebastian/'):

    expt_names = ['historical', 'Paris 1.5C', 'Paris 2C', 'RCP 4.5', 'RCP 8.5']
    expt_codes = ['historical', 'paris1.5C', 'paris2C', 'rcp45', 'rcp85']
    num_ens = [10, 5, 10, 10, 10]

    for n in range(len(expt_names)):
        for e in range(1, num_ens[n]+1):
            out_file = out_dir + 'basal_melting_' + expt_codes[n] + '_ens' + str(e).zfill(2) + '.nc'
            print('Processing '+out_file)
            extract_ismr(expt_names[n], e, out_file, base_dir=base_dir)


# Save the geometry to a file for Sebastian to use
def save_geometry (out_file, base_dir='./'):

    grid = Grid(base_dir+grid_dir)
    x, y = polar_stereo(grid.lon_2d, grid.lat_2d, lat_c=-70)
    ncfile = NCfile(out_file, grid, 'xy')
    ncfile.add_variable('bathymetry', grid.bathy, 'xy', units='m')
    ncfile.add_variable('draft',  grid.draft, 'xy', units='m')
    ncfile.add_variable('x', x, 'xy', units='m')
    ncfile.add_variable('y', y, 'xy', units='m')
    ncfile.close()
    
            
        
        

    

    

    


    
    

    

    
    
