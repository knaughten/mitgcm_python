###########################################################
# Generate atmospheric forcing.
###########################################################

import numpy as np
import sys
import os

from grid import Grid, SOSEGrid, grid_check_split, choose_grid
from file_io import read_netcdf, write_binary, NCfile, netcdf_time, read_binary, find_cmip6_files
from utils import real_dir, fix_lon_range, mask_land_ice, ice_shelf_front_points, dist_btw_points
from interpolation import interp_nonreg_xy, interp_reg, extend_into_mask, discard_and_fill, smooth_xy, interp_slice_helper
from constants import temp_C2K, Lv, Rv, es0, sh_coeff, rho_fw
from calculus import area_integral
from plot_latlon import latlon_plot

# Interpolate the freshwater flux from iceberg melting (monthly climatology from NEMO G07 simulations) to the model grid so it can be used for runoff forcing.

# Arguments:
# grid_path: path to directory containing MITgcm binary grid files
# input_dir: path to directory with iceberg data
# output_file: desired path to binary output file which MITgcm will read

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the interpolated data in, so you can easily check that it looks okay. (The annual mean will also be plotted and shown on screen whether or not you define nc_out.)
# prec: precision to write output_file. Must match exf_iprec in the "data.exf" namelist (default 32)
def iceberg_meltwater (grid_path, input_dir, output_file, nc_out=None, prec=32):

    input_dir = real_dir(input_dir)
    file_head = 'icebergs_'
    file_tail = '.nc'

    print 'Building grids'
    # Read the NEMO grid from the first file
    # It has longitude in the range -180 to 180
    file_path = input_dir + file_head + '01' + file_tail
    nemo_lon = read_netcdf(file_path, 'nav_lon')
    nemo_lat = read_netcdf(file_path, 'nav_lat')
    # Build the model grid
    model_grid = Grid(grid_path, max_lon=180)

    print 'Interpolating'
    icebergs_interp = np.zeros([12, model_grid.ny, model_grid.nx])    
    for month in range(12):
        print '...month ' + str(month+1)
        # Read the data
        file_path = input_dir + file_head + '{0:02d}'.format(month+1) + file_tail
        icebergs = read_netcdf(file_path, 'berg_total_melt', time_index=0)
        # Interpolate
        icebergs_interp_tmp = interp_nonreg_xy(nemo_lon, nemo_lat, icebergs, model_grid.lon_1d, model_grid.lat_1d, fill_value=0)
        # Make sure the land and ice shelf cavities don't get any iceberg melt
        icebergs_interp_tmp[model_grid.land_mask+model_grid.ice_mask] = 0
        # Save to the master array
        icebergs_interp[month,:] = icebergs_interp_tmp    

    write_binary(icebergs_interp, output_file, prec=prec)

    print 'Plotting'
    # Make a nice plot of the annual mean
    latlon_plot(mask_land_ice(np.mean(icebergs_interp, axis=0), model_grid), model_grid, include_shelf=False, vmin=0, title=r'Annual mean iceberg melt (kg/m$^2$/s)')                
    if nc_out is not None:
        # Also write to NetCDF file
        print 'Writing ' + nc_out
        ncfile = NCfile(nc_out, model_grid, 'xyt')
        ncfile.add_time(np.arange(12)+1, units='months')
        ncfile.add_variable('iceberg_melt', icebergs_interp, 'xyt', units='kg/m^2/s')
        ncfile.close()
        

# Set up surface salinity restoring using a monthly climatology interpolated from SOSE. Don't restore on the continental shelf.

# Arguments:
# grid_path: path to directory containing MITgcm binary grid files
# sose_dir: directory containing SOSE monthly climatologies and grid/ subdirectory (available on Scihub at /data/oceans_input/raw_input_data/SOSE_monthly_climatology)
# output_salt_file: desired path to binary file containing salinity values to restore
# output_mask_file: desired path to binary file containing restoring mask

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the salinity and mask in, so you can easily check that they look okay
# h0: threshold bathymetry (negative, in metres) for definition of continental shelf; everything shallower than this will not be restored. Default -1250 (excludes Maud Rise but keeps Filchner Trough).
# obcs_sponge: width of the OBCS sponge layer - no need to restore in that region
# split: as in function sose_ics
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def sose_sss_restoring (grid_path, sose_dir, output_salt_file, output_mask_file, nc_out=None, h0=-1250, obcs_sponge=0, split=180, prec=64):

    sose_dir = real_dir(sose_dir)

    print 'Building grids'
    # First build the model grid and check that we have the right value for split
    model_grid = grid_check_split(grid_path, split)
    # Now build the SOSE grid
    sose_grid = SOSEGrid(sose_dir+'grid/', model_grid=model_grid, split=split)
    # Extract surface land mask
    sose_mask = sose_grid.hfac[0,:] == 0

    print 'Building mask'
    mask_surface = np.ones([model_grid.ny, model_grid.nx])
    # Mask out land and ice shelves
    mask_surface[model_grid.hfac[0,:]==0] = 0
    # Save this for later
    mask_land_ice = np.copy(mask_surface)
    # Mask out continental shelf
    mask_surface[model_grid.bathy > h0] = 0
    # Smooth, and remask the land and ice shelves
    mask_surface = smooth_xy(mask_surface, sigma=2)*mask_land_ice
    if obcs_sponge > 0:
        # Also mask the cells affected by OBCS and/or its sponge
        mask_surface[:obcs_sponge,:] = 0
        mask_surface[-obcs_sponge:,:] = 0
        mask_surface[:,:obcs_sponge] = 0
        mask_surface[:,-obcs_sponge:] = 0
    # Make a 3D version with zeros in deeper layers
    mask_3d = np.zeros([model_grid.nz, model_grid.ny, model_grid.nx])
    mask_3d[0,:] = mask_surface
    
    print 'Reading SOSE salinity'
    # Just keep the surface layer
    sose_sss = sose_grid.read_field(sose_dir+'SALT_climatology.data', 'xyzt')[:,0,:,:]
    
    # Figure out which SOSE points we need for interpolation
    # Restoring mask interpolated to the SOSE grid
    fill = np.ceil(interp_reg(model_grid, sose_grid, mask_3d[0,:], dim=2, fill_value=1))
    # Extend into the mask a few times to make sure there are no artifacts near the coast
    fill = extend_into_mask(fill, missing_val=0, num_iters=3)

    # Process one month at a time
    sss_interp = np.zeros([12, model_grid.nz, model_grid.ny, model_grid.nx])
    for month in range(12):
        print 'Month ' + str(month+1)
        print '...filling missing values'
        sose_sss_filled = discard_and_fill(sose_sss[month,:], sose_mask, fill, use_3d=False)
        print '...interpolating'
        # Mask out land and ice shelves
        sss_interp[month,0,:] = interp_reg(sose_grid, model_grid, sose_sss_filled, dim=2)*mask_land_ice

    write_binary(sss_interp, output_salt_file, prec=prec)
    write_binary(mask_3d, output_mask_file, prec=prec)

    if nc_out is not None:
        print 'Writing ' + nc_out
        ncfile = NCfile(nc_out, model_grid, 'xyzt')
        ncfile.add_time(np.arange(12)+1, units='months')
        ncfile.add_variable('salinity', sss_interp, 'xyzt', units='psu')
        ncfile.add_variable('restoring_mask', mask_3d, 'xyz')
        ncfile.close()


# Convert one year of ERA5 data to the format and units required by MITgcm.
def process_era5 (in_dir, out_dir, year, six_hourly=True, first_year=False, last_year=False, prec=32):

    in_dir = real_dir(in_dir)
    out_dir = real_dir(out_dir)

    if year == 1979 and not first_year:
        print 'Warning (process_era): last we checked, 1979 was the first year of ERA5. Unless this has changed, you need to set first_year=True.'
    if year == 2018 and not last_year:
        print 'Warning (process_era): last we checked, 2018 was the last year of ERA5. Unless this has changed, you need to set last_year=True.'

    # Construct file paths for input and output files
    in_head = in_dir + 'era5_'
    var_in = ['msl', 't2m', 'd2m', 'u10', 'v10', 'tp', 'ssrd', 'strd', 'e']
    if six_hourly:
        accum_flag = '_2'
    in_tail = '_' + str(year) + '.nc'
    out_head = out_dir + 'ERA5_'
    var_out = ['apressure', 'atemp', 'aqh', 'uwind', 'vwind', 'precip', 'swdown', 'lwdown', 'evap']
    out_tail = '_' + str(year)

    # Northermost latitude to keep
    lat0 = -30
    # Length of ERA5 time interval in seconds
    dt = 3600.

    # Read the grid from the first file
    first_file = in_head + var_in[0] + in_tail
    lon = read_netcdf(first_file, 'longitude')
    lat = read_netcdf(first_file, 'latitude')
    # Find the index of the last latitude we don't care about (remember that latitude goes from north to south in ERA files!)
    j_bound = np.nonzero(lat < lat0)[0][0] - 2
    # Trim and flip latitude
    lat = lat[:j_bound:-1]
    # Also read the first time index for the starting date
    start_date = netcdf_time(first_file, monthly=False)[0]

    if first_year:
        # Print grid information to the reader
        print '\n'
        print 'For var in ' + str(var_out) + ', make these changes in input/data.exf:\n'
        print 'varstartdate1 = ' + start_date.strftime('%Y%m%d')
        if six_hourly:
            print 'varperiod = ' + str(6*dt)
        else:
            print 'varperiod = ' + str(dt)
        print 'varfile = ' + 'ERA5_var'
        print 'var_lon0 = ' + str(lon[0])
        print 'var_lon_inc = ' + str(lon[1]-lon[0])
        print 'var_lat0 = ' + str(lat[0])
        print 'var_lat_inc = ' + str(lat.size-1) + '*' + str(lat[1]-lat[0])
        print 'var_nlon = ' + str(lon.size)
        print 'var_nlat = ' + str(lat.size)
        print '\n'

    # Loop over variables
    for i in range(len(var_in)):
        
        in_file = in_head + var_in[i] + in_tail
        print 'Reading ' + in_file
        data = read_netcdf(in_file, var_in[i])
        
        print 'Processing'
        # Trim and flip over latitude
        data = data[:,:j_bound:-1,:]
        
        if var_in[i] == 'msl':
            # Save pressure for later conversions
            press = np.copy(data)

        elif var_in[i] == 't2m':
            # Convert from Kelvin to Celsius
            data -= temp_C2K

        elif var_in[i] == 'd2m':
            # Calculate specific humidity from dew point temperature and pressure
            # Start with vapour pressure
            e = es0*np.exp(Lv/Rv*(1/temp_C2K - 1/data))
            data = sh_coeff*e/(press - (1-sh_coeff)*e)
            
        elif var_in[i] in ['tp', 'ssrd', 'strd', 'e']:
            # Accumulated variables
            # This is more complicated
            
            if six_hourly:
                # Need to read data from the following hour to interpolate to this hour. This was downloaded into separate files.
                in_file_2 = in_head + var_in[i] + accum_flag + in_tail
                print 'Reading ' + in_file_2
                data_2 = read_netcdf(in_file_2, var_in[i])
                data_2 = data_2[:,:j_bound:-1,:]
            # not six_hourly will be dealt with after the first_year check
            
            if first_year:
                # The first 7 hours of the accumulated variables are missing during the first year of ERA5. Fill this missing period with data from the next available time indices.
                if six_hourly:
                    # The first file is missing two indices (hours 0 and 6)
                    data = np.concatenate((data[:2,:], data), axis=0)
                    # The second file is missing one index (hour 1)
                    data_2 = np.concatenate((data_2[:1,:], data_2), axis=0)
                else:
                    # The first file is missing 7 indices (hours 0 to 6)
                    data = np.concatenate((data[:7,:], data), axis=0)
                    
            if not six_hourly:
                # Now get data from the following hour. Just shift one timestep ahead.
                # First need data from the first hour of next year
                if last_year:
                    # There is no such data; just copy the last hour of this year
                    data_next = data[-1,:]
                else:
                    in_file_2 = in_head + var_in[i] + '_' + str(year+1) + '.nc'
                    data_next = read_netcdf(in_file_2, var_in[i], time_index=0)
                    data_next = data_next[:j_bound:-1,:]  
                data_2 = np.concatenate((data[1:,:], np.expand_dims(data_next,0)), axis=0)
                
            # Now we can interpolate to the given hour: just the mean of either side
            data = 0.5*(data + data_2)
            # Convert from integrals to time-averages
            data /= dt
            if var_in[i] in ['ssrd', 'strd', 'e']:
                # Swap sign on fluxes
                data *= -1

        out_file = out_head + var_out[i] + out_tail
        write_binary(data, out_file, prec=prec)


# If you run a simulation that goes until the end of the ERA-Interim or ERA5 record (eg 2017), it will die right before the end, because it needs the first time index of the next year (eg 2018) as an endpoint for interpolation.
# To avoid this error, copy the last time index of the last year of data to a new file named correctly for the next year. So the model will just hold the atmospheric forcing constant for the last forcing step of the simulation.

# Arguments:
# bin_dir: path to directory containing ERA-Interim or ERA5 binary files
# last_year: last year of data (eg 2017)

# Optional keyword arguments:
# option: 'era5' (default) or 'eraint'; this will determine the file naming convention
# nlon, nlat: grid dimensions. Default is 1440x241 for ERA5 (which was cut off at 30S), and 480x241 for ERA-Interim.
# out_dir: if set, will write the new file to this directory instead of bin_dir (useful if you don't have write access for bin_dir).
# prec: precision of binary files; must match exf_iprec in input/data.exf
def era_dummy_year (bin_dir, last_year, option='era5', nlon=None, nlat=None, out_dir=None, prec=32):

    bin_dir = real_dir(bin_dir)
    if out_dir is None:
        out_dir = bin_dir
        
    if nlon is None:
        if option == 'era5':
            nlon = 1440
        elif option == 'eraint':
            nlon = 480
        else:
            print 'Error (era_dummy_year): invalid option ' + option
            sys.exit()
    if nlat is None:
        # The same for both cases, assuming ERA5 was cut off at 30S
        nlat = 241

    # Figure out the file paths
    if option == 'era5':
        var_names = ['apressure', 'atemp', 'aqh', 'uwind', 'vwind', 'precip', 'swdown', 'lwdown', 'evap']
        file_head = 'ERA5_'
    elif option == 'eraint':
        var_names = ['msl', 'tmp2m_degC', 'spfh2m', 'u10m', 'v10m', 'rain', 'dsw', 'dlw']
        file_head = 'ERAinterim_'            

    for var in var_names:
        file_in = bin_dir + file_head + var + '_' + str(last_year)
        # Select the last time index
        data = read_binary(file_in, [nlon, nlat], 'xyt', prec=prec)[-1,:]
        file_out = out_dir + file_head + var + '_' + str(last_year+1)
        write_binary(data, file_out, prec=prec)


# Recalculate ERA-Interim humidity: the original Matlab scripts did the conversion using a reference temperature instead of actual temperature. Also make a dummy last year as above.
# WARNING WARNING THIS IS WRONG THE ORIGINAL WAY WAS RIGHT
def fix_eraint_humidity (in_dir, out_dir, prec=32):

    in_dir = real_dir(in_dir)
    out_dir = real_dir(out_dir)

    # File paths
    in_head = in_dir + 'era_a_'
    in_tail = '_075.nc'
    out_head = out_dir + 'ERAinterim_spfh2m_'
    start_year = 1979
    end_year = 2017

    for year in range(start_year, end_year+1):
        in_file = in_head + str(year) + in_tail
        print 'Reading ' + in_file
        # Need temperature, pressure, and dew point
        temp = read_netcdf(in_file, 't2m')
        press = read_netcdf(in_file, 'msl')
        dewpoint = read_netcdf(in_file, 'd2m')
        # Calculate vapour pressure
        e = es0*np.exp(Lv/Rv*(1/temp - 1/dewpoint))
        # Calculate specific humidity
        spf = sh_coeff*e/(press - (1-sh_coeff)*e)
        # Now flip in latitude to match Matlab-generated files
        spf = spf[:,::-1,:]
        out_file = out_head + str(year)
        write_binary(spf, out_file, prec=prec)
        if year == end_year:
            # Copy the last timestep as in era_dummy_year
            spf_last = spf[-1,:]
            out_file = out_head + str(year+1)
            write_binary(spf_last, out_file, prec=prec)


# Create a mask file to impose polynyas (frcConvMaskFile in input/data.kpp, also switch on useFrcConv in input/data.kpp and define ALLOW_FORCED_CONVECTION in KPP_OPTIONS.h). The mask will be 1 in the polynya region which will tell KPP to mix all the way down to the bottom there.
# The argument "polynya" is a key determining the centre and radii of the ellipse bounding the polynya. Current options are 'maud_rise', 'near_shelf', 'maud_rise_big', and 'maud_rise_small'.
def polynya_mask (grid_path, polynya, mask_file, prec=64):

    from plot_latlon import latlon_plot

    # Define the centre and radii of the ellipse bounding the polynya
    if polynya == 'maud_rise':  # Area 2.6 x 10^5 km^2
        lon0 = 0.
        lat0 = -65.
        rlon = 8.
        rlat = 2.
    elif polynya == 'near_shelf':  # Area 2.6 x 10^5 km^2
        lon0 = -30.
        lat0 = -70.
        rlon = 9.
        rlat = 2.2
    elif polynya == 'maud_rise_big':  # Area 6.2 x 10^5 km^2
        lon0 = 0.
        lat0 = -65.
        rlon = 15.
        rlat = 2.5
    elif polynya == 'maud_rise_small':  # Area 0.34 x 10^5 km^2
        lon0 = 0
        lat0 = -65.
        rlon = 2.8
        rlat = 0.75
    else:
        print 'Error (polynya_mask): invalid polynya option ' + polynya
        sys.exit()

    # Build the grid
    grid = Grid(grid_path)
    # Set up the mask
    mask = np.zeros([grid.ny, grid.nx])
    # Select the polynya region
    index = (grid.lon_2d - lon0)**2/rlon**2 + (grid.lat_2d - lat0)**2/rlat**2 <= 1
    mask[index] = 1

    # Print the area of the polynya
    print 'Polynya area is ' + str(area_integral(mask, grid)*1e-6) + ' km^2'
    # Plot the mask
    latlon_plot(mask_land_ice(mask, grid), grid, include_shelf=False, title='Polynya mask', figsize=(10,6))

    # Write to file
    write_binary(mask, mask_file, prec=prec)


# Create a file with scaling factors for atmosphere/sea-ice drag in each cell (SEAICE_scaleDragFile in input/data.seaice; also switch on SEAICE_scaleDrag). The value of SEAICE_drag will be multiplied by the scaling factor in each cell.
# The arguments rd_scale, bb_scale, and ft_scale are the scaling factors to set over Ronne Depression, Berkner Bank, and Filchner Trough respectively. They must be positive. The code will smooth the mask so there are no sharp boundaries in the scaling.
# Settings from UKESM correction: rd_scale = ft_scale = 2.5, bb_scale = 0.5
def seaice_drag_scaling (grid_path, output_file, rd_scale=1, bb_scale=1, ft_scale=1, prec=64):

    # Cutoff latitude
    max_lat = -74
    # Longitude bounds on each region
    rd_bounds = [-62, -58]
    bb_bounds = [-57, -49]
    ft_bounds = [-48, -35]
    bounds = [rd_bounds, bb_bounds, ft_bounds]
    scale_factors = [rd_scale, bb_scale, ft_scale]
    # Max distance from the coast (km)
    scale_dist = 150
    # Sigma for smoothing
    sigma = 2

    print 'Building grid'
    grid = Grid(grid_path)
    print 'Selecting coastal points'
    coast_mask = grid.get_coast_mask(ignore_iceberg=True)
    lon_coast = grid.lon_2d[coast_mask].ravel()
    lat_coast = grid.lat_2d[coast_mask].ravel()

    print 'Selecting regions'
    scale_coast = np.ones(lon_coast.shape)
    for n in range(3):
        index = (lon_coast >= bounds[n][0])*(lon_coast <= bounds[n][1])*(lat_coast <= max_lat)
        scale_coast[index] = scale_factors[n]

    print 'Calculating distance from the coast'
    min_dist = None
    nearest_scale = None
    # Loop over all the coastal points
    for i in range(lon_coast.size):
        # Calculate distance of every point in the model grid to this specific coastal point, in km
        dist_to_pt = dist_btw_points([lon_coast[i], lat_coast[i]], [grid.lon_2d, grid.lat_2d])*1e-3
        if min_dist is None:
            # Initialise the arrays
            min_dist = dist_to_pt
            nearest_scale = np.zeros(min_dist.shape) + scale_coast[i]
        else:
            # Figure out which cells have this coastal point as the closest one yet, and update the arrays
            index = dist_to_pt < min_dist
            min_dist[index] = dist_to_pt[index]
            nearest_scale[index] = scale_coast[i]
    # Smooth the result, and mask out the land and ice shelves
    min_dist = mask_land_ice(min_dist, grid)
    nearest_scale = mask_land_ice(smooth_xy(nearest_scale, sigma=sigma), grid)

    print 'Extending scale factors offshore'
    # Cosine function moving from scaling factor to 1 over distance of 300 km offshore
    scale_extend = (min_dist < scale_dist)*(nearest_scale - 1)*np.cos(np.pi/2*min_dist/scale_dist) + 1

    print 'Plotting'
    latlon_plot(scale_extend, grid, ctype='ratio', include_shelf=False, title='Scaling factor', figsize=(10,6))
    latlon_plot(scale_extend, grid, ctype='ratio', include_shelf=False, title='Scaling factor', zoom_fris=True)

    print 'Writing to file'
    # Replace mask with zeros
    mask = scale_extend.mask
    scale_extend = scale_extend.data
    scale_extend[mask] = 0
    write_binary(scale_extend, output_file, prec=prec)


# Process one year of daily CMIP6 atmospheric data (in practice, from UKESM1-0-LL) and convert to MITgcm EXF format for forcing simulations.
# Assumes there are 30-day months - true for the DECK experiments in UKESM at least.

# Arguments:
# var: CMIP variable to process. Accepted variables are: tas, huss, uas, vas, psl, pr, rsds, rlds
# expt: experiment name to process, eg abrupt-4xCO2.

# Optional keyword arguments:
# mit_start_year, mit_end_year: years to start and end the processing of forcing. Default is to process all available data.
# model_path: path to the directory for the given model's output, within which all the experiments lie.
# ensemble member: name of the ensemble member to process. Must have daily output available.
# out_dir: path to directory in which to save output files.
# out_file_head: beginning of output filenames. If not set, it will be determined automatically as <expt>_<var>_. Each file will have the year appended.

def cmip6_atm_forcing (var, expt, mit_start_year=None, mit_end_year=None, model_path='/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/', ensemble_member='r1i1p1f2', out_dir='./', out_file_head=None):

    import netCDF4 as nc

    # Days per year (assumes 30-day months)
    days_per_year = 12*30

    # Make sure it's a real variable
    if var not in ['tas', 'huss', 'uas', 'vas', 'psl', 'pr', 'rsds', 'rlds']:
        print 'Error (cmip6_atm_forcing): unknown variable ' + var
        sys.exit()

    # Construct out_file_head if needed
    if out_file_head is None:
        out_file_head = expt+'_'+var+'_'
    elif out_file_head[-1] != '_':
        # Add an underscore if it's not already there
        out_file_head += '_'
    out_dir = real_dir(out_dir)

    # Figure out where all the files are, and which years they cover
    in_files, start_years, end_years = find_cmip6_files(model_path, expt, ensemble_member, var, 'day')
    if mit_start_year is None:
        mit_start_year = start_years[0]
    if mit_end_year is None:
        mit_end_year = end_years[-1]

    # Tell the user what to write about the grid
    lat = read_netcdf(in_files[0], 'lat')
    lon = read_netcdf(in_files[0], 'lon') 
    print '\nChanges to make in data.exf:'
    print '*_lon0='+str(lon[0])
    print '*_lon_inc='+str(lon[1]-lon[0])
    print '*_lat0='+str(lat[0])
    print '*_lat_inc='+str(lat[1]-lat[0])
    print '*_nlon='+str(lon.size)
    print '*_nlat='+str(lat.size)

    # Loop over each file
    for t in range(len(in_files)):

        file_path = in_files[t]
        print 'Processing ' + file_path        
        print 'Covers years '+str(start_years[t])+' to '+str(end_years[t])
        
        # Loop over years
        t_start = 0  # Time index in file
        t_end = t_start+days_per_year
        for year in range(start_years[t], end_years[t]+1):
            if year >= mit_start_year and year <= mit_end_year:
                print 'Processing ' + str(year)

                # Read data
                print 'Reading ' + str(year) + ' from indicies ' + str(t_start) + '-' + str(t_end)
                data = read_netcdf(file_path, var, t_start=t_start, t_end=t_end)
                # Conversions if necessary
                if var == 'tas':
                    # Kelvin to Celsius
                    data -= temp_C2K
                elif var == 'pr':
                    # kg/m^2/s to m/s
                    data /= rho_fw
                elif var in ['rsds', 'rlds']:
                    # Swap sign on radiation fluxes
                    data *= -1
                # Write data
                write_binary(data, out_dir+out_file_head+str(year))
            # Update time range for next time
            t_start = t_end
            t_end = t_start + days_per_year
            
                
        

    

    

    

    
        
        

    

    
    

    
