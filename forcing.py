import numpy as np
import sys

from grid import Grid, SOSEGrid, grid_check_split, choose_grid
from file_io import read_netcdf, write_binary, NCfile, netcdf_time
from utils import real_dir, fix_lon_range, mask_land_ice
from interpolation import interp_nonreg_xy, interp_reg, extend_into_mask, discard_and_fill, smooth_xy
from constants import temp_C2K, Lv, Rv, es0, sh_coeff

try:
    from plot_latlon import latlon_plot
except(ImportError):
    print "Warning (forcing.py): can't import plotting scripts"

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

    print 'Writing ' + output_file
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


# Set up a mask for surface restoring (eg of salinity). Don't restore on the continental shelf. You can also cut out an ellipse to allow a polynya.

# Arguments:
# grid: either a Grid object or a file/directory containing grid variables

# Optional keyword arguments:
# output_mask_file: desired path to binary file containing restoring mask. If not defined (i.e. None), the mask will be returned by the function, rather than written.
# nc_out: path to a NetCDF file to save the mask in, so you can easily check that it looks okay
# h0: threshold bathymetry (negative, in metres) for definition of continental shelf; everything shallower than this will not be restored. Default -1250 (excludes Maud Rise but keeps Filchner Trough).
# obcs_sponge: width of the OBCS sponge layer - no need to restore in that region
# polynya: string indicating an ellipse should be cut out of the restoring mask to allow a polynya to form. Options are 'maud_rise' (centered at 0E, 64S) or 'near_shelf' (centered at 35W, 70S), both with radius 10 degrees in longitude and 2 degrees in latitude.
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def restoring_mask (grid, out_file=None, nc_out=None, h0=-1250, obcs_sponge=0, polynya=None, prec=64):

    # Figure out if we need to add a polynya
    add_polynya = polynya is not None
    if add_polynya:
        if polynya == 'maud_rise':
            # Assumes split=180!
            lon0 = 0.
            lat0 = -65.
            rlon = 10.
            rlat = 2.5
        elif polynya == 'near_shelf':
            # Assumes split=180!
            lon0 = -30.                
            lat0 = -70.
            rlon = 10.
            rlat = 2.5
        else:
            print 'Error (restoring_mask): unrecognised polynya option ' + polynya
            sys.exit() 

    # Build the grid if we need to
    grid = choose_grid(grid, None)

    mask_surface = np.ones([grid.ny, grid.nx])
    # Mask out land and ice shelves
    mask_surface[grid.hfac[0,:]==0] = 0
    # Save this for later
    mask_land_ice = np.copy(mask_surface)
    # Mask out continental shelf
    mask_surface[grid.bathy > h0] = 0
    # Smooth, and remask the land and ice shelves
    mask_surface = smooth_xy(mask_surface, sigma=2)*mask_land_ice
    if add_polynya:
        # Cut a hole for a polynya
        index = (grid.lon_2d - lon0)**2/rlon**2 + (grid.lat_2d - lat0)**2/rlat**2 <= 1
        mask_surface[index] = 0
        # Smooth again with a smaller radius, and remask the land and ice shelves
        mask_surface = smooth_xy(mask_surface, sigma=1)*mask_land_ice
    if obcs_sponge > 0:
        # Also mask the cells affected by OBCS and/or its sponge
        mask_surface[:obcs_sponge,:] = 0
        mask_surface[-obcs_sponge:,:] = 0
        mask_surface[:,:obcs_sponge] = 0
        mask_surface[:,-obcs_sponge:] = 0
    
    # Make a 3D version with zeros in deeper layers
    mask_3d = np.zeros([grid.nz, grid.ny, grid.nx])
    mask_3d[0,:] = mask_surface

    if out_file is None:
        # Return the mask instead of writing it
        return mask_3d
    else:
        print 'Writing ' + out_file
        write_binary(mask_3d, out_file, prec=prec)
        if nc_out is not None:
            print 'Writing ' + nc_out
            ncfile = NCfile(nc_out, grid, 'xyz')
            ncfile.add_variable('restoring_mask', mask_3d, 'xyz')
            ncfile.close()
    

# Set up surface salinity restoring using a monthly climatology interpolated from SOSE.

# Arguments:
# grid_path: path to directory containing MITgcm binary grid files
# sose_dir: directory containing SOSE monthly climatologies and grid/ subdirectory (available on Scihub at /data/oceans_input/raw_input_data/SOSE_monthly_climatology)
# output_salt_file: desired path to binary file containing salinity values to restore
# output_mask_file: desired path to binary file containing restoring mask

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the salinity and mask in, so you can easily check that they look okay
# h0, obcs_sponge, polynya: as in function restoring_mask
# split: as in function sose_ics
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def sose_sss_restoring (grid_path, sose_dir, output_salt_file, output_mask_file, nc_out=None, h0=-1250, obcs_sponge=0, polynya=None, split=180, prec=64):

    sose_dir = real_dir(sose_dir)

    print 'Building grids'
    # First build the model grid and check that we have the right value for split
    model_grid = grid_check_split(grid_path, split)
    # Now build the SOSE grid
    sose_grid = SOSEGrid(sose_dir+'grid/', model_grid=model_grid, split=split)
    # Extract surface land mask
    sose_mask = sose_grid.hfac[0,:] == 0

    print 'Building mask'
    mask_3d = restoring_mask(model_grid, None, h0=h0, obcs_sponge=obcs_sponge, polynya=polynya)

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
        sss_interp[month,0,:] = interp_reg(sose_grid, model_grid, sose_sss_filled, dim=2)*np.ceil(model_grid.hfac[0,:])

    print 'Writing ' + output_salt_file
    write_binary(sss_interp, output_salt_file, prec=prec)
    print 'Writing ' + output_mask_file
    write_binary(mask_3d, output_mask_file, prec=prec)

    if nc_out is not None:
        print 'Writing ' + nc_out
        ncfile = NCfile(nc_out, model_grid, 'xyzt')
        ncfile.add_time(np.arange(12)+1, units='months')
        ncfile.add_variable('salinity', sss_interp, 'xyzt', units='psu')
        ncfile.add_variable('restoring_mask', mask_3d, 'xyz')
        ncfile.close()



def process_era (in_dir, out_dir, year, first_year=False, prec=32):

    in_dir = real_dir(in_dir)
    out_dir = real_dir(out_dir)

    if year == 2000 and not first_year:
        print 'Warning (process_era): last we checked, 2000 was the first year of ERA5. Unless this has changed, you need to set first_year=True.'
        sys.stdout.flush()

    # Construct file paths for input and output files
    in_head = in_dir + 'era5_'
    var_in = ['msl', 't2m', 'd2m', 'u10', 'v10', 'tp', 'ssrd', 'strd']
    in_tail = '_' + str(year) + '.nc'
    out_head = out_dir + 'ERA5_'
    var_out = ['apressure', 'atemp', 'aqh', 'uwind', 'vwind', 'precip', 'swdown', 'lwdown']
    out_tail = '_' + str(year)

    # Northermost latitude to keep
    lat0 = -30
    # Length of ERA5 time interval in seconds
    dt = 3600.  # 1 hour

    # Read the grid from the first file
    first_file = in_head + var_in[0] + in_tail
    lon = read_netcdf(first_file, 'longitude')
    lat = read_netcdf(first_file, 'latitude')
    # Find the index of the last latitude we don't care about (remember that latitude goes from north to south in ERA files!)
    j_bound = np.nonzero(lat < lat0)[0][0] - 2
    # Trim and flip latitude
    lat = lat[:j_bound:-1]
    # Also read the first time index for the starting date
    start_date = netcdf_time(first_file)[0]

    if first_year:
        # Print grid information to the reader
        print 'For var in ' + str(var_out) + ', make these changes in input/data.exf:'
        print 'var_lon0 = ' + str(lon[0])
        print 'var_lon_inc = ' + str(lon[1]-lon[0])
        print 'var_nlon = ' + str(lon.size)
        print 'var_lat0 = ' + str(lat[0])
        print 'var_lat_inc = ' + str(lat[1]-lat[0])
        print 'var_nlat = ' + str(lat.size)
        print 'varperiod = ' + str(dt)
        print 'varstartdate1 = ' + start_date.strftime('%Y%m%d')
        sys.stdout.flush()

    # Loop over variables
    for i in range(len(var_in)):
        
        in_file = in_head + var_in[i] + in_tail
        print 'Reading ' + in_file
        sys.stdout.flush()
        data = read_netcdf(in_file, var_in[i])
        
        print 'Processing'
        sys.stdout.flush()
        # Trim and flip over latitude
        data = data[:,:j_bound:-1,:]
        
        if var_in[i] == 'msl':
            # Save pressure for later conversions
            press = np.copy(data)

        elif var_in[i] == 't2m':
            # Save temperature for later conversions
            temp = np.copy(data)
            # Now convert from Kelvin to Celsius
            data -= temp_C2K

        elif var_in[i] == 'd2m':
            # Calculate specific humidity from dew point temperature, temperature, and pressure
            # Start with vapour pressure
            e = es0*exp(Lv/Rv*(1/temp - 1/data))
            data = sh_coeff*e/(press - (1-sh_coeff)*e)
            
        elif var_in[i] in ['tp', 'ssrd', 'strd']:
            # Accumulated variables: convert from integrals to time-averages
            data /= dt
            if first_year:
                # The first 6 hours of the accumulated variables are missing during the first year of ERA5. Fill this missing period with data from the subsequent 6 hours.
                data = np.concatenate((data[:6,:], data), axis=0)

        out_file = out_head + var_out[i] + out_tail
        print 'Writing ' + out_file
        sys.stdout.flush()
        write_binary(data, out_file, prec=prec)
