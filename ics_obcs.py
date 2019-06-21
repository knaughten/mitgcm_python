###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, grid_check_split, choose_grid
from utils import real_dir, xy_to_xyz, z_to_xyz, rms, select_top, fix_lon_range, mask_land
from file_io import write_binary, read_binary
from interpolation import extend_into_mask, discard_and_fill, neighbours_z, interp_slice_helper, interp_grid
from constants import sec_per_year, gravity, sec_per_day
from diagnostics import density

import numpy as np
import os
import sys


# Calculate a monthly climatology of the given variable in SOSE, from its monthly output over the entire 6-year reanalysis.

# Arguments:
# in_file: binary SOSE file (.data) containing one record for each month of the SOSE period. You can also leave ".data" off as it will get stripped off anyway.
# out_file: desired path to output file
def make_sose_climatology (in_file, out_file):

    from MITgcmutils import rdmds

    # Strip .data from filename before reading
    data = rdmds(in_file.replace('.data', ''))
    climatology = np.zeros(tuple([12]) + data.shape[1:])
    for month in range(12):
        climatology[month,:] = np.mean(data[month::12,:], axis=0)
    write_binary(climatology, out_file)


# Create initial conditions for temperature, salinity, sea ice area, and sea ice thickness using the SOSE monthly climatology for January. Ice shelf cavities will be filled with constant temperature and salinity.

# Arguments:
# grid_path: path to directory containing MITgcm binary grid files
# sose_dir: directory containing SOSE monthly climatologies and grid/ subdirectory (available on Scihub at /data/oceans_input/raw_input_data/SOSE_monthly_climatology)
# output_dir: directory to save the binary MITgcm initial conditions files (binary)

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the initial conditions in, so you can easily check that they look okay
# constant_t, constant_s: temperature and salinity to fill ice shelf cavities with (default -1.9 C and 34.4 psu)
# split: longitude to split the SOSE grid at. Must be 180 (if your domain includes 0E; default) or 0 (if your domain includes 180E). If your domain is circumpolar (i.e. includes both 0E and 180E), try either and hope for the best. You might have points falling in the gap between SOSE's periodic boundary, in which case you'll have to write a few patches to wrap the SOSE data around the boundary (do this in the SOSEGrid class in grid.py).
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def sose_ics (grid_path, sose_dir, output_dir, nc_out=None, constant_t=-1.9, constant_s=34.4, split=180, prec=64):

    from grid import SOSEGrid
    from file_io import NCfile
    from interpolation import interp_reg

    sose_dir = real_dir(sose_dir)
    output_dir = real_dir(output_dir)

    # Fields to interpolate
    fields = ['THETA', 'SALT', 'SIarea', 'SIheff']
    # Flag for 2D or 3D
    dim = [3, 3, 2, 2]
    # Constant values for ice shelf cavities
    constant_value = [constant_t, constant_s, 0, 0]
    # End of filenames for input
    infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_SOSE.ini'
    
    print 'Building grids'
    # First build the model grid and check that we have the right value for split
    model_grid = grid_check_split(grid_path, split)
    # Now build the SOSE grid
    sose_grid = SOSEGrid(sose_dir+'grid/', model_grid=model_grid, split=split)
    # Extract land mask
    sose_mask = sose_grid.hfac == 0
    
    print 'Building mask for SOSE points to fill'
    # Figure out which points we need for interpolation
    # Find open cells according to the model, interpolated to SOSE grid
    model_open = np.ceil(interp_reg(model_grid, sose_grid, np.ceil(model_grid.hfac), fill_value=1))
    # Find ice shelf cavity points according to model, interpolated to SOSE grid
    model_cavity = np.ceil(interp_reg(model_grid, sose_grid, xy_to_xyz(model_grid.ice_mask, model_grid), fill_value=0)).astype(bool)
    # Select open, non-cavity cells
    fill = model_open*np.invert(model_cavity)
    # Extend into the mask a few times to make sure there are no artifacts near the coast
    fill = extend_into_mask(fill, missing_val=0, use_3d=True, num_iters=3)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyz')

    # Process fields
    for n in range(len(fields)):
        print 'Processing ' + fields[n]
        in_file = sose_dir + fields[n] + infile_tail
        out_file = output_dir + fields[n] + outfile_tail
        print '...reading ' + in_file
        # Just keep the January climatology
        if dim[n] == 3:
            sose_data = sose_grid.read_field(in_file, 'xyzt')[0,:]
        else:
            # Fill any missing regions with zero sea ice, as we won't be extrapolating them later
            sose_data = sose_grid.read_field(in_file, 'xyt', fill_value=0)[0,:]
        # Discard the land mask, and extrapolate slightly into missing regions so the interpolation doesn't get messed up.
        print '...extrapolating into missing regions'
        if dim[n] == 3:
            sose_data = discard_and_fill(sose_data, sose_mask, fill)
            # Fill cavity points with constant values
            sose_data[model_cavity] = constant_value[n]
        else:
            # Just care about surface layer
            sose_data = discard_and_fill(sose_data, sose_mask[0,:], fill[0,:], use_3d=False)
        print '...interpolating to model grid'
        data_interp = interp_reg(sose_grid, model_grid, sose_data, dim=dim[n])
        # Fill the land mask with zeros
        if dim[n] == 3:
            data_interp[model_grid.hfac==0] = 0
        else:
            data_interp[model_grid.hfac[0,:]==0] = 0
        write_binary(data_interp, out_file, prec=prec)
        if nc_out is not None:
            print '...adding to ' + nc_out
            if dim[n] == 3:
                ncfile.add_variable(fields[n], data_interp, 'xyz')
            else:
                ncfile.add_variable(fields[n], data_interp, 'xy')

    if nc_out is not None:
        ncfile.close()


# Create initial conditions for temperature, salinity, sea ice area, sea ice thickness, and snow thickness using monthly climatology output from another (larger) MITgcm domain. This larger domain is assumed to include all ice shelf cavities.

# Arguments:
# grid_path: path to directory containing binary grid files for the new domain
# source_file: path to NetCDF file (using xmitgcm conventions) containing a monthly climatology of the larger domain's output
# output_dir: directory to save the binary MITgcm initial conditions files

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the initial conditions in, so you can easily check that they look okay
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def mit_ics (grid_path, source_file, output_dir, nc_out=None, prec=64):

    from file_io import NCfile, read_netcdf
    from interpolation import interp_reg

    output_dir = real_dir(output_dir)

     # Fields to interpolate
    fields = ['THETA', 'SALT', 'SIarea', 'SIheff', 'SIhsnow']
    # Flag for 2D or 3D
    dim = [3, 3, 2, 2, 2]
    # End of filenames for output
    outfile_tail = '_MIT.ini'

    print 'Building grids'
    source_grid = Grid(source_file)
    model_grid = Grid(grid_path)
    # Extract land mask of source grid
    source_mask = source_grid.hfac==0

    print 'Building mask for points to fill'
    # Select open cells according to the model, interpolated to the source grid
    fill = np.ceil(interp_reg(model_grid, source_grid, np.ceil(model_grid.hfac), fill_value=0)).astype(bool)
    # Extend into mask a few times to make sure there are no artifacts near the coast
    fill = extend_into_mask(fill, missing_val=0, use_3d=True, num_iters=3)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyz')

    # Process fields
    for n in range(len(fields)):
        print 'Processing ' + fields[n]
        out_file = output_dir + fields[n] + outfile_tail
        # Read the January climatology
        source_data = read_netcdf(source_file, fields[n], time_index=0)
        # Discard the land mask, and extrapolate slightly into missing regions so the interpolation doesn't get messed up.
        print '...extrapolating into missing regions'
        if dim[n] == 3:
            source_data = discard_and_fill(source_data, source_mask, fill)
        else:
            # Just care about the surface layer
            source_data = discard_and_fill(source_data, source_mask[0,:], fill[0,:], use_3d=False)
        print '...interpolating to model grid'
        data_interp = interp_reg(source_grid, model_grid, source_data, dim=dim[n])
        # Fill the land mask with zeros
        if dim[n] == 3:
            data_interp[model_grid.hfac==0] = 0
        else:
            data_interp[model_grid.hfac[0,:]==0] = 0
        write_binary(data_interp, out_file, prec=prec)
        if nc_out is not None:
            print '...adding to ' + nc_out
            if dim[n] == 3:
                ncfile.add_variable(fields[n], data_interp, 'xyz')
            else:
                ncfile.add_variable(fields[n], data_interp, 'xy')

    if nc_out is not None:
        ncfile.close()    


# Calculate the initial pressure loading anomaly of the ice shelf. This depends on the density of the hypothetical seawater displaced by the ice shelf. There are three different assumptions we could make:
# 1. Assume the displaced water has a constant temperature and salinity (default)
# 2. Use nearest-neighbour extrapolation within the cavity to set the temperature and salinity of the displaced water. This is equivalent to finding the temperature and salinity of the surface layer immediately beneath the ice base, and extrapolating it up vertically at every point.
# 3. The user did something else ahead of time, and their initial temperature and salinity files already have the mask filled in.

# Arguments:
# grid: Grid object OR path to grid directory OR path to NetCDF file
# out_file: path to desired output file

# Optional keyword arguments:
# option: 'constant', 'nearest', or 'precomputed' as described above
# ini_temp_file, ini_salt_file: paths to initial conditions (binary) for temperature and salinity
# ini_temp, ini_salt: alternatively, just pass the 3D arrays for initial temperature and salinity
# constant_t, constant_s: if option='constant', temperature and salinity to use
# eosType: 'MDJWF', 'JMD95', or 'LINEAR'. Must match value in "data" namelist.
# rhoConst: reference density as in MITgcm's "data" namelist
# tAlpha, sBeta, Tref, Sref: if eosType='LINEAR', set these to match your "data" namelist.
# hfac: 3D array of hFacC values, if the values stored in the grid are out of date (eg when coupling)
# prec: as in function sose_ics
# check_grid: boolean indicating that grid might be a file/directory path rather than a Grid object. Switch this off if you're using a dummy grid which has all the necessary variables but is not a Grid object.

def calc_load_anomaly (grid, out_file, option='constant', ini_temp_file=None, ini_salt_file=None, ini_temp=None, ini_salt=None, constant_t=-1.9, constant_s=34.4, eosType='MDJWF', rhoConst=1035, tAlpha=None, sBeta=None, Tref=None, Sref=None, hfac=None, prec=64, check_grid=True):

    errorTol = 1e-13  # convergence criteria

    # Build the grid if needed
    if check_grid:
        grid = choose_grid(grid, None)
    # Decide which hfac to use
    if hfac is None:
        hfac = grid.hfac

    # Set temperature and salinity
    if ini_temp is not None and ini_salt is not None:
        # Deep copy of the arrays
        temp = np.copy(ini_temp)
        salt = np.copy(ini_salt)
    elif ini_temp_file is not None and ini_salt_file is not None:
        # Read from file
        temp = read_binary(ini_temp_file, [grid.nx, grid.ny, grid.nz], 'xyz', prec=prec)
        salt = read_binary(ini_salt_file, [grid.nx, grid.ny, grid.nz], 'xyz', prec=prec)
    else:
        print 'Error (calc_load_anomaly): Must either specify ini_temp and ini_salt OR ini_temp_file and ini_salt_file'
        sys.exit()

    # Fill in the ice shelves
    # The bathymetry will get filled too, but that doesn't matter because pressure is integrated from the top down
    closed = hfac==0
    if option == 'constant':
        # Fill with constant values
        temp[closed] = constant_t
        salt[closed] = constant_s
    elif option == 'nearest':
        # Select the layer immediately below the ice shelves and tile to make it 3D
        temp_top = xy_to_xyz(select_top(np.ma.masked_where(closed, temp), return_masked=False), grid)
        salt_top = xy_to_xyz(select_top(np.ma.masked_where(closed, salt), return_masked=False), grid)
        # Fill the mask with these values
        temp[closed] = temp_top[closed]
        salt[closed] = salt_top[closed]    
    elif option == 'precomputed':
        for data in [temp, salt]:
            # Make sure there are no missing values
            if (data[~closed]==0).any():
                print 'Error (calc_load_anomaly): you selected the precomputed option, but there are appear to be missing values in the land mask.'
                sys.exit()
            # Make sure it's not a masked array as this will break the rms
            if isinstance(data, np.ma.MaskedArray):
                # Fill the mask with zeros
                data[data.mask] = 0
                data = data.data
    else:
        print 'Error (calc_load_anomaly): invalid option ' + option
        sys.exit()

    # Get vertical integrands considering z at both centres and edges of layers
    dz_merged = np.zeros(2*grid.nz)
    dz_merged[::2] = abs(grid.z - grid.z_edges[:-1])  # dz of top half of each cell
    dz_merged[1::2] = abs(grid.z_edges[1:] - grid.z)  # dz of bottom half of each cell
    # Tile to make 3D
    z = z_to_xyz(grid.z, grid)
    dz_merged = z_to_xyz(dz_merged, grid)

    # Initial guess for pressure (dbar) at centres of cells
    press = abs(z)*gravity*rhoConst*1e-4

    # Iteratively calculate pressure load anomaly until it converges
    press_old = np.zeros(press.shape)  # Dummy initial value for pressure from last iteration
    rms_error = 0
    while True:
        rms_old = rms_error
        rms_error = rms(press, press_old)
        print 'RMS error = ' + str(rms_error)
        if rms_error < errorTol or np.abs(rms_error-rms_old) < 0.1*errorTol:
            print 'Converged'
            break
        # Save old pressure
        press_old = np.copy(press)
        # Calculate density anomaly at centres of cells
        drho_c = density(eosType, salt, temp, press, rhoConst=rhoConst, Tref=Tref, Sref=Sref, tAlpha=tAlpha, sBeta=sBeta) - rhoConst
        # Use this for both centres and edges of cells
        drho = np.zeros(dz_merged.shape)
        drho[::2,...] = drho_c
        drho[1::2,...] = drho_c
        # Integrate pressure load anomaly (Pa)
        pload_full = np.cumsum(drho*gravity*dz_merged, axis=0)
        # Update estimate of pressure
        press = (abs(z)*gravity*rhoConst + pload_full[1::2,...])*1e-4

    # Extract pload at each level edge (don't care about centres anymore)
    pload_edges = pload_full[::2,...]

    # Now find pload at the ice shelf base
    # For each xy point, calculate three variables:
    # (1) pload at the base of the last fully dry ice shelf cell
    # (2) pload at the base of the cell beneath that
    # (3) hFacC for that cell
    # To calculate (1) we have to shift pload_3d_edges upward by 1 cell
    pload_edges_above = neighbours_z(pload_edges)[0]
    pload_above = select_top(np.ma.masked_where(closed, pload_edges_above), return_masked=False)
    pload_below = select_top(np.ma.masked_where(closed, pload_edges), return_masked=False)
    hfac_below = select_top(np.ma.masked_where(closed, hfac), return_masked=False)
    # Now we can interpolate to the ice base
    pload = pload_above + (1-hfac_below)*(pload_below - pload_above)

    # Write to file
    write_binary(pload, out_file, prec=prec)


# Create open boundary conditions for temperature, salinity, horizontal velocities, and (if you want sea ice) sea ice area, thickness, and velocities. Use either the SOSE monthly climatology (source='SOSE') or output from another MITgcm model, stored in a NetCDF file (source='MIT').

# Arguments:
# location: 'N', 'S', 'E', or 'W' corresponding to the open boundary to process (north, south, east, west). So, run this function once for each open boundary in your domain.
# grid_path: path to directory containing MITgcm binary grid files. The latitude or longitude of the open boundary will be determined from this file.
# input_path: either a directory containing SOSE data (if source='SOSE'; as in function sose_ics) or a NetCDF file (with xmitgcm conventions) containing a monthly climatology from another MITgcm model (if source='MIT').
# output_dir: directory to save binary MITgcm OBCS files

# Optional keyword arguments:
# source: 'SOSE' or 'MIT' as described above.
# use_seaice: True if you want sea ice OBCS (default), False if you don't
# nc_out: path to a NetCDF file to save the interpolated boundary conditions to, so you can easily check that they look okay
# prec: precision to write binary files (32 or 64, must match exf_iprec_obcs in the "data.exf" namelist. If you don't have EXF turned on, it must match readBinaryPrec in "data").

def make_obcs (location, grid_path, input_path, output_dir, source='SOSE', use_seaice=True, nc_out=None, prec=32, split=180):

    from grid import SOSEGrid
    from file_io import NCfile, read_netcdf
    from interpolation import interp_bdry

    if source == 'SOSE':
        input_path = real_dir(input_path)
    output_dir = real_dir(output_dir)

    # Fields to interpolate
    # Important: SIarea has to be before SIuice and SIvice so it can be used for masking
    fields = ['THETA', 'SALT', 'UVEL', 'VVEL', 'SIarea', 'SIheff', 'SIuice', 'SIvice', 'ETAN']  
    # Flag for 2D or 3D
    dim = [3, 3, 3, 3, 2, 2, 2, 2, 2]
    # Flag for grid type
    gtype = ['t', 't', 'u', 'v', 't', 't', 'u', 'v', 't']
    if source == 'MIT':
        # Also consider snow thickness
        fields += ['SIhsnow']
        dim += [2]
        gtype += ['t']
    # End of filenames for input
    infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_'+source+'.OBCS_'+location

    print 'Building MITgcm grid'
    if source == 'SOSE':
        model_grid = grid_check_split(grid_path, split)
    elif source == 'MIT':
        model_grid = Grid(grid_path)
    # Figure out what the latitude or longitude is on the boundary, both on the centres and outside edges of those cells
    if location == 'S':
        lat0 = model_grid.lat_1d[0]
        lat0_e = model_grid.lat_corners_1d[0]
        print 'Southern boundary at ' + str(lat0) + ' (cell centre), ' + str(lat0_e) + ' (cell edge)'
    elif location == 'N':
        lat0 = model_grid.lat_1d[-1]
        lat0_e = 2*model_grid.lat_corners_1d[-1] - model_grid.lat_corners_1d[-2]
        print 'Northern boundary at ' + str(lat0) + ' (cell centre), ' + str(lat0_e) + ' (cell edge)'
    elif location == 'W':
        lon0 = model_grid.lon_1d[0]
        lon0_e = model_grid.lon_corners_1d[0]
        print 'Western boundary at ' + str(lon0) + ' (cell centre), ' + str(lon0_e) + ' (cell edge)'
    elif location == 'E':
        lon0 = model_grid.lon_1d[-1]
        lon0_e = 2*model_grid.lon_corners_1d[-1] - model_grid.lon_corners_1d[-2]
        print 'Eastern boundary at ' + str(lon0) + ' (cell centre), ' + str(lon0_e) + ' (cell edge)'
    else:
        print 'Error (make_obcs): invalid location ' + str(location)
        sys.exit()

    if source == 'SOSE':
        print 'Building SOSE grid'
        source_grid = SOSEGrid(input_path+'grid/', model_grid=model_grid, split=split)
    elif source == 'MIT':
        print 'Building grid from source model'
        source_grid = Grid(input_path)
    else:
        print 'Error (make_obcs): invalid source ' + source
        sys.exit()
    # Calculate interpolation indices and coefficients to the boundary latitude or longitude
    if location in ['N', 'S']:
        # Cell centre
        j1, j2, c1, c2 = interp_slice_helper(source_grid.lat_1d, lat0)
        # Cell edge
        j1_e, j2_e, c1_e, c2_e = interp_slice_helper(source_grid.lat_corners_1d, lat0_e)
    else:
        # Pass lon=True to consider the possibility of boundary near 0E
        i1, i2, c1, c2 = interp_slice_helper(source_grid.lon_1d, lon0, lon=True)
        i1_e, i2_e, c1_e, c2_e = interp_slice_helper(source_grid.lon_corners_1d, lon0_e, lon=True)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyzt')
        ncfile.add_time(np.arange(12)+1, units='months')  

    # Process fields
    for n in range(len(fields)):
        if fields[n].startswith('SI') and not use_seaice:
            continue

        print 'Processing ' + fields[n]
        if source == 'SOSE':
            in_file = input_path + fields[n] + infile_tail
        out_file = output_dir + fields[n] + outfile_tail
        # Read the monthly climatology at all points
        if source == 'SOSE':
            if dim[n] == 3:
                source_data = source_grid.read_field(in_file, 'xyzt')
            else:
                source_data = source_grid.read_field(in_file, 'xyt')
        else:
            source_data = read_netcdf(input_path, fields[n])

        if fields[n] == 'SIarea' and source == 'SOSE':
            # We'll need this field later for SIuice and SIvice, as SOSE didn't mask those variables properly
            print 'Interpolating sea ice area to u and v grids for masking of sea ice velocity'
            source_aice_u = interp_grid(source_data, source_grid, 't', 'u', time_dependent=True, mask_with_zeros=True, periodic=True)
            source_aice_v = interp_grid(source_data, source_grid, 't', 'v', time_dependent=True, mask_with_zeros=True, periodic=True)
        # Set sea ice velocity to zero wherever sea ice area is zero
        if fields[n] in ['SIuice', 'SIvice'] and source == 'SOSE':
            print 'Masking sea ice velocity with sea ice area'
            if fields[n] == 'SIuice':
                index = source_aice_u==0
            else:
                index = source_aice_v==0
            source_data[index] = 0            

        # Choose the correct grid for lat, lon, hfac
        source_lon, source_lat = source_grid.get_lon_lat(gtype=gtype[n], dim=1)
        source_hfac = source_grid.get_hfac(gtype=gtype[n])
        model_lon, model_lat = model_grid.get_lon_lat(gtype=gtype[n], dim=1)
        model_hfac = model_grid.get_hfac(gtype=gtype[n])
        # Interpolate to the correct grid and choose the correct horizontal axis
        if location in ['N', 'S']:
            if gtype[n] == 'v':
                source_data = c1_e*source_data[...,j1_e,:] + c2_e*source_data[...,j2_e,:]
                # Multiply hfac by the ceiling of hfac on each side, to make sure we're not averaging over land
                source_hfac = (c1_e*source_hfac[...,j1_e,:] + c2_e*source_hfac[...,j2_e,:])*np.ceil(source_hfac[...,j1_e,:])*np.ceil(source_hfac[...,j2_e,:])
            else:
                source_data = c1*source_data[...,j1,:] + c2*source_data[...,j2,:]
                source_hfac = (c1*source_hfac[...,j1,:] + c2*source_hfac[...,j2,:])*np.ceil(source_hfac[...,j1,:])*np.ceil(source_hfac[...,j2,:])
            source_haxis = source_lon
            model_haxis = model_lon
            if location == 'S':
                model_hfac = model_hfac[:,0,:]
            else:
                model_hfac = model_hfac[:,-1,:]
        else:
            if gtype[n] == 'u':
                source_data = c1_e*source_data[...,i1_e] + c2_e*source_data[...,i2_e]
                source_hfac = (c1_e*source_hfac[...,i1_e] + c2_e*source_hfac[...,i2_e])*np.ceil(source_hfac[...,i1_e])*np.ceil(source_hfac[...,i2_e])
            else:
                source_data = c1*source_data[...,i1] + c2*source_data[...,i2]
                source_hfac = (c1*source_hfac[...,i1] + c2*source_hfac[...,i2])*np.ceil(source_hfac[...,i1])*np.ceil(source_hfac[...,i2])
            source_haxis = source_lat
            model_haxis = model_lat
            if location == 'W':
                model_hfac = model_hfac[...,0]
            else:
                model_hfac = model_hfac[...,-1]
        if source == 'MIT' and model_haxis[0] < source_haxis[0]:
            # Need to extend source data to the west or south. Just add one row.
            source_haxis = np.concatenate(([model_haxis[0]-0.1], source_haxis))
            source_data = np.concatenate((np.expand_dims(source_data[:,...,0], -1), source_data), axis=-1)
            source_hfac = np.concatenate((np.expand_dims(source_hfac[:,0], 1), source_hfac), axis=1)
        # For 2D variables, just need surface hfac
        if dim[n] == 2:
            source_hfac = source_hfac[0,:]
            model_hfac = model_hfac[0,:]

        # Now interpolate each month to the model grid
        if dim[n] == 3:
            data_interp = np.zeros([12, model_grid.nz, model_haxis.size])
        else:
            data_interp = np.zeros([12, model_haxis.size])
        for month in range(12):
            print '...interpolating month ' + str(month+1)
            data_interp_tmp = interp_bdry(source_haxis, source_grid.z, source_data[month,:], source_hfac, model_haxis, model_grid.z, model_hfac, depth_dependent=(dim[n]==3))
            if fields[n] not in ['THETA', 'SALT']:
                # Zero in land mask is more physical than extrapolated data
                index = model_hfac==0
                data_interp_tmp[index] = 0
            data_interp[month,:] = data_interp_tmp

        write_binary(data_interp, out_file, prec=prec)
        
        if nc_out is not None:
            print '...adding to ' + nc_out
            # Construct the dimension code
            if location in ['S', 'N']:
                dimension = 'x'
            else:
                dimension = 'y'
            if dim[n] == 3:
                dimension += 'z'
            dimension += 't'
            ncfile.add_variable(fields[n] + '_' + location, data_interp, dimension)

    if nc_out is not None:
        ncfile.close()

        
# Correct the normal velocity in OBCS files to prevent massive sea level drift.
# Option 1 ('balance'): Calculate net transport into the domain based on OBCS velocities alone. This can be done before simulations even start, and should work well if you have useRealFreshwaterFlux turned off.
# Option 2 ('correct'): Calculate net transport based on the mean change in sea surface height over a test simulation. Run the model for a while, see how much the area-averaged eta changes over some number of years (timeseries.py should be helpful here), and then run this script to counteract any drift with OBCS corrections.
# Option 3 ('dampen'): Dampen large seasonal cycles in net transport, by correcting the velocities on a monthly-varying basis. A maximum change in mean sea surface height in any one month is specified (default 0.25 m), and a scaling factor for transport is determined such that the maximum absolute value of transport is no more than this threshold. This assumes the OBCS are monthly!!

# Arguments:
# grid_path: path to Grid directory

# Optional keyword arguments:
# option: 'balance' or 'correct' (as above)
# obcs_file_w_u, obcs_file_e_u, obcs_file_s_v, obcs_file_n_v: paths to OBCS files for UVEL (western and eastern boundaries) or VVEL (southern and northern boundaries). You only have to set the filenames for the boundaries which are actually open in your domain. They will be overwritten with corrected versions.
# d_eta: if option='correct', change in area-averaged sea surface height over the test simulation (m)
# d_t: if option='correct', length of the test simulation (years)
# max_deta_dt: if option='dampen', maximum allowable change in mean sea surface height in any given month (default 0.25 m/month)
# prec: precision of the OBCS files (as in function sose_obcs)

def balance_obcs (grid_path, option='balance', obcs_file_w_u=None, obcs_file_e_u=None, obcs_file_s_v=None, obcs_file_n_v=None, d_eta=None, d_t=None, max_deta_dt=0.25, prec=32):

    if option == 'correct' and (d_eta is None or d_t is None):
        print 'Error (balance_obcs): must set d_eta and d_t for option="correct"'
        sys.exit()
    
    print 'Building grid'
    grid = Grid(grid_path)

    # Calculate integrands of area, scaled by hFacC
    # Note that dx and dy are only available on western and southern edges of cells respectively; for the eastern and northern boundary, will just have to use 1 cell in. Not perfect, but this correction wouldn't perfectly conserve anyway.
    # Area of western face = dy*dz*hfac
    dA_w = xy_to_xyz(grid.dy_w, grid)*z_to_xyz(grid.dz, grid)*grid.hfac
    # Area of southern face = dx*dz*hfac
    dA_s = xy_to_xyz(grid.dx_s, grid)*z_to_xyz(grid.dz, grid)*grid.hfac

    # Now extract the area array at each boundary, and wrap up into a list for easy iteration later
    dA_bdry = [dA_w[:,:,0], dA_w[:,:,-1], dA_s[:,0,:], dA_s[:,-1,:]]
    # Some more lists:
    bdry_key = ['W', 'E', 'S', 'N']
    files = [obcs_file_w_u, obcs_file_e_u, obcs_file_s_v, obcs_file_n_v]
    dimensions = ['yzt', 'yzt', 'xzt', 'xzt']
    sign = [1, -1, 1, -1]  # Multiply velocity variable by this to get incoming transport
    # Initialise number of timesteps
    num_time = None

    # Integrate the total area of ocean cells on boundaries
    total_area = 0
    for i in range(len(files)):
        if files[i] is not None:
            print 'Calculating area of ' + bdry_key[i] + ' boundary'
            total_area += np.sum(dA_bdry[i])

    # Calculate the net transport into the domain
    if option in ['balance', 'dampen']:
        # Transport based on OBCS normal velocities
        if option == 'balance':
            net_transport = 0
        elif option == 'dampen':
            net_transport = None
        for i in range(len(files)):
            if files[i] is not None:
                print 'Processing ' + bdry_key[i] + ' boundary from ' + files[i]
                # Read data
                vel = read_binary(files[i], [grid.nx, grid.ny, grid.nz], dimensions[i], prec=prec)
                if num_time is None:
                    # Find number of time indices
                    num_time = vel.shape[0]
                elif num_time != vel.shape[0]:
                    print 'Error (balance_obcs): inconsistent number of time indices between OBCS files'
                    sys.exit()
                if option == 'dampen' and net_transport is None:
                    # Initialise transport per month
                    net_transport = np.zeros(num_time)
                if option == 'balance':
                    # Time-average velocity (this is equivalent to calculating the transport at each month and then time-averaging at the end - it's all just sums)
                    vel = np.mean(vel, axis=0)
                    # Integrate net transport through this boundary into the domain, and add to global sum
                    net_transport += np.sum(sign[i]*vel*dA_bdry[i])
                elif option == 'dampen':
                    # Integrate net transport at each month
                    for t in range(num_time):
                        net_transport[t,:] += np.sum(sign[i]*vel[t,:]*dA_bdry[i])
    elif option == 'correct':
        # Transport based on simulated changes in sea surface height
        # Need area of sea surface
        dA_sfc = np.sum(grid.dA*np.invert(grid.land_mask).astype(float))
        # Calculate transport in m^3/s
        net_transport = d_eta*dA_sfc/(d_t*sec_per_year)        

    # Inner function to nicely print the net transport to the user
    def print_net_transport (transport):
        if transport < 0:
            direction = 'out of the domain'
        else:
            direction = 'into the domain'
        print 'Net transport is ' + str(abs(transport*1e-6)) + ' Sv ' + direction

    if option == 'dampen':
        for t in range(num_time):
            print 'Month ' + str(t+1)
            print_net_transport(net_transport[t])
    else:
        print_net_transport(net_transport)

    if option == 'dampen':
        # Calculate the acceptable maximum absolute transport
        # First need total area of sea surface (including cavities) in domain
        surface_area = np.sum(mask_land(grid.dA), grid)
        max_transport = max_deta_dt*surface_area*1e-6/(sec_per_day*30)
        print 'Maximum allowable transport is ' + str(max_transport) + ' Sv'
        if np.max(np.abs(net_transport)) <= max_transport:
            print 'OBCS satisfy this; nothing to do'
            return
        # Work out by what factor to dampen the transports
        scale_factor = max_transport/np.max(np.abs(net_transport))
        print 'Will scale transports by ' + str(scale_factor)
        # Calculate corresponding velocity correction at each month
        correction = np.zeros(num_time)
        for t in range(num_time):
            correction[t] = (scale_factor-1)*net_transport[t]/total_area
            print 'Month ' + str(t+1) + ': will apply correction of ' + str(correction[t]) + ' m/s to normal velocity at each boundary'
    else:
        # Calculate single correction in m/s
        correction = -1*net_transport/total_area
        print 'Will apply correction of ' + str(correction) + ' m/s to normal velocity at each boundary'

    # Now apply the correction
    for i in range(len(files)):
        if files[i] is not None:
            print 'Correcting ' + files[i]
            # Read all the data again
            vel = read_binary(files[i], [grid.nx, grid.ny, grid.nz], dimensions[i], prec=prec)
            # Apply the correction
            if option == 'dampen':
                for t in range(num_time):
                    vel[t,:] += sign[i]*correction[t]
            else:
                vel += sign[i]*correction
            # Overwrite the file
            write_binary(vel, files[i], prec=prec)

    if option in ['balance', 'dampen']:
        # Recalculate the transport to make sure it worked
        if option == 'balance':
            net_transport_new = 0
        elif option == 'dampen':
            net_transport_new = np.zeros(num_time)
        for i in range(len(files)):
            if files[i] is not None:
                vel = read_binary(files[i], [grid.nx, grid.ny, grid.nz], dimensions[i], prec=prec)
                if option == 'balance':
                    vel = np.mean(vel, axis=0)
                    net_transport_new += np.sum(sign[i]*vel*dA_bdry[i])
                elif option == 'dampen':
                    for t in range(num_time):
                        net_transport_new[t,:] += np.sum(sign[i]*vel[t,:]*dA_bdry[i])
        if option == 'balance':
            print_net_transport(net_transport_new)
        elif option == 'dampen':
            for t in range(num_time):
                print 'Month ' + str(t+1)
                print_net_transport(net_transport_new[t,:])
