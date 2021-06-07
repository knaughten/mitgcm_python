###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from .grid import Grid, grid_check_split, choose_grid
from .utils import real_dir, xy_to_xyz, z_to_xyz, rms, select_top, fix_lon_range, mask_land, add_time_dim, is_depth_dependent
from .file_io import write_binary, read_binary, find_cmip6_files, write_netcdf_basic
from .interpolation import extend_into_mask, discard_and_fill, neighbours_z, interp_slice_helper, interp_grid
from .constants import sec_per_year, gravity, sec_per_day, months_per_year
from .diagnostics import density

import numpy as np
import os
import sys


# Helper function for make_sose_climatology and make_bsose_climatology.
# Given an array of monthly data for multiple years, calculate the monthly climatology.
def calc_climatology (data):
    climatology = np.zeros(tuple([12]) + data.shape[1:])
    for month in range(12):
        climatology[month,:] = np.mean(data[month::12,:], axis=0)
    return climatology  
    

# Calculate a monthly climatology of the given variable in SOSE, from its monthly output over the entire 6-year reanalysis.

# Arguments:
# in_file: binary SOSE file (.data) containing one record for each month of the SOSE period. You can also leave ".data" off as it will get stripped off anyway.
# out_file: desired path to output file
def make_sose_climatology (in_file, out_file):
    
    from MITgcmutils import rdmds
    # Strip .data from filename before reading
    data = rdmds(in_file.replace('.data', ''))
    climatology = calc_climatology(data)    
    write_binary(climatology, out_file)


# Do the same for SOSE versions stored in NetCDF files (like B-SOSE). You must also supply the variable name in the file, and the path to the complete NetCDF grid file.
def make_sose_climatology_netcdf (in_file, var_name, out_file, units=None):
    
    from .file_io import read_netcdf
    import netCDF4 as nc
    data = read_netcdf(in_file, var_name)
    climatology = calc_climatology(data)
    write_netcdf_basic(climatology, var_name, out_file, units=units)


# Helper function for initial conditions: figure out which points on the source grid will be needed for interpolation. Does not include ice shelf cavities, unless missing_cavities=False.
def get_fill_mask (source_grid, model_grid, missing_cavities=True):

    from .interpolation import interp_reg

    # Find open cells according to the model, interpolated to source grid
    model_open = np.ceil(interp_reg(model_grid, source_grid, np.ceil(model_grid.hfac), fill_value=0))
    if missing_cavities:    
        # Find ice shelf cavity points according to model, interpolated to source grid
        model_cavity = np.ceil(interp_reg(model_grid, source_grid, xy_to_xyz(model_grid.ice_mask, model_grid), fill_value=0)).astype(bool)
        # Select open, non-cavity cells
        fill = model_open*np.invert(model_cavity)
    else:
        fill = model_open
    # Extend into the mask a few times to make sure there are no artifacts near the coast
    fill = extend_into_mask(fill, missing_val=0, use_3d=True, num_iters=3)
    if missing_cavities:
        return fill, model_cavity
    else:
        return fill


# Helper function for initial conditions: process and interpolate a field from the source grid to the model grid, and write to file (binary plus NetCDF if needed).
def process_ini_field (source_data, source_mask, fill, source_grid, model_grid, dim, field_name, out_file, missing_cavities=True, model_cavity=None, cavity_value=None, regular=True, nc_out=None, ncfile=None, prec=64, missing_val=-9999):

    from .interpolation import interp_reg, interp_nonreg

    # Error checking
    if missing_cavities and dim==3 and (model_cavity is None or cavity_value is None):
        print('Error (process_ini_field): must provide model_cavity and cavity_value')
        sys.exit()
    if nc_out is not None and ncfile is None:
        print('Error (process_ini_field): must provide ncfile')
        sys.exit()
        
    print('...extrapolating into missing regions')
    if dim == 3:
        source_data = discard_and_fill(source_data, source_mask, fill, missing_val=missing_val)
        if missing_cavities:
            source_data[model_cavity] = cavity_value
    else:
        # Just the surface layer
        source_data = discard_and_fill(source_data, source_mask[0,:], fill[0,:], use_3d=False, missing_val=missing_val)

    print('...interpolating to model grid')
    if regular:
        data_interp = interp_reg(source_grid, model_grid, source_data, dim=dim)
    else:
        data_interp = interp_nonreg(source_grid, model_grid, source_data, dim=dim, fill_value=missing_val)
    if dim==3 and missing_cavities:
        # Fill the cavities with constant values again, because there may be artifacts near the grounding line (cavity points not included in extend_into_mask call in get_fill_mask)
        data_interp[xy_to_xyz(model_grid.ice_mask, model_grid)] = cavity_value
        
    # Fill the land mask with zeros
    if dim == 3:
        data_interp[model_grid.hfac==0] = 0
    else:
        data_interp[model_grid.hfac[0,:]==0] = 0
        
    # Write file
    write_binary(data_interp, out_file, prec=prec)
    if nc_out is not None:
        print(('...adding to ' + nc_out))
        if dim == 3:
            ncfile.add_variable(field_name, data_interp, 'xyz')
        else:
            ncfile.add_variable(field_name, data_interp, 'xy')


# Create initial conditions for temperature, salinity, sea ice area, and sea ice thickness using the SOSE monthly climatology for January. Ice shelf cavities will be filled with constant temperature and salinity.

# Arguments:
# grid_path: path to directory containing MITgcm binary grid files
# sose_dir: directory containing SOSE monthly climatologies and grid/ subdirectory (available on Scihub at /data/oceans_input/raw_input_data/SOSE_monthly_climatology)
# output_dir: directory to save the binary MITgcm initial conditions files (binary)

# Optional keyword arguments:
# bsose: set to True if it's the BSOSE version, including SIhsnow and saved in NetCDF files.
# nc_out: path to a NetCDF file to save the initial conditions in, so you can easily check that they look okay
# constant_t, constant_s: temperature and salinity to fill ice shelf cavities with (default -1.9 C and 34.4 psu)
# split: longitude to split the SOSE grid at. Must be 180 (if your domain includes 0E; default) or 0 (if your domain includes 180E). If your domain is circumpolar (i.e. includes both 0E and 180E), try either and hope for the best. You might have points falling in the gap between SOSE's periodic boundary, in which case you'll have to write a few patches to wrap the SOSE data around the boundary (do this in the SOSEGrid class in grid.py).
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def sose_ics (grid_path, sose_dir, output_dir, bsose=False, nc_out=None, constant_t=-1.9, constant_s=34.4, split=180, prec=64):

    from .grid import SOSEGrid
    from .file_io import NCfile

    sose_dir = real_dir(sose_dir)
    output_dir = real_dir(output_dir)

    # Fields to interpolate
    fields = ['THETA', 'SALT', 'SIarea', 'SIheff']
    # Flag for 2D or 3D
    dim = [3, 3, 2, 2]
    # Constant values for ice shelf cavities
    constant_value = [constant_t, constant_s, 0, 0]
    if bsose:
        # Add snow depth
        fields += ['SIhsnow']
        dim += [2]
        constant_value += [0]
    # End of filenames for input
    if bsose:
        infile_tail = '_climatology.nc'
    else:
        infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_BSOSE.ini'
    
    print('Building grids')
    # First build the model grid and check that we have the right value for split
    model_grid = grid_check_split(grid_path, split)
    # Now build the SOSE grid
    if bsose:
        sose_grid_path = sose_dir+'grid.nc'
    else:
        sose_grid_path = sose_dir+'grid/'
    sose_grid = SOSEGrid(sose_grid_path, model_grid=model_grid, split=split)
    # Extract land mask
    sose_mask = sose_grid.hfac == 0
    
    print('Building mask for SOSE points to fill')
    fill, model_cavity = get_fill_mask(sose_grid, model_grid)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyz')
    else:
        ncfile = None

    # Process fields
    for n in range(len(fields)):
        print(('Processing ' + fields[n]))
        in_file = sose_dir + fields[n] + infile_tail
        out_file = output_dir + fields[n] + outfile_tail
        print(('...reading ' + in_file))
        # Just keep the January climatology
        if dim[n] == 3:
            sose_data = sose_grid.read_field(in_file, 'xyzt', var_name=fields[n])[0,:]
        else:
            # Fill any missing regions with zero sea ice, as we won't be extrapolating them later
            sose_data = sose_grid.read_field(in_file, 'xyt', var_name=fields[n], fill_value=0)[0,:]
        process_ini_field(sose_data, sose_mask, fill, sose_grid, model_grid, dim[n], fields[n], out_file, model_cavity=model_cavity, cavity_value=constant_value[n], nc_out=nc_out, ncfile=ncfile, prec=prec)

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

    from .file_io import NCfile, read_netcdf

    output_dir = real_dir(output_dir)

     # Fields to interpolate
    fields = ['THETA', 'SALT', 'SIarea', 'SIheff', 'SIhsnow']
    # Flag for 2D or 3D
    dim = [3, 3, 2, 2, 2]
    # End of filenames for output
    outfile_tail = '_MIT.ini'

    print('Building grids')
    source_grid = Grid(source_file)
    model_grid = Grid(grid_path)
    # Extract land mask of source grid
    source_mask = source_grid.hfac==0

    print('Building mask for points to fill')
    fill = get_fill_mask(source_grid, model_grid, missing_cavities=False)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyz')

    # Process fields
    for n in range(len(fields)):
        print(('Processing ' + fields[n]))
        out_file = output_dir + fields[n] + outfile_tail
        # Read the January climatology
        source_data = read_netcdf(source_file, fields[n], time_index=0)
        process_ini_field(source_data, source_mask, fill, source_grid, model_grid, dim[n], fields[n], out_file, missing_cavities=False, nc_out=nc_out, ncfile=ncfile, prec=prec)


# Create initial conditions for temperature, salinity, sea ice area, sea ice thickness, and snow thickness using January output from the given year of a CMIP6 simulation. 
def cmip6_ics (grid_path, year0, expt='piControl', cmip_model_path='/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/', ensemble_member='r1i1p1f2', constant_t=-1.9, constant_s=34.4, output_dir='./', nc_out=None, prec=64):

    from .file_io import NCfile, read_netcdf
    from .grid import CMIPGrid

    output_dir = real_dir(output_dir)

    # Fields to interpolate
    fields_mit = ['THETA', 'SALT', 'SIarea', 'SIheff', 'SIhsnow']
    fields_cmip = ['thetao', 'so', 'siconc', 'sithick', 'sisnthick']
    # Flag for number of dimensions
    dim = [3, 3, 2, 2, 2]
    # Flag for realm
    realm = ['Omon', 'Omon', 'SImon', 'SImon', 'SImon']
    # Constant values for ice shelf cavities
    constant_value = [constant_t, constant_s, 0, 0, 0]
    # End of filenames for output
    outfile_tail = '_'+expt+'.ini'

    print('Building MITgcm grid')
    model_grid = Grid(grid_path)
    print('Building CMIP6 model grid')
    cmip_grid = CMIPGrid(cmip_model_path, expt, ensemble_member)

    print('Building mask for CMIP points to fill')
    fill, model_cavity = get_fill_mask(cmip_grid, model_grid)

    # Set up NetCDF file
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyz')

    # Process fields
    for n in range(len(fields_mit)):
        print(('Processing ' + fields_mit[n]))
        # Figure out where all the files are, and which years they cover
        in_files, start_years, end_years = find_cmip6_files(cmip_model_path, expt, ensemble_member, fields_cmip[n], realm[n])
        # Find which file includes the year we want
        if start_years[-1] <= year0 and end_years[-1] >= year0:
            file_index = len(start_years)-1
        else:
            file_index = np.where(np.array(start_years) > year0)[0][0]-1
        file_path = in_files[file_index]
        # Find time index in that file for January of year0
        time_index = (year0-start_years[file_index])*months_per_year
        if fields_mit[n] == 'SIarea':
            # Save file path and time index for sea ice area - will need it later
            file_path_aice = file_path
            time_index_aice = time_index
        # Read data
        print(('Reading ' + file_path + ' at index ' + str(time_index)))
        cmip_data = read_netcdf(file_path, fields_cmip[n], time_index=time_index)
        if fields_mit[n] == 'SIarea':
            # Convert from percent to fraction
            cmip_data *= 1e-2
        if fields_mit[n] in ['SIheff', 'SIhsnow']:
            # These variables are masked in regions of zero sea ice. Fill these regions with zeros instead.
            index = np.where(cmip_data.mask*np.invert(cmip_grid.mask[0,:]))
            cmip_data[index] = 0
            # Also need to weight them with sea ice concentration.
            cmip_data_aice = read_netcdf(file_path_aice, 'siconc', time_index=time_index_aice)*1e-2
            cmip_data *= cmip_data_aice
        out_file = output_dir + fields_mit[n] + outfile_tail
        process_ini_field(cmip_data, cmip_grid.mask, fill, cmip_grid, model_grid, dim[n], fields_mit[n], out_file, model_cavity=model_cavity, cavity_value=constant_value[n], regular=False, nc_out=nc_out, ncfile=ncfile, prec=prec)

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
        print('Error (calc_load_anomaly): Must either specify ini_temp and ini_salt OR ini_temp_file and ini_salt_file')
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
                print('Error (calc_load_anomaly): you selected the precomputed option, but there are appear to be missing values in the land mask.')
                sys.exit()
            # Make sure it's not a masked array as this will break the rms
            if isinstance(data, np.ma.MaskedArray):
                # Fill the mask with zeros
                data[data.mask] = 0
                data = data.data
    else:
        print(('Error (calc_load_anomaly): invalid option ' + option))
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
        print(('RMS error = ' + str(rms_error)))
        if rms_error < errorTol or np.abs(rms_error-rms_old) < 0.1*errorTol:
            print('Converged')
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


# Find the latitude or longitude on the boundary of the MITgcm grid, both on the centres and outside edges of these cells.
def find_obcs_boundary (grid, location):

    if location == 'S':
        loc0 = grid.lat_1d[0]
        loc0_e = grid.lat_corners_1d[0]
    elif location == 'N':
        loc0 = grid.lat_1d[-1]
        loc0_e = 2*grid.lat_corners_1d[-1] - grid.lat_corners_1d[-2]
    elif location == 'W':
        loc0 = grid.lon_1d[0]
        loc0_e = grid.lon_corners_1d[0]
    elif location == 'E':
        loc0 = grid.lon_1d[-1]
        loc0_e = 2*grid.lon_corners_1d[-1] - grid.lon_corners_1d[-2]
    else:
        print(('Error (find_obcs_boundary): invalid location '+str(location)))
        sys.exit()
    return loc0, loc0_e


# Create open boundary conditions for temperature, salinity, horizontal velocities, and (if you want sea ice) sea ice area, thickness, and velocities. Use either the SOSE monthly climatology (source='BSOSE' or 'SOSE' depending on version) or output from another MITgcm model, stored in a NetCDF file (source='MIT').

# Arguments:
# location: 'N', 'S', 'E', or 'W' corresponding to the open boundary to process (north, south, east, west). So, run this function once for each open boundary in your domain.
# grid_path: path to directory containing MITgcm binary grid files. The latitude or longitude of the open boundary will be determined from this file.
# input_path: either a directory containing SOSE data (if source='SOSE' or 'BSOSE'; as in function sose_ics) or a NetCDF file (with xmitgcm conventions) containing a monthly climatology from another MITgcm model (if source='MIT').
# output_dir: directory to save binary MITgcm OBCS files

# Optional keyword arguments:
# source: 'BSOSE', 'SOSE' or 'MIT' as described above.
# use_seaice: True if you want sea ice OBCS (default), False if you don't
# nc_out: path to a NetCDF file to save the interpolated boundary conditions to, so you can easily check that they look okay
# prec: precision to write binary files (32 or 64, must match exf_iprec_obcs in the "data.exf" namelist. If you don't have EXF turned on, it must match readBinaryPrec in "data").

def make_obcs (location, grid_path, input_path, output_dir, source='SOSE', use_seaice=True, nc_out=None, prec=32, split=180):

    from .grid import SOSEGrid
    from .file_io import NCfile, read_netcdf
    from .interpolation import interp_bdry

    if source in ['SOSE', 'BSOSE']:
        input_path = real_dir(input_path)
    output_dir = real_dir(output_dir)

    # Fields to interpolate
    # Important: SIarea has to be before SIuice and SIvice so it can be used for masking
    fields = ['THETA', 'SALT', 'UVEL', 'VVEL', 'SIarea', 'SIheff', 'SIuice', 'SIvice']  
    # Flag for 2D or 3D
    dim = [3, 3, 3, 3, 2, 2, 2, 2]
    # Flag for grid type
    gtype = ['t', 't', 'u', 'v', 't', 't', 'u', 'v']
    if source in ['BSOSE', 'MIT']:
        # Also consider snow thickness
        fields += ['SIhsnow']
        dim += [2]
        gtype += ['t']
    # End of filenames for input
    if source in ['SOSE', 'MIT']:
        infile_tail = '_climatology.data'
    elif source == 'BSOSE':
        infile_tail = '_climatology.nc'
    # End of filenames for output
    outfile_tail = '_'+source+'.OBCS_'+location

    print('Building MITgcm grid')
    if source in ['SOSE', 'BSOSE']:
        model_grid = grid_check_split(grid_path, split)
    elif source == 'MIT':
        model_grid = Grid(grid_path)

    # Identify boundary
    loc0, loc0_e = find_obcs_boundary(model_grid, location)
    print((location+' boundary at '+str(loc0)+' (cell centre), '+str(loc0_e)+' (cell edge)'))

    if source == 'SOSE':
        print('Building SOSE grid')
        source_grid = SOSEGrid(input_path+'grid/', model_grid=model_grid, split=split)
    elif source == 'BSOSE':
        print('Building B-SOSE grid')
        source_grid = SOSEGrid(input_path+'grid.nc', model_grid=model_grid, split=split)
    elif source == 'MIT':
        print('Building grid from source model')
        source_grid = Grid(input_path)
    else:
        print(('Error (make_obcs): invalid source ' + source))
        sys.exit()
    # Calculate interpolation indices and coefficients to the boundary latitude or longitude
    if location in ['N', 'S']:
        # Cell centre
        j1, j2, c1, c2 = interp_slice_helper(source_grid.lat_1d, loc0)
        # Cell edge
        j1_e, j2_e, c1_e, c2_e = interp_slice_helper(source_grid.lat_corners_1d, loc0_e)
    else:
        # Pass lon=True to consider the possibility of boundary near 0E
        i1, i2, c1, c2 = interp_slice_helper(source_grid.lon_1d, loc0, lon=True)
        i1_e, i2_e, c1_e, c2_e = interp_slice_helper(source_grid.lon_corners_1d, loc0_e, lon=True)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyzt')
        ncfile.add_time(np.arange(12)+1, units='months')  

    # Process fields
    for n in range(len(fields)):
        if fields[n].startswith('SI') and not use_seaice:
            continue

        print(('Processing ' + fields[n]))
        if source in ['SOSE', 'BSOSE']:
            in_file = input_path + fields[n] + infile_tail
        out_file = output_dir + fields[n] + outfile_tail
        # Read the monthly climatology at all points
        if source in ['SOSE', 'BSOSE']:
            if dim[n] == 3:
                source_data = source_grid.read_field(in_file, 'xyzt', var_name=fields[n])
            else:
                source_data = source_grid.read_field(in_file, 'xyt', var_name=fields[n])
        else:
            source_data = read_netcdf(input_path, fields[n])

        if fields[n] == 'SIarea' and source == 'SOSE':
            # We'll need this field later for SIuice and SIvice, as SOSE didn't mask those variables properly
            print('Interpolating sea ice area to u and v grids for masking of sea ice velocity')
            source_aice_u = interp_grid(source_data, source_grid, 't', 'u', time_dependent=True, mask_with_zeros=True, periodic=True)
            source_aice_v = interp_grid(source_data, source_grid, 't', 'v', time_dependent=True, mask_with_zeros=True, periodic=True)
        # Set sea ice velocity to zero wherever sea ice area is zero
        if fields[n] in ['SIuice', 'SIvice'] and source == 'SOSE':
            print('Masking sea ice velocity with sea ice area')
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
            print(('...interpolating month ' + str(month+1)))
            data_interp_tmp = interp_bdry(source_haxis, source_grid.z, source_data[month,:], source_hfac, model_haxis, model_grid.z, model_hfac, depth_dependent=(dim[n]==3))
            if fields[n] not in ['THETA', 'SALT']:
                # Zero in land mask is more physical than extrapolated data
                index = model_hfac==0
                data_interp_tmp[index] = 0
            data_interp[month,:] = data_interp_tmp

        write_binary(data_interp, out_file, prec=prec)
        
        if nc_out is not None:
            print(('...adding to ' + nc_out))
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


# Helper functions for cmip6_obcs:
                

# Given the slice weighting coefficients (as calculated in next function, and tiled to match the data's dimensions), and an array of data (can be 2D, 3D, or 4D as long as the last 2 dimensions are lat and lon), extract the boundary slice from the given data.
# Assumes "data" is a MaskedArray so we don't interpolate into the mask.
def extract_slice (data, weights, location):
    # Multiply the data by the weights and sum in the correct dimension
    if location in ['N', 'S']:
        axis = -2
    else:
        axis = -1
    data_slice = np.ma.sum(data*weights, axis=axis)
    if isinstance(data, np.ma.MaskedArray):
        # Do the same for the mask attached to the data. Any cells which end up as nonzero have interpolated into the mask.
        data_mask_slice = np.ma.sum(data.mask*weights, axis=axis)
        # Mask out these regions.
        data_slice = np.ma.masked_where(data_mask_slice>0, data_slice)
    return data_slice


# Find the weighting coefficients for the 2D CMIP lat-lon grid (structured but not necessarily regular, eg ORCA1 grid) to interpolate to the given boundary.
# Also interpolate the other axis (latitude for E/W boundary, longitude for N/S) to this slice.
def find_slice_weights (cmip_grid, model_grid, location, gtype):

    # Find the lon/lat value at the boundary
    loc0_centre, loc0_edge = find_obcs_boundary(model_grid, location)
    if (location in ['N','S'] and gtype=='v') or (location in ['E','W'] and gtype=='u'):
        loc0 = loc0_edge
    else:
        loc0 = loc0_centre
    cmip_lon, cmip_lat = cmip_grid.get_lon_lat(gtype=gtype)
    if gtype == 'v':
        # There is something weird with the northernmost row of longitude on the v-grid. So replace it with the second last row.
        cmip_lon[-1,:] = cmip_lon[-2,:]
    # Set up an array of zeros, which will be filled with nonzero-weights where needed
    weights = np.zeros(cmip_lon.shape)
    # Unfortunately there is necessary code repetition here...
    if location in ['E', 'W']:
        # Loop from south to north
        for j in range(cmip_grid.ny):
            # Find the coefficients to interpolate to the boundary longitude in this row
            i1, i2, c1, c2 = interp_slice_helper(cmip_lon[j,:], loc0, lon=True)
            # Save these coefficients in the weighting array
            if i1==i2:
                # Value is in array
                weights[j,i1] = 1
            else:
                weights[j,i1] = c1
                weights[j,i2] = c2
        haxis = extract_slice(cmip_lat, weights, location)
        # Parameters for error checking
        N = cmip_grid.ny
        axis = 1
    elif location in ['N', 'S']:
        # Loop from west to east
        for i in range(cmip_grid.nx):
            j1, j2, c1, c2 = interp_slice_helper(cmip_lat[:,i], loc0)
            weights[j1,i] = c1
            weights[j2,i] = c2
        haxis = extract_slice(cmip_lon, weights, location)
        N = cmip_grid.nx
        axis = 0
    # Error checking
    if np.count_nonzero(np.sum(weights, axis=axis)) != N:
        print('Error (find_slice_weights): Something went wrong')
        sys.exit()
    return weights, haxis


# Create open boundary conditions from a CMIP6 model (in practice, UKESM1-0-LL). This is a bit more complicated as they're time-varying (rather than a single climatology), not on a regular lat-lon grid, and not from another MITgcm model.
# Assumes 30-day months, and ocean longitude in the range (-180, 180).
def cmip6_obcs (location, grid_path, expt, mit_start_year=None, mit_end_year=None, cmip_model_path='/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/', ensemble_member='r1i1p1f2', output_dir='./', nc_out=None, prec=32):

    from .file_io import NCfile, read_netcdf
    from .grid import CMIPGrid
    from .interpolation import interp_bdry

    output_dir = real_dir(output_dir)

    # Fields to interpolate
    fields_mit = ['THETA', 'SALT', 'UVEL', 'VVEL', 'SIarea', 'SIheff', 'SIhsnow', 'SIuice', 'SIvice']
    fields_cmip = ['thetao', 'so', 'uo', 'vo', 'siconc', 'sithick', 'sisnthick', 'siu', 'siv']
    # Flag for number of dimensions
    dim = [3, 3, 3, 3, 2, 2, 2, 2, 2]
    # Flag for grid type
    gtype = ['t', 't', 'u', 'v', 't', 't', 't', 'u', 'v']
    # Flag for realm
    realm = ['Omon', 'Omon', 'Omon', 'Omon', 'SImon', 'SImon', 'SImon', 'SImon', 'SImon']
    # Middle of filenames for output (will be preceded by variable, and followed by year)
    outfile_mid = '_'+expt+'.OBCS_'+location+'_'

    print('Building MITgcm grid')
    model_grid = Grid(grid_path)
    print('Building CMIP6 model grid')
    cmip_grid = CMIPGrid(cmip_model_path, expt, ensemble_member)
    print('Calculating weighting coefficients to extract slice')
    # Do this for each grid
    weights_t, haxis_t = find_slice_weights(cmip_grid, model_grid, location, 't')
    weights_u, haxis_u = find_slice_weights(cmip_grid, model_grid, location, 'u')
    weights_v, haxis_v = find_slice_weights(cmip_grid, model_grid, location, 'v')

    # Inner function to tile the weights in the z and time dimensions (12 months)
    def tile_weights (weights):
        weights_2d_time = add_time_dim(weights, 12)
        weights_3d_time = add_time_dim(xy_to_xyz(weights, cmip_grid), 12)
        return weights_2d_time, weights_3d_time
    
    # Inner function to choose the right weights/haxis for grid and dimension
    def get_weights_haxis (gtype, num_dim):
        if gtype == 't':
            weights = weights_t
            haxis = haxis_t
        elif gtype == 'u':
            weights = weights_u
            haxis = haxis_u
        elif gtype == 'v':
            weights = weights_v
            haxis = haxis_v
        weights_2d_time, weights_3d_time = tile_weights(weights)
        if num_dim==2:
            return weights_2d_time, haxis
        elif num_dim==3:
            return weights_3d_time, haxis

    if nc_out is not None:
        # Set up a NetCDF file just for the first year
        ncfile = NCfile(nc_out, model_grid, 'xyzt')
        ncfile.add_time(np.arange(12)+1, units='months')

    # Process each field
    for n in range(len(fields_mit)):
        print(('Variable ' + fields_mit[n]))

        # Organise grids
        print('Tiling weights to correct dimensions')
        weights, cmip_haxis = get_weights_haxis(gtype[n], dim[n])
        model_lon, model_lat = model_grid.get_lon_lat(gtype=gtype[n], dim=1)
        model_hfac = model_grid.get_hfac(gtype=gtype[n])
        if location in ['N', 'S']:
            model_haxis = model_lon
        else:
            model_haxis = model_lat
        # Get hfac on the boundary
        if location == 'S':
            model_hfac = model_hfac[:,0,:]
        elif location == 'N':
            model_hfac = model_hfac[:,-1,:]
        elif location == 'W':
            model_hfac = model_hfac[...,0]
        elif location == 'E':
            model_hfac = model_hfac[...,-1]
        if dim[n] == 2:
            # Just need surface hfac
            model_hfac = model_hfac[0,:]
        h_is_lon = location in ['N', 'S']
        extend_south = (model_haxis[0] < cmip_haxis[0]) and not h_is_lon
        if extend_south:
            # Will need to extend CMIP data to the south. Just add one row.
            cmip_haxis = np.concatenate(([model_haxis[0]-0.1], cmip_haxis))
        
        # Figure out where all the files are, and which years they cover
        in_files, start_years, end_years = find_cmip6_files(cmip_model_path, expt, ensemble_member, fields_cmip[n], realm[n])
        if mit_start_year is None:
            mit_start_year = start_years[0]
        if mit_end_year is None:
            mit_end_year = end_years[-1]
        if fields_mit[n] == 'SIarea':
            # Save file list for sea ice area
            in_files_aice = in_files
            start_years_aice = start_years
            end_years_aice = end_years
        
        # Loop over each file
        for t in range(len(in_files)):
            file_path = in_files[t]
            print(('Processing ' + file_path))
            print(('Covers years '+str(start_years[t])+' to '+str(end_years[t])))
            
            # Loop over years
            t_start = 0  # Time index in file
            t_end = t_start+months_per_year
            for year in range(start_years[t], end_years[t]+1):
                if year >= mit_start_year and year <= mit_end_year:
                    print(('Reading ' + str(year) + ' from indices ' + str(t_start) + '-' + str(t_end)))
                    # Read data
                    data = read_netcdf(file_path, fields_cmip[n], t_start=t_start, t_end=t_end)
                    if fields_mit[n] == 'SIarea':
                        # Convert from percent to fraction
                        data *= 1e-2
                    if fields_mit[n] in ['SIheff', 'SIhsnow', 'SIuice', 'SIvice']:
                        # These variables are masked in regions of zero sea ice. Fill those regions with zeros instead.
                        mask = cmip_grid.get_mask(gtype=gtype[n], surface=True)
                        index = np.where(data.mask*np.invert(mask))
                        data[index] = 0
                    if fields_mit[n] in ['SIheff', 'SIhsnow']:
                        # These variables need to be weighted by sea ice concentration.
                        # Make sure the concentration files (saved from before) line up with these files.
                        if (start_years != start_years_aice) or (end_years != end_years_aice):
                            print(('Error (cmip6_obcs): siconc files do not line up with ' + fields_cmip[n] + ' files. You will need to edit the code.'))
                            sys.exit()
                        data_aice = read_netcdf(in_files_aice[t], 'siconc', t_start=t_start, t_end=t_end)*1e-2
                        data *= data_aice
                        
                    # Extract the slice
                    data_slice = extract_slice(data, weights, location)
                    # Get mask as 1s and 0s
                    data_mask = np.invert(data_slice[0,:].mask).astype(int)
                    if extend_south:                    
                        data_slice = np.ma.concatenate((np.expand_dims(data_slice[...,0],-1), data_slice), axis=-1)
                        data_mask = np.concatenate((np.expand_dims(data_mask[...,0],-1), data_mask), axis=-1)

                    # Interpolate each month in turn
                    if dim[n] == 3:
                        data_interp = np.zeros([12, model_grid.nz, model_haxis.size])
                    else:
                        data_interp = np.zeros([12, model_haxis.size])
                    for month in range(12):
                        print(('Interpolating ' + str(year) + '/' + str(month+1)))
                        data_interp_tmp = interp_bdry(cmip_haxis, cmip_grid.z, data_slice[month,:], data_mask, model_haxis, model_grid.z, model_hfac, lon=h_is_lon, depth_dependent=(dim[n]==3))
                        if fields_mit[n] not in ['THETA', 'SALT']:
                            # Zero in land mask is more physical than extrapolated data
                            index = model_hfac==0
                            data_interp_tmp[index] = 0
                        data_interp[month,:] = data_interp_tmp

                    # Write the data
                    write_binary(data_interp, output_dir+fields_mit[n]+outfile_mid+str(year), prec=prec)
                    if nc_out is not None and year==mit_start_year:
                        # Save the first year in NetCDF file
                        if location in ['N', 'S']:
                            dimension = 'x'
                        else:
                            dimension = 'y'
                        if dim[n] == 3:
                            dimension += 'z'
                        dimension += 't'
                        ncfile.add_variable(fields_mit[n]+'_'+location, data_interp, dimension)

                # Update time range for next time
                t_start = t_end
                t_end = t_start + months_per_year

    if nc_out is not None:
        ncfile.close()


# Correct the normal velocity in OBCS files to prevent massive sea level drift.
# Option 1 ('balance'): Calculate net transport into the domain based on OBCS velocities alone. This can be done before simulations even start, and should work well if you have useRealFreshwaterFlux turned off.
# Option 2 ('correct'): Calculate net transport based on the mean change in sea surface height over a test simulation. Run the model for a while, see how much the area-averaged eta changes over some number of years (timeseries.py should be helpful here), and then run this script to counteract any drift with OBCS corrections.
# Option 3 ('dampen'): Dampen large seasonal cycles in net transport, by correcting the velocities on a monthly-varying basis. A maximum change in mean sea surface height in any one month is specified (default 0.5 m), and a scaling factor for transport is determined such that the maximum absolute value of transport is no more than this threshold. This assumes the OBCS are monthly!!

# Arguments:
# grid: Grid OR path to grid directory

# Optional keyword arguments:
# option: 'balance', 'correct', or 'dampen' (as above)
# obcs_file_w_u, obcs_file_e_u, obcs_file_s_v, obcs_file_n_v: paths to OBCS files for UVEL (western and eastern boundaries) or VVEL (southern and northern boundaries). You only have to set the filenames for the boundaries which are actually open in your domain. They will be overwritten with corrected versions.
# d_eta: if option='correct', change in area-averaged sea surface height over the test simulation (m)
# d_t: if option='correct', length of the test simulation (years)
# max_deta_dt: if option='dampen', maximum allowable change in mean sea surface height in any given month (default 0.5 m/month)
# multi_year: process many files, one per boundary per year
# start_year, end_year: range of years to process, if multi_year=True. The filenames will be appended by these years.
# prec: precision of the OBCS files (as in function sose_obcs
def balance_obcs (grid, option='balance', in_dir='./', obcs_file_w_u=None, obcs_file_e_u=None, obcs_file_s_v=None, obcs_file_n_v=None, d_eta=None, d_t=None, max_deta_dt=0.5, multi_year=False, start_year=None, end_year=None, prec=32):

    if option == 'correct' and (d_eta is None or d_t is None):
        print('Error (balance_obcs): must set d_eta and d_t for option="correct"')
        sys.exit()
    if multi_year and (start_year is None or end_year is None):
        print('Error (balance_obcs): must set start_year and end_year when multi_year=True')
        sys.exit()        

    in_dir = real_dir(in_dir)
    grid = choose_grid(grid, None)

    # Set up the filenames as lists
    def make_file_list (file_head):
        if file_head is None:
            if multi_year:
                return [None for year in range(start_year, end_year+1)]
            else:
                return [None]
        else:
            if multi_year:
                return [in_dir+file_head+str(year) for year in range(start_year, end_year+1)]
            else:
                return [in_dir+file_head]
            
    obcs_files_w_u = make_file_list(obcs_file_w_u)
    obcs_files_e_u = make_file_list(obcs_file_e_u)
    obcs_files_s_v = make_file_list(obcs_file_s_v)
    obcs_files_n_v = make_file_list(obcs_file_n_v)
    if multi_year:
        num_years = end_year-start_year+1
    else:
        num_years = 1

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
    files = [obcs_files_w_u, obcs_files_e_u, obcs_files_s_v, obcs_files_n_v]
    dimensions = ['yzt', 'yzt', 'xzt', 'xzt']
    sign = [1, -1, 1, -1]  # Multiply velocity variable by this to get incoming transport
    # Initialise number of timesteps per file
    num_months = None

    # Integrate the total area of ocean cells on boundaries
    # Should not change over time
    total_area = 0
    for i in range(len(files)):
        if files[i][0] is not None:
            print(('Calculating area of ' + bdry_key[i] + ' boundary'))
            total_area += np.sum(dA_bdry[i])            

    # Calculate the net transport into the domain
    if option in ['balance', 'dampen']:
        # Transport based on OBCS normal velocities
        if option == 'balance':
            net_transport = np.zeros(num_years)
        elif option == 'dampen':
            # Set size later once num_months is set
            net_transport = None
        for t in range(num_years):
            for i in range(len(files)):
                if files[i][t] is not None:
                    print(('Processing ' + bdry_key[i] + ' boundary from ' + files[i][t]))
                    # Read data
                    vel = read_binary(files[i][t], [grid.nx, grid.ny, grid.nz], dimensions[i], prec=prec)
                    if num_months is None:
                        # Find number of time indices
                        num_months = vel.shape[0]
                    elif num_months != vel.shape[0]:
                        print('Error (balance_obcs): inconsistent number of time indices between OBCS files')
                        sys.exit()
                    if option == 'dampen' and net_transport is None:
                        # Initialise transport per month
                        net_transport = np.zeros([num_years, num_months])
                    if option == 'balance':
                        # Time-average velocity (this is equivalent to calculating the transport at each month and then time-averaging at the end - it's all just sums)
                        vel = np.mean(vel, axis=0)
                        # Integrate net transport through this boundary into the domain, and add to global sum
                        net_transport[t] += np.sum(sign[i]*vel*dA_bdry[i])
                    elif option == 'dampen':
                        # Integrate net transport at each month
                        for tt in range(num_months):
                            net_transport[t,tt] += np.sum(sign[i]*vel[tt,:]*dA_bdry[i])
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
        print(('Net transport is ' + str(abs(transport*1e-6)) + ' Sv ' + direction))

    if option == 'correct':
        print_net_transport(net_transport)
    else:
        for t in range(num_years):
            if multi_year:
                print(('Year ' + str(start_year+t)))
            if option == 'dampen':
                for tt in range(num_months):
                    print(('Month ' + str(tt+1)))
                    print_net_transport(net_transport[t,tt])
            else:
                print_net_transport(net_transport[t])        

    if option == 'dampen':
        # Calculate the acceptable maximum absolute transport
        # First need total area of sea surface (including cavities) in domain
        surface_area = np.sum(mask_land(grid.dA, grid))
        max_transport = max_deta_dt*surface_area/(sec_per_day*30)
        print(('Maximum allowable transport is ' + str(max_transport*1e-6) + ' Sv'))
        correction = np.zeros([num_years, num_months])
        for t in range(num_years):
            if multi_year:
                print(('Year ' + str(start_year+t)))
            if np.max(np.abs(net_transport[t,:])) <= max_transport:
                print('OBCS satisfy max allowable transport; nothing to do')
                continue
            # Work out by what factor to dampen the transports
            scale_factor = max_transport/np.max(np.abs(net_transport[t,:]))
            print(('Will scale transports by ' + str(scale_factor)))
            # Calculate corresponding velocity correction at each month
            for tt in range(num_months):
                correction[t,tt] = (scale_factor-1)*net_transport[t,tt]/total_area
                print(('Month ' + str(tt+1) + ': will apply correction of ' + str(correction[t,tt]) + ' m/s to normal velocity at each boundary'))
    else:
        # Calculate correction in m/s
        correction = -1*net_transport/total_area
        # Print results
        if option == 'correct':
            # Just a single value
            print(('Will apply correction of ' + str(correction) + ' m/s to normal velocity at each boundary'))
        elif option == 'balance':
            # Loop over years (even if just one)
            for t in range(num_years):
                if multi_year:
                    print(('Year ' + str(start_year+t)))
                print(('Will apply correction of ' + str(correction[t]) + ' m/s to normal velocity at each boundary'))

    # Now apply the correction
    for t in range(num_years):
        for i in range(len(files)):
            if files[i][t] is not None:
                print(('Correcting ' + files[i][t]))
                # Read all the data again
                vel = read_binary(files[i][t], [grid.nx, grid.ny, grid.nz], dimensions[i], prec=prec)
                # Apply the correction
                if option == 'dampen':
                    for tt in range(num_months):
                        vel[tt,:] += sign[i]*correction[t,tt]
                elif option == 'balance':
                    vel += sign[i]*correction[t]
                else:
                    vel += sign[i]*correction
                # Overwrite the file
                write_binary(vel, files[i][t], prec=prec)

    if option in ['balance', 'dampen']:
        # Recalculate the transport to make sure it worked
        if option == 'balance':
            net_transport_new = np.zeros(num_years)
        elif option == 'dampen':
            net_transport_new = np.zeros([num_years, num_months])
        for t in range(num_years):
            for i in range(len(files)):
                if files[i][t] is not None:
                    vel = read_binary(files[i][t], [grid.nx, grid.ny, grid.nz], dimensions[i], prec=prec)
                    if option == 'balance':
                        vel = np.mean(vel, axis=0)
                        net_transport_new[t] += np.sum(sign[i]*vel*dA_bdry[i])
                    elif option == 'dampen':
                        for tt in range(num_months):
                            net_transport_new[t,tt] += np.sum(sign[i]*vel[tt,:]*dA_bdry[i])
            if multi_year:
                print(('Year ' + str(start_year+t)))
            if option == 'balance':
                print_net_transport(net_transport_new[t])
            elif option == 'dampen':
                for tt in range(num_months):
                    print(('Month ' + str(tt+1)))
                    print_net_transport(net_transport_new[t,tt])


# Merge two sets of initial conditions for temperature and salinity, to keep the values from the first set in the deep ocean, and the values for the second set on the continental shelf (defined by the 2500 m isobath plus ice shelf cavities).
def ics_merge (grid_path, temp_file_deep, salt_file_deep, temp_file_shelf, salt_file_shelf, temp_file_out, salt_file_out, h0=-2500, prec=64, nc_out=None):

    from .file_io import NCfile

    # Build the grid
    grid = Grid(grid_path)
    # Read the existing fields
    temp_deep = read_binary(temp_file_deep, [grid.nx, grid.ny, grid.nz], 'xyz', prec=prec)
    salt_deep = read_binary(salt_file_deep, [grid.nx, grid.ny, grid.nz], 'xyz', prec=prec)
    temp_shelf = read_binary(temp_file_shelf, [grid.nx, grid.ny, grid.nz], 'xyz', prec=prec)
    salt_shelf = read_binary(salt_file_shelf, [grid.nx, grid.ny, grid.nz], 'xyz', prec=prec)
    # Get 3D index of deep ocean
    bathy_3d = xy_to_xyz(grid.bathy, grid)
    ice_mask_3d = xy_to_xyz(grid.ice_mask, grid)
    index_deep = (bathy_3d <= h0)*np.invert(ice_mask_3d)
    # Merge the two arrays
    temp = temp_shelf
    temp[index_deep] = temp_deep[index_deep]
    salt = salt_shelf
    salt[index_deep] = salt_deep[index_deep]
    # Write to file
    write_binary(temp, temp_file_out, prec=prec)
    write_binary(salt, salt_file_out, prec=prec)
    if nc_out is not None:
        ncfile = NCfile(nc_out, grid, 'xyz')
        ncfile.add_variable('THETA', temp, 'xyz')
        ncfile.add_variable('SALT', salt, 'xyz')
        ncfile.close()
    

    
    
