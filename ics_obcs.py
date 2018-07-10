###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, SOSEGrid
from utils import real_dir, xy_to_xyz, z_to_xyz, rms, select_top, fix_lon_range
from file_io import read_binary, write_binary, NCfile
from interpolation import interp_reg, extend_into_mask, discard_and_fill, neighbours_z, interp_bdry, interp_slice_helper
from constants import sose_nx, sose_ny, sose_nz

import numpy as np
import os
import sys


# Calculate a monthly climatology of the given variable in SOSE, from its monthly output over the entire 6-year reanalysis.

# Arguments:
# in_file: binary SOSE file (.data) containing one record for each month of the SOSE period
# out_file: desired path to output file
# dimensions: 'xy' or 'xyz' for 2D and 3D variables respectively
def make_sose_climatology (in_file, out_file, dimensions):

    sose_dim = [sose_nx, sose_ny, sose_nz]
    data = read_binary(in_file, sose_dim, dimensions+'t')    
    climatology = np.zeros(tuple([12]) + data.shape[1:])
    for month in range(12):
        climatology[month,:] = np.mean(data[month::12,:], axis=0)
    write_binary(climatology, out_file)


# Create initial conditions for temperature, salinity, sea ice area, and sea ice thickness using the SOSE monthly climatology for January. Temperature and salinity will be extrapolated into coastal regions where SOSE is prone to artifacts. Ice shelf cavities will be filled with constant temperature and salinity.

# Arguments:
# grid_file: NetCDF grid file for your MITgcm configuration
# sose_dir: directory containing SOSE monthly climatologies and grid/ subdirectory (available on Scihub at /data/oceans_input/raw_input_data/SOSE_monthly_climatology)
# output_dir: directory to save the binary MITgcm initial conditions files (binary)

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the initial conditions in, so you can easily check that they look okay
# constant_t, constant_s: temperature and salinity to fill ice shelf cavities with (default -1.9 C and 34.4 psu)
# split: longitude to split the SOSE grid at. Must be 180 (if your domain includes 0E; default) or 0 (if your domain includes 180E). If your domain is circumpolar (i.e. includes both 0E and 180E), try either and hope for the best. You might have points falling in the gap between SOSE's periodic boundary, in which case you'll have to write a few patches to wrap the SOSE data around the boundary (do this in the SOSEGrid class in grid.py).
# prec: precision to write binary files (64 or 32, must match readBinaryPrec in "data" namelist)

def sose_ics (grid_file, sose_dir, output_dir, nc_out=None, constant_t=-1.9, constant_s=34.4, split=180, prec=64):

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

    # Number of iterations to remove coastal points from SOSE
    coast_iters = 10
    
    print 'Building grids'
    # First build the model grid and check that we have the right value for split
    if split == 180:
        model_grid = Grid(grid_file)
        if model_grid.lon_1d[0] > model_grid.lon_1d[-1]:
            print 'Error (sose_ics): Looks like your domain crosses 180E. Run this again with split=0.'
            sys.exit()
    elif split == 0:
        model_grid = Grid(grid_file, max_lon=360)
        if model_grid.lon_1d[0] > model_grid.lon_1d[-1]:
            print 'Error (sose_ics): Looks like your domain crosses 0E. Run this again with split=180.'
            sys.exit()
    else:
        print 'Error (sose_ics): split must be 180 or 0'
        sys.exit()
    # Now build the SOSE grid
    sose_grid = SOSEGrid(sose_dir+'grid/', model_grid=model_grid, split=split)

    print 'Building mask for SOSE points to discard'
    # Figure out which points we don't trust
    # (1) Closed cells according to SOSE
    sose_mask = sose_grid.hfac == 0
    # (2) Closed cells according to model, interpolated to SOSE grid
    # Only consider a cell to be open if all the points used to interpolate it are open. But, there are some oscillatory interpolation errors which prevent some of these cells from being exactly 1. So set a threshold of 0.99 instead.
    # Use a fill_value of 1 so that the boundaries of the domain are still considered ocean cells (since sose_grid is slightly larger than model_grid). Boundaries which should be closed will get masked in the next step.
    model_open = interp_reg(model_grid, sose_grid, np.ceil(model_grid.hfac), fill_value=1)
    model_mask = model_open < 0.99
    # (3) Points near the coast (which SOSE tends to say are around 0C, even if this makes no sense). Extend the surface model_mask by coast_iters cells, and tile to be 3D. This will also remove all ice shelf cavities.
    coast_mask = xy_to_xyz(extend_into_mask(model_mask[0,:], missing_val=0, num_iters=coast_iters), sose_grid)
    # Put them all together into one mask
    discard = (sose_mask + model_mask + coast_mask).astype(bool)

    print 'Building mask for SOSE points to fill'
    # Figure out which points we need for interpolation
    # Find ice shelf cavity points according to model, interpolated to SOSE grid
    model_cavity = np.ceil(interp_reg(model_grid, sose_grid, xy_to_xyz(model_grid.zice_mask, model_grid), fill_value=0)).astype(bool)
    # Find open, non-cavity cells
    # This time, consider a cell to be open if any of the points used to interpolate it are open (i.e. ceiling)
    fill = np.ceil(model_open)*np.invert(model_cavity)
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
        # Temperature and salinity should have some values discarded, and extrapolated into cavities. There's no need to do this for the 2D sea ice variables.
        if dim[n] == 3:
            print '...extrapolating into missing regions'
            sose_data = discard_and_fill(sose_data, discard, fill)
            # Fill cavity points with constant values
            sose_data[model_cavity] = constant_value[n]
        print '...interpolating to model grid'
        data_interp = interp_reg(sose_grid, model_grid, sose_data, dim=dim[n])
        # Fill the land mask with zeros
        if dim[n] == 3:
            data_interp[model_grid.hfac==0] = 0
        else:
            data_interp[model_grid.hfac[0,:]==0] = 0
        print '...writing ' + out_file
        write_binary(data_interp, out_file, prec=prec)
        if nc_out is not None:
            print '...adding to ' + nc_out
            if dim[n] == 3:
                ncfile.add_variable(fields[n], data_interp, 'xyz')
            else:
                ncfile.add_variable(fields[n], data_interp, 'xy')

    if nc_out is not None:
        ncfile.close()


# Calculate the initial pressure loading anomaly of the ice shelf. Assume that the water displaced by the ice shelf has the same temperature and salinity as the constant values we filled the ice shelf cavities with in sose_ics.

# Arguments:
# grid_path: path to NetCDF grid file
# mitgcm_code_path: path to your copy of the MITgcm source code repository. This is needed so we can access the official function for MDJWF density calculation.
# out_file: path to desired output file

# Optional keyword arguments:
# constant_t, constant_s: as in function sose_ics
# rhoConst: reference density as in MITgcm's "data" namelist
# prec: as in function sose_ics

def calc_load_anomaly (grid_path, mitgcm_code_path, out_file, constant_t=-1.9, constant_s=34.4, rhoConst=1035, prec=64):

    print 'Things to check in your "data" namelist:'
    print "eosType='MDJWF'"
    print 'rhoConst='+str(rhoConst)
    print 'readBinaryPrec=' + str(prec)

    g = 9.81  # gravity (m/s^2)
    errorTol = 1e-13  # convergence criteria

    # Load the MDJWF density function
    mitgcm_utils_path = real_dir(mitgcm_code_path) + 'utils/python/MITgcmutils/MITgcmutils/'
    if not os.path.isfile(mitgcm_utils_path+'mdjwf.py'):
        print 'Error (calc_load_anomaly): ' + mitgcm_utils_path + ' does not contain the script mdjwf.py.'
        sys.exit()    
    sys.path.insert(0, mitgcm_utils_path)
    from mdjwf import densmdjwf

    # Build the grid
    grid = Grid(grid_path)

    # Get vertical integrands considering z at both centres and edges of layers
    dz_merged = np.zeros(2*grid.nz)
    dz_merged[::2] = abs(grid.z - grid.z_edges[:-1])  # dz of top half of each cell
    dz_merged[1::2] = abs(grid.z_edges[1:] - grid.z)  # dz of bottom half of each cell
    # Initial guess for pressure (dbar) at centres of cells
    press = abs(grid.z)*g*rhoConst*1e-4
    # Get depth arrays of T and S
    temp = np.zeros(grid.nz) + constant_t
    salt = np.zeros(grid.nz) + constant_s

    # Iteratively calculate pressure load anomaly until it converges
    press_old = np.zeros(press.shape)  # Dummy initial value for pressure from last iteration
    while rms(press, press_old) > errorTol:
        print 'RMS error = ' + str(rms(press, press_old))
        # Save old pressure
        press_old = np.copy(press)
        # Calculate density anomaly at centres of cells
        drho_c = densmdjwf(salt, temp, press) - rhoConst
        # Use this for both centres and edges of cells
        drho = np.zeros(dz_merged.shape)
        drho[::2] = drho_c
        drho[1::2] = drho_c
        # Integrate pressure load anomaly (Pa)
        pload_full = np.cumsum(drho*g*dz_merged)
        # Update estimate of pressure
        press = (abs(grid.z)*g*rhoConst + pload_full[1::2])*1e-4
    print 'Converged'

    # Now extract pload at the ice shelf base
    # First tile pload at level edges to be 3D
    pload_3d = z_to_xyz(pload_full[::2], grid)
    # For each xy point, calculate three variables:
    # (1) pload at the base of the last fully dry ice shelf cell
    # (2) pload at the base of the cell beneath that
    # (3) hFacC for that cell
    # To calculate (1) we have to shift pload_3d_edges upward by 1 cell
    pload_3d_above = neighbours_z(pload_3d)[0]
    pload_above = select_top(pload_3d_above, masked=False, grid=grid)
    pload_below = select_top(pload_3d, masked=False, grid=grid)
    hfac_below = select_top(grid.hfac, masked=False, grid=grid)
    # Now we can interpolate to the ice base
    pload = pload_above + (1-hfac_below)*(pload_below - pload_above)

    # Write to file
    write_binary(pload, out_file, prec=prec)


def sose_obcs (location, grid_file, sose_dir, output_dir, nc_out=None, prec=32):

    sose_dir = real_dir(sose_dir)
    output_dir = real_dir(output_dir)

    # Fields to interpolate
    fields = ['THETA', 'SALT', 'U', 'V', 'ETAN', 'SIarea', 'SIheff', 'SIuice', 'SIvice']
    # Flag for 2D or 3D
    dim = [3, 3, 3, 3, 2, 2, 2, 2, 2]
    # Flag for grid type
    gtype = ['t', 't', 'u', 'v', 't', 't', 't', 'u', 'v']
    # End of filenames for input
    infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_SOSE.ini'

    print 'Building MITgcm grid'
    model_grid = Grid(grid_file)
    # Figure out what the latitude or longitude is on the boundary, both on the centres and outside edges of those cells
    if location == 'S':
        lat0 = model_grid.lat_1d[0]
        lat0_e = model_grid.lat_corners_1d[0]
    elif location == 'N':
        lat0 = model_grid.lat_1d[-1]
        lat0_e = 2*model_grid.lat_corners_1d[-1] - model_grid.lat_corners_1d[-2]
    else:
        # Make sure longitude is in the range (0, 360) first to agree with SOSE
        lon = fix_lon_range(model_grid.lon_1d, max_lon=360)
        lon_corners = fix_lon_range(model_grid.lon_corners_1d, max_lon=360)
        if location == 'W':
            lon0 = lon[0]
            lon0_e = lon_corners[0]
        elif location == 'E':
            lon0 = lon[-1]
            lon0_e = 2*lon_corners[-1] - lon_corners[-2]
        else:
            print 'Error (sose_obcs): invalid location ' + str(location)
            sys.exit()

    print 'Building SOSE grid'
    sose_grid = SOSEGrid(sose_dir+'grid/')
    # Find interpolation coefficients to the boundary latitude or longitude
    if location in ['N', 'S']:
        # Cell centre
        j1, j2, c1, c2 = interp_slice_helper(sose_grid.lat, lat0)
        # Cell edge
        j1_e, j2_e, c1_e, c2_e = interp_slice_helper(sose_grid.lat_corners, lat0_e)
    else:
        # Pass lon=True to consider the possibility of boundary near 0E
        i1, i2, c1, c2 = interp_slice_helper(sose_grid.lon, lon0, lon=True)
        i1_e, i2_e, c1_e, c2_e = interp_slice_helper(sose_grid.lon_corners, lon0_e, lon=True)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, model_grid, 'xyzt')
        ncfile.add_time(np.arange(12)+1, units='months')

    # Process fields
    for n in range(len(fields)):
        
        print 'Processing ' + fields[n]
        in_file = sose_dir + fields[n] + infile_tail
        out_file = output_dir + fields[n] + outfile_tail
        # Read the monthly climatology at all points
        if dim[n] == 3:
            sose_data = sose_grid.read_field(in_file, 'xyzt')
        else:
            sose_data = sose_grid.read_field(in_file, 'xyt')

        # Mask the SOSE data
        sose_hfac = sose_grid.get_hfac(gtype=gtype[n])
        if dim == 3:
            sose_hfac = np.tile(sose_hfac, (12, 1, 1, 1))
        else:
            sose_hfac = np.tile(sose_hfac[0,:], (12, 1, 1))
        sose_data = np.ma.masked_where(sose_hfac==0, sose_data)
        
        # Choose the correct grid
        sose_lon, sose_lat = sose_grid.get_lon_lat(gtype=gtype[n])
        model_lon, model_lat = model_grid.get_lon_lat(gtype=gtype[n], dim=1)
        # Interpolate to the correct grid and choose the correct horizontal axis
        if location in ['N', 'S']:
            if gtype[n] == 'v':
                sose_data = c1_e*sose_data[...,j1_e,:] + c2_e*sose_data[...,j2_e,:]
            else:
                sose_data = c1*sose_data[...,j1,:] + c2*sose_data[...,j2,:]
            sose_haxis = sose_lon
            model_haxis = model_lon
        else:
            if gtype[n] == 'u':
                sose_data = c1_e*sose_data[...,i1_e] + c2_e*sose_data[...,i2_e]
            else:
                sose_data = c1*sose_data[...,i1] + c2*sose_data[...,i2]
            sose_haxis = sose_lat
            model_haxis = model_lat

        # Get the model's hFac on the boundary
        if location == 'S':
            model_hfac = model_grid.get_hfac(gtype=gtype[n])[:,0,:]
        elif location == 'N':
            model_hfac = model_grid.get_hfac(gtype=gtype[n])[:,-1,:]
        elif location == 'W':
            model_hfac = model_grid.get_hfac(gtype=gtype[n])[:,:,0]
        elif location == 'E':
            model_hfac = model_grid.get_hfac(gtype=gtype[n])[:,:,-1]
        # For 2D variables, select just the top level
        if dim[n] == 2:
            model_hfac = model_hfac[0,:]

        # Now interpolate each month to the model grid
        if dim[n] == 3:
            data_interp = np.zeros([12, model_grid.nz, model_haxis.size])
        else:
            data_interp = np.zeros([12, model_haxis.size])
        for month in range(12):
            print '...interpolating month ' + str(month+1)
            data_interp[month,:] = interp_bdry(sose_haxis, sose_grid.z, sose_data[month,:], model_haxis, model_grid.z, model_hfac, depth_dependent=(dim[n]==3))

        print '...writing ' + out_file
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
                
                    
    
        

    
