###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, SOSEGrid
from utils import real_dir, xy_to_xyz, z_to_xyz, rms, select_top, fix_lon_range
from file_io import write_binary, NCfile
from interpolation import interp_reg, extend_into_mask, discard_and_fill, neighbours_z, interp_slice_helper, interp_bdry, interp_grid

from MITgcmutils import rdmds
from MITgcmutils.mdjwf import densmdjwf

import numpy as np
import os
import sys


# Calculate a monthly climatology of the given variable in SOSE, from its monthly output over the entire 6-year reanalysis.

# Arguments:
# in_file: binary SOSE file (.data) containing one record for each month of the SOSE period. You can also leave ".data" off as it will get stripped off anyway.
# out_file: desired path to output file
def make_sose_climatology (in_file, out_file):

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
    if split == 180:
        model_grid = Grid(grid_path, max_lon=180)
        if model_grid.lon_1d[0] > model_grid.lon_1d[-1]:
            print 'Error (sose_ics): Looks like your domain crosses 180E. Run this again with split=0.'
            sys.exit()
    elif split == 0:
        model_grid = Grid(grid_path, max_lon=360)
        if model_grid.lon_1d[0] > model_grid.lon_1d[-1]:
            print 'Error (sose_ics): Looks like your domain crosses 0E. Run this again with split=180.'
            sys.exit()
    else:
        print 'Error (sose_ics): split must be 180 or 0'
        sys.exit()
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
# out_file: path to desired output file

# Optional keyword arguments:
# constant_t, constant_s: as in function sose_ics
# rhoConst: reference density as in MITgcm's "data" namelist
# prec: as in function sose_ics

def calc_load_anomaly (grid_path, out_file, constant_t=-1.9, constant_s=34.4, rhoConst=1035, prec=64):

    print 'Things to check in your "data" namelist:'
    print "eosType='MDJWF'"
    print 'rhoConst='+str(rhoConst)
    print 'readBinaryPrec=' + str(prec)

    g = 9.81  # gravity (m/s^2)
    errorTol = 1e-13  # convergence criteria

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


# Create open boundary conditions for temperature, salinity, horizontal ocean and sea ice velocities, sea surface height, sea ice area and thickness, using the SOSE monthly climatology.

# Arguments:
# location: 'N', 'S', 'E', or 'W' corresponding to the open boundary to process (north, south, east, west). So, run this function once for each open boundary in your domain.
# grid_path: path to directory containing MITgcm binary grid files. The latitude or longitude of the open boundary will be determined from this file.
# sose_dir: as in function sose_ics
# output_dir: directory to save binary MITgcm OBCS files

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the interpolated boundary conditions to, so you can easily check that they look okay
# prec: precision to write binary files (32 or 64, must match exf_iprec_obcs in the "data.exf" namelist. If you don't have EXF turned on, it must match readBinaryPrec in "data").

def sose_obcs (location, grid_path, sose_dir, output_dir, nc_out=None, prec=32):

    sose_dir = real_dir(sose_dir)
    output_dir = real_dir(output_dir)

    # Fields to interpolate
    # Important: SIarea has to be before SIuice and SIvice so it can be used for masking
    fields = ['THETA', 'SALT', 'UVEL', 'VVEL', 'ETAN', 'SIarea', 'SIheff', 'SIuice', 'SIvice']  
    # Flag for 2D or 3D
    dim = [3, 3, 3, 3, 2, 2, 2, 2, 2]
    # Flag for grid type
    gtype = ['t', 't', 'u', 'v', 't', 't', 't', 'u', 'v']
    # End of filenames for input
    infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_SOSE.OBCS_' + location

    print 'Building MITgcm grid'
    # Make sure longitude is in the range (0, 360) to match SOSE
    model_grid = Grid(grid_path, max_lon=360)
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
        print 'Error (sose_obcs): invalid location ' + str(location)
        sys.exit()

    print 'Building SOSE grid'
    sose_grid = SOSEGrid(sose_dir+'grid/')
    # Calculate interpolation indices and coefficients to the boundary latitude or longitude
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

        if fields[n] == 'SIarea':
            # We'll need this field later for SIuice and SIvice
            print 'Interpolating sea ice area to u and v grids for masking of sea ice velocity'
            sose_aice_u = interp_grid(sose_data, sose_grid, 't', 'u', time_dependent=True, mask_with_zeros=True, periodic=True)
            sose_aice_v = interp_grid(sose_data, sose_grid, 't', 'v', time_dependent=True, mask_with_zeros=True, periodic=True)
        # Set sea ice velocity to zero wherever sea ice area is zero
        if fields[n] in ['SIuice', 'SIvice']:
            print 'Masking sea ice velocity with sea ice area'
            if fields[n] == 'SIuice':
                index = sose_aice_u==0
            else:
                index = sose_aice_v==0
            sose_data[index] = 0            
        
        # Choose the correct grid for lat, lon, hfac
        sose_lon, sose_lat = sose_grid.get_lon_lat(gtype=gtype[n])
        sose_hfac = sose_grid.get_hfac(gtype=gtype[n])
        model_lon, model_lat = model_grid.get_lon_lat(gtype=gtype[n], dim=1)
        model_hfac = model_grid.get_hfac(gtype=gtype[n])
        # Interpolate to the correct grid and choose the correct horizontal axis
        if location in ['N', 'S']:
            if gtype[n] == 'v':
                sose_data = c1_e*sose_data[...,j1_e,:] + c2_e*sose_data[...,j2_e,:]
                # Multiply hfac by the ceiling of hfac on each side, to make sure we're not averaging over land
                sose_hfac = (c1_e*sose_hfac[...,j1_e,:] + c2_e*sose_hfac[...,j2_e,:])*np.ceil(sose_hfac[...,j1_e,:])*np.ceil(sose_hfac[...,j2_e,:])
            else:
                sose_data = c1*sose_data[...,j1,:] + c2*sose_data[...,j2,:]
                sose_hfac = (c1*sose_hfac[...,j1,:] + c2*sose_hfac[...,j2,:])*np.ceil(sose_hfac[...,j1,:])*np.ceil(sose_hfac[...,j2,:])
            sose_haxis = sose_lon
            model_haxis = model_lon
            if location == 'S':
                model_hfac = model_hfac[:,0,:]
            else:
                model_hfac = model_hfac[:,-1,:]
        else:
            if gtype[n] == 'u':
                sose_data = c1_e*sose_data[...,i1_e] + c2_e*sose_data[...,i2_e]
                sose_hfac = (c1_e*sose_hfac[...,i1_e] + c2_e*sose_hfac[...,i2_e])*np.ceil(sose_hfac[...,i1_e])*np.ceil(sose_hfac[...,i2_e])
            else:
                sose_data = c1*sose_data[...,i1] + c2*sose_data[...,i2]
                sose_hfac = (c1*sose_hfac[...,i1] + c2*sose_hfac[...,i2])*np.ceil(sose_hfac[...,i1])*np.ceil(sose_hfac[...,i2])
            sose_haxis = sose_lat
            model_haxis = model_lat
            if location == 'W':
                model_hfac = model_hfac[...,0]
            else:
                model_hfac = model_hfac[...,-1]
        # For 2D variables, just need surface hfac
        if dim[n] == 2:
            sose_hfac = sose_hfac[0,:]
            model_hfac = model_hfac[0,:]

        # Now interpolate each month to the model grid
        if dim[n] == 3:
            data_interp = np.zeros([12, model_grid.nz, model_haxis.size])
        else:
            data_interp = np.zeros([12, model_haxis.size])
        for month in range(12):
            print '...interpolating month ' + str(month+1)
            data_interp[month,:] = interp_bdry(sose_haxis, sose_grid.z, sose_data[month,:], sose_hfac, model_haxis, model_grid.z, model_hfac, depth_dependent=(dim[n]==3))

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
                
                    
    
        

    
