###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, SOSEGrid
from utils import real_dir, xy_to_xyz, select_top, z_to_xyz
from file_io import read_binary, write_binary, NCfile
from interpolation import interp_reg, extend_into_mask, discard_and_fill
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
    sose_grid = SOSEGrid(sose_dir+'grid/', model_grid, split=split)

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

    # Load the MDJWF density function
    mitgcm_utils_path = real_dir(mitgcm_code_path) + 'utils/python/MITgcmutils/MITgcmutils/'
    if not os.path.isfile(mitgcm_utils_path+'mdjwf.py'):
        print 'Error (calc_load_anomaly): ' + mitgcm_utils_path + ' does not contain the script mdjwf.py.'
        sys.exit()    
    sys.path.insert(0, mitgcm_utils_path)
    from mdjwf import densmdjwf

    # Build the grid
    grid = Grid(grid_path)

    # Calculate the (potential) density of the given T and S.
    rho_cavity = densmdjwf(constant_s, constant_t, 0)
    # Analytical solution to the density integral
    pload = g*abs(grid.zice)*(rho_cavity - rhoConst)

    # Write to file
    write_binary(pload, out_file, prec=prec)

    
