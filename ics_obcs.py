###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, SOSEGrid
from utils import real_dir, xy_to_xyz
from file_io import read_binary, write_binary, NCfile
from interpolation import interp_reg

import numpy as np


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


def sose_ics (grid_file, sose_dir, output_dir, nc_out=None, split=180):

    sose_dir = real_dir(sose_dir)
    output_dir = real_dir(output_dir)

    # 3D fields to interpolate
    fields_3d = ['THETA', 'SALT']
    # 2D fields to interpolate
    fields_2d = ['SIarea', 'SIheff']
    # End of filenames for input
    infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_SOSE.ini'
    
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

    # Figure out which points we don't trust
    # Closed cells according to SOSE
    sose_mask = sose_grid.hfac==0
    # Closed cells according to model, interpolated to SOSE grid
    # Take the floor so that a cell is only considered open if all points used to interpolate it were open
    model_mask = np.floor(interp_reg(model_grid, sose_grid, np.ceil(model_grid.hfac), dim=3, fill_value=0))==0
    # Ice shelf cavity points according to model, interpolated to SOSE grid and tiled to be 3D
    model_zice = xy_to_xyz(interp_reg(model_grid, sose_grid, model_grid.zice, dim=2, fill_value=0), sose_grid)!=0
    # Put them all together into one mask
    discard = sose_mask*model_mask*model_zice

    # Remaining steps:
    # Figure out which points we don't trust
    # Figure out which points we need to fill
    # Loop over variables:
    #   Read the data
    #   Remove the points we don't trust
    #   Fill the points we need to fill
    #   Interpolate
    #   Write to file
    


    
    '''print 'Interpolating mask'
    # Figure out which points on the model grid can't be reliably interpolated from SOSE output (as they are outside the bounds, within the land/ice-shelf mask, or too near the coast)
    interp_mask = interp_reg_3d_mask(grid, sose_grid)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, grid, 'xyz')

    # Process 3D fields
    for n in range(len(fields_3d)):
        print 'Processing ' + fields_3d[n]
        in_file = sose_dir + fields_3d[n] + infile_tail
        out_file = output_dir + fields_2d[n] + outfile_tail
        print '...reading ' + in_file
        # Just keep the January climatology
        sose_data = read_binary(in_file, sose_grid, 'xyzt')[0,:]
        data_interp = interp_fill_reg_3d(grid, sose_grid, sose_data, interp_mask)
        print '...writing ' + out_file
        write_binary(data_interp, out_file)
        if nc_out is not None:
            print '...adding to ' + nc_out
            ncfile.add_variable(fields_3d[n], data_interp, 'xyz')

    if nc_out is not None:
        ncfile.finished()'''
