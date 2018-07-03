###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, BinaryGrid
from utils import real_dir
from constants import sose_nx, sose_ny, sose_nz, sose_res
from file_io import read_binary, write_binary, NCfile
from interpolation import interp_reg_3d_mask, interp_fill_reg_3d

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
    fields_2d = ['AREA', 'HEFF']
    # End of filenames for input
    infile_tail = '_climatology.data'
    # End of filenames for output
    outfile_tail = '_SOSE.ini'

    sose_dims = [sose_nx, sose_ny, sose_nz]

    print 'Building grids'
    if split == 180:
        grid = Grid(grid_file)
        if grid.lon_1d[0] > grid.lon_1d[-1]:
            print 'Error (sose_ics): Looks like your domain crosses 180E. Run this again with split=0.'
            sys.exit()
    elif split == 0:
        grid = Grid(grid_file, max_lon=360)
        if grid.lon_1d[0] > grid.lon_1d[-1]:
            print 'Error (sose_ics): Looks like your domain crosses 0E. Run this again with split=180.'
            sys.exit()
    else:
        print 'Error (sose_ics): split must be 180 or 0'
        sys.exit()
            

    


    
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
