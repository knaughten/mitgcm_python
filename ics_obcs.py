###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, BinaryGrid
from utils import real_dir
from constants import sose_nx, sose_ny, sose_nz
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


def sose_ics (grid_file, sose_dir, output_dir, nc_out=None, split_lon=180):

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

    # Make the grids, with longitude from 0 to 360
    grid = Grid(grid_file, max_lon=360)
    sose_grid = BinaryGrid(sose_dir+'grid/', sose_nx, sose_ny, sose_nz, max_lon=360)
    # Figure out which points on the model grid can't be reliably interpolated from SOSE output (as they are outside the bounds, within the land mask, or too near the coast)
    interp_mask = interp_reg_3d_mask(grid, sose_grid)

    # Set up a NetCDF file so the user can check the results
    if nc_out is not None:
        ncfile = NCfile(nc_out, grid, 'xyz')

    # Process 3D fields
    for n in range(len(fields_3d)):
        print 'Processing ' + fields_3d[n]
        print '...reading ' + fields_3d[n]+infile_tail
        # Just keep the January climatology
        sose_data = read_binary(sose_dir+fields_3d[n]+infile_tail, sose_grid, 'xyzt')[0,:]
        data_interp = interp_fill_reg_3d(grid, sose_grid, sose_data, interp_mask)
        print '...writing ' + fields_3d[n]+outfile_tail
        write_binary(data_interp, output_dir+fields_3d[n]+outfile_tail)
        if nc_out is not None:
            print '...adding to ' + nc_out
            ncfile.add_variable(fields_3d[n], data_interp, 'xyz')

    if nc_out is not None:
        ncfile.finished()
