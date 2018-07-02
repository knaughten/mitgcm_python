###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, BinaryGrid
from utils import real_dir
from constants import sose_nx, sose_ny, sose_nz
from file_io import read_binary, write_binary
import numpy as np

def make_sose_climatology (in_file, out_file, dimensions):

    sose_dim = [sose_nx, sose_ny, sose_nz]
    data = read_binary(in_file, sose_dim, dimensions)
    climatology = np.zeros(data.shape[1:])
    for month in range(12):
        climatology[month,:] = np.mean(data[month::12,:], axis=0)
    write_binary(climatology, out_file)


def sose_ics (grid_file, sose_dir, temp_out, salt_out, aice_out, hice_out, nc_out, split_lon=180):

    temp_file = 'THETA_mnthlyBar.0000000100.data'
    salt_file = 'SALT_mnthlyBar.0000000100.data'

    sose_dir = real_dir(sose_dir)
    
    model_grid = Grid(grid_file, max_lon=360)
    sose_grid = BinaryGrid(sose_dir+'grid/', sose_nx, sose_ny, sose_nz, max_lon=360)

    # Read the data we need
    
    
    # Read SOSE data for all the Januarys
    # Average over all the Januarys
    # Interpolate hFac
    # Interpolate T and S (one 3D function):
    #    Extend data into land mask
    #    Make RegularGridInterpolator (how to deal with missing values?) 
    #    Interpolate one depth level at a time (probably - test memory). Fill missing values with -9999 and apply hFac mask.
    #    Fill missing values with average of their non-missing neighbours. Repeat until all the missing values are filled. Make sure land mask doesn't get in the way.
    # Interpolate sea ice variables (one 2D function):
    #    Make RegularGridInterpolator
    #    Interpolate to surface grid. Fill missing values with 0 and apply land_zice mask.
    # Write each variable to binary
    # Write a NetCDF file for checking
    # Print something to the user

    
