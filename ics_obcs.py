###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

from grid import Grid, BinaryGrid
from utils import revert_lon_range, real_dir
from file_io import read_binary
from constants import sose_nx, sose_ny, sose_nz

def sose_ics (grid_file, sose_dir, temp_out, salt_out, aice_out, hice_out, nc_out, split_lon=180):

    sose_dir = real_dir(sose_dir)
    
    model_grid = Grid(grid_file)
    sose_grid = BinaryGrid(sose_dir + 'grid/', sose_nx, sose_ny, sose_nz)

    if split_lon == 180:
        pass
    elif split_lon == 0:
        # Want longitude values between -180 and 180 so there is no gap at 0
        pass
    else:
        print 'Error (sose_ics): split_lon must be 0 or 180'
        sys.exit()

    
    # Read SOSE data for all the Januarys
    # Average over all the Januarys
    # Interpolate T and S (one 3D function):
    #    Make RegularGridInterpolator (how to deal with missing values?) Maybe interp2d instead?
    #    Interpolate one depth level at a time (probably - test memory). Fill missing values with -9999 and apply hFac mask.
    #    Fill missing values with average of their non-missing neighbours. Repeat until all the missing values are filled. Make sure land mask doesn't get in the way.
    # Interpolate sea ice variables (one 2D function):
    #    Make RegularGridInterpolator
    #    Interpolate to surface grid. Fill missing values with 0 and apply land_zice mask.
    # Write each variable to binary
    # Write a NetCDF file for checking
    # Print something to the user

    
