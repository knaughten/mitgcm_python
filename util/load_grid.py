import numpy as np
import sys

from read_netcdf import read_netcdf
from fix_lon_range import fix_lon_range

# Load all the useful grid variables and store them in a Grid object.
# Disclaimer: I wrote this for an ocean application using a polar spherical (regular lat-lon) grid. It might not work for other sorts of grids, or atmospheric applications.

class Grid:

    # Initialistaion
    
    # Arguments:
    # path: path to NetCDF grid file (if nc=True) or run directory containing all the binary grid files (if nc=False)
    # nc: Default True. If False, reads the grid from binary files instead of from a NetCDF file.

    # Output: Grid object containing lots of grid variables - read comments in code to find them all.

    # Examples:
    # Read grid from NetCDF file:
    # grid = Grid('grid.nc')
    # Read grid from binary files in run directory
    # grid = Grid('../run/', nc=False)
    # A few of the grid variables:
    # grid.lon_2d
    # grid.lat_2d
    # grid.z
    # grid.dA
    # grid.hfac
    
    def __init__ (self, path, nc=True):

        if nc:
            # Read grid from a NetCDF file

            # 1D lon and lat axes on regular grids
            # Make sure longitude is between -180 and 180
            # Cell centres
            self.lon_1d = fix_lon_range(read_netcdf(path, 'X'))
            self.lat_1d = read_netcdf(path, 'Y')
            # Cell corners
            self.lon_psi_1d = fix_lon_range(read_netcdf(path, 'Xp1'))
            self.lat_psi_1d = read_netcdf(path, 'Yp1')

            # 2D lon and lat fields on any grid
            # Cell centres
            self.lon_2d = fix_lon_range(read_netcdf(path, 'XC'))
            self.lat_2d = read_netcdf(path, 'YC')
            # Cell corners
            self.lon_psi_2d = fix_lon_range(read_netcdf(path, 'XG'))
            self.lat_psi_2d = read_netcdf(path, 'YG')

            # 2D integrands of distance
            # Across faces
            self.dx = read_netcdf(path, 'dxF')
            self.dy = read_netcdf(path, 'dyF')
            # Between centres
            self.dx_t = read_netcdf(path, 'dxC')
            self.dy_t = read_netcdf(path, 'dyC')
            # Between u-points
            self.dx_u = self.dx  # Equivalent to distance across face
            self.dy_u = read_netcdf(path, 'dyU')
            # Between v-points
            self.dx_v = read_netcdf(path, 'dxV')
            self.dy_v = self.dy  # Equivalent to distance across face
            # Between corners
            self.dx_psi = read_netcdf(path, 'dxG')
            self.dy_psi = read_netcdf(path, 'dyG')

            # 2D integrands of area
            # Area of faces
            self.dA = read_netcdf(path, 'rA')
            # Centered on u-points
            self.dA_u = read_netcdf(path, 'rAw')
            # Centered on v-points
            self.dA_v = read_netcdf(path, 'rAs')
            # Centered on corners
            self.dA_psi = read_netcdf(path, 'rAz')

            # Vertical grid
            # Assumes we're in the ocean so using z-levels - not sure how this
            # would handle atmospheric pressure levels.
            # Flip everything so it starts from the surface.
            # Depth axis at centres of z-levels
            self.z = flip(read_netcdf(path, 'Z'), 0)
            # Depth axis at edges of z-levels
            self.z_edges = flip(read_netcdf(path, 'Zp1'), 0)

            # Vertical integrands of distance
            # Across cells
            self.dz = flip(read_netcdf(path, 'drF'), 0)
            # Between centres
            self.dz_t = flip(read_netcdf(path, 'drC'), 0)

            # Partial cell fractions
            # At centres
            self.hfac = flip(read_netcdf(path, 'HFacC'), 0)
            # At u-points
            self.hfac_u = flip(read_netcdf(path, 'HFacW'), 0)
            # At v-points
            self.hfac_v = flip(read_netcdf(path, 'HFacS'), 0)

            # Topography
            # Bathymetry (bottom depth)
            self.bathy = read_netcdf(path, 'R_low')
            # Ice shelf draft (surface depth)
            self.zice = read_netcdf(path, 'Ro_surf')
            # Water column thickness
            self.wct = read_netcdf(path, 'Depth')

        else:

            print 'Error (load_grid_netcdf.py): the code has only been written for NetCDF grids so far.'
            sys.exit()
        


