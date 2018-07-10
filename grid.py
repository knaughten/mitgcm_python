#######################################################
# Everything to do with reading the grid
# This relies on a NetCDF grid file. To create it, run MITgcm just long enough to produce one grid.t*.nc file for each tile, and then glue them together using gluemnc (utils/scripts/gluemnc in the MITgcm distribution).
#######################################################

import numpy as np
import sys

from file_io import read_netcdf, read_binary
from utils import fix_lon_range, real_dir, split_longitude
from constants import fris_bounds, sose_nx, sose_ny, sose_nz, sose_res


# Grid object containing lots of grid variables.
class Grid:

    # Initialisation arguments:
    # file_path: path to NetCDF grid file    
    def __init__ (self, file_path, max_lon=180):

        # 1D lon and lat axes on regular grids
        # Make sure longitude is between -180 and 180
        # Cell centres
        self.lon_1d = fix_lon_range(read_netcdf(file_path, 'X'), max_lon=max_lon)
        self.lat_1d = read_netcdf(file_path, 'Y')
        # Cell corners (southwest)
        self.lon_corners_1d = fix_lon_range(read_netcdf(file_path, 'Xp1'), max_lon=max_lon)
        self.lat_corners_1d = read_netcdf(file_path, 'Yp1')

        # 2D lon and lat fields on any grid
        # Cell centres
        self.lon_2d = fix_lon_range(read_netcdf(file_path, 'XC'), max_lon=max_lon)
        self.lat_2d = read_netcdf(file_path, 'YC')
        # Cell corners
        self.lon_corners_2d = fix_lon_range(read_netcdf(file_path, 'XG'), max_lon=max_lon)
        self.lat_corners_2d = read_netcdf(file_path, 'YG')

        # 2D integrands of distance
        # Across faces
        self.dx = read_netcdf(file_path, 'dxF')
        self.dy = read_netcdf(file_path, 'dyF')
        # Between centres
        self.dx_t = read_netcdf(file_path, 'dxC')
        self.dy_t = read_netcdf(file_path, 'dyC')
        # Between u-points
        self.dx_u = self.dx  # Equivalent to distance across face
        self.dy_u = read_netcdf(file_path, 'dyU')
        # Between v-points
        self.dx_v = read_netcdf(file_path, 'dxV')
        self.dy_v = self.dy  # Equivalent to distance across face
        # Between corners
        self.dx_psi = read_netcdf(file_path, 'dxG')
        self.dy_psi = read_netcdf(file_path, 'dyG')

        # 2D integrands of area
        # Area of faces
        self.dA = read_netcdf(file_path, 'rA')
        # Centered on u-points
        self.dA_u = read_netcdf(file_path, 'rAw')
        # Centered on v-points
        self.dA_v = read_netcdf(file_path, 'rAs')
        # Centered on corners
        self.dA_psi = read_netcdf(file_path, 'rAz')

        # Vertical grid
        # Assumes we're in the ocean so using z-levels - not sure how this
        # would handle atmospheric pressure levels.
        # Depth axis at centres of z-levels
        self.z = read_netcdf(file_path, 'Z')
        # Depth axis at edges of z-levels
        self.z_edges = read_netcdf(file_path, 'Zp1')
        # Depth axis at w-points
        self.z_w = read_netcdf(file_path, 'Zl')

        # Vertical integrands of distance
        # Across cells
        self.dz = read_netcdf(file_path, 'drF')
        # Between centres
        self.dz_t = read_netcdf(file_path, 'drC')

        # Dimension lengths (on tracer grid)
        self.nx = self.lon_1d.size
        self.ny = self.lat_1d.size
        self.nz = self.z.size

        # Partial cell fractions
        # At centres
        self.hfac = read_netcdf(file_path, 'HFacC')
        # On western edges
        self.hfac_w = read_netcdf(file_path, 'HFacW')
        # On southern edges
        self.hfac_s = read_netcdf(file_path, 'HFacS')

        # Create masks on the t, u, and v grids
        # We can't do the psi grid because there is no hfac there
        # Land masks
        self.land_mask = self.build_land_mask(self.hfac)
        self.land_mask_u = self.build_land_mask(self.hfac_w)
        self.land_mask_v = self.build_land_mask(self.hfac_s)
        # Ice shelf masks
        self.zice_mask = self.build_zice_mask(self.hfac)
        self.zice_mask_u = self.build_zice_mask(self.hfac_w)
        self.zice_mask_v = self.build_zice_mask(self.hfac_s)
        # FRIS masks
        self.fris_mask = self.build_fris_mask(self.zice_mask, self.lon_2d, self.lat_2d)
        self.fris_mask_u = self.build_fris_mask(self.zice_mask_u, self.lon_corners_2d, self.lat_2d)
        self.fris_mask_v = self.build_fris_mask(self.zice_mask_v, self.lon_2d, self.lat_corners_2d)

        # Topography (as seen by the model after adjustment for eg hfacMin - not necessarily equal to what is specified by the user)
        # Bathymetry (bottom depth)
        self.bathy = read_netcdf(file_path, 'R_low')
        # Ice shelf draft (surface depth, enforce 0 in land or open-ocean points)
        self.zice = read_netcdf(file_path, 'Ro_surf')
        self.zice[np.invert(self.zice_mask)] = 0
        # Water column thickness
        self.wct = read_netcdf(file_path, 'Depth')

        
    # Given a 3D hfac array on any grid, create the land mask.
    def build_land_mask (self, hfac):

        return np.sum(hfac, axis=0)==0


    # Given a 3D hfac array on any grid, create the ice shelf mask.
    def build_zice_mask (self, hfac):

        return (np.sum(hfac, axis=0)!=0)*(hfac[0,:]==0)


    # Create a mask just containing FRIS ice shelf points.
    # Arguments:
    # zice_mask, lon, lat: 2D arrays of the ice shelf mask, longitude, and latitude on any grid
    def build_fris_mask (self, zice_mask, lon, lat):

        fris_mask = np.zeros(zice_mask.shape, dtype='bool')
        # Identify FRIS in two parts, split along the line 45W
        # Each set of 4 bounds is in form [lon_min, lon_max, lat_min, lat_max]
        regions = [[fris_bounds[0], -45, fris_bounds[2], -74.7], [-45, fris_bounds[1], fris_bounds[2], -77.85]]
        for bounds in regions:
            # Select the ice shelf points within these bounds
            index = zice_mask*(lon >= bounds[0])*(lon <= bounds[1])*(lat >= bounds[2])*(lat <= bounds[3])
            fris_mask[index] = True
        return fris_mask

        
    # Return the longitude and latitude arrays for the given grid type.
    # 't' (default), 'u', 'v', 'psi', and 'w' are all supported.
    # Default returns the 2D meshed arrays; can set dim=1 to get 1D axes.
    def get_lon_lat (self, gtype='t', dim=2):

        if dim == 1:
            lon = self.lon_1d
            lon_corners = self.lon_corners_1d
            lat = self.lat_1d
            lat_corners = self.lat_corners_1d
        elif dim == 2:
            lon = self.lon_2d
            lon_corners = self.lon_corners_2d
            lat = self.lat_2d
            lat_corners = self.lat_corners_2d
        else:
            print 'Error (get_lon_lat): dim must be 1 or 2'
            sys.exit()

        if gtype in ['t', 'w']:
            return lon, lat
        elif gtype == 'u':
            return lon_corners, lat
        elif gtype == 'v':
            return lon, lat_corners
        elif gtype == 'psi':
            return lon_corners, lat_corners
        else:
            print 'Error (get_lon_lat): invalid gtype ' + gtype
            sys.exit()


    # Return the hfac array for the given grid type.
    # 'psi' and 'w' have no hfac arrays so they are not supported
    def get_hfac (self, gtype='t'):

        if gtype == 't':
            return self.hfac
        elif gtype == 'u':
            return self.hfac_w
        elif gtype == 'v':
            return self.hfac_s
        else:
            print 'Error (get_hfac): no hfac exists for the ' + gtype + ' grid'
            sys.exit()


    # Return the land mask for the given grid type.
    def get_land_mask (self, gtype='t'):

        if gtype == 't':
            return self.land_mask
        elif gtype == 'u':
            return self.land_mask_u
        elif gtype == 'v':
            return self.land_mask_v
        else:
            print 'Error (get_land_mask): no mask exists for the ' + gtype + ' grid'
            sys.exit()

            
    # Return the ice shelf mask for the given grid type.
    def get_zice_mask (self, gtype='t'):

        if gtype == 't':
            return self.zice_mask
        elif gtype == 'u':
            return self.zice_mask_u
        elif gtype == 'v':
            return self.zice_mask_v
        else:
            print 'Error (get_zice_mask): no mask exists for the ' + gtype + ' grid'
            sys.exit()


    # Return the FRIS mask for the given grid type.
    def get_fris_mask (self, gtype='t'):

        if gtype == 't':
            return self.fris_mask
        elif gtype == 'u':
            return self.fris_mask_u
        elif gtype == 'v':
            return self.fris_mask_v
        else:
            print 'Error (get_fris_mask): no mask exists for the ' + gtype + ' grid'
            sys.exit()


# Special class for the SOSE grid, which is read from a few binary files. It inherits many functions from Grid.

# To speed up interpolation, trim and/or extend the SOSE grid to agree with the bounds of model_grid (Grid object for the model which you'll be interpolating SOSE data to).
# Depending on the longitude range within the model grid, it might also be necessary to rearrange the SOSE grid so it splits at 180E=180W (split=180, default, implying longitude ranges from -180 to 180 and max_lon=180 when creating model_grid) instead of its native split at 0E (split=0, implying longitude ranges from 0 to 360 and max_lon=360 when creating model_grid).
# The rule of thumb is, if your model grid includes 0E, split at 180E, and vice versa. A circumpolar model should be fine either way as long as it doesn't have any points in the SOSE periodic boundary gap (in which case you'll have to write a patch). 
# MOST IMPORTANTLY, if you are reading a SOSE binary file, don't use read_binary from file_io. Use the class function read_field (defined below) which will repeat the trimming/extending/splitting/rearranging correctly.

# If you don't want to do any trimming or extending, just set model_grid=None and split=360 (or nothing as 360 is the default).
class SOSEGrid(Grid):

    def __init__ (self, grid_dir, model_grid=None, split=360):

        grid_dir = real_dir(grid_dir)
        self.orig_dims = [sose_nx, sose_ny, sose_nz]

        self.trim_extend = True
        if model_grid is None:
            self.trim_extend = False

        if self.trim_extend:
            # Error checking for which longitude range we're in
            if split == 180:
                max_lon = 180
                if np.amax(model_grid.lon_2d) > max_lon:
                    print 'Error (SOSEGrid): split=180 does not match model grid'
            elif split == 0:
                max_lon = 360
                if np.zmin(model_grid.lon_2d) < 0:
                    print 'Error (SOSEGrid): split=0 does not match model grid'
            else:
                print 'Error (SOSEGrid): split must be 180 or 0'
                sys.exit()
        else:
            # Make sure we're not splitting
            if split != 360:
                print "Error (SOSEGrid): can't split unless model_grid is defined"
                sys.exit()            

        # Read longitude at cell centres (make the 2D grid 1D as it's regular)
        self.lon = fix_lon_range(read_binary(grid_dir+'XC.data', self.orig_dims, 'xy'), max_lon=max_lon)[0,:]
        if split == 180:
            # Split the domain at 180E=180W and rearrange the two halves so longitude is strictly ascending
            self.i_split = np.nonzero(self.lon < 0)[0][0]
            self.lon = split_longitude(self.lon, self.i_split)
        else:
            # Set i_split to 0 which won't actually do anything
            self.i_split = 0
            
        # Read longitude at cell corners, splitting as before
        self.lon_corners = split_longitude(fix_lon_range(read_binary(grid_dir+'XG.data', self.orig_dims, 'xy'), max_lon=max_lon), self.i_split)[0,:]
        if self.lon_corners[0] > 0:
            # The split happened between lon_corners[i_split] and lon[i_split].
            # Take mod 360 on this index of lon_corners to make sure it's strictly increasing.
            self.lon_corners[0] -= 360

        # Make sure the longitude axes are strictly increasing after the splitting
        if not np.all(np.diff(self.lon)>0) or not np.all(np.diff(self.lon_corners)>0):
            print 'Error (SOSEGrid): longitude is not strictly increasing'
            sys.exit()
            
        # Read latitude at cell centres and corners
        self.lat = read_binary(grid_dir+'YC.data', self.orig_dims, 'xy')[:,0]
        self.lat_corners = read_binary(grid_dir+'YG.data', self.orig_dims, 'xy')[:,0]
        # Read depth
        self.z = read_binary(grid_dir+'RC.data', self.orig_dims, 'z')

        if self.trim_extend:
        
            # Trim and/or extend the axes
            # Notes about this:
            # Longitude can only be trimmed as SOSE considers all longitudes (someone doing a high-resolution circumpolar model with points in the gap might need to write a patch to wrap the SOSE grid around)
            # Latitude can be trimmed in both directions, or extended to the south (not extended to the north - if you need to do this, SOSE is not the right product for you!)
            # Depth can be extended by one level in both directions, and the deeper bound can also be trimmed
            # The indices i, j, and k will be kept track of with 4 variables each. For example, with longitude:
            # i0_before = first index we care about
            #           = how many cells to trim at beginning
            # i0_after = i0_before's position in the new grid
            #          = how many cells to extend at beginning
            # i1_before = first index we don't care about
            #           sose_nx - i1_before = how many cells to trim at end
            # i1_after = i1_before's position in the new grid
            #          = i1_before - i0_before + i0_after
            # nx = length of new grid
            #      nx - i1_after = how many cells to extend at end

            # Find bounds on model grid
            xmin = np.amin(model_grid.lon_corners_2d)
            xmax = np.amax(model_grid.lon_2d)
            ymin = np.amin(model_grid.lat_corners_2d)
            ymax = np.amax(model_grid.lat_2d)
            z_shallow = model_grid.z[0]
            z_deep = model_grid.z[-1]

            # Western bound (use longitude at cell centres to make sure all grid types clear the bound)
            if xmin == self.lon[0]:
                # Nothing to do
                self.i0_before = 0            
            elif xmin > self.lon[0]:
                # Trim
                self.i0_before = np.nonzero(self.lon > xmin)[0][0] - 1
            else:
                print 'Error (SOSEGrid): not allowed to extend westward'
                sys.exit()
            self.i0_after = 0

            # Eastern bound (use longitude at cell corners, i.e. western edge)
            if xmax == self.lon_corners[-1]:
                # Nothing to do
                self.i1_before = sose_nx
            elif xmax < self.lon_corners[-1]:
                # Trim
                self.i1_before = np.nonzero(self.lon_corners > xmax)[0][0] + 1
            else:
                print 'Error (SOSEGrid): not allowed to extend eastward'
                sys.exit()
            self.i1_after = self.i1_before - self.i0_before + self.i0_after
            self.nx = self.i1_after

            # Southern bound (use latitude at cell centres)
            if ymin == self.lat[0]:
                # Nothing to do
                self.j0_before = 0
                self.j0_after = 0
            elif ymin > self.lat[0]:
                # Trim
                self.j0_before = np.nonzero(self.lat > ymin)[0][0] - 1
                self.j0_after = 0
            elif ymin < self.lat[0]:
                # Extend
                self.j0_after = int(np.ceil((self.lat[0]-ymin)/sose_res))
                self.j0_before = 0

            # Northern bound (use latitude at cell corners, i.e. southern edge)
            if ymax == self.lat_corners[-1]:
                # Nothing to do
                self.j1_before = sose_ny
            elif ymax < self.lat_corners[-1]:
                # Trim
                self.j1_before = np.nonzero(self.lat_corners > ymax)[0][0] + 1
            else:
                print 'Error (SOSEGrid): not allowed to extend northward'
                sys.exit()
            self.j1_after = self.j1_before - self.j0_before + self.j0_after
            self.ny = self.j1_after

            # Depth
            self.k0_before = 0
            if z_shallow <= self.z[0]:
                # Nothing to do
                self.k0_after = 0
            else:
                # Extend
                self.k0_after = 1
            if z_deep > self.z[-1]:
                # Trim
                self.k1_before = np.nonzero(self.z < z_deep)[0][0]
            else:
                # Either extend or do nothing
                self.k1_before = sose_nz
            self.k1_after = self.k1_before + self.k0_after
            if z_deep < self.z[-1]:
                # Extend
                self.nz = self.k1_after + 1
            else:
                self.nz = self.k1_after

            # Now we have the indices we need, so trim/extend the axes as needed
            # Longitude: can only trim
            self.lon = self.lon[self.i0_before:self.i1_before]
            self.lon_corners = self.lon_corners[self.i0_before:self.i1_before]
            # Latitude: can extend on south side, trim on both sides
            lat_extend = np.flipud(-1*(np.arange(self.j0_after)+1)*sose_res + self.lat[self.j0_before])
            lat_trim = self.lat[self.j0_before:self.j1_before]        
            self.lat = np.concatenate((lat_extend, lat_trim))
            lat_corners_extend = np.flipud(-1*(np.arange(self.j0_after)+1)*sose_res + self.lat_corners[self.j0_before])
            lat_corners_trim = self.lat_corners[self.j0_before:self.j1_before]        
            self.lat_corners = np.concatenate((lat_corners_extend, lat_corners_trim))
            # Depth: can extend on both sides (depth 0 at top and extrapolated at bottom to clear the deepest model depth), trim on deep side
            z_above = 0*np.ones([self.k0_after])  # Will either be [0] or empty
            z_middle = self.z[self.k0_before:self.k1_before]
            z_below = (2*model_grid.z[-1] - model_grid.z[-2])*np.ones([self.nz-self.k1_after])   # Will either be [something deeper than z_deep] or empty
            self.z = np.concatenate((z_above, z_middle, z_below))

            # Make sure we cleared those bounds
            if self.lon_corners[0] > xmin:
                print 'Error (SOSEGrid): western bound not cleared'
                sys.exit()
            if self.lon_corners[-1] < xmax:
                print 'Error (SOSEGrid): eastern bound not cleared'
                sys.exit()
            if self.lat_corners[0] > ymin:
                print 'Error (SOSEGrid): southern bound not cleared'
                sys.exit()
            if self.lat_corners[-1] < ymax:
                print 'Error (SOSEGrid): northern bound not cleared'
                sys.exit()
            if self.z[0] < z_shallow:
                print 'Error (SOSEGrid): shallow bound not cleared'
                sys.exit()
            if self.z[-1] > z_deep:
                print 'Error (SOSEGrid): deep bound not cleared'
                sys.exit()

            # Now read the rest of the variables we need, splitting/trimming/extending them as needed
            self.hfac = self.read_field(grid_dir+'hFacC.data', 'xyz', fill_value=0)
            self.hfac_w = self.read_field(grid_dir+'hFacW.data', 'xyz', fill_value=0)
            self.hfac_s = self.read_field(grid_dir+'hFacS.data', 'xyz', fill_value=0)

        else:

            # Nothing fancy to do, so read the rest of the fields
            self.hfac = read_binary(grid_dir+'hFacC.data', self.orig_dims, 'xyz')
            self.hfac_w = read_binary(grid_dir+'hFacW.data', self.orig_dims, 'xyz')
            self.hfac_s = read_binary(grid_dir+'hFacS.data', self.orig_dims, 'xyz')
            self.nx = sose_nx
            self.ny = sose_ny
            self.nz = sose_nz

        # Create land masks
        self.land_mask = self.build_land_mask(self.hfac)
        self.land_mask_u = self.build_land_mask(self.hfac_w)
        self.land_mask_v = self.build_land_mask(self.hfac_s)
    


    # Read a field from a binary MDS file and split, trim, extend as needed.
    # The field can be time dependent: dimensions must be one of 'xy', 'xyt', 'xyz', or 'xyzt'.
    # Extended regions will just be filled with fill_value for now. See function discard_and_fill in interpolation.py for how to extrapolate data into these regions.
    def read_field (self, file_path, dimensions, fill_value=-9999):

        # Expect to have xy in the dimensions. The only case which won't get caught by read_binary is z alone.
        if dimensions == 'z':
            print 'Error (read_field): not set up to read fields of dimension ' + dimensions
            sys.exit()

        if self.trim_extend:

            # Read the field and split along longitude
            data_orig = split_longitude(read_binary(file_path, self.orig_dims, dimensions), self.i_split)
            # Create a new array of the correct dimension (including extended regions)
            data_shape = [self.ny, self.nx]
            if 'z' in dimensions:
                data_shape = [self.nz] + data_shape        
            if 't' in dimensions:
                num_time = data_orig.shape[0]
                data_shape = [num_time] + data_shape
            data = np.zeros(data_shape) + fill_value

            # Trim
            if 'z' in dimensions:
                data[..., self.k0_after:self.k1_after, self.j0_after:self.j1_after, self.i0_after:self.i1_after] = data_orig[..., self.k0_before:self.k1_before, self.j0_before:self.j1_before, self.i0_before:self.i1_before]
            else:
                data[..., self.j0_after:self.j1_after, self.i0_after:self.i1_after] = data_orig[..., self.j0_before:self.j1_before, self.i0_before:self.i1_before]

        else:
            # Nothing fancy to do
            data = read_binary(file_path, self, dimensions)

        return data
            

    # Return the longitude and latitude arrays for the given grid type.
    def get_lon_lat (self, gtype='t', dim=1):

        # We need to have dim as a keyword argument so this agrees with the Grid class function, but there is no option for dim=2
        if dim != 1:
            print 'Error (get_lon_lat): must have dim=1 for SOSE grid'
            sys.exit()

        if gtype in ['t', 'w']:
            return self.lon, self.lat
        elif gtype == 'u':
            return self.lon_corners, self.lat
        elif gtype == 'v':
            return self.lon, self.lat_corners
        elif gtype == 'psi':
            return self.lon_corners, self.lat_corners
        else:
            print 'Error (get_lon_lat): invalid gtype ' + gtype
            sys.exit()


    # Dummy definitions for functions we don't want, which would otherwise be inhertied from Grid
    def build_zice_mask (self, hfac):
        print 'Error (SOSEGrid): no ice shelves to mask'
        sys.exit()
    def build_fris_mask (self, hfac):
        print 'Error (SOSEGrid): no ice shelves to mask'
        sys.exit()
    def get_zice_mask (self, gtype='t'):
        print 'Error (SOSEGrid): no ice shelves to mask'
        sys.exit()
    def get_fris_mask (self, gtype='t'):
        print 'Error (SOSEGrid): no ice shelves to mask'
        sys.exit()
    
