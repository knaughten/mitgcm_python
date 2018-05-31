#############################################################
# Utilities specific to slice plots (lat-depth or lon-depth).
#############################################################

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import sys


# Create the rectangular Polygon patches for plotting a slice (necessary to properly represent partial cells) and the corresponding data values. This is done with 4 helper functions:
# 1) get_slice_values: slice the data and the grid values to the given longitude or latitude
# 2) get_slice_boundaries: determine the boundaries of each sliced cell: latitude or longitude to the left and right, and depth above and below
# 3) get_slice_minmax: find the minimum and maximum values in the slice, as well as the spatial bounds of unmasked data; this is necessary because automatic axes limits and colour mapping is not supported by general patch plots
# 4) get_slice_patches: build the array of Polygon patches

# There are two APIs you can use depending on the situation:
# slice_patches builds everything. It outputs the array of patches and their corresponding values, the modified value of lon0 or lat0 (nearest-neighbour), the spatial bounds hmin, hmax, zmin, zmax (if they were specified, they are not modified), and the min and max values within these bounds.
# slice_values assumes that the patches have already been built using get_slice_values (with return_bdry=True so that the boundary arrays left, right, below, above are saved) and you are plotting another field on the same patches. It only calls step 1 (with return_grid_vars=False so that only the new data is sliced) and 3 (with return_spatial=False so that only the min and max values are calculated). It outputs the values corresponding to the already-known patches, and the min and max values.

# Keyword arguments:
# lon0, lat0: longitude or latitude to slice along (its nearest-neighbour). Exactly one must be specified.
# hmin, hmax, zmin, zmax: desired bounds on horizontal axis (latitude or longitude) and depth axis (negative, in metres).


def get_slice_values (data, grid, gtype='t', lon0=None, lat0=None, return_grid_vars=True):

    if gtype not in ['t', 'u', 'v', 'psi']:
        print 'Error (get_slice_values): the ' + gtype + '-grid is not supported for slices'

    # Figure out direction of slice
    if lon0 is not None and lat0 is None:
        h_axis = 'lat'
    elif lat0 is not None and lon0 is None:
        h_axis = 'lon'
    else:
        print 'Error (get_slice_values): must specify exactly one of lon0, lat0'
        sys.exit()

    # Find nearest neighbour to lon0 and slice the data here
    lon, lat = grid.get_lon_lat(gtype=gtype, dim=1)
    if h_axis == 'lat':
        i0 = np.argmin(abs(lon-lon0))
        data_slice = data[:,:,i0]
        if return_grid_vars:
            # Save the real location of the slice
            loc0 = lon[i0]
    elif h_axis == 'lon':
        j0 = np.argmin(abs(lat-lat0))
        data_slice = data[:,j0,:]
        if return_grid_vars:
            loc0 = lat[j0]

    # Get horizontal boundaries, as well as hfac and surface depth (grid.zice)
    # Also throw away one row of data so all points are bounded
    if h_axis == 'lat':
        if gtype in ['t', 'u']:
            # Centered in y
            # Throw away northernmost row of data
            data_slice = data_slice[:,:-1]
            if return_grid_vars:
                # Boundaries are southern edges of cells in y            
                h_bdry = grid.lat_corners_1d            
                # Get hfac and zice at centres
                hfac = grid.hfac[:,:-1,i0]
                zice = grid.zice[:-1,i0]
        elif gtype in ['v', 'psi']:
            # Edges in y
            # Throw away southernmost row of data
            data_slice = data_slice[:,1:]
            if return_grid_vars:
                # Boundaries are centres of cells in y
                h_bdry = grid.lat_1d
                # Get hfac at edges
                hfac = grid.hfac_s[:,1:,i0]
                # Ice shelf draft at these edges is the minimum of the tracer points on either side
                zice = np.minimum(grid.zice[:-1,i0], grid.zice[1:,i0])
    elif h_axis == 'lon':
        if gtype in ['t', 'v']:
            # Centered in x
            # Throw away easternmost row of data
            data_slice = data_slice[:,:-1]
            if return_grid_vars:
                # Boundaries are western edges of cells in x
                h_bdry = grid.lon_corners_1d
                # Get hfac and zice at centres
                hfac = grid.hfac[:,j0,:-1]
                zice = grid.zice[j0,:-1]
        elif gtype in ['u', 'psi']:
            # Edges in x
            # Throw away westernmost row of data
            data_slice = data_slice[:,1:]
            if return_grid_vars:
                # Boundaries are centres of cells in x
                h_bdry = grid.lon_1d
                # Get hfac at edges
                hfac = grid.hfac_w[:,j0,1:]
                # Ice shelf draft at these edges is the minimum of the tracer points on either side
                zice = np.minimum(grid.zice[j0,:-1], grid.zice[j0,1:])

    if return_grid_vars:
        return data_slice, h_bdry, hfac, zice, loc0
    else:
        return data_slice


def get_slice_boundaries (data_slice, grid, h_bdry, hfac, zice):

    # Set up a bunch of information about the grid, all stored in arrays with the same dimension as data_slice. This helps with vectorisation later.
    nh = data_slice.shape[1]
    # Left and right boundaries (lat or lon)
    left = np.tile(h_bdry[:-1], (grid.nz, 1))
    right = np.tile(h_bdry[1:], (grid.nz, 1))
    # Ice shelf draft
    zice = np.tile(zice, (grid.nz, 1))    
    # Depth of vertical layer above
    lev_above = np.tile(np.expand_dims(grid.z_edges[:-1], 1), (1, nh))
    # Depth of vertical layer below
    lev_below = np.tile(np.expand_dims(grid.z_edges[1:], 1), (1, nh))
    # Vertical thickness
    dz = np.tile(np.expand_dims(grid.dz, 1), (1, nh))
    # hfac one row above (zeros for air)
    hfac_above = np.zeros(data_slice.shape)
    hfac_above[1:,:] = hfac[:-1,:]
    # hfac one row below (zeros for seafloor)
    hfac_below = np.zeros(data_slice.shape)
    hfac_below[:-1,:] = hfac[1:,:]

    # Now find the true upper and lower boundaries, taking partial cells into account.
    # Start with zeros everywhere and deal with partial cells first.
    depth_above = np.zeros(data_slice.shape)
    depth_below = np.zeros(data_slice.shape)
    # Partial cells with ice shelves but no seafloor: wet portion is at the bottom
    index = np.nonzero((hfac>0)*(hfac<1)*(hfac_above==0)*(hfac_below>0))
    depth_above[index] = lev_below[index] + dz[index]*hfac[index]
    # Partial cells with seafloor but no ice shelves: wet portion is at the top
    index = np.nonzero((hfac>0)*(hfac<1)*(hfac_above>0)*(hfac_below==0))
    depth_below[index] = lev_above[index] - dz[index]*hfac[index]
    # Partial cells with both ice shelves and seafloor - now we need to use the surface depth as seen by the model, to properly position the wet portion of the cell.
    index = np.nonzero((hfac>0)*(hfac<1)*(hfac_above==0)*(hfac_below==0))
    depth_above[index] = zice[index]
    depth_below[index] = depth_above[index] - dz[index]*hfac[index]
    
    # Now we need to merge depth_above and depth_below, because depth_above for one cell is equal to depth_below for the cell above, and vice versa.
    # Figure out the other option for depth_above based on depth_below
    depth_above_2 = np.zeros(data_slice.shape)
    depth_above_2[1:,:] = depth_below[:-1,:]
    depth_above_2[0,:] = depth_above[0,:]  # No other option for surface
    # Should never be nonzero in the same place
    if np.any(depth_above*depth_above_2 != 0):
        print 'Error (get_slice_boundaries): something went wrong in calculation of partial cells'
        sys.exit()
    # Add them together to capture all the nonzero values
    above = depth_above + depth_above_2
    # Anything still zero is just the regular z levels
    index = above == 0
    above[index] = lev_above[index]
    # Similarly for depth_below
    depth_below_2 = np.zeros(data_slice.shape)
    depth_below_2[:-1,:] = depth_above[1:,:]
    depth_below_2[-1,:] = depth_below[-1,:]
    if np.any(depth_below*depth_below_2 != 0):
        print 'Error (get_slice_boundaries): something went wrong in calculation of partial cells'
        sys.exit()
    below = depth_below + depth_below_2
    index = below == 0
    below[index] = lev_below[index]

    return left, right, below, above


def get_slice_minmax (data_slice, left, right, below, above, hmin=None, hmax=None, zmin=None, zmax=None, return_spatial=True):

    # Figure out if we'll need to determine spatial bounds, and if so, set temporary ones
    calc_hmin = hmin is None
    if calc_hmin:
        hmin = np.amin(left)
    calc_hmax = hmax is None
    if calc_hmax:
        hmax = np.amax(right)
    calc_zmin = zmin is None
    if calc_zmin:
        zmin = np.amin(below)
    calc_zmax = zmax is None
    if calc_zmax:
        zmax = np.amax(above)
    # Select all the unmasked entries between these bounds
    index = np.nonzero((left >= hmin)*(right <= hmax)*(below >= zmin)*(above <= zmax)*(np.invert(data_slice.mask)))
    # Find the min and max values
    vmin = np.amin(data_slice[index])
    vmax = np.amax(data_slice[index])
    if return_spatial:
        # Find any unset spatial bounds on unmasked data
        if calc_hmin:
            hmin = np.amin(left[index])
        if calc_hmax:
            hmax = np.amax(right[index])
        if calc_zmin:
            zmin = np.amin(below[index])
        if calc_zmax:
            zmax = np.amax(above[index])
        # Pad the left and/or bottom with a bit of the mask
        if calc_hmin:
            hmin -= 0.015*(hmax-hmin)
        if calc_zmin:
            zmin -= 0.015*(zmax-zmin)
        return vmin, vmax, hmin, hmax, zmin, zmax
    else:
        return vmin, vmax


def get_slice_patches (data_slice, left, right, below, above):

    num_pts = data_slice.size
    # Set up coordinates, tracing around outside of patches
    coord = np.zeros([num_pts, 4, 2])
    # Top left corner
    coord[:,0,0] = left.ravel()
    coord[:,0,1] = above.ravel()
    # Top right corner
    coord[:,1,0] = right.ravel()
    coord[:,1,1] = above.ravel()
    # Bottom right corner
    coord[:,2,0] = right.ravel()
    coord[:,2,1] = below.ravel()
    # Bottom left corner
    coord[:,3,0] = left.ravel()
    coord[:,3,1] = below.ravel()
    # We have to make one patch at a time
    patches = []    
    for i in range(num_pts):
        patches.append(Polygon(coord[i,:], True, linewidth=0.))

    return patches


def slice_patches (data, grid, gtype='t', lon0=None, lat0=None, hmin=None, hmax=None, zmin=None, zmax=None, return_bdry=False):

    data_slice, h_bdry, hfac, zice, loc0 = get_slice_values(data, grid, gtype=gtype, lon0=lon0, lat0=lat0)
    left, right, below, above = get_slice_boundaries(data_slice, grid, h_bdry, hfac, zice)
    vmin, vmax, hmin, hmax, zmin, zmax = get_slice_minmax(data_slice, left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax)
    patches = get_slice_patches(data_slice, left, right, below, above)

    if return_bdry:
        return patches, data_slice.ravel(), loc0, hmin, hmax, zmin, zmax, vmin, vmax, left, right, below, above
    else:
        return patches, data_slice.ravel(), loc0, hmin, hmax, zmin, zmax, vmin, vmax


def slice_values (data, grid, left, right, below, above, hmin, hmax, zmin, zmax, gtype='t', lon0=None, lat0=None):

    data_slice = get_slice_values(data, grid, gtype=gtype, lon0=lon0, lat0=lat0, return_grid_vars=False)
    vmin, vmax = get_slice_minmax(data_slice, left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_spatial=False)
    return data_slice.ravel(), vmin, vmax


# Add Polygon patches to a slice plot. Outputs the image returned by PatchCollection, which can be used to make a colourbar later.
def plot_slice_patches (ax, patches, values, hmin, hmax, zmin, zmax, vmin, vmax, cmap='jet'):

    img = PatchCollection(patches, cmap=cmap)
    img.set_array(values)
    img.set_clim(vmin=vmin, vmax=vmax)
    img.set_edgecolor('face')
    ax.add_collection(img)
    ax.set_xlim([hmin, hmax])
    ax.set_ylim([zmin, zmax])    
    return img

