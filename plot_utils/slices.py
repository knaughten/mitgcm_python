#############################################################
# Utilities specific to slice plots (lat-depth or lon-depth).
# Also general transect between two points, or data along
# ice shelf front (scroll down)
#############################################################

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import sys

from ..utils import dist_btw_points, ice_shelf_front_points


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
        print(('Error (get_slice_values): the ' + gtype + '-grid is not supported for slices'))

    # Figure out direction of slice
    if lon0 is not None and lat0 is None:
        h_axis = 'lat'
    elif lat0 is not None and lon0 is None:
        h_axis = 'lon'
    else:
        print('Error (get_slice_values): must specify exactly one of lon0, lat0')
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

    # Get horizontal boundaries and hfac
    # Also throw away one row of data so all points are bounded
    if h_axis == 'lat':
        if gtype in ['t', 'u']:
            # Centered in y
            # Throw away northernmost row of data
            data_slice = data_slice[:,:-1]
            if return_grid_vars:
                # Boundaries are southern edges of cells in y            
                h_bdry = grid.lat_corners_1d            
                # Get hfac at centres
                hfac = grid.hfac[:,:-1,i0]
        elif gtype in ['v', 'psi']:
            # Edges in y
            # Throw away southernmost row of data
            data_slice = data_slice[:,1:]
            if return_grid_vars:
                # Boundaries are centres of cells in y
                h_bdry = grid.lat_1d
                # Get hfac at edges
                hfac = grid.hfac_s[:,1:,i0]
    elif h_axis == 'lon':
        if gtype in ['t', 'v']:
            # Centered in x
            # Throw away easternmost row of data
            data_slice = data_slice[:,:-1]
            if return_grid_vars:
                # Boundaries are western edges of cells in x
                h_bdry = grid.lon_corners_1d
                # Get hfac at centres
                hfac = grid.hfac[:,j0,:-1]
        elif gtype in ['u', 'psi']:
            # Edges in x
            # Throw away westernmost row of data
            data_slice = data_slice[:,1:]
            if return_grid_vars:
                # Boundaries are centres of cells in x
                h_bdry = grid.lon_1d
                # Get hfac at edges
                hfac = grid.hfac_w[:,j0,1:]

    if return_grid_vars:
        return data_slice, h_bdry, hfac, loc0
    else:
        return data_slice


def get_slice_boundaries (data_slice, grid, h_bdry, hfac):

    # Set up a bunch of information about the grid, all stored in arrays with the same dimension as data_slice. This helps with vectorisation later.
    nh = data_slice.shape[1]
    # Left and right boundaries (lat or lon)
    left = np.tile(h_bdry[:-1], (grid.nz, 1))
    right = np.tile(h_bdry[1:], (grid.nz, 1))
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
    index = (hfac>0)*(hfac<1)*(hfac_above==0)*(hfac_below>0)
    depth_above[index] = lev_below[index] + dz[index]*hfac[index]
    # Partial cells with seafloor but no ice shelves: wet portion is at the top
    index = (hfac>0)*(hfac<1)*(hfac_above>0)*(hfac_below==0)
    depth_below[index] = lev_above[index] - dz[index]*hfac[index]    
    # Partial cells with both ice shelves and seafloor - this is a problem with the grid - print a warning, and assume the wet portion is at the bottom.
    index = (hfac>0)*(hfac<1)*(hfac_above==0)*(hfac_below==0)
    if np.count_nonzero(index) > 0:
        print('Warning (get_slice_boundaries): this grid has partial cells with both ice shelves and seafloor. They will be pinched, and the position of the wet portion may not be accurate in this plot.')
        depth_above[index] = lev_below[index] + dz[index]*hfac[index]
    
    # Now we need to merge depth_above and depth_below, because depth_above for one cell is equal to depth_below for the cell above, and vice versa.
    # Figure out the other option for depth_above based on depth_below
    depth_above_2 = np.zeros(data_slice.shape)
    depth_above_2[1:,:] = depth_below[:-1,:]
    depth_above_2[0,:] = depth_above[0,:]  # No other option for surface
    # Should never be nonzero in the same place
    if np.any(depth_above*depth_above_2 != 0):
        print('Warning (get_slice_boundaries): something went wrong in calculation of partial cells')
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
        print('Warning (get_slice_boundaries): something went wrong in calculation of partial cells')
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


def slice_patches (data, grid, gtype='t', lon0=None, lat0=None, hmin=None, hmax=None, zmin=None, zmax=None, return_bdry=False, return_gridded=False):

    data_slice, h_bdry, hfac, loc0 = get_slice_values(data, grid, gtype=gtype, lon0=lon0, lat0=lat0)
    left, right, below, above = get_slice_boundaries(data_slice, grid, h_bdry, hfac)
    vmin, vmax, hmin, hmax, zmin, zmax = get_slice_minmax(data_slice, left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax)
    patches = get_slice_patches(data_slice, left, right, below, above)

    if return_gridded:
        haxis = (left[0,:]+right[0,:])/2.

    if return_bdry:
        if return_gridded:
            return patches, data_slice.ravel(), loc0, hmin, hmax, zmin, zmax, vmin, vmax, left, right, below, above, data_slice, haxis, grid.z
        else:
            return patches, data_slice.ravel(), loc0, hmin, hmax, zmin, zmax, vmin, vmax, left, right, below, above
    else:
        if return_gridded:
            return patches, data_slice.ravel(), loc0, hmin, hmax, zmin, zmax, vmin, vmax, data_slice, haxis, grid.z
        else:
            return patches, data_slice.ravel(), loc0, hmin, hmax, zmin, zmax, vmin, vmax            


def slice_values (data, grid, left, right, below, above, hmin, hmax, zmin, zmax, gtype='t', lon0=None, lat0=None, return_gridded=False):

    data_slice = get_slice_values(data, grid, gtype=gtype, lon0=lon0, lat0=lat0, return_grid_vars=False)
    vmin, vmax = get_slice_minmax(data_slice, left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_spatial=False)
    if return_gridded:
        return data_slice.ravel(), vmin, vmax, data_slice
    else:
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


# Extract the data and boundaries along a general transect between two (lon,lat) points. This replaces get_slice_values and get_slice_boundaries.
def get_transect (data, grid, point0, point1, gtype='t', return_grid_vars=True, time_dependent=False):

    # Extract the coordinates from the start and end points, so that we start at the southernmost point
    flip = point1[1] < point0[1]
    if flip:
        [lon0, lat0] = point1
        [lon1, lat1] = point0
    else:
        [lon0, lat0] = point0
        [lon1, lat1] = point1
        
    # Boolean indicating direction of slope
    pos_slope = lon0 < lon1
    
    # Some error checking
    if lon0 == lon1:
        print('Error (get_transect): This is a line of constant longitude. Use the regular slice scripts instead.')
        sys.exit()
    if lat0 == lat1:
        print('Error (get_transect): This is a line of constant latitude. Use the regular slice scripts instead.')
        sys.exit()
    if min(lon0, lon1) < np.amin(grid.lon_corners_1d) or max(lon0, lon1) > np.amax(grid.lon_1d) or lat0 < np.amin(grid.lat_corners_1d) or lat1 > np.amax(grid.lat_1d):
        print('Error (get_transect): This line falls outside of the domain.')
        sys.exit()
    if gtype != 't':
        print('Error (get_transect): gtypes other than t are not yet supported.')
        sys.exit()
    # Save the slope of the line
    slope = float((lat1-lat0))/(lon1-lon0)

    # Find the limits on latitude to search
    # Last edge latitude south of the line
    j_start = np.nonzero(grid.lat_corners_1d > lat0)[0][0] - 1
    # First edge latitude north of the line
    j_end = np.nonzero(grid.lat_corners_1d > lat1)[0][0]
    # Consider edge cases
    j_start = max(j_start, 0)
    j_end = min(j_end, grid.ny-1)

    # Make a list of indices for cells that are intersected by the line.
    cells_intersect = []
    for j in range(j_start, j_end+1):
        # Find the longitude of intersection between the line and this latitude
        lon_star = (grid.lat_corners_1d[j]-lat0)/slope + lon0
        # Find the longitude index of the cell this intersects: last lon_corners west of lon_star, considering the edge case
        i_new = max(np.nonzero(grid.lon_corners_1d > lon_star)[0][0] - 1, 0)
        if j > j_start:
            # Get the cell most recently saved
            [j_old, i_old] = cells_intersect[-1]
            if j_old != j-1:
                print('Error: j_old is not j-1')
                sys.exit()
            # Add the cells between it and the new one, in the right order
            if pos_slope:
                i_range = list(range(i_old+1, i_new+1))
            else:
                i_range = list(range(i_old-1, i_new-1, -1))
            for i in i_range:
                cells_intersect.append((j-1,i))
        # Add the new cell
        if j < j_end:
            cells_intersect.append((j,i_new))

    if flip:
        # Reverse the order of the list
        cells_intersect.reverse()

    # Set up array to save the extracted transects
    if time_dependent:
        num_time = data.shape[0]
        data_trans = np.ma.empty([num_time, grid.nz, len(cells_intersect)])
    else:
        data_trans = np.ma.empty([grid.nz, len(cells_intersect)])
    if return_grid_vars:
        # Also their horizontal boundaries (distance from lon0, lat0) and hfac
        left = np.ma.empty(data_trans.shape[-2:])
        right = np.ma.empty(data_trans.shape[-2:])
        hfac_trans = np.ma.empty(data_trans.shape[-2:])
    # Finally, a counter for the position in axis 1, because we might not keep all the intersected cells
    posn = 0
    for cell in cells_intersect:
        [j, i] = cell
        # Find the intersections between the line and the cell boundaries.
        intersections = []
        lon_left = grid.lon_corners_1d[i]
        lat_bottom = grid.lat_corners_1d[j]
        if i < grid.nx:
            lon_right = grid.lon_corners_1d[i+1]
        else:
            # Extrapolate the boundary
            lon_right = 2*grid.lon_corners_1d[i] - grid.lon_corners_1d[i-1]
        if j < grid.ny:
            lat_top = grid.lat_corners_1d[j+1]
        else:
            lat_top = 2*grid.lat_corners_1d[j] - grid.lat_corners_1d[j-1]
        # Check if the line intersects the bottom
        lon_star = (lat_bottom-lat0)/slope + lon0
        if lon_star >= lon_left and lon_star <= lon_right:
            intersections.append((lon_star, lat_bottom))
        # Check the top
        lon_star = (lat_top-lat0)/slope + lon0
        if lon_star >= lon_left and lon_star <= lon_right:
            intersections.append((lon_star, lat_top))
        # Check the left
        lat_star = (lon_left-lon0)*slope + lat0
        if lat_star >= lat_bottom and lat_star <= lat_top:
            intersections.append((lon_left, lat_star))
        # Check the right
        lat_star = (lon_right-lon0)*slope + lat0
        if lat_star >= lat_bottom and lat_star <= lat_top:
            intersections.append((lon_right, lat_star))
        # We expect there to be 2 items in the list. Check the other cases:
        if len(intersections) == 1:
            # The line just skimmed the corner of the cell. Discard this cell, as its two diagonal neighbours will be picked up.
            continue
        elif len(intersections) in [0,3,4]:
            # This should never happen.
            print(('Error (get_transect): ' + str(len(intersections)) + ' intersections. Something went wrong.'))
            sys.exit()
        # Now save data from this water column to the transect
        data_trans[...,posn] = data[...,j,i]
        if return_grid_vars:
            # Calculate the distance from each intersection to the start of the line
            dist_a = dist_btw_points(intersections[0], point0)
            dist_b = dist_btw_points(intersections[1], point0)
            # Choose the left and right boundaries and convert to km
            left_dist = min(dist_a, dist_b)*1e-3
            right_dist = max(dist_a, dist_b)*1e-3
            # Now save the boundaries and hfac
            left[:,posn] = np.zeros(grid.nz)+left_dist
            right[:,posn] = np.zeros(grid.nz)+right_dist
            hfac_trans[:,posn] = grid.hfac[:,j,i]        
        posn += 1

    # Trim the transects in case we didn't use all the cells
    data_trans = data_trans[...,:posn]
    if not return_grid_vars:
        return data_trans
    else:
        left = left[:,:posn]
        right = right[:,:posn]
        hfac_trans = hfac_trans[:,:posn]
        # Now calculate the top and bottom boundaries
        if time_dependent:
            data_dummy = data_trans[0,:]
        else:
            data_dummy = data_trans
        below, above = get_slice_boundaries(data_dummy, grid, left, hfac_trans)[2:]
        return data_trans, left, right, below, above


# API to build everything for a transect. Equivalent to slice_patches.
def transect_patches (data, grid, point0, point1, gtype='t', zmin=None, zmax=None, return_bdry=False, return_gridded=False):

    data_trans, left, right, below, above = get_transect(data, grid, point0, point1, gtype=gtype)
    vmin, vmax, hmin, hmax, zmin, zmax = get_slice_minmax(data_trans, left, right, below, above, zmin=zmin, zmax=zmax)
    # We don't want hmin to be padded on the left, so set it back to zero if needed
    hmin = max(hmin, 0)
    patches = get_slice_patches(data_trans, left, right, below, above)

    if return_gridded:
        haxis = (left[0,:]+right[0,:])/2

    if return_bdry:
        if return_gridded:
            return patches, data_trans.ravel(), hmin, hmax, zmin, zmax, vmin, vmax, left, right, below, above, data_trans, haxis, grid.z
        else:
            return patches, data_trans.ravel(), hmin, hmax, zmin, zmax, vmin, vmax, left, right, below, above
    else:
        if return_gridded:
            return patches, data_trans.ravel(), hmin, hmax, zmin, zmax, vmin, vmax, data_trans, haxis, grid.z
        else:
            return patches, data_trans.ravel(), hmin, hmax, zmin, zmax, vmin, vmax


# API to get values for already-known patches along a transect. Equivalent to slice_values.
def transect_values (data, grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, gtype='t', return_gridded=False):

    data_trans = get_transect(data, grid, point0, point1, gtype=gtype, return_grid_vars=False)
    vmin, vmax = get_slice_minmax(data_trans, left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_spatial=False)
    if return_gridded:
        return data_trans.ravel(), vmin, vmax, data_trans
    else:
        return data_trans.ravel(), vmin, vmax


def get_iceshelf_front (data, grid, xmin=None, xmax=None, ymin=None, ymax=None, shelf='other', primary_start='W', secondary_start='N', gtype='t'):

    # Check the primary/secondary start variables make sense:
    if primary_start not in ['W', 'E', 'S', 'N']:
        print(('Error (get_iceshelf_front): invalid primary_start ' + primary_start))
        sys.exit()
    if secondary_start not in ['W', 'E', 'S', 'N']:
        print(('Error (get_iceshelf_front): invalid secondary_start ' + secondary_start))
        sys.exit()
    if (primary_start in ['W', 'E'] and secondary_start in ['W', 'E']) or (primary_start in ['S', 'N'] and secondary_start in ['S', 'N']):
        print('Error (get_iceshelf_front): primary_start and secondary_start must be along different dimensions.')
        sys.exit()

    # Threshold distance after which to say the ice shelf is done
    dist_max = 10

    # Set up some variables for this grid
    lon, lat = grid.get_lon_lat(gtype=gtype)
    i_vals, j_vals = np.meshgrid(list(range(grid.nx)), list(range(grid.ny)))
    land_mask = grid.get_land_mask(gtype=gtype)
    fris_mask = grid.get_ice_mask(shelf='fris', gtype=gtype)
    if shelf == 'ronne':
        ice_mask = fris_mask*(lon < -46)
    elif shelf == 'filchner':
        ice_mask = fris_mask*(lon > -46)
    else:
        ice_mask = grid.get_ice_mask(gtype=gtype)
    hfac = grid.get_hfac(gtype=gtype)

    # Find ice shelf front points
    front_points = ice_shelf_front_points(grid, ice_mask=ice_mask, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    # Find first point
    # First narrow down based on the primary direction to start with
    if primary_start == 'W':
        # Find the cell(s) which are furthest west
        index = lon[front_points]==np.amin(lon[front_points])
    elif primary_start == 'E':
        index = lon[front_points]==np.amax(lon[front_points])
    elif primary_start == 'S':
        index = lat[front_points]==np.amin(lat[front_points])
    elif primary_start == 'N':
        index = lat[front_points]==np.amax(lat[front_points])
    # Now consider the secondary direction
    if secondary_start == 'W':
        posn = np.argmin(lon[front_points][index])
    elif secondary_start == 'E':
        posn = np.argmax(lon[front_points][index])
    elif secondary_start == 'S':
        posn = np.argmin(lat[front_points][index])
    elif secondary_start == 'N':
        posn = np.argmax(lat[front_points][index])
    # Now get the i and j values of the first point
    i0 = i_vals[front_points][index][posn]
    j0 = j_vals[front_points][index][posn]
    # Save their lat and lon for labelling
    point_start = (lon[j0,i0], lat[j0,i0])

    # Set up array to save extracted data and hfac
    data_front = np.ma.empty([grid.nz, np.count_nonzero(front_points)])
    hfac_front = np.ma.empty(data_front.shape)
    # Counter for position along axis 1
    counter = 0
    # Make a copy of front_points to keep track of which points remain
    remaining_points = np.copy(front_points)

    # Extract data from the ice shelf points, in order
    while True:
        # Extract the data
        data_front[:,counter] = data[:,j0,i0]
        hfac_front[:,counter] = hfac[:,j0,i0]
        counter += 1
        # Remove this point from the list
        remaining_points[j0,i0] = False
        if np.count_nonzero(remaining_points) == 0:
            # We've done all the points
            break
        # Find the next point (closest point in i-j space)
        dist = np.sqrt((i_vals[remaining_points]-i0)**2 + (j_vals[remaining_points]-j0)**2)
        if np.amin(dist) > dist_max:
            # There are no points within the distance threshold. This will cut off weird artifacts (like sticking-out bits of the ice shelf, 1 cell thick) that weren't captured before.
            break
        posn = np.argmin(dist)
        i0 = i_vals[remaining_points][posn]
        j0 = j_vals[remaining_points][posn]
        
    # Trim if needed
    data_front = data_front[:,:counter]
    hfac_front = hfac_front[:,:counter]
    # Save lat and lon from the last point
    point_end = (lon[j0,i0], lat[j0,i0])

    # Dummy horizontal axis
    left = np.tile(np.arange(data_front.shape[-1]), (grid.nz,1))
    right = left + 1

    # Get vertical boundaries
    below, above = get_slice_boundaries(data_front, grid, left, hfac_front)[2:]

    return data_front, left, right, below, above, point_start, point_end        
            
            


