#######################################################
# All things interpolation
#######################################################

import numpy as np
import sys

from .utils import mask_land, mask_land_ice, mask_3d, xy_to_xyz, z_to_xyz, is_depth_dependent
from .grid import Grid


# Interpolate from one grid type to another. 

# Arguments:
# data: array of dimension (maybe time) x (maybe depth) x lat x lon
# grid: Grid object
# gtype_in: grid type of "data". As in function Grid.get_lon_lat.
# gtype_out: grid type to interpolate to

# Optional keyword arguments:
# time_dependent: as in function apply_mask
# mask_shelf: indicates to mask the ice shelves as well as land. Only valid if "data" isn't depth-dependent.
# mask_with_zeros: indicates to fill the mask with zeros instead of making a MaskedArray (better for interpolation)
# periodic: indicates the grid has an east/west periodic boundary

# Output: array of the same dimension as "data", interpolated to the new grid type

def interp_grid (data, grid, gtype_in, gtype_out, time_dependent=False, mask=True, mask_shelf=False, mask_with_zeros=False, periodic=False):

    depth_dependent = is_depth_dependent(data, time_dependent=time_dependent)
    # Make sure we're not trying to mask the ice shelf from a depth-dependent field
    if mask_shelf and depth_dependent:
        print("Error (interp_grid): can't set mask_shelf=True for a depth-dependent field.")
        sys.exit()

    if mask and gtype_in in ['u', 'v', 'psi', 'w']:
        # Fill the mask with zeros (okay because no-slip boundary condition)
        data_tmp = np.copy(data)
        data_tmp[data.mask] = 0.0
    else:
        # Tracer land mask is the least restrictive, so it doesn't matter what the masked values are - they will definitely get re-masked at the end.
        data_tmp = data        

    # Interpolate
    data_interp = np.empty(data_tmp.shape)
    if gtype_in == 'u' and gtype_out == 't':
        # Midpoints in the x direction
        data_interp[...,:-1] = 0.5*(data_tmp[...,:-1] + data_tmp[...,1:])
        # Extend/wrap the easternmost column
        if periodic:
            data_interp[...,-1] = 0.5*(data_tmp[...,-1] + data_tmp[...,0])
        else:
            data_interp[...,-1] = data_tmp[...,-1]
    elif gtype_in == 'v' and gtype_out == 't':
        # Midpoints in the y direction
        data_interp[...,:-1,:] = 0.5*(data_tmp[...,:-1,:] + data_tmp[...,1:,:])
        # Extend the northernmost row
        data_interp[...,-1,:] = data_tmp[...,-1,:]
    elif gtype_in == 't' and gtype_out == 'u':
        # Midpoints in the x direction
        data_interp[...,1:] = 0.5*(data_tmp[...,:-1] + data_tmp[...,1:])
        # Extend/wrap the westernmost column
        if periodic:
            data_interp[...,0] = data_interp[...,-1]
        else:
            data_interp[...,0] = data_tmp[...,0]
    elif gtype_in == 't' and gtype_out == 'v':
        # Midpoints in the y direction
        data_interp[...,1:,:] = 0.5*(data_tmp[...,:-1,:] + data_tmp[...,:-1,:])
        # Extend the southernmost row
        data_interp[...,0,:] = data_tmp[...,0,:]
    else:
        print('Error (interp_grid): interpolation from the ' + gtype_in + '-grid to the ' + gtype_out + '-grid is not yet supported')
        sys.exit()

    if mask:
        # Now apply the mask
        if depth_dependent:
            data_interp = mask_3d(data_interp, grid, gtype=gtype_out, time_dependent=time_dependent)
        else:
            if mask_shelf:
                data_interp = mask_land_ice(data_interp, grid, gtype=gtype_out, time_dependent=time_dependent)
            else:
                data_interp = mask_land(data_interp, grid, gtype=gtype_out, time_dependent=time_dependent)

        if mask_with_zeros:
            # Remove mask and fill with zeros
            data_interp[data_interp.mask] = 0
            data_interp = data_interp.data

    return data_interp


# Finds the value of the given array to the west, east, south, north of every point, as well as which neighbours are non-missing, and how many neighbours are non-missing.
# Can also do 1D arrays (so just neighbours to the left and right) if you pass use_1d=True.
def neighbours (data, missing_val=-9999, use_1d=False):

    # Find the value to the west, east, south, north of every point
    # Just copy the boundaries
    data_w = np.empty(data.shape)
    data_w[...,1:] = data[...,:-1]
    data_w[...,0] = data[...,0]
    data_e = np.empty(data.shape)
    data_e[...,:-1] = data[...,1:]
    data_e[...,-1] = data[...,-1]
    if not use_1d:
        data_s = np.empty(data.shape)
        data_s[...,1:,:] = data[...,:-1,:]
        data_s[...,0,:] = data[...,0,:]
        data_n = np.empty(data.shape)
        data_n[...,:-1,:] = data[...,1:,:]
        data_n[...,-1,:] = data[...,-1,:]     
    # Arrays of 1s and 0s indicating whether these neighbours are non-missing
    valid_w = (data_w != missing_val).astype(float)
    valid_e = (data_e != missing_val).astype(float)
    if use_1d:
        # Number of valid neighoburs of each point
        num_valid_neighbours = valid_w + valid_e
        # Finished
        return data_w, data_e, valid_w, valid_e, num_valid_neighbours
    valid_s = (data_s != missing_val).astype(float)
    valid_n = (data_n != missing_val).astype(float)
    num_valid_neighbours = valid_w + valid_e + valid_s + valid_n
    return data_w, data_e, data_s, data_n, valid_w, valid_e, valid_s, valid_n, num_valid_neighbours


# Like the neighbours function, but in the vertical dimension: neighbours above and below
def neighbours_z (data, missing_val=-9999):

    data_u = np.empty(data.shape)
    data_u[...,1:,:,:] = data[...,:-1,:,:]
    data_u[...,0,:,:] = data[...,0,:,:]
    data_d = np.empty(data.shape)
    data_d[...,:-1,:,:] = data[...,1:,:,:]
    data_d[...,-1,:,:] = data[...,-1,:,:]
    valid_u = (data_u != missing_val).astype(float)
    valid_d = (data_d != missing_val).astype(float)
    num_valid_neighbours_z = valid_u + valid_d
    return data_u, data_d, valid_u, valid_d, num_valid_neighbours_z

    
# Given an array with missing values, extend the data into the mask by setting missing values to the average of their non-missing neighbours, and repeating as many times as the user wants.
# If "data" is a regular array with specific missing values, set missing_val (default -9999). If "data" is a MaskedArray, set masked=True instead.
# Setting use_3d=True indicates this is a 3D array, and where there are no valid neighbours on the 2D plane, neighbours above and below should be used.
# Setting preference='vertical' (instead of default 'horizontal') indicates that if use_3d=True, vertical neighbours should be preferenced over horizontal ones.
# Setting use_1d=True indicates this is a 1D array.
def extend_into_mask (data, missing_val=-9999, masked=False, use_1d=False, use_3d=False, preference='horizontal', num_iters=1):

    if missing_val != -9999 and masked:
        print("Error (extend_into_mask): can't set a missing value for a masked array")
        sys.exit()
    if use_1d and use_3d:
        print("Error (extend_into_mask): can't have use_1d and use_3d at the same time")
        sys.exit()
    if use_3d and preference not in ['horizontal', 'vertical']:
        print('Error (extend_into_mask): invalid preference ' + preference)

    if masked:
        # MaskedArrays will mess up the extending
        # Unmask the array and fill the mask with missing values
        data_unmasked = data.data
        data_unmasked[data.mask] = missing_val
        data = data_unmasked

    for iter in range(num_iters):
        # Find the neighbours of each point, whether or not they are missing, and how many non-missing neighbours there are.
        # Then choose the points that can be filled.
        # Then set them to the average of their non-missing neighbours.
        if use_1d:
            # Just consider horizontal neighbours in one direction
            data_w, data_e, valid_w, valid_e, num_valid_neighbours = neighbours(data, missing_val=missing_val, use_1d=True)
            index = (data == missing_val)*(num_valid_neighbours > 0)
            data[index] = (data_w[index]*valid_w[index] + data_e[index]*valid_e[index])/num_valid_neighbours[index]
        elif use_3d and preference == 'vertical':
            # Consider vertical neighbours
            data_d, data_u, valid_d, valid_u, num_valid_neighbours = neighbours_z(data, missing_val=missing_val)
            index = (data == missing_val)*(num_valid_neighbours > 0)
            data[index] = (data_u[index]*valid_u[index] + data_d[index]*valid_d[index])/num_valid_neighbours[index]
        else:
            # Consider horizontal neighbours in both directions
            data_w, data_e, data_s, data_n, valid_w, valid_e, valid_s, valid_n, num_valid_neighbours = neighbours(data, missing_val=missing_val)
            index = (data == missing_val)*(num_valid_neighbours > 0)
            data[index] = (data_w[index]*valid_w[index] + data_e[index]*valid_e[index] + data_s[index]*valid_s[index] + data_n[index]*valid_n[index])/num_valid_neighbours[index]
        if use_3d:
            # Consider the other dimension(s). Find the points that haven't already been filled based on the first dimension(s) we checked, but could be filled now.
            if preference == 'vertical':
                # Look for horizontal neighbours
                data_w, data_e, data_s, data_n, valid_w, valid_e, valid_s, valid_n, num_valid_neighbours_new = neighbours(data, missing_val=missing_val)
                index = (data == missing_val)*(num_valid_neighbours == 0)*(num_valid_neighbours_new > 0)
                data[index] = (data_w[index]*valid_w[index] + data_e[index]*valid_e[index] + data_s[index]*valid_s[index] + data_n[index]*valid_n[index])/num_valid_neighbours_new[index]
            elif preference == 'horizontal':
                # Look for vertical neighbours
                data_d, data_u, valid_d, valid_u, num_valid_neighbours_new = neighbours_z(data, missing_val=missing_val)
                index = (data == missing_val)*(num_valid_neighbours == 0)*(num_valid_neighbours_new > 0)
                data[index] = (data_u[index]*valid_u[index] + data_d[index]*valid_d[index])/num_valid_neighbours_new[index]
                
    if masked:
        # Remask the MaskedArray
        data = ma.masked_where(data==missing_val, data)

    return data    


# Interpolate a topography field "data" (eg bathymetry, ice shelf draft, mask) to grid cells. We want the area-averaged value over each grid cell. So it's not enough to just interpolate to a point (because the source data might be much higher resolution than the new grid) or to average all points within the cell (because the source data might be lower or comparable resolution). Instead, interpolate to a finer grid within each grid cell (default 10x10) and then average over these points.

# Arguments:
# x, y: 1D arrays with x and y coordinates of source data (polar stereographic for BEDMAP2, lon and lat for GEBCO)
# data: 2D array of source data
# x_interp, y_interp: 2D arrays with x and y coordinates of the EDGES of grid cells - the output array will be 1 smaller in each dimension

# Optional keyword argument:
# n_subgrid: dimension of finer grid within each grid cell (default 10, i.e. 10 x 10 points per grid cell)

# Output: data on centres of new grid

def interp_topo (x, y, data, x_interp, y_interp, n_subgrid=10):

    from scipy.interpolate import RectBivariateSpline

    # x_interp and y_interp are the edges of the grid cells, so the number of cells is 1 less
    num_j = y_interp.shape[0] -1
    num_i = x_interp.shape[1] - 1
    data_interp = np.empty([num_j, num_i])

    # RectBivariateSpline needs (y,x) not (x,y) - this can really mess you up when BEDMAP2 is square!!
    interpolant = RectBivariateSpline(y, x, data)

    # Loop over grid cells (can't find a vectorised way to do this without overflowing memory)
    for j in range(num_j):
        for i in range(num_i):
            # Make a finer grid within this grid cell (regular in x and y)
            # First identify the boundaries so that x and y are strictly increasing
            if x_interp[j,i] < x_interp[j,i+1]:
                x_start = x_interp[j,i]
                x_end = x_interp[j,i+1]
            else:
                x_start = x_interp[j,i+1]
                x_end = x_interp[j,i]
            if y_interp[j,i] < y_interp[j+1,i]:
                y_start = y_interp[j,i]
                y_end = y_interp[j+1,i]
            else:
                y_start = y_interp[j+1,i]
                y_end = y_interp[j,i]
            # Define edges of the sub-cells
            x_edges = np.linspace(x_start, x_end, num=n_subgrid+1)
            y_edges = np.linspace(y_start, y_end, num=n_subgrid+1)
            # Calculate centres of the sub-cells
            x_vals = 0.5*(x_edges[1:] + x_edges[:-1])
            y_vals = 0.5*(y_edges[1:] + y_edges[:-1])
            # Interpolate to the finer grid, then average over those points to estimate the mean value of the original field over the entire grid cell
            data_interp[j,i] = np.mean(interpolant(y_vals, x_vals))

    return data_interp


# Given an array representing a mask (e.g. ocean mask where 1 is ocean, 0 is land), identify any isolated cells (i.e. 1 cell of ocean with land on 4 sides) and remove them (i.e. recategorise them as land).
def remove_isolated_cells (data, mask_val=0):

    num_valid_neighbours = neighbours(data, missing_val=mask_val)[-1]
    index = (data!=mask_val)*(num_valid_neighbours==0)
    print('...' + str(np.count_nonzero(index)) + ' isolated cells')
    data[index] = mask_val
    return data


# Interpolate from a regular lat-lon grid to another regular lat-lon grid.
# source_lon and source_lat should be 1D arrays; target_lon and target_lat can be either 1D or 2D.
# Fill anything outside the bounds of the source grid with fill_value, but assume there are no missing values within the bounds of the source grid.
def interp_reg_xy (source_lon, source_lat, source_data, target_lon, target_lat, fill_value=-9999, method='linear'):

    from scipy.interpolate import RegularGridInterpolator

    # Build an interpolant
    interpolant = RegularGridInterpolator((source_lat, source_lon), source_data, bounds_error=False, fill_value=fill_value, method=method)
    if len(target_lon.shape) == 1:
        # Make target lat/lon arrays 2D
        target_lon, target_lat = np.meshgrid(target_lon, target_lat)
    # Interpolate
    data_interp = interpolant((target_lat, target_lon))
    return data_interp


# Like interp_reg_xy, but for lat-lon-depth grids.
def interp_reg_xyz (source_lon, source_lat, source_z, source_data, target_lon, target_lat, target_z, fill_value=-9999):

    from scipy.interpolate import RegularGridInterpolator

    # Build an interpolant
    # Make depth positive so it's strictly increasing
    interpolant = RegularGridInterpolator((-source_z, source_lat, source_lon), source_data, bounds_error=False, fill_value=fill_value)
    # Make target axes 3D    
    if len(target_lon.shape) == 1:
        target_lon, target_lat = np.meshgrid(target_lon, target_lat)
    dimensions = [target_lon.shape[1], target_lon.shape[0], target_z.size]
    target_lon = xy_to_xyz(target_lon, dimensions)
    target_lat = xy_to_xyz(target_lat, dimensions)
    target_z = z_to_xyz(target_z, dimensions)
    # Interpolate
    data_interp = interpolant((-target_z, target_lat, target_lon))
    return data_interp    


# Interpolate a field on a regular MITgcm grid, to another regular MITgcm grid. Anything outside the bounds of the source grid will be filled with fill_value.
# source_grid and target_grid can be either Grid or SOSEGrid objects.
# Set dim=3 for 3D fields (xyz), dim=2 for 2D fields (xy).
def interp_reg (source_grid, target_grid, source_data, dim=3, gtype='t', fill_value=-9999):

    # Get the correct lat and lon on the source grid
    source_lon, source_lat = source_grid.get_lon_lat(gtype=gtype, dim=1)
    # Get the correct lat and lon on the target grid
    target_lon, target_lat = target_grid.get_lon_lat(gtype=gtype)
    
    if dim == 2:
        return interp_reg_xy(source_lon, source_lat, source_data, target_lon, target_lat, fill_value=fill_value)
    elif dim == 3:
        return interp_reg_xyz(source_lon, source_lat, source_grid.z, source_data, target_lon, target_lat, target_grid.z, fill_value=fill_value)
    else:
        print('Error (interp_reg): dim must be 2 or 3')
        sys.exit()


# Given data on a 3D grid (or 2D if you set use_3d=False), throw away any points indicated by the "discard" boolean mask (i.e. fill them with missing_val), and then extrapolate into any points indicated by the "fill" boolean mask (by calling extend_into_mask as many times as needed).
def discard_and_fill (data, discard, fill, missing_val=-9999, use_1d=False, use_3d=True, preference='horizontal', log=True):

    # import file_io for basic error output
    #import .file_io as fio
 
    # First throw away the points we don't trust
    data[discard] = missing_val
    # Now fill the values we need to fill
    num_missing = np.count_nonzero((data==missing_val)*fill)
    while num_missing > 0:
        if log:
            print('......' + str(num_missing) + ' points to fill')
        data = extend_into_mask(data, missing_val=missing_val, use_1d=use_1d, use_3d=use_3d, preference=preference)
        num_missing_old = num_missing
        num_missing = np.count_nonzero((data==missing_val)*fill)
        if num_missing == num_missing_old:
            # There are some disconnected regions. This can happen with coupling sometimes. Try using more iterations of extend_into_mask.
            for iters in range(2, 100):
                data = extend_into_mask(data, missing_val=missing_val, use_1d=use_1d, use_3d=use_3d, preference=preference, num_iters=iters)
                num_missing_old = num_missing
                num_missing = np.count_nonzero((data==missing_val)*fill)
                if num_missing != num_missing_old:
                    break
            if num_missing == num_missing_old:
                # If cannot complete discard and fill, write errors out to very basic file 
                print('Error (discard_and_fill): some missing values cannot be filled')
                print('Dumping data, discard, and fill data to error_fill_dump.nc') 
                #fio.write_netcdf_very_basic(data,    'data',    'error_dump_data.nc', use_3d=use_3d)
                #fio.write_netcdf_very_basic(discard, 'discard', 'error_dump_discard.nc', use_3d=use_3d)
                #fio.write_netcdf_very_basic(fill,    'fill',    'error_dump_fill.nc', use_3d=use_3d)
                sys.exit()
    return data


# Wrapper to completely fill the mask (either provided as a separate mask/hfac where 0 means masked, or as a MaskedArray within the data)
def fill_into_mask (data, mask=None, missing_val=-9999, use_1d=False, use_3d=True, preference='horizontal', log=True):

    if mask is not None:
        discard = mask==0
    elif isinstance(data, np.ma.MaskedArray):
        discard = data.mask
    else:
        print('Error (fill_into_mask): must provide a mask or use a MaskedArray')
        sys.exit()
    fill = np.ones(data.shape).astype(bool)
    return discard_and_fill(data, discard, fill, missing_val=missing_val, use_1d=use_1d, use_3d=use_3d, preference=preference, log=log)
    


# Given a monotonically increasing 1D array "data", and a scalar value "val0", find the indicies i1, i2 and interpolation coefficients c1, c2 such that c1*data[i1] + c2*data[i2] = val0.
# If the array is longitude and may not be strictly increasing, and/or there is the possibility of val0 in the gap between the periodic boundary, set lon=True.
def interp_slice_helper (data, val0, lon=False, warn=True):

    if lon:
        # Transformation to make sure longitude array is strictly increasing
        data0 = data[0]
        data = (data-data0)%360
        val0 = (val0-data0)%360
    # Case that val0 is in the array
    if val0 in data:
        i = np.argwhere(data==val0)[0][0]
        return i, i, 1, 0    

    # Find the last index less than val0
    i1 = np.nonzero(data < val0)[0][-1]
    if lon and i1==data.size-1:
        # Special case where value falls in the periodic boundary
        i2 = 0
        c2 = (val0 - data[i1])/(data[i2]+360 - data[i1])
        c1 = 1 - c2
        return i1, i2, c1, c2
    # Find the first index greater than val0
    i2 = np.nonzero(data > val0)[0][0]
    if i2 != i1+1:
        if data[i1+1] is np.ma.masked or data[i2-1] is np.ma.masked:
            # This can happen with tripolar grids eg CICE...
            # Usually it's within the land mask of MITgcm so doesn't matter, but warn the user just in case
            if warn:
                print('Warning (interp_slice_helper): hit the boundary of masked part of grid - hopefully this is within your land mask!')
                i2 = i1
        else:
            # This can happen with curved grids near the north pole where the jump in longitude changes over space.
            if warn:
                print('Warning (interp_slice_helper): i1='+str(i1)+' but i2='+str(i2))
                print('Setting i2='+str(i1+1)+', is this what you want?')
            i2 = i1+1
    # Calculate the weighting coefficients
    if i1 == i2:
        c1 = 1
        c2 = 0
    else:
        c2 = (val0 - data[i1])/(data[i2] - data[i1])
        c1 = 1 - c2
    return i1, i2, c1, c2


# Interpolate an array "data" to a point (lon0, lat0). Other dimensions (eg time, depth) will be preserved.
# Can also set return_hfac=True to return the column of hFac values interpolated to this point. If any of the neighbouring points are fully closed (i.e. land), the interpolated hFac will be zero there too.
def interp_bilinear (data, lon0, lat0, grid, gtype='t', return_hfac=False):

    lon, lat = grid.get_lon_lat(gtype=gtype, dim=1)
    i1, i2, a1, a2 = interp_slice_helper(lon, lon0, lon=True)
    j1, j2, b1, b2 = interp_slice_helper(lat, lat0)
    data_interp = a1*b1*data[...,j1,i1] + a2*b1*data[...,j1,i2] + a1*b2*data[...,j2,i1] + a2*b2*data[...,j2,i2]
    if return_hfac:
        hfac = grid.get_hfac(gtype=gtype)
        hfac_interp = interp_bilinear(hfac, lon0, lat0, grid, gtype=gtype)
        # Check for closed points
        index = hfac[:,j1,i1]*hfac[:,j1,i2]*hfac[:,j2,i1]*hfac[:,j2,i2] == 0
        hfac_interp[index] = 0
        return data_interp, hfac_interp
    else:
        return data_interp


# Interpolate a lateral boundary field from a regular grid to another regular grid. Prior to interpolation, extend the source data all the way into the mask so there will be no missing values (which MITgcm doesn't handle very well).
# This routine can be called for a depth-dependent field (latitude or longitude versus depth, i.e. a 3D variable which was sliced) or a depth-independent field (just dependent on latitude or longitude, i.e. a 2D variable which was sliced).

# Arguments:
# source_h: 1D array of latitude or longitude on the source grid.
# source_z: 1D array of depth (negative) on the source grid. If depth_dependent=False, you can pass None for this argument.
# source_data: slice of data on the source grid (dimension depth x latitude/longitude if depth_dependent=True; dimension latitude/longitude if depth_dependent=False)
# source_hfac: slice of hFac on the source grid; just used for masking
# target_h, target_z, target_hfac: similar for the target grid
# IMPORTANT: Make sure that source_h, source_hfac, target_h, target_hfac are all on the correct grid (t, u, v) corresponding to the data.

# Optional keyword arguments:
# lon: indicates that "h" is a longitude array, and we might need to rearrange things so source_h is strictly increasing
# depth_dependent: boolean indicating whether the data is depth-dependent
# missing_val: missing value to use for checking mask; just make sure it doesn't equal a value that real data might hold

def interp_bdry (source_h, source_z, source_data, source_hfac, target_h, target_z, target_hfac, lon=False, depth_dependent=True, missing_val=-9999):

    from scipy.interpolate import RegularGridInterpolator, interp1d

    if lon:
        # Transformation to make sure source_h is strictly increasing
        h0 = source_h[0]
        source_h = (source_h-h0)%360
        target_h = (target_h-h0)%360

    if depth_dependent:
        # Extend the source axes at the top and/or bottom if needed
        if abs(target_z[0]) < abs(source_z[0]):
            # Add a row of zero depth
            source_z = np.concatenate(([0], source_z))
            # Copy the top row of data
            source_data = np.concatenate((np.expand_dims(source_data[0,:], 0), source_data), axis=0)
            source_hfac = np.concatenate((np.expand_dims(source_hfac[0,:], 0), source_hfac), axis=0)
        if abs(target_z[-1]) > abs(source_z[-1]):
            # Add a row of sufficiently deep depth
            source_z = np.concatenate((source_z, [2*target_z[-1] - target_z[-2]]))
            # Copy the bottom row of data
            source_data = np.concatenate((source_data, np.expand_dims(source_data[-1,:], 0)), axis=0)
            source_hfac = np.concatenate((source_hfac, np.expand_dims(source_hfac[-1,:], 0)), axis=0)

    # Extend all the way into the mask
    source_data = fill_into_mask(source_data, mask=source_hfac, missing_val=missing_val, use_1d=(not depth_dependent), use_3d=False, log=False)
    
    # Interpolate
    if depth_dependent:
        interpolant = RegularGridInterpolator((-source_z, source_h), source_data, bounds_error=False, fill_value=missing_val)
        target_h, target_z = np.meshgrid(target_h, target_z)
        data_interp = interpolant((-target_z, target_h))
    else:
        interpolant = interp1d(source_h, source_data, bounds_error=False, fill_value=missing_val)
        data_interp = interpolant(target_h)
    
    if np.count_nonzero(data_interp==missing_val) > 0:
        print('Error (interp_bdry): missing values remain in the interpolated data.')
        if np.amin(target_h) < np.amin(source_h) or np.amax(target_h) > np.amax(source_h):
            print('Need to extend the horizontal axis for the source data.')
        sys.exit()
        
    return data_interp


# Interpolate the given field to the given depth (constant). It can be any dimension as long as depth is the first axis (if time_dependent=False) or the second axis (if time_dependent=True).
# grid can either be a Grid object, or an array of depth values
def interp_to_depth (data, z0, grid, time_dependent=False, gtype='t'):

    if isinstance(grid, Grid):
        z = grid.z
    else:
        z = grid
        
    if gtype == 'w':
        print('Error (interp_to_depth): w-grids not supported yet')
        sys.exit()
    if z0 > z[0]:
        # Return surface layer
        k1 = 0
        k2 = 0
        c1 = 1
        c2 = 0
    elif z0 < z[-1]:
        # Return bottom layer
        k1 = -1
        k2 = -1
        c1 = 1
        c2 = 0
    else:
        # Make depth positive so array is increasing and we can get right coefficients
        k1, k2, c1, c2 = interp_slice_helper(-z, -z0)
    if time_dependent:
        return c1*data[:,k1,...] + c2*data[:,k2,...]
    else:
        return c1*data[k1,...] + c2*data[k2,...]


# Interpolate from a non-regular grid (structured but not regular in lat-lon, e.g. curvilinear) to a another grid (regular or non-regular is fine).
# The input lat and lon arrays should be 2D for the source grid, and either 1D (if regular) or 2D for the target grid.
# Fill anything outside the bounds of the source grid with fill_value. If fill_mask=True, fill them with the nearest neighbours instead.
def interp_nonreg_xy (source_lon, source_lat, source_data, target_lon, target_lat, fill_value=-9999, fill_mask=False, method='linear'):

    from scipy.interpolate import griddata

    # Check for missing values
    if isinstance(source_data, np.ma.MaskedArray):
        missing = source_data.mask + (source_data == fill_value)
    else:
        missing = source_data == fill_value

    # Figure out if target lon and lat are 1D or 2D
    if len(target_lon.shape) == 1 and len(target_lat.shape) == 1:
        # Make them 2D
        target_lon, target_lat = np.meshgrid(target_lon, target_lat)

    # Set up an nx2 array containing the coordinates of each point in the source grid
    source_points = np.stack((np.ravel(source_lon[~missing]), np.ravel(source_lat[~missing])), axis=-1)
    # Same for the target grid
    target_points = np.stack((np.ravel(target_lon), np.ravel(target_lat)), axis=-1)
    # Also flatten the data
    source_values = np.ravel(source_data[~missing])
    
    # Interpolate
    data_interp = griddata(source_points, source_values, target_points, fill_value=fill_value, method=method)
    if fill_mask:
        data_interp_2 = griddata(source_points, source_values, target_points, method='nearest')
        index = data_interp==fill_value
        data_interp[index] = data_interp_2[index]
    # Un-flatten the result
    return np.reshape(data_interp, target_lon.shape)


# Interpolate a 3D field on a grid which is non-regular in the lat-lon direction, but has a normal z-axis.
def interp_nonreg_xyz (source_lon, source_lat, source_z, source_data, target_lon, target_lat, target_z, fill_value=-9999):

    # Find dimensions of grid
    if len(target_lon.shape) == 1 and len(target_lat.shape) == 1:
        nx = target_lon.size
        ny = target_lat.size
    else:
        nx = target_lon.shape[1]
        ny = target_lon.shape[0]
    nz = target_z.size

    # Interpolate each depth individually
    data_interp = np.ma.empty([nz, ny, nx])
    for k in range(nz):
        print('...depth ' + str(k+1) + ' of ' + str(nz))
        if k==0 and target_z[k] > source_z[0]:
            # Target grid's surface layer is too shallow - extrapolate
            source_data_2d = source_data[0,:]
        elif k==nz-1 and target_z[-1] < source_z[-1]:
            # Target grid's bottom layer is too deep - extrapolate
            source_data_2d = source_data[-1,:]
        else:        
            # Extract a lon-lat slice of source data, at this depth
            source_data_2d = interp_to_depth(source_data, target_z[k], source_z)
        # Now interpolate in 2D
        data_interp[k,:] = interp_nonreg_xy(source_lon, source_lat, source_data_2d, target_lon, target_lat, fill_value=fill_value)

    return data_interp


# API to interpolate from a non-regular lat-lon grid, on either 2 or 3 dimensions.
def interp_nonreg (source_grid, target_grid, source_data, dim=3, gtype='t', fill_value=-9999):

    # Get the correct lat and lon on source and target grids
    source_lon, source_lat = source_grid.get_lon_lat(gtype=gtype, dim=2)
    target_lon, target_lat = target_grid.get_lon_lat(gtype=gtype, dim=2)

    if dim == 2:
        return interp_nonreg_xy(source_lon, source_lat, source_data, target_lon, target_lat, fill_value=fill_value)
    elif dim == 3:
        return interp_nonreg_xyz(source_lon, source_lat, source_grid.z, source_data, target_lon, target_lat, target_grid.z, fill_value=fill_value)
    else:
        print('Error (interp_nonreg): dim must be 2 or 3')
        sys.exit()


# Smooth a lat-lon field with a 2D Gaussian filter. Default radius of 2 grid cells (1/2 degree for a quarter-degree grid).
def smooth_xy (data, sigma=2):

    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(data, sigma)


# Calculate the interpolation coefficients to extract a slice of constant latitude or longitude from an arbitrary (not necessarily regular) lat-lon grid.
def interp_slice_helper_nonreg (lon, lat, loc0, direction):

    # Error checking
    if direction not in ['lat', 'lon']:
        print('Error (interp_slice_helper_nonreg): invalid direction '+direction)
        sys.exit()
    # Get grid sizes
    
    nx = lon.shape[1]
    ny = lat.shape[0]
    if direction == 'lat':
        num_pts = nx
    elif direction == 'lon':
        num_pts = ny
    # Loop over the correct axis
    i1 = np.empty(num_pts)
    i2 = np.empty(num_pts)
    c1 = np.empty(num_pts)
    c2 = np.empty(num_pts)
    for j in range(num_pts):
        if direction == 'lat':
            i1[j], i2[j], c1[j], c2[j] = interp_slice_helper(lat[:,j], loc0)
        elif direction == 'lon':
            i1[j], i2[j], c1[j], c2[j] = interp_slice_helper(lon[j,:], loc0, lon=True)
    return i1, i2, c1, c2


# Given these coefficients, extract the slice of the given data, which may or may not be time- or depth-dependent.
def extract_slice_nonreg (data, direction, i1, i2, c1, c2):

    # Set up an array of the correct shape, collapsing one dimension down to a single index
    if direction == 'lat':
        shape = data[...,0,:].shape
    elif direction == 'lon':
        shape = data[...,0].shape
    else:
        print('Error (extract_slice_nonreg): invalid direction '+direction)
        sys.exit()
    data_slice = np.ma.empty(shape)
    num_pts = i1.size
    for j in range(num_pts):
        if direction == 'lat':
            data_slice[...,j] = c1[j]*data[...,int(i1[j]),j] + c2[j]*data[...,int(i2[j]),j]
        elif direction == 'lon':
            data_slice[...,j] = c1[j]*data[...,j,int(i1[j])] + c2[j]*data[...,j,int(i2[j])]
    return data_slice


# Fill missing values in the given array with a distance-weighted mean of its num_neighbours nearest neighbours (default 10).
# For now only works with a 2D array and using index values as distance (rather than lat/lon or Cartesian distance or something), for the purposes of T/S space.
# You can either pass a MaskedArray or set a specific missing value.
# Can also pass an additional weighting array (eg log of volume)
def distance_weighted_nearest_neighbours (data, weights=None, num_neighbours=10, missing_val=-9999):

    if isinstance(data, np.ma.MaskedArray):
        mask = data.mask
    else:
        mask = data==missing_val
    if weights is None:
        weights = np.ones(data.shape)
    i_vals, j_vals = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    data_filled = np.empty(data.shape)
    # Loop over missing points
    for i0, j0 in zip(i_vals[mask], j_vals[mask]):
        # Get the distance of every other point to this point
        distance = np.sqrt((i_vals-i0)**2 + (j_vals-j0)**2)
        # Apply the mask so we only consider distances to valid points
        distance = np.ma.masked_where(mask, distance)
        # Select the num_neighbours closest points
        neighbours = distance <= np.sort(distance[~mask])[num_neighbours-1]
        # Calculate 1/distance for weighting
        inv_distance = 1/distance
        # Calculate the distance-weighted mean over these points, including additional weights
        data_filled[j0,i0] = np.sum(data[neighbours]*inv_distance[neighbours]*weights[neighbours])/np.sum(inv_distance[neighbours]*weights[neighbours])
    data_filled[~mask] = data[~mask]
    return data_filled
    
    
    

