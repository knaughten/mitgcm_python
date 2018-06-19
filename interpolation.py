#######################################################
# All things interpolation
#######################################################

import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline

from utils import mask_land, mask_land_zice, mask_3d


# Interpolate from one grid type to another. Currently only u-grid to t-grid and v-grid to t-grid are supported.

# Arguments:
# data: array of dimension (maybe time) x (maybe depth) x lat x lon
# grid: Grid object
# gtype_in: grid type of "data". As in function Grid.get_lon_lat.
# gtype_out: grid type to interpolate to

# Optional keyword arguments:
# time_dependent: as in function apply_mask
# mask_shelf: indicates to mask the ice shelves as well as land. Only valid if "data" isn't depth-dependent.

# Output: array of the same dimension as "data", interpolated to the new grid type

def interp_grid (data, grid, gtype_in, gtype_out, time_dependent=False, mask_shelf=False):

    # Figure out if the field is depth-dependent
    if (time_dependent and len(data.shape)==4) or (not time_dependent and len(data.shape)==3):
        depth_dependent=True
    else:
        depth_dependent=False
    # Make sure we're not trying to mask the ice shelf from a depth-dependent field
    if mask_shelf and depth_dependent:
        print "Error (interp_grid): can't set mask_shelf=True for a depth-dependent field."
        sys.exit()

    if gtype_in in ['u', 'v', 'psi', 'w']:
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
        # Extend the easternmost column
        data_interp[...,-1] = data_tmp[...,-1]
    elif gtype_in == 'v' and gtype_out == 't':
        # Midpoints in the y direction
        data_interp[...,:-1,:] = 0.5*(data_tmp[...,:-1,:] + data_tmp[...,1:,:])
        # Extend the northernmost row
        data_interp[...,-1,:] = data_tmp[...,-1,:]
    else:
        print 'Error (interp_grid): interpolation from the ' + gtype_in + '-grid to the ' + gtype_out + '-grid is not yet supported'
        sys.exit()

    # Now apply the mask
    if depth_dependent:
        data_interp = mask_3d(data_interp, grid, gtype=gtype_out, time_dependent=time_dependent)
    else:
        if mask_shelf:
            data_interp = mask_land_zice(data_interp, grid, gtype=gtype_out, time_dependent=time_dependent)
        else:
            data_interp = mask_land(data_interp, grid, gtype=gtype_out, time_dependent=time_dependent)

    return data_interp


# Given an array with missing values, extend the data into the mask by setting missing values to the average of their non-missing neighbours, and repeating as many times as the user wants.
# If "data" is a regular array with specific missing values, set missing_val (default -9999). If "data" is a MaskedArray, set masked=True instead.
# num_iters indicates the number of times the data is extended into the mask (default 5).
def extend_into_mask (data, missing_val=-9999, masked=False, num_iters=5):

    if missing_val != -9999 and masked:
        print "Error (extend_into_mask): can't set a missing value for a masked array"
        sys.exit()

    if masked:
        # MaskedArrays will mess up the extending
        # Unmask the array and fill the mask with missing values
        data_unmasked = data.data
        data_unmasked[data.mask] = missing_val
        data = data_unmasked

    for iter in range(num_iters):
        print '...iteration ' + str(iter+1) + ' of ' + str(num_iters)
        # Find the value to the left, right, down, up of every point
        # Just copy the boundaries
        data_l = np.empty(data.shape)
        data_l[:,1:] = data[:,:-1]
        data_l[:,0] = data[:,0]
        data_r = np.empty(data.shape)
        data_r[:,:-1] = data[:,1:]
        data_r[:,-1] = data[:,-1]
        data_d = np.empty(data.shape)
        data_d[1:,:] = data[:-1,:]
        data_d[0,:] = data[0,:]
        data_u = np.empty(data.shape)
        data_u[:-1,:] = data[1:,:]
        data_u[-1,:] = data[-1,:]
        # Arrays of 1s and 0s indicating whether these neighbours are non-missing
        valid_l = (data_l != missing_val).astype(float)
        valid_r = (data_r != missing_val).astype(float)
        valid_d = (data_d != missing_val).astype(float)
        valid_u = (data_u != missing_val).astype(float)
        # Number of valid neighbours of each point
        num_valid_neighbours = valid_l + valid_r + valid_d + valid_u
        # Choose the points that can be filled
        index = np.nonzero((data == missing_val)*(num_valid_neighbours > 0))
        # Set them to the average of their non-missing neighbours
        data[index] = (data_l[index]*valid_l[index] + data_r[index]*valid_r[index] + data_d[index]*valid_d[index] + data_u[index]*valid_u[index])/num_valid_neighbours[index]

    if masked:
        # Remask the MaskedArray
        data = ma.masked_where(data==missing_val, data)

    return data



def interp_topo (x, y, data, x_interp, y_interp, n_subgrid=10):

    # x_interp and y_interp are the edges of the grid cells, so the number of cells is 1 less
    num_j = y_interp.shape[0] -1
    num_i = x_interp.shape[1] - 1
    data_interp = np.empty([num_j, num_i])

    print 'Creating interpolant'
    interpolant = RectBivariateSpline(y, x, data)

    # Loop over grid cells (can't find a vectorised way to do this without overflowing memory)
    for j in range(num_j):
        print '...latitude index ' + str(j+1) + ' of ' + str(num_j)
        for i in range(num_i):
            # Make a finer grid within this grid cell (regular in x and y)
            # Edges of the sub-cells
            x_edges = np.linspace(x_interp[j,i], x_interp[j,i+1], num=n_subgrid+1)
            y_edges = np.linspace(y_interp[j,i], y_interp[j+1,i], num=n_subgrid+1)
            # Centres of the sub-cells
            x_vals = 0.5*(x_edges[1:] + x_edges[:-1])
            y_vals = 0.5*(y_edges[1:] + y_edges[:-1])
            # Interpolate to the finer grid, then average over those points to estimate the mean value of the original field over the entire grid cell
            data_interp[j,i] = np.mean(interpolant(y_vals, x_vals))

    return data_interp
                
    

    
