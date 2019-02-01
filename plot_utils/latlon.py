#######################################################
# Utilities specific to lat-lon plots.
#######################################################

import numpy as np
import matplotlib.colors as cl

from ..utils import mask_land, select_top, select_bottom
from ..calculus import vertical_average
from ..interpolation import interp_grid


# Determine longitude and latitude on the boundaries of cells for the given grid type (tracer, u, v, psi), and do one of two things:
# (1) if extrapolate=True, extrapolate one row and one column of the latitude and longitude
# (2) if extrapolate=False, throw away one row and one column of the given data field
# Either way, each data point will then have latitude and longitude boundaries defined on 4 sides. This is needed for pcolormesh so that the coordinates of the quadrilateral patches are correctly defined.
# The data array can have more than 2 dimensions, as long as the second last dimension is latitude (size M), and the last dimension is longitude (size N).
# Outputs longitude and latitude at the boundary of each cell (size (M+1)x(N+1), or MxN) and the data (size ...xMxN, or ...x(M-1)x(N-1)).
def cell_boundaries (data, grid, gtype='t', extrapolate=True):

    # Inner function to pad the given array in the given direction(s), either extrapolating or copying.
    def extend_array (A, south=None, north=None, west=None, east=None):
        if south is not None:
            if south == 'extrapolate':
                s_bdry = 2*A[0,:]-A[1,:]
            elif south == 'copy':
                s_bdry = A[0,:]
            A = np.concatenate((s_bdry[None,:], A), axis=0)
        if north is not None:
            if north == 'extrapolate':
                n_bdry = 2*A[-1,:]-A[-2,:]
            elif north == 'copy':
                n_bdry = A[-1,:]
            A = np.concatenate((A, n_bdry[None,:]), axis=0)
        if west is not None:
            if west == 'extrapolate':
                w_bdry = 2*A[:,0]-A[:,1]
            elif west == 'copy':
                w_bdry = A[:,0]
            A = np.concatenate((w_bdry[:,None], A), axis=1)
        if east is not None:
            if east == 'extrapolate':
                e_bdry = 2*A[:,-1]-A[:,-2]
            elif east == 'copy':
                e_bdry = A[:,-1]
            A = np.concatenate((A, e_bdry[:,None]), axis=1)
        return A

    if gtype in ['t', 'w']:
        # Tracer grid: at centres of cells
        # Boundaries are corners of cells
        # Throw away eastern and northern edges
        return grid.lon_corners_2d, grid.lat_corners_2d, data[...,:-1,:-1]
    elif gtype == 'u':
        # U-grid: on left edges of cells
        # Boundaries are centres of cells in X, corners of cells in Y
        # Throw away western and northern edges
        return grid.lon_2d, grid.lat_corners_2d, data[...,:-1,1:]
    elif gtype == 'v':
        # V-grid: on bottom edges of cells
        # Boundaries are corners of cells in X, centres of cells in Y
        # Throw away eastern and southern edges
        return grid.lon_corners_2d, grid.lat_2d, data[...,1:,:-1]
    elif gtype == 'psi':
        # Psi-grid: on southwest corners of cells
        # Boundaries are centres of cells
        # Throw away western and southern edges
        return grid.lon_2d, grid.lat_2d, data[...,1:,1:]


# Shade various masks in grey on the plot: just the land mask, or the land and ice shelves.
# shade_mask is the helper function; shade_land and shade_land_ice are the APIs.

def shade_mask (ax, mask, grid, gtype='t'):

    # Properly mask all the False values, so that only True values are unmasked
    mask_plot = np.ma.masked_where(np.invert(mask), mask)
    # Prepare quadrilateral patches
    lon, lat, mask_plot = cell_boundaries(mask_plot, grid, gtype=gtype)
    # Add to plot
    ax.pcolormesh(lon, lat, mask_plot, cmap=cl.ListedColormap([(0.6, 0.6, 0.6)]))

    
def shade_land (ax, grid, gtype='t'):
    
    shade_mask(ax, grid.get_land_mask(gtype=gtype), grid, gtype=gtype)

    
def shade_land_ice (ax, grid, gtype='t'):
    
    shade_mask(ax, grid.get_land_mask(gtype=gtype)+grid.get_ice_mask(gtype=gtype), grid, gtype=gtype)


# Contour the ice shelf front in black.
def contour_iceshelf_front (ax, grid):

    # Mask land out of ice shelf draft, so that grounding line isn't contoured
    draft = mask_land(grid.draft, grid)
    # Find the shallowest non-zero ice shelf draft
    draft0 = np.amax(draft[draft!=0])
    # Add to plot
    ax.contour(grid.lon_2d, grid.lat_2d, draft, levels=[draft0], colors=('black'), linestyles='solid')


# Prepare the velocity vectors for plotting: given 3D arrays of u and v on their original grids, and already masked, do a vertical transformation vel_option and interpolate to the tracer grid. Options for vel_option are:
# 'vel': vertically average (default)
# 'sfc': select the top layer
# 'bottom': select the bottom layer
# 'ice': indicates u and v are already 2D (usually because they're sea ice velocity) so no vertical transformation is needed
# Returns the speed as well as both vector components.
def prepare_vel (u, v, grid, vel_option='avg'):

    # Get the correct 2D velocity field
    if vel_option == 'avg':
        u_2d = vertical_average(u, grid, gtype='u')
        v_2d = vertical_average(v, grid, gtype='v')
    elif vel_option == 'sfc':
        u_2d = select_top(u)
        v_2d = select_top(v)
    elif vel_option == 'bottom':
        u_2d = select_bottom(u)
        v_2d = select_top(v)
    elif vel_option == 'ice':
        u_2d = u
        v_2d = v

    # Interpolate to the tracer grid
    if vel_option == 'ice':
        # This is sea ice velocity so we need to mask the ice shelves
        mask_shelf = True
    else:
        mask_shelf = False
    u_interp = interp_grid(u_2d, grid, 'u', 't', mask_shelf=mask_shelf)
    v_interp = interp_grid(v_2d, grid, 'v', 't', mask_shelf=mask_shelf)

    # Calculate speed
    speed = np.sqrt(u_interp**2 + v_interp**2)

    return speed, u_interp, v_interp


# Overlay vectors (typically velocity). u_vec and v_vec must have already been processed using prepare_vec. You can tune the appearance of the arrows using the keyword arguments chunk (size of block to average velocity vectors over, so the plot isn't too crowded), scale, headwidth, and headlength (all from the quiver function).
# average_blocks is the helper function, overlay_vectors is the API.

def average_blocks (data, chunk):

    # Check if there is a mask
    if np.ma.is_masked(data):
        mask = True
    else:
        mask = False

    # Figure out dimensions of output array
    ny_chunks, nx_chunks = np.ceil(np.array(data.shape)/float(chunk)).astype(int)    
    data_blocked = np.zeros([ny_chunks, nx_chunks])
    if mask:
        data_blocked = np.ma.MaskedArray(data_blocked)

    # Average over blocks
    for j in range(ny_chunks):
        start_j = j*chunk
        end_j = min((j+1)*chunk, data.shape[0])
        for i in range(nx_chunks):
            start_i = i*chunk
            end_i = min((i+1)*chunk, data.shape[1])
            data_blocked[j,i] = np.mean(data[start_j:end_j, start_i:end_i])

    return data_blocked


def overlay_vectors (ax, u_vec, v_vec, grid, chunk=10, scale=0.8, headwidth=6, headlength=7):

    lon, lat = grid.get_lon_lat()
    lon_plot = average_blocks(lon, chunk)
    lat_plot = average_blocks(lat, chunk)
    u_plot = average_blocks(u_vec, chunk)
    v_plot = average_blocks(v_vec, chunk)
    ax.quiver(lon_plot, lat_plot, u_plot, v_plot, scale=scale, headwidth=headwidth, headlength=headlength)
