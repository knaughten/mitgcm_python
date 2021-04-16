#######################################################
# Utilities specific to lat-lon plots.
#######################################################

import numpy as np
import matplotlib.colors as cl
import sys

from ..utils import mask_land, select_top, select_bottom, get_x_y
from ..calculus import vertical_average
from ..interpolation import interp_grid, interp_to_depth


# Determine longitude and latitude on the boundaries of cells for the given grid type (tracer, u, v, psi), and do one of two things:
# (1) if extrapolate=True, extrapolate one row and one column of the latitude and longitude
# (2) if extrapolate=False, throw away one row and one column of the given data field
# Either way, each data point will then have latitude and longitude boundaries defined on 4 sides. This is needed for pcolormesh so that the coordinates of the quadrilateral patches are correctly defined.
# The data array can have more than 2 dimensions, as long as the second last dimension is latitude (size M), and the last dimension is longitude (size N).
# Outputs longitude and latitude at the boundary of each cell (size (M+1)x(N+1), or MxN) and the data (size ...xMxN, or ...x(M-1)x(N-1)).
# If you want a polar stereographic project, set pster=True. It will return x and y instead of lon and lat.
def cell_boundaries (data, grid, gtype='t', extrapolate=True, pster=False):

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
        lon = grid.lon_corners_2d
        lat = grid.lat_corners_2d
        # Care about eastern and northern edges
        if extrapolate:
            lon = extend_array(lon, north='copy', east='extrapolate')
            lat = extend_array(lat, north='extrapolate', east='copy')
        else:
            data = data[...,:-1,:-1]
    elif gtype == 'u':
        # U-grid: on left edges of cells
        # Boundaries are centres of cells in X, corners of cells in Y
        lon = grid.lon_2d
        lat = grid.lat_corners_2d
        # Care about western and northern edges
        if extrapolate:
            lon = extend_array(lon, north='copy', west='extrapolate')
            lat = extend_array(lat, north='extrapolate', west='copy')
        else:
            data = data[...,:-1,1:]
    elif gtype == 'v':
        # V-grid: on bottom edges of cells
        # Boundaries are corners of cells in X, centres of cells in Y
        lon = grid.lon_corners_2d
        lat = grid.lat_2d
        # Care about eastern and southern edges
        if extrapolate:
            lon = extend_array(lon, south='copy', east='extrapolate')
            lat = extend_array(lat, south='extrapolate', east='copy')
        else:
            data = data[...,1:,:-1]
    elif gtype == 'psi':
        # Psi-grid: on southwest corners of cells
        # Boundaries are centres of cells
        lon = grid.lon_2d
        lat = grid.lat_2d
        # Care about western and southern edges
        if extrapolate:
            lon = extend_array(lon, south='copy', west='extrapolate')
            lat = extend_array(lat, south='extrapolate', west='copy')
        else:
            data = data[...,1:,1:]

    # Convert to polar stereographic if needed
    x, y = get_x_y(lon, lat, pster=pster)
    return x, y, data


# Shade various masks on the plot: just the land mask, the land and ice shelves, or the ocean. Default is to shade in grey, can also do white.
# shade_mask is the helper function; shade_land and shade_land_ice are the APIs.

def shade_mask (ax, mask, grid, gtype='t', pster=False, colour='grey', rasterized=False):

    # Properly mask all the False values, so that only True values are unmasked
    mask_plot = np.ma.masked_where(np.invert(mask), mask)
    # Prepare quadrilateral patches
    x, y, mask_plot = cell_boundaries(mask_plot, grid, gtype=gtype, pster=pster)
    if colour == 'grey':
        rgb = (0.6, 0.6, 0.6)
    elif colour == 'white':
        rgb = (1, 1, 1)
    else:
        print 'Error (shade_mask): invalid colour ' + colour
        sys.exit()
    # Add to plot        
    img = ax.pcolormesh(x, y, mask_plot, cmap=cl.ListedColormap([rgb]), linewidth=0, rasterized=rasterized)
    img.set_edgecolor('face')

    
def shade_land (ax, grid, gtype='t', pster=False, land_mask=None, rasterized=False):
    if land_mask is None:
        land_mask = grid.get_land_mask(gtype=gtype)
    shade_mask(ax, land_mask, grid, gtype=gtype, pster=pster, rasterized=rasterized)

    
def shade_land_ice (ax, grid, gtype='t', pster=False, land_mask=None, ice_mask=None, rasterized=False):
    if land_mask is None:
        land_mask = grid.get_land_mask(gtype=gtype)
    if ice_mask is None:
        ice_mask = grid.get_ice_mask(gtype=gtype)
    shade_mask(ax, land_mask+ice_mask, grid, gtype=gtype, pster=pster, rasterized=rasterized)


def clear_ocean (ax, grid, gtype='t', pster=False, land_mask=None, rasterized=False):
    if land_mask is None:
        land_mask = grid.get_land_mask(gtype=gtype)
    shade_mask(ax, np.invert(land_mask), grid, gtype=gtype, pster=pster, colour='white', rasterized=rasterized)


# Fill the background of the plot with grey.
def shade_background (ax):
    ax.patch.set_facecolor((0.6, 0.6, 0.6))


# Contour the ice shelf front in black.
def contour_iceshelf_front (ax, grid, pster=False):

    # Mask land out of ice shelf draft, so that grounding line isn't contoured
    draft = mask_land(grid.draft, grid)
    # Find the shallowest non-zero ice shelf draft
    draft0 = np.amax(draft[draft!=0])
    # Convert to polar stereographic if needed
    x, y = get_x_y(grid.lon_2d, grid.lat_2d, pster=pster)
    # Add to plot
    ax.contour(x, y, draft, levels=[draft0], colors=('black'), linestyles='solid')


# Prepare the velocity vectors for plotting: given 3D arrays of u and v on their original grids, and already masked, do a vertical transformation vel_option and interpolate to the tracer grid. Options for vel_option are:
# 'vel': vertically average (default)
# 'sfc': select the top layer
# 'bottom': select the bottom layer
# 'interp': interpolate to depth z0 (positive, in metres)
# 'ice': indicates u and v are already 2D (usually because they're sea ice velocity) so no vertical transformation is needed
# Returns the speed as well as both vector components.
def prepare_vel (u, v, grid, vel_option='avg', z0=None, time_dependent=False):

    # Get the correct 2D velocity field
    if vel_option == 'avg':
        u_2d = vertical_average(u, grid, gtype='u', time_dependent=time_dependent)
        v_2d = vertical_average(v, grid, gtype='v', time_dependent=time_dependent)
    elif vel_option == 'sfc':
        u_2d = select_top(u, time_dependent=time_dependent)
        v_2d = select_top(v, time_dependent=time_dependent)
    elif vel_option == 'bottom':
        u_2d = select_bottom(u, time_dependent=time_dependent)
        v_2d = select_bottom(v, time_dependent=time_dependent)
    elif vel_option == 'ice':
        u_2d = u
        v_2d = v
    elif vel_option == 'interp':
        if z0 is None:
            print "Error (prepare_vel): Must set z0 if option='interp'."
            sys.exit()
        u_2d = interp_to_depth(u, z0, grid, gtype='u', time_dependent=time_dependent)
        v_2d = interp_to_depth(v, z0, grid, gtype='v', time_dependent=time_dependent)

    # Interpolate to the tracer grid
    if vel_option == 'ice':
        # This is sea ice velocity so we need to mask the ice shelves
        mask_shelf = True
    else:
        mask_shelf = False
    u_interp = interp_grid(u_2d, grid, 'u', 't', mask_shelf=mask_shelf, time_dependent=time_dependent)
    v_interp = interp_grid(v_2d, grid, 'v', 't', mask_shelf=mask_shelf, time_dependent=time_dependent)

    # Calculate speed
    speed = np.sqrt(u_interp**2 + v_interp**2)

    return speed, u_interp, v_interp


# Overlay vectors (typically velocity). u_vec and v_vec must have already been processed using prepare_vec. You can tune the appearance of the arrows using the keyword arguments chunk (size of block to average velocity vectors over, so the plot isn't too crowded), scale, headwidth, and headlength (all from the quiver function).
# average_blocks is the helper function, overlay_vectors is the API.

def average_blocks (lon, lat, data_x, data_y, chunk_x, chunk_y, option):

    # Check if there is a mask
    if np.ma.is_masked(data_x):
        mask = True
    else:
        mask = False
    if option == 'max':
        # Get magnitude
        data_mag = np.sqrt(data_x**2 + data_y**2)

    # Figure out dimensions of output array
    nx_chunks = int(np.ceil(data_x.shape[1]/float(chunk_x)))
    ny_chunks = int(np.ceil(data_x.shape[0]/float(chunk_y)))
    lon_blocked = np.zeros([ny_chunks, nx_chunks])
    lat_blocked = np.zeros([ny_chunks, nx_chunks])
    data_x_blocked = np.zeros([ny_chunks, nx_chunks])
    data_y_blocked = np.zeros([ny_chunks, nx_chunks])
    if mask:
        lon_blocked = np.ma.MaskedArray(lon_blocked)
        lat_blocked = np.ma.MaskedArray(lat_blocked)
        data_x_blocked = np.ma.MaskedArray(data_x_blocked)

    # Average over blocks or select location of maximum magnitude
    for j in range(ny_chunks):
        start_j = j*chunk_y
        end_j = min((j+1)*chunk_y, data_x.shape[0])
        for i in range(nx_chunks):
            start_i = i*chunk_x
            end_i = min((i+1)*chunk_x, data_x.shape[1])
            lon_block = lon[start_j:end_j, start_i:end_i]
            lat_block = lat[start_j:end_j, start_i:end_i]
            data_x_block = data_x[start_j:end_j, start_i:end_i]
            data_y_block = data_y[start_j:end_j, start_i:end_i]
            if option == 'avg':
                lon_blocked[j,i] = np.mean(lon_block)
                lat_blocked[j,i] = np.mean(lat_block)
                data_x_blocked[j,i] = np.mean(data_x_block)
                data_y_blocked[j,i] = np.mean(data_y_block)
            elif option == 'max':
                data_mag_block = data_mag[start_j:end_j, start_i:end_i]
                if np.count_nonzero(~data_mag_block.mask) == 0:
                    # Block is entirely masked
                    lon_blocked[j,i] = np.mean(lon_block)
                    lat_blocked[j,i] = np.mean(lat_block)
                    data_x_blocked[j,i] = np.ma.masked
                    data_y_blocked[j,i] = np.ma.masked
                else:
                    [j0, i0] = np.argwhere(data_mag_block==np.amax(data_mag_block))[0]
                    lon_blocked[j,i] = lon_block[j0, i0]
                    lat_blocked[j,i] = lat_block[j0, i0]
                    data_x_blocked[j,i] = data_x_block[j0, i0]
                    data_y_blocked[j,i] = data_y_block[j0, i0]                

    return lon_blocked, lat_blocked, data_x_blocked, data_y_blocked


def overlay_vectors (ax, u_vec, v_vec, grid, option='avg', chunk=10, chunk_x=None, chunk_y=None, scale=0.8, headwidth=6, headlength=7):

    if chunk_x is None:
        chunk_x = chunk
    if chunk_y is None:
        chunk_y = chunk

    lon, lat = grid.get_lon_lat()
    lon_plot, lat_plot, u_plot, v_plot = average_blocks(lon, lat, u_vec, v_vec, chunk_x, chunk_y, option)
    ax.quiver(lon_plot, lat_plot, u_plot, v_plot, scale=scale, headwidth=headwidth, headlength=headlength)
