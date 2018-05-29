#######################################################
# Helper functions for plotting
#######################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import matplotlib.colors as cl
from matplotlib.patches import Polygon
import sys

from utils import mask_land, select_top, select_bottom
import constants as const
from io import netcdf_time
from averaging import vertical_average
from interpolation import interp_grid


# On a timeseries plot, label every month
def monthly_ticks (ax):

    ax.xaxis.set_major_locator(dt.MonthLocator())
    ax.xaxis.set_major_formatter(dt.DateFormatter("%b '%y"))


# On a timeseries plot, label every year
def yearly_ticks (ax):

    ax.xaxis.set_major_locator(dt.YearLocator())
    ax.xaxis.set_major_formatter(dt.DateFormatter('%Y'))


# If a figure name is defined, save the figure to that file. Otherwise, display the figure on screen.
def finished_plot (fig, fig_name=None):

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        fig.show()


# Determine longitude and latitude on the boundaries of cells for the given grid type (tracer, u, v, psi), and throw away one row and one column of the given data field so that every remaining point has latitude and longitude boundaries defined on 4 sides. This is needed for pcolormesh so that the coordinates of the quadrilateral patches are correctly defined.

# Arguments:
# data: array of at least 2 dimensions, where the second last dimension is latitude (size M), and the last dimension is longitude (size N).
# grid: Grid object

# Optional keyword argument:
# gtype: as in function Grid.get_lon_lat

# Output:
# lon: longitude at the boundary of each cell (size MxN)
# lat: latitude at the boundary of each cell (size MxN)
# data: data within each cell (size ...x(M-1)x(N-1), note one row and one column have been removed depending on the grid type)

def cell_boundaries (data, grid, gtype='t'):

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


# Set the limits of the longitude and latitude axes, and give them nice labels.

# Arguments:
# ax: Axes object
# lon, lat: values on x and y axes

# Optional keyword arguments:
# zoom_fris: zoom into the FRIS cavity (bounds set in constants.py)
# xmin, xmax, ymin, ymax: specific limits on longitude and latitude

def latlon_axes (ax, lon, lat, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None):
    
    # Set limits on axes
    if zoom_fris:
        xmin = const.fris_bounds[0]
        xmax = const.fris_bounds[1]
        ymin = const.fris_bounds[2]
        ymax = const.fris_bounds[3]
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # Check location of ticks
    lon_ticks = ax.get_xticks()
    lat_ticks = ax.get_yticks()
    # Often there are way more longitude ticks than latitude ticks
    if float(len(lon_ticks))/float(len(lat_ticks)) > 1.5:
        # Automatic tick locations can disagree with limits of axes, but this doesn't change the axes limits unless you get and then set the tick locations. So make sure there are no disagreements now.
        lon_ticks = lon_ticks[(lon_ticks >= ax.get_xlim()[0])*(lon_ticks <= ax.get_xlim()[1])]
        # Remove every second one
        lon_ticks = lon_ticks[1::2]        
        ax.set_xticks(lon_ticks)

    # Set nice tick labels
    lon_labels = []
    for x in lon_ticks:
        # Decide whether it's west or east
        if x <= 0:
            x = -x
            suff = r'$^{\circ}$W'
        else:
            suff = r'$^{\circ}$E'
        # Decide how to format the number
        if round(x) == x:
            # No decimal places needed
            label = str(int(round(x)))
        elif round(x,1) == x:
            # One decimal place
            label = '{0:.1f}'.format(x)
        else:
            # Round to two decimal places
            label = '{0:.2f}'.format(round(x,2))
        lon_labels.append(label+suff)
    ax.set_xticklabels(lon_labels)
    # Repeat for latitude
    lat_labels = []
    for y in lat_ticks:
        if y <= 0:
            y = -y
            suff = r'$^{\circ}$S'
        else:
            suff = r'$^{\circ}$N'
        if round(y) == y:
            label = str(int(round(y)))
        elif round(y,1) == y:
            label = '{0:.1f}'.format(y)
        else:
            label = '{0:.2f}'.format(round(y,2))
        lat_labels.append(label+suff)
    ax.set_yticklabels(lat_labels)


# Truncate colourmap function from https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n== -1:
        n = cmap.N
    new_cmap = cl.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

    
# Create colourmaps.

# Arguments:
# data: array of data the colourmap will apply to

# Optional keyword arguments:
# ctype: 'basic' is just the 'jet' colourmap
#        'plusminus' creates a red/blue colour map where 0 is white
#        'vel' is the 'cool' colourmap starting at 0; good for plotting velocity
#        'ismr' creates a special colour map for ice shelf melting/refreezing, with negative values in blue, 0 in white, and positive values moving from yellow to orange to red to pink
# vmin, vmax: if defined, enforce these minimum and/or maximum values for the colour map. vmin might get modified for 'ismr' colour map if there is no refreezing (i.e. set to 0).
# change_points: list of size 3 containing values where the 'ismr' colourmap should hit the colours yellow, orange, and red. It should not include the minimum value, 0, or the maximum value. Setting these parameters allows for a nonlinear transition between colours, and enhanced visibility of the melt rate. If it is not defined, the change points will be determined linearly.

# Output:
# vmin, vmax: min and max values for colourmap
# cmap: colourmap to plot with

def set_colours (data, ctype='basic', vmin=None, vmax=None, change_points=None):

    # Work out bounds
    if vmin is None:
        vmin = np.amin(data)
    else:
        # Make sure it's not an integer
        vmin = float(vmin)
    if vmax is None:
        vmax = np.amax(data)
    else:
        vmax = float(vmax)

    if ctype == 'basic':
        return plt.get_cmap('jet'), vmin, vmax

    elif ctype == 'plusminus':
        # Truncate the RdBu_r colourmap as needed, so that 0 is white and no unnecessary colours are shown
        if abs(vmin) > vmax:
            min_colour = 0
            max_colour = 0.5*(1 - vmax/vmin)
        else:
            min_colour = 0.5*(1 + vmin/vmax)
            max_colour = 1
        return truncate_colormap(plt.get_cmap('RdBu_r'), min_colour, max_colour), vmin, vmax

    elif ctype == 'vel':
        # Make sure it starts at 0
        return plt.get_cmap('cool'), 0, vmax

    elif ctype == 'ismr':
        # Fancy colourmap for ice shelf melting and refreezing
        
        # First define the colours we'll use
        ismr_blue = (0.26, 0.45, 0.86)
        ismr_white = (1, 1, 1)
        ismr_yellow = (1, 0.9, 0.4)
        ismr_orange = (0.99, 0.59, 0.18)
        ismr_red = (0.5, 0.0, 0.08)
        ismr_pink = (0.96, 0.17, 0.89)
        
        if change_points is None:            
            # Set change points to yield a linear transition between colours
            change_points = 0.25*vmax*np.arange(1,3+1)
        if len(change_points) != 3:
            print 'Error (set_colours): wrong size for change_points list'
            sys.exit()
            
        if vmin < 0:
            # There is refreezing here; include blue for elements < 0
            cmap_vals = np.concatenate(([vmin], [0], change_points, [vmax]))
            cmap_colours = [ismr_blue, ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]            
            cmap_vals_norm = (cmap_vals-vmin)/(vmax-vmin)
        else:
            # No refreezing; start at 0
            cmap_vals = np.concatenate(([0], change_points, [vmax]))
            cmap_colours = [ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]
            cmap_vals_norm = cmap_vals/vmax
        cmap_vals_norm[-1] = 1
        cmap_list = []
        for i in range(cmap_vals.size):
            cmap_list.append((cmap_vals_norm[i], cmap_colours[i]))

        # Make sure vmin isn't greater than 0
        return cl.LinearSegmentedColormap.from_list('ismr', cmap_list), min(vmin,0), vmax


# Shade the given boolean mask in grey on the plot.
def shade_mask (ax, mask, grid, gtype='t'):

    # Properly mask all the False values, so that only True values are unmasked
    mask_plot = np.ma.masked_where(np.invert(mask), mask)
    # Prepare quadrilateral patches
    lon, lat, mask_plot = cell_boundaries(mask_plot, grid, gtype=gtype)
    # Add to plot
    ax.pcolormesh(lon, lat, mask_plot, cmap=cl.ListedColormap([(0.6, 0.6, 0.6)]))


# Shade the land in grey
def shade_land (ax, grid, gtype='t'):

    shade_mask(ax, grid.get_land_mask(gtype=gtype), grid, gtype=gtype)


# Shade the land and ice shelves in grey
def shade_land_zice (ax, grid, gtype='t'):

    shade_mask(ax, grid.get_land_mask(gtype=gtype)+grid.get_zice_mask(gtype=gtype), grid, gtype=gtype)
    

# Contour the ice shelf front in black
def contour_iceshelf_front (ax, grid):

    # Mask land out of ice shelf draft, so that grounding line isn't contoured
    zice = mask_land(grid.zice, grid)
    # Find the shallowest non-zero ice shelf draft
    zice0 = np.amax(zice[zice!=0])
    # Add to plot
    ax.contour(grid.lon_2d, grid.lat_2d, zice, levels=[zice0], colors=('black'), linestyles='solid')


# Find the minimum and maximum values of an array in the given region.

# Arguments:
# data: 2D array (lat x lon), already masked as desired
# grid: Grid object

# Optional keyword arguments:
# zoom_fris: as in function latlon_axes
# xmin, xmax, ymin, ymax: as in function latlon_axes
# gtype: as in function Grid.get_lon_lat

# Output:
# vmin, vmax: min and max values of data in the given region

def set_colour_bounds (data, grid, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, gtype='t'):

    # Choose the correct longitude and latitude arrays
    lon, lat = grid.get_lon_lat(gtype=gtype)

    # Set limits on axes
    if zoom_fris:
        xmin = const.fris_bounds[0]
        xmax = const.fris_bounds[1]
        ymin = const.fris_bounds[2]
        ymax = const.fris_bounds[3]
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lon)
    if ymax is None:
        ymax = np.amax(lon)

    # Select the correct indices
    loc = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)
    # Find the min and max values
    return np.amin(data[loc]), np.amax(data[loc])


# Given a date, return a nice string that can be added to plots.
# Option 1: set keyword argument "date" with a Datetime object.
# Option 2: set keyword arguments "file_path" and "time_index" to read the date from a NetCDF file.
def parse_date (file_path=None, time_index=None, date=None):

    # Create the Datetime object if needed
    if date is None:
        date = netcdf_time(file_path)[time_index]
    return date.strftime('%d %b %Y')


# Given 3D arrays of u and v on their original grids, do a vertical transformation (vertically average, select top layer, or select bottom layer) and interpolate to the tracer grid. Return the speed as well as both vector components.

# Arguments:
# u, v: 3D (depth x lat x lon) arrays of u and v, on the u-grid and v-grid respectively, already masked with hfac
# grid: Grid option

# Optional keyword argument:
# vel_option: 'vel' (vertically average, default), 'sfc' (select the top layer), 'bottom' (select the bottom layer), or 'ice' (sea ice velocity so no vertical transformation is needed)

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


# Average a 2D array into blocks of size chunk x chunk. This is good for plotting vectors so the plot isn't too crowded.

# Arguments:
# data: 2D array, either masked or unmasked
# chunk: integer representing the side length of each chunk to average. It doesn't have to evenly divide the array; the last row and column of chunks will just be smaller if necessary.

# Output: 2D array of smaller dimension (ceiling of original dimensions divided by chunk). If "data" has masked values, any blocks which are completely masked will also be masked in the output array.

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
        

# Overlay vectors (typically velocity).

# Arguments:
# ax: Axes object
# u_vec, v_vec: 2D velocity components to overlay, already interpolated to the tracer grid
# grid: Grid object

# Optional keyword arguments:
# chunk: size of block to average velocity vectors over (so plot isn't too crowded)
# scale, headwidth, headlength: arguments to the "quiver" function, to fine-tune the appearance of the arrows

def overlay_vectors (ax, u_vec, v_vec, grid, chunk=10, scale=0.8, headwidth=6, headlength=7):

    lon, lat = grid.get_lon_lat()
    lon_plot = average_blocks(lon, chunk)
    lat_plot = average_blocks(lat, chunk)
    u_plot = average_blocks(u_vec, chunk)
    v_plot = average_blocks(v_vec, chunk)
    ax.quiver(lon_plot, lat_plot, u_plot, v_plot, scale=scale, headwidth=headwidth, headlength=headlength)



def slice_patches (data, grid, gtype='t', lon0=None, lat0=None):    

    if gtype not in ['t', 'u', 'v', 'psi']:
        print 'Error (slice_patches): the ' + gtype + '-grid is not supported for slices'

    # Figure out direction of slice
    if lon0 is not None and lat0 is None:
        h_axis = 'lat'
    elif lat0 is not None and lon0 is None:
        h_axis = 'lon'
    else:
        print 'Error (slice_cell_boundaries): must specify exactly one of lon0, lat0'
        sys.exit()

    # Find nearest neighbour to lon0 and slice the data here
    lon, lat = grid.get_lon_lat(gtype=gtype, dim=1)
    if h_axis == 'lat':
        i0 = np.argmin(abs(lon-lon0))
        data_slice = data[:,:,i0]
        # Save the real location of the slice
        loc0 = lon[i0]
    elif h_axis == 'lon':
        j0 = np.argmin(abs(lat-lat0))
        data_slice = data[:,j0,:]
        loc0 = lat[j0]

    # Get horizontal boundaries, as well as hfac and surface depth (grid.zice)
    # Also throw away one row of data so all points are bounded
    if h_axis == 'lat':
        if gtype in ['t', 'u']:
            # Centered in y
            # Boundaries are southern edges of cells in y            
            h_bdry = grid.lat_corners_1d
            # Throw away northernmost row of data
            data_slice = data_slice[:,:-1]
            # Get hfac and zice at centres
            hfac = grid.hfac[:,:-1,i0]
            zice = grid.zice[:-1,i0]
        elif gtype in ['v', 'psi']:
            # Edges in y
            # Boundaries are centres of cells in y
            h_bdry = grid.lat_1d
            # Throw away southernmost row of data
            data_slice = data_slice[:,1:]
            # Get hfac at edges
            hfac = grid.hfac_s[:,1:,i0]
            # Ice shelf draft at these edges is the minimum of the tracer points on either side
            zice = np.minimum(grid.zice[:-1,i0], grid.zice[1:,i0])
    elif h_axis == 'lon':
        if gtype in ['t', 'v']:
            # Centered in x
            # Boundaries are western edges of cells in x
            h_bdry = grid.lon_corners_1d
            # Throw away easternmost row of data
            data_slice = data_slice[:,:-1]
            # Get hfac and zice at centres
            hfac = grid.hfac[:,j0,:-1]
            zice = grid.zice[j0,:-1]
        elif gtype in ['u', 'psi']:
            # Edges in x
            # Boundaries are centres of cells in x
            h_bdry = grid.lon_1d
            # Throw away westernmost row of data
            data_slice = data_slice[:,1:]
            # Get hfac at edges
            hfac = grid.hfac_w[:,j0,1:]
            # Ice shelf draft at these edges is the minimum of the tracer points on either side
            zice = np.minimum(grid.zice[j0,:-1], grid.zice[j0,1:])
    nh = data_slice.shape[1]

    # Now set up a bunch of information about the grid, all stored in arrays with the same dimension as data_slice. This helps with vectorisation later.        
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
        print 'Error (slice_cell_boundaries): something went wrong in calculation of partial cells'
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
        print 'Error (slice_cell_boundaries): something went wrong in calculation of partial cells'
        sys.exit()
    below = depth_below + depth_below_2
    index = below == 0
    below[index] = lev_below[index]

    # Now make the rectangular patches, using flattened arrays
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
    
    return loc0, patches, data_slice.ravel()


# Set things up for complicated multi-panelled plots. Initialise a figure window of the correct size and set the locations of panels and colourbar(s). The exact output depends on the single argument, which is a string containing the key for the type of plot you want. Read the comments to choose one.
def set_panels (key):

    if key == '1x2C1':
        # Two side-by-side plots with one colourbar below
        fig = plt.figure(figsize=(12,6))
        gs = plt.GridSpec(1,2)
        gs.update(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.05)
        cbaxes = fig.add_axes([0.3, 0.05, 0.4, 0.04])
        return fig, gs, cbaxes
    
            
        
        

    
        
