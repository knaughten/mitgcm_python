#######################################################
# Miscellaneous useful tools
#######################################################

import numpy as np
import sys

from .constants import rho_fw, sec_per_year, region_bounds, deg2rad, rEarth


# Given an array containing longitude, make sure it's in the range (max_lon-360, max_lon). Default is (-180, 180). If max_lon is None, nothing will be done to the array.
def fix_lon_range (lon, max_lon=180):

    if max_lon is not None:
        index = lon >= max_lon
        lon[index] = lon[index] - 360
        index = lon < max_lon-360
        lon[index] = lon[index] + 360
    return lon


# Convert freshwater flux into the ice shelf (diagnostic SHIfwFlx) (kg/m^2/s, positive means freezing) to ice shelf melt rate (m/y, positive means melting).
def convert_ismr (shifwflx):

    return -shifwflx/rho_fw*sec_per_year


# Tile a 2D (lat x lon) array in depth so it is 3D (depth x lat x lon).
# grid can either be a Grid object or an array of grid dimensions [nx, ny, nz].
def xy_to_xyz (data, grid):

    if isinstance(grid, list):
        nz = grid[2]
    else:
        nz = grid.nz

    return np.tile(data, (nz, 1, 1))


# Tile a 1D depth array in lat and lon so it is 3D (depth x lat x lon).
def z_to_xyz (data, grid):

    if isinstance(grid, list):
        nx = grid[0]
        ny = grid[1]
    else:
        nx = grid.nx
        ny = grid.ny

    return np.tile(np.expand_dims(np.expand_dims(np.copy(data),1),2), (1, ny, nx))


# Tile any array (of any dimension) in time, with num_time records. Time will be the first dimension in the new array.
def add_time_dim (data, num_time):

    shape = [num_time]
    for i in range(len(data.shape)):
        shape += [1]
    return np.tile(data, shape)


# Helper function for select_top and select_bottom
def select_level (option, data, masked=True, grid=None, gtype='t', time_dependent=False, return_masked=None):

    if not masked:
        if grid is None:
            print('Error (select_level): need to supply grid if masked=False')
            sys.exit()
        data_masked = mask_3d(np.copy(data), grid, gtype=gtype, time_dependent=time_dependent)
    else:
        data_masked = data
    if return_masked is None:
        return_masked = masked

    # Figure out the dimensions of the data when the vertical dimension is removed
    collapsed_shape = data_masked.shape[:-3] + data_masked.shape[-2:]
    # Array which will hold values at the given level, initialised to NaN
    data_lev = np.zeros(collapsed_shape)
    data_lev[:] = np.nan
    if option == 'top':
        # Loop from surface to bottom
        k_vals = list(range(data_masked.shape[-3]))
    elif option == 'bottom':
        # Loop from bottom to top
        k_vals = list(range(data.shape[-3]-1, -1, -1))
    else:
        print(('Error (select_level): invalid option ' + option))
        sys.exit()
    for k in k_vals:
        curr_data = data_masked[...,k,:,:]
        # Find points which are unmasked at this vertical level, and which
        # haven't already been assigned a top level
        index = np.nonzero(np.invert(curr_data.mask)*np.isnan(data_lev))
        data_lev[index] = curr_data[index]
    # Anything still NaN is land; mask it out
    data_lev = np.ma.masked_where(np.isnan(data_lev), data_lev)

    if not return_masked:
        # Fill the mask with zeros
        data_lev[data_lev.mask] = 0
        data_lev = data_lev.data

    return data_lev


# Select the top layer from the given array of data. This is useful to see conditions immediately beneath ice shelves.
# If masked=True (default), the input array is already masked with hfac (see mask_3d below). If masked=False, you need to supply the keyword arguments grid, gtype, and time_dependent (as in mask_3d). You can also control whether or not the output array is a masked array using return_masked (default the same value as masked).
# The only assumption about the input array dimensions is that the third last dimension is the vertical dimension. So it can be depth x lat x lon, or time x depth x lat x lon, or even something like experiment x time x depth x lat x lon.
def select_top (data, masked=True, grid=None, gtype='t', time_dependent=False, return_masked=None):
    return select_level('top', data, masked=masked, grid=grid, gtype=gtype, time_dependent=time_dependent, return_masked=return_masked)

    
# Select the bottom layer from the given array of data. See select_top for more documentation.
def select_bottom (data, masked=True, grid=None, gtype='t', time_dependent=False, return_masked=None):
    return select_level('bottom', data, masked=masked, grid=grid, gtype=gtype, time_dependent=time_dependent, return_masked=return_masked)


# Helper function for masking functions below
# depth_dependent only has an effect if the mask is 2D.
def apply_mask (data, mask, time_dependent=False, depth_dependent=False):

    if depth_dependent and len(mask.shape)==2:
        # Tile a 2D mask in the depth dimension
        grid_dim = [data.shape[-1], data.shape[-2], data.shape[-3]]
        mask = xy_to_xyz(mask, grid_dim)
    if time_dependent:
        # Tile the mask in the time dimension
        mask = add_time_dim(mask, data.shape[0])

    if len(mask.shape) != len(data.shape):
        print('Error (apply_mask): invalid dimensions of data')
        sys.exit()

    data = np.ma.masked_where(mask, data)
    return data


# Mask land out of an array.

# Arguments:
# data: array of data to mask, assumed to be 2D unless time_dependent or depth_dependent say otherwise
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask
# depth_dependent: as in function apply_mask

def mask_land (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, grid.get_land_mask(gtype=gtype), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Mask land and ice shelves out of an array, just leaving the open ocean.
def mask_land_ice (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, grid.get_land_mask(gtype=gtype)+grid.get_ice_mask(gtype=gtype), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Mask land and open ocean out of an array, just leaving the ice shelves.
def mask_except_ice (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, np.invert(grid.get_ice_mask(gtype=gtype)), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Mask everything except FRIS out of an array.
def mask_except_fris (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, np.invert(grid.get_ice_mask(shelf='fris', gtype=gtype)), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Apply the 3D hfac mask. Dry cells are masked out; partial cells are untouched.

# Arguments:
# data: array of data to mask, assumed to be 3D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_3d (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, grid.get_hfac(gtype=gtype)==0, time_dependent=time_dependent)


# Find the indices bounding the given year in the given time array. This script doesn't check that the entire year is within the array! Partial years are supported.

# Arguments:
# time: array of Datetime objects (can be created by the function netcdf_time)
# year: integer containing the year we care about

# Output: two integers containing the first index of year in time, and the first index of the next year (i.e. the last index of the year plus one, following python convention).

def select_year (time, year):

    t_start = -1
    for t in range(time.size):
        if time[t].year == year:
            t_start = t
            break
    if t_start == -1:
        print(('Error (trim_year): this array contains no instances of the year ' + str(year)))
        sys.exit()
    t_end = time.size
    for t in range(t_start+1, time.size):
        if time[t].year == year+1:
            t_end = t
            break
    return t_start, t_end


# Convert longitude and latitude to polar stereographic projection used by BEDMAP2. Adapted from polarstereo_fwd.m in the MITgcm Matlab toolbox for Bedmap.
def polar_stereo (lon, lat, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    # Deep copies of arrays in case they are reused
    lon = np.copy(lon)
    lat = np.copy(lat)

    if lat_c < 0:
        # Southern hemisphere
        pm = -1
    else:
        # Northern hemisphere
        pm = 1

    # Prepare input
    lon = lon*pm*deg2rad
    lat = lat*pm*deg2rad
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*pm*deg2rad

    # Calculations
    t = np.tan(np.pi/4 - lat/2)/((1 - e*np.sin(lat))/(1 + e*np.sin(lat)))**(e/2)
    t_c = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    m_c = np.cos(lat_c)/np.sqrt(1 - (e*np.sin(lat_c))**2)
    rho = a*m_c*t/t_c
    x = pm*rho*np.sin(lon - lon0)
    y = -pm*rho*np.cos(lon - lon0)

    return x, y


# Determine the x and y coordinates based on whether the user wants polar stereographic or not.
def get_x_y (lon, lat, pster=False):
    if pster:
        x, y = polar_stereo(lon, lat)
    else:
        x = lon
        y = lat
    return x, y


# Find the minimum and maximum values of a 2D (lat x lon) array in the given region.
def var_min_max (data, grid, pster=False, zoom_fris=False, xmin=None, xmax=None, ymin=None, ymax=None, gtype='t', ua=False):

    if ua:
        # grid is a list with x and y wrapped up in it
        [x, y] = grid
    else:
        # Choose the correct longitude and latitude arrays
        lon, lat = grid.get_lon_lat(gtype=gtype)
        # Convert to polar stereographic if needed
        x, y = get_x_y(lon, lat, pster=pster)

    # Set limits on axes
    if zoom_fris:
        if pster:
            [xmin, xmax, ymin, ymax] = region_bounds['fris_pster_plot']
        else:
            [xmin, xmax, ymin, ymax] = region_bounds['fris_plot']
    if xmin is None:
        xmin = np.amin(x)
    if xmax is None:
        xmax = np.amax(x)
    if ymin is None:
        ymin = np.amin(y)
    if ymax is None:
        ymax = np.amax(y)

    # Select the correct indices
    loc = (x >= xmin)*(x <= xmax)*(y >= ymin)*(y <= ymax)
    # Find the min and max values
    return np.amin(data[loc]), np.amax(data[loc])


# As above, but for a time x depth array, where the depth axis may be zoomed.
# Assumes not on the w-grid.
def var_min_max_zt (data, grid, zmin=None, zmax=None):

    if zmin is None:
        zmin = grid.z[-1]
    if zmax is None:
        zmax = grid.z[0]
    # Make z 2D
    z = add_time_dim(np.copy(grid.z), data.shape[0])
    loc = (z >= zmin)*(z <= zmax)
    return np.amin(data[loc]), np.amax(data[loc])


# Find all the factors of the integer n.
def factors (n):

    factors = []
    for i in range(1, n+1):
        if n % i == 0:
            factors.append(i)
    return factors


# Given a path to a directory, make sure it ends with /
def real_dir (dir_path):

    if not dir_path.endswith('/'):
        dir_path += '/'
    return dir_path


# Given an array representing a mask (as above) and 2D arrays of longitude and latitude, mask out the points between the given lat/lon bounds.
def mask_box (data, lon, lat, xmin=None, xmax=None, ymin=None, ymax=None, mask_val=0):

    # Set any bounds which aren't already set
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    index = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)
    data[index] = mask_val
    return data


# Mask out the points above or below the line segment bounded by the given points.
def mask_line (data, lon, lat, p_start, p_end, direction, mask_val=0):

    limit = (p_end[1] - p_start[1])/float(p_end[0] - p_start[0])*(lon - p_start[0]) + p_start[1]
    west_bound = min(p_start[0], p_end[0])
    east_bound = max(p_start[0], p_end[0])
    if direction == 'above':
        index = (lat >= limit)*(lon >= west_bound)*(lon <= east_bound)
    elif direction == 'below':
        index = (lat <= limit)*(lon >= west_bound)*(lon <= east_bound)
    else:
        print(('Error (mask_line): invalid direction ' + direction))
        sys.exit()
    data[index] = mask_val
    return data


# Interface to mask_line: mask points above line segment (to the north)
def mask_above_line (data, lon, lat, p_start, p_end, mask_val=0):

    return mask_line(data, lon, lat, p_start, p_end, 'above', mask_val=mask_val)


# Interface to mask_line: mask points below line segment (to the south)
def mask_below_line (data, lon, lat, p_start, p_end, mask_val=0):

    return mask_line(data, lon, lat, p_start, p_end, 'below', mask_val=mask_val)


# Like mask_box, but only mask out ice shelf points within the given box.
def mask_iceshelf_box (omask, imask, lon, lat, xmin=None, xmax=None, ymin=None, ymax=None, mask_val=0, option='land'):

    # Set any bounds which aren't already set
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    index = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)*(imask == 1)
    if option == 'land':
        # Turn ice shelf points into land
        mask = omask
    elif option == 'ocean':
        # Turn ice shelf points into open ocean
        mask = imask
    else:
        print(('Error (mask_iceshelf_box): Invalid option ' + option))
        sys.exit()
    mask[index] = mask_val
    return mask


# Split and rearrange the given array along the given index in the longitude axis (last axis). This is useful when converting from longitude ranges (0, 360) to (-180, 180) if the longitude array needs to be strictly increasing for later interpolation.
def split_longitude (array, split):

    return np.concatenate((array[...,split:], array[...,:split]), axis=-1)


# Return the root mean squared difference between the two arrays (assumed to be the same size), summed over all entries.
def rms (array1, array2):

    return np.sqrt(np.sum((array1 - array2)**2))


# Work out whether the given year is a leap year.
def is_leap_year (year):
    return year%4 == 0 and (year%100 != 0 or year%400 == 0)


# Return the number of days in the given month (indexed 1-12) of the given year.
def days_per_month (month, year, allow_leap=True):

    # Days per month in non-leap years
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Special case for February in leap years
    if month == 2 and is_leap_year(year) and allow_leap:
        return days[month-1]+1
    else:
        return days[month-1]


# Make sure the given field isn't time-dependent, based on the expected number of dimensions.
def check_time_dependent (var, num_dim=3):

    if len(var.shape) == num_dim+1:
        print('Error (check_time_dependent): variable cannot be time dependent.')
        sys.exit()


# Calculate hFacC, hFacW, or hFacS (depending on value of gtype) without knowing the full grid, i.e. just from the bathymetry and ice shelf draft on the tracer grid.
def calc_hfac (bathy, draft, z_edges, hFacMin=0.1, hFacMinDr=20., gtype='t'):

    if gtype == 'u':
        # Need to get bathy and draft on the western edge of each cell
        # Choose the shallowest bathymetry from the adjacent tracer cells
        bathy = np.concatenate((np.expand_dims(bathy[:,0],1), np.maximum(bathy[:,:-1], bathy[:,1:])), axis=1)
        # Choose the deepest ice shelf draft from the adjacent tracer cells
        draft = np.concatenate((np.expand_dims(draft[:,0],1), np.minimum(draft[:,:-1], draft[:,1:])), axis=1)
        # Now correct for negative wct
        draft = np.maximum(draft, bathy)
    elif gtype == 'v':
        # Need to get bathy and draft on the southern edge of each cell
        bathy = np.concatenate((np.expand_dims(bathy[0,:],0), np.maximum(bathy[:-1,:], bathy[1:,:])), axis=0)
        draft = np.concatenate((np.expand_dims(draft[0,:],0), np.minimum(draft[:-1,:], draft[1:,:])), axis=0)
        draft = np.maximum(draft, bathy)        

    # Calculate a few grid variables
    z_above = z_edges[:-1]
    z_below = z_edges[1:]
    dz = np.abs(z_edges[1:] - z_edges[:-1])
    nz = dz.size
    ny = bathy.shape[0]
    nx = bathy.shape[1]    
    
    # Tile all arrays to be 3D
    bathy = xy_to_xyz(bathy, [nx, ny, nz])
    draft = xy_to_xyz(draft, [nx, ny, nz])
    dz = z_to_xyz(dz, [nx, ny, ny])
    z_above = z_to_xyz(z_above, [nx, ny, nz])
    z_below = z_to_xyz(z_below, [nx, ny, nz])
    
    # Start out with all cells closed
    hfac = np.zeros([nz, ny, nx])
    # Find fully open cells
    index = (z_below >= bathy)*(z_above <= draft)
    hfac[index] = 1
    # Find partial cells due to bathymetry alone
    index = (z_below < bathy)*(z_above <= draft)
    hfac[index] = (z_above[index] - bathy[index])/dz[index]
    # Find partial cells due to ice shelf draft alone
    index = (z_below >= bathy)*(z_above > draft)
    hfac[index] = (draft[index] - z_below[index])/dz[index]
    # Find partial cells which are intersected by both
    index = (z_below < bathy)*(z_above > draft)
    hfac[index] = (draft[index] - bathy[index])/dz[index]

    # Now apply hFac limitations
    hfac_limit = np.maximum(hFacMin, np.minimum(hFacMinDr/dz, 1))    
    index = hfac < hfac_limit/2
    hfac[index] = 0
    index = (hfac >= hfac_limit/2)*(hfac < hfac_limit)
    hfac[index] = hfac_limit[index]

    return hfac


# Calculate bathymetry or ice shelf draft from hFacC.
def bdry_from_hfac (option, hfac, z_edges):

    nz = hfac.shape[0]
    ny = hfac.shape[1]
    nx = hfac.shape[2]
    dz = z_edges[:-1]-z_edges[1:]

    bdry = np.zeros([ny, nx])
    bdry[:,:] = np.nan
    if option == 'bathy':
        # Loop from bottom to top
        k_vals = list(range(nz-1, -1, -1))
    elif option == 'draft':
        # Loop from top to bottom
        k_vals = list(range(nz))
    else:
        print(('Error (bdry_from_hfac): invalid option ' + option))
        sys.exit()
    for k in k_vals:
        hfac_tmp = hfac[k,:]
        # Identify wet cells with no boundary assigned yet
        index = (hfac_tmp!=0)*np.isnan(bdry)
        if option == 'bathy':
            bdry[index] = z_edges[k] - dz[k]*hfac_tmp[index]
        elif option == 'draft':
            bdry[index] = z_edges[k] - dz[k]*(1-hfac_tmp[index])
    # Anything still NaN is land mask and should be zero
    index = np.isnan(bdry)
    bdry[index] = 0

    return bdry


# Modify the given bathymetry or ice shelf draft to make it reflect what the model will actually see, based on hFac constraints.
def model_bdry (option, bathy, draft, z_edges, hFacMin=0.1, hFacMinDr=20.):

    # First calculate the hFacC
    hfac = calc_hfac(bathy, draft, z_edges, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    # Now calculate the new boundary
    return bdry_from_hfac(option, hfac, z_edges)


# Determine if a string is an integer.
def str_is_int (s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Find the Cartesian distance between two lon-lat points.
# This also works if one of point0, point1 is a 2D array of many points.
def dist_btw_points (point0, point1):
    [lon0, lat0] = point0
    [lon1, lat1] = point1
    dx = rEarth*np.cos((lat0+lat1)/2*deg2rad)*(lon1-lon0)*deg2rad
    dy = rEarth*(lat1-lat0)*deg2rad
    return np.sqrt(dx**2 + dy**2)


# Find all ice shelf front points and return them as a list.
# For a specific ice shelf, pass a special ice_mask 
def ice_shelf_front_points (grid, ice_mask=None, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None):

    from .interpolation import neighbours

    # Build masks
    if ice_mask is None:
        ice_mask = grid.get_ice_mask(gtype=gtype)
    open_ocean = grid.get_open_ocean_mask(gtype=gtype)

    # Set any remaining bounds
    lon, lat = grid.get_lon_lat(gtype=gtype)
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)

    # Find number of open-ocean neighbours for each point
    num_open_ocean_neighbours = neighbours(open_ocean, missing_val=0)[-1]
    # Find all ice shelf points within bounds that have at least 1 open-ocean neighbour
    return ice_mask*(lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)*(num_open_ocean_neighbours > 0)


# Given an axis with values in the centre of each cell, find the locations of the boundaries of each cell (extrapolating for the outer boundaries).
def axis_edges (x):
    x_bound = 0.5*(x[:-1]+x[1:])
    x_bound = np.concatenate(([2*x_bound[0]-x_bound[1]], x_bound, [2*x_bound[-1]-x_bound[-2]]))
    return x_bound


# Given an array (or two), find the min and max value (unless these are already defined), and pad with the given percentage (default 2%) of the difference between them.
def choose_range (x1, x2=None, xmin=None, xmax=None, pad=0.02):

    xmin_set = xmin is not None
    xmax_set = xmax is not None

    if not xmin_set:
        xmin = np.amin(x1)
        if x2 is not None:
            xmin = min(xmin, np.amin(x2))
    if not xmax_set:
        xmax = np.amax(x1)
        if x2 is not None:
            xmax = max(xmax, np.amax(x2))
            
    delta = pad*(xmax-xmin)
    if not xmin_set:
        xmin -= delta
    if not xmax_set:
        xmax += delta
    return xmin, xmax


# Figure out if a field is depth-dependent, given the last two dimensions being lat and lon, and the possibility of time-dependency.
def is_depth_dependent (data, time_dependent=False):
    return (time_dependent and len(data.shape)==4) or (not time_dependent and len(data.shape)==3)


# Mask everything outside the given bounds. The array must include latitude and longitude dimensions; depth and time are optional.
def mask_outside_box (data, grid, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, time_dependent=False):
    depth_dependent = is_depth_dependent(data, time_dependent=time_dependent)
    lon, lat = grid.get_lon_lat(gtype=gtype)
    if depth_dependent:
        lon = xy_to_xyz(lon, grid)
        lat = xy_to_xyz(lat, grid)
    if time_dependent:
        lon = add_time_dim(lon, data.shape[0])
        lat = add_time_dim(lat, data.shape[0])
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    index = np.invert((lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax))
    return np.ma.masked_where(index, data)


# Given a field with a periodic boundary (in longitude), wrap it on either end so we can interpolate with  no gaps in the middle. If is_lon, add/subtract 360 from these values, if needed, to make sure it's monotonic.
def wrap_periodic (data, is_lon=False):

    # Add 1 column to the beginning and 1 to the end of the longitude dimension
    new_shape = list(data.shape[:-1]) + [data.shape[-1]+2]
    data_wrap = np.empty(new_shape)
    # Copy the middle
    data_wrap[...,1:-1] = data
    # Wrap the edges from either end
    data_wrap[...,0] = data[...,-1]
    data_wrap[...,-1] = data[...,0]
    if is_lon:
        if np.amin(np.diff(data, axis=-1)) < 0:
            print('Error (wrap_periodic): longitude array is not monotonic')
            sys.exit()
        # Add/subtract 360, if needed
        data_wrap[...,0] -= 360
        data_wrap[...,-1] += 360
    return data_wrap


# Given an array of one year of data where the first dimension is time, convert from daily averages to monthly averages.
# If you want to consider leap years, pass the year argument. The default is a year with no leap (1979).
# If there is more than one record per day, set the per_day argument.
def daily_to_monthly (data, year=1979, per_day=1):

    if data.shape[0]//per_day not in [365, 366]:
        print('Error (daily_to_monthly): The first dimension is not time, or else this is not one year of data.')
        sys.exit()
    new_shape = [12] + list(data.shape[1:])
    if isinstance(data, np.ma.MaskedArray):
        data_monthly = np.ma.empty(new_shape)
    else:
        data_monthly = np.empty(new_shape)
    t = 0
    for month in range(12):
        nt = days_per_month(month+1, year)*per_day
        data_monthly[month,...] = np.mean(data[t:t+nt,...], axis=0)
        t += nt
    return data_monthly


# Given a set of titles, find the common parts from the beginning and the end of each title. Trim them and return the master beginning title (trimmed of unnecessary prepositions) as well as the trimmed individual titles.
# For example, the list
# ['Basal mass balance of Pine Island Glacier Ice Shelf',
#  'Basal mass balance of Dotson and Crosson Ice Shelves',
#  'Basal mass balance of Thwaites Ice Shelf']
# would return the master title 'Basal mass balance' and the trimmed titles
# ['Pine Island Glacier', 'Dotson and Crosson', 'Thwaites']
def trim_titles (titles):

    # First replace "shelves" with "shelf" (ignore s so not case sensitive)
    for n in range(len(titles)):
        titles[n] = titles[n].replace('helves', 'helf')
    # Trim the common starts and ends, saving the starts
    title_start = ''
    found = True
    while found:
        found = False
        if all([s[0]==titles[0][0] for s in titles]):
            found = True
            title_start += titles[0][0]
            titles = [s[1:] for s in titles]
        if all([s[-1]==titles[0][-1] for s in titles]):
            found = True
            titles = [s[:-1] for s in titles]
    # Trim any white space
    title_start = title_start.strip()
    # Remove prepositions
    for s in [' of', ' in', ' from']:
        if title_start.endswith(s):
            title_start = title_start[:title_start.index(s)]
    return title_start, titles


# Smooth the given data with a moving average of the given window, and trim and/or shift the time axis too if it's given. The data can be of any number of dimensions; the smoothing will happen on the first dimension.
def moving_average (data, window, time=None, keep_edges=False):

    if window == 0:
        if time is not None:
            return data, time
        else:
            return data

    centered = window%2==1
    if centered:
        radius = (window-1)//2
    else:
        radius = window//2

    # Will have to trim each end by one radius
    t_first = radius
    t_last = data.shape[0] - radius  # First one not selected, as per python convention
    # Need to set up an array of zeros of the same shape as a single time index of data
    shape = [1]
    for t in range(1, len(data.shape)):
        shape.append(data.shape[t])
    zero_base = np.zeros(shape)
    # Do the smoothing in two steps
    data_cumsum = np.ma.concatenate((zero_base, np.ma.cumsum(data, axis=0)), axis=0)
    if centered:
        data_smoothed = (data_cumsum[t_first+radius+1:t_last+radius+1,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius+1)
    else:
        data_smoothed = (data_cumsum[t_first+radius:t_last+radius,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius)
    if keep_edges:
        # Add the edges back on, smoothing them as much as we can with smaller windows.
        if centered:
            data_smoothed_full = np.ma.empty(data.shape)
            data_smoothed_full[t_first:t_last,...] = data_smoothed
            for n in range(radius):
                # Edges at beginning
                data_smoothed_full[n,...] = np.mean(data[:2*n+1,...], axis=0)
                # Edges at end
                data_smoothed_full[-(n+1),...] = np.mean(data[-(2*n+1):,...], axis=0)
            data_smoothed = data_smoothed_full
        else:
            print('Error (moving_average): have not yet coded keep_edges=False for even windows. Want to figure it out?')
            sys.exit()
    if time is not None and not keep_edges:
        if centered:
            time_trimmed = time[radius:time.size-radius]
        else:
            # Need to shift time array half an index forward
            # This will work whether it's datetime or numerical values
            time1 = time[radius-1:time.size-radius-1]
            time2 = time[radius:time.size-radius]
            if isinstance(time[0], int):
                time_trimmed = time1 + (time2-time1)/2.
            else:
                time_trimmed = time1 + (time2-time1)//2 # Can't have a float for datetime            
        return data_smoothed, time_trimmed
    else:
        return data_smoothed


# Return the index of the given start year in the array of Datetime objects.
def index_year_start (time, year0):
    years = np.array([t.year for t in time])
    return np.where(years==year0)[0][0]

# Return the first index after the given end year in the array of Datetime objects.
def index_year_end (time, year0):
    years = np.array([t.year for t in time])
    if years[-1] == year0:
        return years.size
    else:
        return np.where(years>year0)[0][0]

# Do both at once
def index_period (time, year_start, year_end):
    return index_year_start(time, year_start), index_year_end(time, year_end)


# Helper function to make a 2D mask 3D, with masking of bathymetry and optional depth bounds (zmin=deep, zmax=shallow, both negative in metres)
def mask_2d_to_3d (mask, grid, zmin=None, zmax=None):

    if zmin is None:
        zmin = grid.z[-1]
    if zmax is None:
        zmax = grid.z[0]
    mask = xy_to_xyz(mask, grid)
    # Mask out closed cells
    mask *= grid.hfac!=0
    # Mask out everything outside of depth bounds
    z_3d = z_to_xyz(grid.z, grid)
    mask *= (z_3d >= zmin)*(z_3d <= zmax)
    return mask


# Helper function to average 1 year of monthly data from a variable (starting with time index index t0), of any dimension (as long as time is first), with proper monthly weighting for the given calendar (360-day, noleap, or standard - if standard need to provide the year).
def average_12_months (data, t0, calendar='standard', year=None):

    if calendar == 'standard' and year is None:
        print('Error (average_12_months): must provide year')
    if calendar in ['360-day', '360_day']:
        days = None
    else:
        if calendar == 'noleap':
            # Dummy year
            year = 1979
        days = np.array([days_per_month(n, year) for n in np.arange(1,12+1)])
    return np.ma.average(data[t0:t0+12,...], axis=0, weights=days)


# Calculate the depth of the maximum value of the 3D field at each x-y point.
def depth_of_max (data, grid, gtype='t'):

    z_3d = z_to_xyz(grid.z, grid)
    data = mask_3d(data, grid, gtype=gtype)
    # Calculate the maximum value at each point and tile to make 3D
    max_val = np.amax(data, axis=0)
    max_val = xy_to_xyz(max_val, grid)    
    # Get a mask of 1s and 0s which is 1 in the locations where the value equals the maximum in that water column
    max_mask = (data==max_val).astype(float)
    # Make sure there's exactly one such point in each water column
    if np.amax(np.sum(max_mask, axis=0)) > 1:
        # Loop over any problem points
        indices = np.argwhere(np.sum(max_mask,axis=0)>1)
        for index in indices:
            [j,i] = index
        # Choose the shallowest one
        k = np.argmax(max_mask[:,j,i])
        max_mask[:,j,i] = 0
        max_mask[k,j,i] = 1
    # Select z at these points and collapse the vertical dimension
    return np.sum(z_3d*max_mask, axis=-3)


# Calculate the shallowest depth of the given isoline, below the given depth z0.
# Regions where the entire water column is below the given isoline will be set to the seafloor depth; regions where it is entirely above the isoline will trigger an error.
def depth_of_isoline (data, z, val0, z0=None):

    [nz, ny, nx] = data.shape
    if len(z.shape) == 1:
        # Make z 3D
        z = z_to_xyz(z, [nx, ny, nz])
    if z0 is None:
        z0 = 0
    # Get data and depth below each level
    z_bottom = z[-1,:]
    z_below = np.ma.concatenate((z[1:,:], z_bottom[None,:]), axis=0)
    data_bottom = np.ma.masked_where(True, data[-1,:])
    data_below = np.ma.concatenate((data[1:,:], data_bottom[None,:]), axis=0)
    # Find points where the isoline is crossed, in either direction
    mask1 = (data < val0)*(data_below >= val0)*(z <= z0)
    mask2 = (data >= val0)*(data_below < val0)*(z <= z0)
    mask = (mask1.astype(bool) + mask2.astype(bool)).astype(float)
    # Find points where the entire water column below z0 is below or above val0
    mask_below = (np.amax(np.ma.masked_where(z > z0, data), axis=0) < val0)
    mask_above = (np.amin(np.ma.masked_where(z > z0, data), axis=0) > val0)
    # Find the seafloor depth at each point
    bathy = np.amin(np.ma.masked_where(data.mask, z), axis=0)
    # And the land mask
    land_mask = np.sum(np.invert(data.mask), axis=0) == 0
    # Make sure there's at most one point in each water column
    if np.amax(np.sum(mask, axis=0)) > 1:
        # Loop over any problem points
        indices = np.argwhere(np.sum(mask, axis=0)>1)
        for index in indices:
            [j,i] = index
            # Choose the shallowest one
            k = np.argmax(mask[:,j,i])
            mask[:,j,i] = 0
            mask[k,j,i] = 1
    # Select data and depth at these points and collapse the vertical dimension
    data_cross = np.sum(data*mask, axis=0)
    data_below_cross = np.sum(data_below*mask, axis=0)
    z_cross = np.sum(z*mask, axis=0)
    z_below_cross = np.sum(z_below*mask, axis=0)
    # Now interpolate to the given isotherm
    depth_iso = (z_cross - z_below_cross)/(data_cross - data_below_cross)*(val0 - data_cross) + z_cross
    # Deal with regions where there is no such isotherm
    # Mask out the land mask
    depth_iso = np.ma.masked_where(land_mask, depth_iso)
    # Mask out regions shallower than z0
    depth_iso = np.ma.masked_where(bathy > z0, depth_iso)
    # Set to seafloor depth where the entire water column is below val0
    depth_iso[mask_below] = bathy[mask_below]
    # Mask where the entire water column is above val0
    depth_iso = np.ma.masked_where(mask_above, depth_iso)
    return depth_iso
    

    
    

    

        
