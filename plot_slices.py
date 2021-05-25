#######################################################
# Zonal or meridional slices (lat-depth or lon-depth)
# or general transects between points!
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np

from grid import choose_grid, Grid
from file_io import read_netcdf, find_variable, check_single_time
from utils import mask_3d, z_to_xyz
from plot_utils.windows import set_panels, finished_plot
from plot_utils.labels import slice_axes, lon_label, lat_label, check_date_string, reduce_cbar_labels
from plot_utils.colours import set_colours, get_extend
from plot_utils.slices import slice_patches, slice_values, plot_slice_patches, get_slice_minmax, transect_patches, transect_values
from plot_utils.latlon import shade_background
from diagnostics import t_minus_tf, density, normal_vector, parallel_vector
from constants import deg_string


# Helper function to determine whether this is a slice along latitude, longitude, or a general transect, and format the string describing the slice.
def get_loc (loc0, lon0=None, lat0=None, point0=None, point1=None):

    if lon0 is not None:
        h_axis = 'lat'
        loc_string = lon_label(loc0, 3)
    elif lat0 is not None:
        h_axis = 'lon'
        loc_string = lat_label(loc0, 3)
    elif point0 is not None and point1 is not None:
        h_axis = 'trans'
        loc_string = '('+lon_label(point0[0],0)+', '+lat_label(point0[1],0)+') to ('+lon_label(point1[0],0)+', '+lat_label(point1[1],0)+')'
    return h_axis, loc_string


# Helper function to make a basic slice plot.
# Reduces duplicated code between slice_plot and slice_plot_diff.
def make_slice_plot (patches, values, loc0, hmin, hmax, zmin, zmax, vmin, vmax, lon0=None, lat0=None, point0=None, point1=None, ax=None, make_cbar=True, contours=None, data_grid=None, haxis=None, zaxis=None, ctype='basic', extend='neither', title='', titlesize=18, date_string=None, fig_name=None, dpi=None):

    # Set colour map
    cmap, vmin, vmax = set_colours(values, ctype=ctype, vmin=vmin, vmax=vmax)
    # Figure out orientation and format slice location
    h_axis, loc_string = get_loc(loc0, lon0=lon0, lat0=lat0, point0=point0, point1=point1)
    # Set up the title
    if title is not None:
        if h_axis in ['lat', 'lon']:
            title += ' at ' + loc_string
        elif h_axis == 'trans':
            title += ' from ' + loc_string
    # Plot
    existing_ax = ax is not None
    if not existing_ax:
        fig, ax = plt.subplots()
    shade_background(ax)
    # Add patches
    img = plot_slice_patches(ax, patches, values, hmin, hmax, zmin, zmax, vmin, vmax, cmap=cmap)
    if contours is not None:
        # Overlay contours
        if None in [data_grid, haxis, zaxis]:
            print 'Error (make_slice_plot): need to specify data_grid, haxis, and zaxis to do contours'
            sys.exit()
        plt.contour(haxis, zaxis, data_grid, levels=contours, colors='black', linestyles='solid')
    # Make nice axis labels
    slice_axes(ax, h_axis=h_axis)
    # Add a colourbar
    if make_cbar:
        plt.colorbar(img, extend=extend)
    # Add a title
    plt.title(title, fontsize=titlesize)
    if date_string is not None:
        # Add the date in the bottom right corner
        plt.text(.99, .01, date_string, fontsize=14, ha='right', va='bottom', transform=fig.transFigure)
    if existing_ax:
        return img
    else:
        finished_plot(fig, fig_name=fig_name, dpi=dpi)


# Basic slice plot of any variable.

# Arguments:
# data: 3D (depth x lat x lon) array of data to plot, already masked with mask_3d
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_lon_lat
# lon0, lat0: as in function slice_patches
# hmin, hmax, zmin, zmax: as in function slice_patches
# vmin, vmax: desired min and max values for colour map
# contours: list of values for which to overlay black contours
# ctype: 'basic' or 'plusminus', as in function set_colours
# title: a title to add to the plot (not including lon0 or lat0, this will be added)
# date_string: as in function latlon_plot
# fig_name: as in function finished_plot

def slice_plot (data, grid, gtype='t', lon0=None, lat0=None, point0=None, point1=None, hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, ctype='basic', title='', titlesize=18, date_string=None, fig_name=None):

    # Choose what the endpoints of the colourbar should do
    extend = get_extend(vmin=vmin, vmax=vmax)

    # Build the patches and get the bounds
    if lon0 is not None or lat0 is not None:
        # Lat-lon slices
        # Get gridded data and axes just in case
        patches, values, loc0, hmin, hmax, zmin, zmax, vmin_tmp, vmax_tmp, data_grid, haxis, zaxis = slice_patches(data, grid, gtype=gtype, lon0=lon0, lat0=lat0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_gridded=True)
    elif point0 is not None and point1 is not None:
        # Transect
        loc0 = None
        patches, values, hmin, hmax, zmin, zmax, vmin_tmp, vmax_tmp, data_grid, haxis, zaxis = transect_patches(data, grid, point0, point1, gtype=gtype, zmin=zmin, zmax=zmax, return_gridded=True)
    else:
        print 'Error (slice_plot): must specify either lon0, lat0, or point0 and point1'
        sys.exit()
    # Update any colour bounds which aren't already set
    if vmin is None:
        vmin = vmin_tmp
    if vmax is None:
        vmax = vmax_tmp
        
    # Plot
    make_slice_plot(patches, values, loc0, hmin, hmax, zmin, zmax, vmin, vmax, lon0=lon0, lat0=lat0, point0=point0, point1=point1, contours=contours, data_grid=data_grid, haxis=haxis, zaxis=zaxis, ctype=ctype, extend=extend, title=title, titlesize=titlesize, date_string=date_string, fig_name=fig_name)


# Slice plot showing difference between two simulations (2 minus 1). It is assumed the corresponding data arrays cover the same period of time.
def slice_plot_diff (data_1, data_2, grid, gtype='t', lon0=None, lat0=None, point0=None, point1=None, hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, title='', date_string=None, fig_name=None):

    # Choose what the endpoints of the colourbar should do
    extend = get_extend(vmin=vmin, vmax=vmax)

    # Build the patches for the first array
    # vmin and vmax don't matter, so just store them as temporary variables
    # Then get the values for the second array on the same patches
    if lon0 is not None or lat0 is not None:
        # Lat-lon slices
        # Get gridded data and axes just in case
        patches, values_1, loc0, hmin, hmax, zmin, zmax, tmp1, tmp2, left, right, below, above, data_grid_1, haxis, zaxis = slice_patches(data_1, grid, gtype=gtype, lon0=lon0, lat0=lat0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_bdry=True, return_gridded=True)
        values_2, tmp3, tmp4, data_grid_2 = slice_values(data_2, grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, lat0=lat0, return_gridded=True)
    elif point0 is not None and point1 is not None:
        # Transect
        loc0 = None
        patches, values_1, hmin, hmax, zmin, zmax, tmp1, tmp2, left, right, below, above, data_grid_1, haxis, zaxis = transect_patches(data_1, grid, point0, point1, gtype=gtype, zmin=zmin, zmax=zmax, return_bdry=True, return_gridded=True)
        values_2, tmp3, tmp4, data_grid_2 = transect_values(data_2, grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, gtype=gtype, return_gridded=True)
    else:
        print 'Error (slice_plot_diff): must specify either lon0, lat0, or point0 and point1'
        sys.exit()
            
    # Calculate the difference
    values_diff = values_2 - values_1
    data_grid = data_grid_2 - data_grid_1
    
    # Now figure out the colour bounds
    # Note we need to reshape the values array to be 2D again
    vmin_tmp, vmax_tmp = get_slice_minmax(np.reshape(values_diff, left.shape), left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_spatial=False)
    # Update any colour bounds which aren't already set
    if vmin is None:
        vmin = vmin_tmp
    if vmax is None:
        vmax = vmax_tmp

    # Plot
    make_slice_plot(patches, values_diff, loc0, hmin, hmax, zmin, zmax, vmin, vmax, lon0=lon0, lat0=lat0, point0=point0, point1=point1, contours=contours, data_grid=data_grid, haxis=haxis, zaxis=zaxis, ctype='plusminus', extend=extend, title=title, date_string=date_string, fig_name=fig_name)    


# NetCDF interface. Call this function with a specific variable key and information about the necessary NetCDF file, to get a nice slice plot.

# Arguments:
# var: keyword indicating which special variable to plot. The options are:
#      'temp': temperature
#      'salt': salinity
#      'tminustf': difference from in-situ freezing point
#      'rho': density (referenced to ref_depth)
#      'u': zonal velocity
#      'v': meridional velocity
#      'vnorm': normal velocity (only for transects). See function normal_vector in diagnostics.py to explain the sign convention.
#      'valong': along-transect velocity.
#      'tadv_along': along-transect advective heat flux
#      'tdif_along': along-transect diffusive heat flux
# file_path: path to NetCDF file containing the necessary variable:
#      'temp': THETA
#      'salt': SALT
#      'tminustf': THETA and SALT
#      'rho': THETA and SALT
#      'u': UVEL
#      'v': VVEL
#      'vnorm', 'valong': UVEL and VVEL
#      'tadv_along': ADVx_TH and ADVy_TH
#      'tdif_along': DFxE_TH and DFyE_TH
# If there are two variables needed (eg THETA and SALT for 'tminustf') and they are stored in separate files, you can put the other file in second_file_path (see below).

# Optional keyword arguments:
# grid: as in function read_plot_latlon
# lon0, lat0: as in function slice_patches
# point0, point1: as in function transect_patches
# time_index, t_start, t_end, time_average: as in function read_netcdf. You must either define time_index or set time_average=True, so it collapses to a single record.
# hmin, hmax, zmin, zmax: as in function slice_patches
# vmin, vmax: as in function slice_plot
# contours: list of values for which to overlay black contours
# date_string: as in function slice_plot. If time_index is defined and date_string isn't, date_string will be automatically determined based on the calendar in file_path.
# fig_name: as in function finished_plot
# second_file_path: path to NetCDF file containing a second variable which is necessary and not contained in file_path. It doesn't matter which is which.
# eosType, rhoConst, Tref, Sref, tAlpha, sBeta: as in function density. Default MDJWF, so none of the others matter.
# ref_depth: reference depth for density (positive, metres - assumed equal to dbar)

def read_plot_slice (var, file_path, grid=None, lon0=None, lat0=None, point0=None, point1=None, time_index=None, t_start=None, t_end=None, time_average=False, hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, date_string=None, fig_name=None, second_file_path=None, eosType='MDJWF', rhoConst=None, Tref=None, Sref=None, tAlpha=None, sBeta=None, ref_depth=0):

    # Build the grid if needed
    grid = choose_grid(grid, file_path)
    # Make sure we'll end up with a single record in time
    check_single_time(time_index, time_average)
    # Determine what to write about the date
    date_string = check_date_string(date_string, file_path, time_index)

    # Inner function to read a variable from the correct NetCDF file and mask appropriately
    def read_and_mask (var_name, check_second=False, gtype='t'):
        # Do we need to choose the right file?
        if check_second and second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, var_name)
        else:
            file_path_use = file_path
        # Read and mask the data
        return mask_3d(read_netcdf(file_path_use, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype=gtype)

    # Read necessary variables from NetCDF file and mask appropriately
    if var in ['temp', 'tminustf', 'rho']:
        temp = read_and_mask('THETA', check_second=True)
    if var in ['salt', 'tminustf', 'rho']:
        salt = read_and_mask('SALT', check_second=True)
    if var in ['u', 'vnorm', 'valong']:
        u = read_and_mask('UVEL', gtype='u')
    if var in ['v', 'vnorm', 'valong']:
        v = read_and_mask('VVEL', gtype='v')
    if var == 'tadv_along':
        tadv_x = read_and_mask('ADVx_TH')
        tadv_y = read_and_mask('ADVy_TH')
    if var == 'tdif_along':
        tdif_x = read_and_mask('DFxE_TH')
        tdif_y = read_and_mask('DFyE_TH')

    if var in ['vnorm', 'valong', 'tadv_along', 'tdif_along'] and None in [point0, point1]:
        print 'Error (read_plot_slice): normal or along-transect variables require point0 and point1 to be specified.'
        sys.exit()
            
    # Plot
    if var == 'temp':
        slice_plot(temp, grid, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Temperature ('+deg_string+'C)', date_string=date_string, fig_name=fig_name)
    elif var == 'salt':
        slice_plot(salt, grid, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Salinity (psu)', date_string=date_string, fig_name=fig_name)
    elif var == 'tminustf':
        slice_plot(t_minus_tf(temp, salt, grid), grid, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, ctype='plusminus', title='Difference from in-situ freezing point ('+deg_string+'C)', date_string=date_string, fig_name=fig_name)
    elif var == 'rho':
        # Calculate density
        rho = mask_3d(density(eosType, salt, temp, ref_depth, rhoConst=rhoConst, Tref=Tref, Sref=Sref, tAlpha=tAlpha, sBeta=sBeta), grid)
        slice_plot(rho, grid, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title=r'Density (kg/m$^3$)', date_string=date_string, fig_name=fig_name)
    elif var == 'u':
        slice_plot(u, grid, gtype='u', lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, ctype='plusminus', contours=contours, title='Zonal velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'v':
        slice_plot(v, grid, gtype='v', lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, ctype='plusminus', contours=contours, title='Meridional velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'vnorm':
        vnorm = normal_vector(u, v, grid, point0, point1)
        slice_plot(vnorm, grid, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, ctype='plusminus', contours=contours, title='Normal velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'valong':
        valong = parallel_vector(u, v, grid, point0, point1)
        slice_plot(valong, grid, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, ctype='plusminus', contours=contours, title='Along-transect velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'tadv_along':
        tadv_along = parallel_vector(tadv_x, tadv_y, grid, point0, point1)
        slice_plot(tadv_along, grid, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, ctype='plusminus', contours=contours, title=r'Along-transect advective heat transport (Km$^3$/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'tdif_along':
        tdif_along = parallel_vector(tdif_x, tdif_y, grid, point0, point1)
        slice_plot(tdif_along, grid, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, ctype='plusminus', contours=contours, title=r'Along-transect diffusive heat transport (Km$^3$/s)', date_string=date_string, fig_name=fig_name)
    else:
        print 'Error (read_plot_slice): variable key ' + str(var) + ' does not exist'
        sys.exit()


# Similar to read_plot_slice, but plots differences between two simulations (2 minus 1). If the two simulations cover different periods of time, set time_index_2 etc. as in function read_plot_latlon_diff.
def read_plot_slice_diff (var, file_path_1, file_path_2, grid=None, lon0=None, lat0=None, point0=None, point1=None, time_index=None, t_start=None, t_end=None, time_average=False, time_index_2=None, t_start_2=None, t_end_2=None, hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, contours=None, date_string=None, fig_name=None, eosType='MDJWF', rhoConst=None, Tref=None, Sref=None, tAlpha=None, sBeta=None, ref_depth=0, coupled=False):

    # Figure out if the two files use different time indices
    diff_time = (time_index_2 is not None) or (time_average and (t_start_2 is not None or t_end_2 is not None))

    if coupled:
        grid_1 = Grid(file_path_1)
        grid_2 = Grid(file_path_2)
    else:
        grid_1 = choose_grid(grid, file_path_1)
        grid_2 = grid_1
    check_single_time(time_index, time_average)
    date_string = check_date_string(date_string, file_path_1, time_index)

    # Inner function to read a variable from a NetCDF file and mask appropriately
    def read_and_mask (var_name, file_path, grid, check_diff_time=False, gtype='t'):
        if var_name in ['tminustf', 'rho']:
            # Need to read 2 variables
            temp = read_and_mask('THETA', file_path, check_diff_time=check_diff_time)
            salt = read_and_mask('SALT', file_path, check_diff_time=check_diff_time)
            if var_name == 'rho':
                return mask_3d(density(eosType, salt, temp, ref_depth, rhoConst=rhoConst, Tref=Tref, Sref=Sref, tAlpha=tAlpha, sBeta=sBeta), grid)
            elif var_name == 'tminustf':
                return t_minus_tf(temp, salt, grid)
        elif var_name in ['vnorm', 'valong']:
            u = read_and_mask('UVEL', file_path, check_diff_time=check_diff_time, gtype='u')
            v = read_and_mask('VVEL', file_path, check_diff_time=check_diff_time, gtype='v')
            if var_name == 'vnorm':
                return normal_vector(u, v, grid, point0, point1)
            elif var_name == 'valong':
                return parallel_vector(u, v, grid, point0, point1)
        elif var_name == 'tadv_along':
            tadv_x = read_and_mask('ADVx_TH', file_path, check_diff_time=check_diff_time)
            tadv_y = read_and_mask('ADVy_TH', file_path, check_diff_time=check_diff_time)
            return parallel_vector(tadv_x, tadv_y, grid, point0, point1)
        elif var_name == 'tdif_along':
            tdif_x = read_and_mask('DFxE_TH', file_path, check_diff_time=check_diff_time)
            tdif_y = read_and_mask('DFyE_TH', file_path, check_diff_time=check_diff_time)
            return parallel_vector(tdif_x, tdif_y, grid, point0, point1)
        else:
            if check_diff_time and diff_time:
                return mask_3d(read_netcdf(file_path, var_name, time_index=time_index_2, t_start=t_start_2, t_end=t_end_2, time_average=time_average), grid, gtype=gtype)
            else:
                return mask_3d(read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype=gtype)

    # Interface to call read_and_mask for each variable
    def read_and_mask_both (var_name, gtype='t'):
        data1 = read_and_mask(var_name, file_path_1, grid_1, gtype=gtype)
        data2 = read_and_mask(var_name, file_path_2, grid_2, check_diff_time=True, gtype=gtype)
        return data1, data2

    if var in ['vnorm', 'valong', 'tadv_along', 'tdif_along'] and None in [point0, point1]:
        print 'Error (read_plot_slice_diff): normal or along-transect variables require point0 and point1 to be specified.'
        sys.exit()

    # Read variables and make plots
    if var == 'temp':
        temp_1, temp_2 = read_and_mask_both('THETA')
        slice_plot_diff(temp_1, temp_2, grid_1, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in temperature ('+deg_string+'C)', date_string=date_string, fig_name=fig_name)
    elif var == 'salt':
        salt_1, salt_2 = read_and_mask_both('SALT')     
        slice_plot_diff(salt_1, salt_2, grid_1, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in salinity (psu)', date_string=date_string, fig_name=fig_name)
    elif var == 'tminustf':
        tmtf_1, tmtf_2 = read_and_mask_both('tminustf')
        slice_plot_diff(tmtf_1, tmtf_2, grid_1, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in difference from in-situ freezing point ('+deg_string+')', date_string=date_string, fig_name=fig_name)
    elif var == 'rho':
        rho_1, rho_2 = read_and_mask_both('rho')
        slice_plot_diff(rho_1, rho_2, grid_1, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title=r'Change in density (kg/m$^3$)', date_string=date_string, fig_name=fig_name)
    elif var == 'u':
        u_1, u_2 = read_and_mask_both('UVEL', gtype='u')
        slice_plot_diff(u_1, u_2, grid_1, gtype='u', lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in zonal velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'v':
        v_1, v_2 = read_and_mask_both('VVEL', gtype='v')
        slice_plot_diff(v_1, v_2, grid_1, gtype='v', lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in meridional velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'vnorm':
        vnorm_1, vnorm_2 = read_and_mask_both(var)
        slice_plot_diff(vnorm_1, vnorm_2, grid_1, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in normal velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'valong':
        valong_1, valong_2 = read_and_mask_both(var)
        slice_plot_diff(valong_1, valong_2, grid_1, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title='Change in along-transect velocity (m/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'tadv_along':
        tadv_along_1, tadv_along_2 = read_and_mask_both(var)
        slice_plot_diff(tadv_along_1, tadv_along_2, grid_1, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title=r'Change in along-transect advective heat transport (Km$^3$/s)', date_string=date_string, fig_name=fig_name)
    elif var == 'tdif_along':
        tdif_along_1, tdif_along_2 = read_and_mask_both(var)
        slice_plot_diff(tdif_along_1, tdif_along_2, grid_1, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, contours=contours, title=r'Change in along-transect diffusive heat transport (Km$^3$/s)', date_string=date_string, fig_name=fig_name)
    else:
        print 'Error (read_plot_slice_diff): variable key ' + str(var) + ' does not exist'
        sys.exit()


# Similar to make_slice_plot, but creates a 2x1 plot containing temperature and salinity.
def make_ts_slice_plot (patches, temp_values, salt_values, loc0, hmin, hmax, zmin, zmax, tmin, tmax, smin, smax, lon0=None, lat0=None, point0=None, point1=None, tcontours=None, scontours=None, temp_grid=None, salt_grid=None, haxis=None, zaxis=None, extend=['neither', 'neither'], diff=False, date_string=None, fig_name=None):

    # Set colour map
    if diff:
        ctype = 'plusminus'
    else:
        ctype = 'basic'
    cmap_t, tmin, tmax = set_colours(temp_values, ctype=ctype, vmin=tmin, vmax=tmax)
    cmap_s, smin, smax = set_colours(salt_values, ctype=ctype, vmin=smin, vmax=smax)

    # Figure out orientation and format slice location
    h_axis, loc_string = get_loc(loc0, lon0=lon0, lat0=lat0, point0=point0, point1=point1)

    # Set panels
    fig, gs, cax_t, cax_s = set_panels('1x2C2')
    # Wrap some things up in lists for easier iteration
    values = [temp_values, salt_values]
    vmin = [tmin, smin]
    vmax = [tmax, smax]
    cmap = [cmap_t, cmap_s]
    cax = [cax_t, cax_s]
    contours = [tcontours, scontours]
    data_grid = [temp_grid, salt_grid]
    if diff:
        title = ['Change in temperature ('+deg_string+'C)', 'Change in salinity (psu)']
    else:
        title = ['Temperature ('+deg_string+'C)', 'Salinity (psu)']
    for i in range(2):
        ax = plt.subplot(gs[0,i])
        # Plot patches
        img = plot_slice_patches(ax, patches, values[i], hmin, hmax, zmin, zmax, vmin[i], vmax[i], cmap=cmap[i])
        if contours[i] is not None:
            # Overlay contours
            if None in [data_grid[i], haxis, zaxis]:
                print 'Error (make_ts_slice_plot): need to specify temp_grid/salt_grid, haxis, and zaxis to do tcontours/scontours'
                sys.exit()
            plt.contour(haxis, zaxis, data_grid[i], levels=contours[i], colors='black', linestyles='solid')
        # Nice axes
        slice_axes(ax, h_axis=h_axis)
        if i == 1:
            # Don't need depth labels a second time
            ax.set_yticklabels([])
            ax.set_ylabel('')
        # Add a colourbar and hide every second label so they're not squished
        cbar = plt.colorbar(img, cax=cax[i], extend=extend[i], orientation='horizontal')
        reduce_cbar_labels(cbar)
        # Variable title
        plt.title(title[i], fontsize=18)
    if date_string is not None:
        # Add date to main title
        loc_string += ', ' + date_string
    # Main title
    plt.suptitle(loc_string, fontsize=20)
    finished_plot(fig, fig_name=fig_name)


# Similar to slice_plot, but creates a 2x1 plot containing temperature and salinity.        
def ts_slice_plot (temp, salt, grid, lon0=None, lat0=None, point0=None, point1=None, hmin=None, hmax=None, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, tcontours=None, scontours=None, date_string=None, fig_name=None):

    # Choose what the endpoints of the colourbars should do
    extend = [get_extend(vmin=tmin, vmax=tmax), get_extend(vmin=smin, vmax=smax)]
    # Build the temperature patches and get the bounds
    if lon0 is not None or lat0 is not None:
        # Lat-lon slices
        # Get gridded data and axes just in case
        patches, temp_values, loc0, hmin, hmax, zmin, zmax, tmin_tmp, tmax_tmp, left, right, below, above, temp_grid, haxis, zaxis = slice_patches(temp, grid, lon0=lon0, lat0=lat0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_bdry=True, return_gridded=True)
        # Get the salinity values on the same patches, and their colour bounds
        salt_values, smin_tmp, smax_tmp, salt_grid = slice_values(salt, grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, lat0=lat0, return_gridded=True)
    elif point0 is not None and point1 is not None:
        # Transect
        loc0 = None
        patches, temp_values, hmin, hmax, zmin, zmax, tmin_tmp, tmax_tmp, left, right, below, above, temp_grid, haxis, zaxis = transect_patches(temp, grid, point0, point1, zmin=zmin, zmax=zmax, return_bdry=True, return_gridded=True)
        salt_values, smin_tmp, smax_tmp, salt_grid = transect_values(salt, grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
    else:
        print 'Error (ts_slice_plot): must specify either lon0, lat0, or point0 and point1'
        sys.exit()        

    # Update any colour bounds which aren't already set
    if tmin is None:
        tmin = tmin_tmp
    if tmax is None:
        tmax = tmax_tmp
    if smin is None:
        smin = smin_tmp
    if smax is None:
        smax = smax_tmp        

    # Make the plot
    make_ts_slice_plot(patches, temp_values, salt_values, loc0, hmin, hmax, zmin, zmax, tmin, tmax, smin, smax, lon0=lon0, lat0=lat0, point0=point0, point1=point1, tcontours=tcontours, scontours=scontours, temp_grid=temp_grid, salt_grid=salt_grid, haxis=haxis, zaxis=zaxis, extend=extend, date_string=date_string, fig_name=fig_name)


# Difference plot for temperature and salinity, between two simulations (2 minus 1).
def ts_slice_plot_diff (temp_1, temp_2, salt_1, salt_2, grid, lon0=None, lat0=None, point0=None, point1=None, hmin=None, hmax=None, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, tcontours=None, scontours=None, date_string=None, fig_name=None):

    # Choose what the endpoints of the colourbars should do
    extend = [get_extend(vmin=tmin, vmax=tmax), get_extend(vmin=smin, vmax=smax)]
    
    # Build the patches and temperature values for the first simulation
    # vmin and vmax don't matter, so just store them as temporary variables
    if lon0 is not None or lat0 is not None:
        # Lat-lon slices
        # Get gridded data and axes just in case
        patches, temp_values_1, loc0, hmin, hmax, zmin, zmax, tmp1, tmp2, left, right, below, above, temp_grid_1, haxis, zaxis = slice_patches(temp_1, grid, lon0=lon0, lat0=lat0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_bdry=True, return_gridded=True)
        # Get the temperature values for the second simulation on the same patches
        temp_values_2, tmp3, tmp4, temp_grid_2 = slice_values(temp_2, grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, lat0=lat0, return_gridded=True)
        # Get the salinity values for both simulations
        salt_values_1, tmp5, tmp6, salt_grid_1 = slice_values(salt_1, grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, lat0=lat0, return_gridded=True)
        salt_values_2, tmp7, tmp8, salt_grid_2 = slice_values(salt_2, grid, left, right, below, above, hmin, hmax, zmin, zmax, lon0=lon0, lat0=lat0, return_gridded=True)
    elif point0 is not None and point1 is not None:
        # Transect
        loc0 = None
        patches, temp_values_1, hmin, hmax, zmin, zmax, tmp1, tmp2, left, right, below, above, temp_grid_1, haxis, zaxis = transect_patches(temp_1, grid, point0, point1, zmin=zmin, zmax=zmax, return_bdry=True, return_gridded=True)
        temp_values_2, tmp3, tmp4, temp_grid_2 = transect_values(temp_2, grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
        salt_values_1, tmp5, tmp6, salt_grid_1 = transect_values(salt_1, grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
        salt_values_2, tmp7, tmp8, salt_grid_2 = transect_values(salt_2, grid, point0, point1, left, right, below, above, hmin, hmax, zmin, zmax, return_gridded=True)
    else:
        print 'Error (ts_slice_plot_diff): must specify either lon0, lat0, or point0 and point1'
        sys.exit()
        
    # Calculate the differences
    temp_values_diff = temp_values_2 - temp_values_1
    salt_values_diff = salt_values_2 - salt_values_1
    temp_grid = temp_grid_2 - temp_grid_1
    salt_grid = salt_grid_2 - salt_grid_1

    # Now figure out the colour bounds
    tmin_tmp, tmax_tmp = get_slice_minmax(np.reshape(temp_values_diff, left.shape), left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_spatial=False)
    smin_tmp, smax_tmp = get_slice_minmax(np.reshape(salt_values_diff, left.shape), left, right, below, above, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, return_spatial=False)
    # Update any colour bounds which aren't already set
    if tmin is None:
        tmin = tmin_tmp
    if tmax is None:
        tmax = tmax_tmp
    if smin is None:
        smin = smin_tmp
    if smax is None:
        smax = smax_tmp

    # Plot
    make_ts_slice_plot(patches, temp_values_diff, salt_values_diff, loc0, hmin, hmax, zmin, zmax, tmin, tmax, smin, smax, lon0=lon0, lat0=lat0, point0=point0, point1=point1, tcontours=tcontours, scontours=scontours, temp_grid=temp_grid, salt_grid=salt_grid, haxis=haxis, zaxis=zaxis, extend=extend, diff=True, date_string=date_string, fig_name=fig_name)
    

# Similar to read_plot_slice, but creates a 2x1 plot containing temperature and salinity.

# Argument:
# file_path: path to NetCDF file containing THETA and SALT. If these variables are stored in two separate files, you can put the other file in second_file_path (see below).

# Optional keyword arguments:
# grid: as in function read_plot_latlon
# lon0, lat0: as in function slice_patches
# time_index, t_start, t_end, time_average: as in function read_netcdf. You must either define time_index or set time_average=True, so it collapses to a single record.
# hmin, hmax, zmin, zmax: as in function slice_patches
# tmin, tmax, smin, smax: bounds on temperature and salinity, for the colourbars
# tcontours, scontours: lists of temperature/salinity values for which to overlay contours
# date_string: as in function slice_plot. If time_index is defined and date_string isn't, date_string will be automatically determined based on the calendar in file_path.
# fig_name: as in function finished_plot
# second_file_path: path to NetCDF file containing a THETA or SALT if this is not contained in file_path. It doesn't matter which is which.

def read_plot_ts_slice (file_path, grid=None, lon0=None, lat0=None, point0=None, point1=None, time_index=None, t_start=None, t_end=None, time_average=False, hmin=None, hmax=None, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, tcontours=None, scontours=None, date_string=None, fig_name=None, second_file_path=None):

    grid = choose_grid(grid, file_path)
    check_single_time(time_index, time_average)
    date_string = check_date_string(date_string, file_path, time_index)

    # Inner function to read a variable from the correct NetCDF file and mask appropriately
    def read_and_mask (var_name):
        # Do we need to choose the right file?
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, var_name)
        else:
            file_path_use = file_path
        # Read and mask the data
        data = mask_3d(read_netcdf(file_path_use, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)
        return data

    # Read temperature and salinity
    temp = read_and_mask('THETA')
    salt = read_and_mask('SALT')

    # Plot
    ts_slice_plot(temp, salt, grid, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, tmin=tmin, tmax=tmax, smin=smin, smax=smax, tcontours=tcontours, scontours=scontours, date_string=date_string, fig_name=fig_name)


# Similar to read_plot_ts_slice, but plots the differences between two simulations (2 minus 1). It is assumed that the two files cover the same time period. Otherwise you can set time_index_2 etc. as in function read_plot_latlon_diff.
def read_plot_ts_slice_diff (file_path_1, file_path_2, grid=None, lon0=None, lat0=None, point0=None, point1=None, time_index=None, t_start=None, t_end=None, time_average=False, time_index_2=None, t_start_2=None, t_end_2=None, hmin=None, hmax=None, zmin=None, zmax=None, tmin=None, tmax=None, smin=None, smax=None, tcontours=None, scontours=None, date_string=None, fig_name=None, second_file_path_1=None, second_file_path_2=None, coupled=False):

    diff_time = (time_index_2 is not None) or (time_average and (t_start_2 is not None or t_end_2 is not None))

    if coupled:
        grid_1 = Grid(file_path_1)
        grid_2 = Grid(file_path_2)
    else:
        grid_1 = choose_grid(grid, file_path_1)
        grid_2 = grid_1
    check_single_time(time_index, time_average)
    date_string = check_date_string(date_string, file_path_1, time_index)

    # Inner function to read a variable from the correct NetCDF file and mask appropriately
    def read_and_mask (var_name, file_path, grid, second_file_path=None, check_diff_time=False):
        # Do we need to choose the right file?
        if second_file_path is not None:
            file_path_use = find_variable(file_path, second_file_path, var_name)
        else:
            file_path_use = file_path
        # Read and mask the data
        if check_diff_time and diff_time:
            return mask_3d(read_netcdf(file_path_use, var_name, time_index=time_index_2, t_start=t_start_2, t_end=t_end_2, time_average=time_average), grid)
        else:
            return mask_3d(read_netcdf(file_path_use, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid)


    # Interface to call read_and_mask for each variable
    def read_and_mask_both (var_name):
        data1 = read_and_mask(var_name, file_path_1, grid_1, second_file_path=second_file_path_1)
        data2 = read_and_mask(var_name, file_path_2, grid_2, second_file_path=second_file_path_2, check_diff_time=True)
        return data1, data2

    # Read temperature and salinity for each simulation
    temp_1, temp_2 = read_and_mask_both('THETA')
    salt_1, salt_2 = read_and_mask_both('SALT')

    # Plot
    ts_slice_plot_diff(temp_1, temp_2, salt_1, salt_2, grid_1, lon0=lon0, lat0=lat0, point0=point0, point1=point1, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, tmin=tmin, tmax=tmax, smin=smin, smax=smax, tcontours=tcontours, scontours=scontours, date_string=date_string, fig_name=fig_name)    
    

# Plot a slice of vertical resolution (dz).
# "grid" can be either a Grid object or a path.
# All other keyword arguments as in slice_plot.
def vertical_resolution (grid, lon0=None, lat0=None, hmin=None, hmax=None, zmin=None, zmax=None, vmin=None, vmax=None, fig_name=None):

    if not isinstance(grid, Grid):
        # Create a Grid object from the given path
        grid = Grid(grid)

    # Tile dz so it's 3D, and apply mask and hFac
    dz = mask_3d(z_to_xyz(grid.dz, grid), grid)*grid.hfac

    # Plot
    slice_plot(dz, grid, lon0=lon0, lat0=lat0, hmin=hmin, hmax=hmax, zmin=zmin, zmax=zmax, vmin=vmin, vmax=vmax, title='Vertical resolution (m)', fig_name=fig_name)

    
    
