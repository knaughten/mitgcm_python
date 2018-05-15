#######################################################
# Miscellaneous useful tools
#######################################################

import numpy as np
import sys
import constants as const

# Given an array containing longitude, make sure it's in the range (-180, 180) as opposed to (0, 360).
def fix_lon_range (lon):

    index = lon > 180
    lon[index] = lon[index] - 360
    index = lon < -180
    lon[index] = lon[index] + 360

    return lon


# Convert freshwater flux into the ice shelf (diagnostic SHIfwFlx) (kg/m^2/s, positive means freezing) to ice shelf melt rate (m/y, positive means melting).
def convert_ismr (shifwflx):

    return -shifwflx/const.rho_fw*const.sec_per_year


# Select the top layer from the given array of data. This is useful to see conditions immediately beneath ice shelves.
# The only assumptions about the input array are that 1) it is masked with hfac (see mask_3d below), 2) the third last dimension is the vertical dimension. So it can be depth x lat x lon, or time x depth x lat x lon, or even something like experiment x time x depth x lat x lon.
def select_top (data):

    # Figure out the dimensions of the data when the vertical dimension is removed
    collapsed_shape = data.shape[:-3] + data.shape[-2:]
    # Array which will hold values at the top level, initialised to NaN
    data_top = np.zeros(collapsed_shape)
    data_top[:] = np.nan
    # Loop from surface to bottom
    for k in range(data.shape[-3]):
        curr_data = data[...,k,:,:]
        # Find points which are unmasked at this vertical level, and which
        # haven't already been assigned a top level
        index = np.nonzero(np.invert(curr_data.mask)*np.isnan(data_top))
        data_top[index] = curr_data[index]
    # Anything still NaN is land; mask it out
    data_top = np.ma.masked_where(np.isnan(data_top), data_top)

    return data_top


# Select the bottom layer from the given array of data. See select_top for more documentation.
def select_bottom (data):

    # Same as select_top, but loop from bottom to top.
    collapsed_shape = data.shape[:-3] + data.shape[-2:]
    data_bottom = np.zeros(collapsed_shape)
    data_bottom[:] = np.nan
    for k in range(data.shape[-3]-1, -1, -1):
        curr_data = data[...,k,:,:]
        index = np.nonzero(np.invert(curr_data.mask)*np.isnan(data_bottom))
        data_bottom[index] = curr_data[index]
    data_top = np.ma.masked_where(np.isnan(data_bottom), data_bottom)

    return data_bottom


# Helper function for masking functions below
def apply_mask (data, mask, time_dependent=False):

    if time_dependent:
        # Tile the mask in the time dimension
        num_time = data.shape[0]
        if len(mask.shape) == 2:
            # Starting with a 2D mask
            mask = np.tile(mask, (num_time, 1, 1))
        elif len(mask.shape) == 3:
            # Starting with a 3D mask
            mask = np.tile(mask, (num_time, 1, 1, 1))
        else:
            print 'Error (apply_mask): invalid dimensions of mask'
            sys.exit()

    if len(mask.shape) != len(data.shape):
        print 'Error (apply_mask): invalid dimensions of data'

    data = np.ma.masked_where(mask, data)
    return data


# Mask land out of a 2D field. It can be time-dependent (i.e. 3D) with the optional keyword argument.
def mask_land (data, grid, time_dependent=False):

    return apply_mask(data, grid.land_mask, time_dependent=time_dependent)


# Mask land and ice shelves out of a 2D field, just leaving the open ocean. It can be time-dependent (i.e. 3D) with the optional keyword argument.
def mask_land_zice (data, grid, time_dependent=False):

    return apply_mask(data, grid.land_mask+grid.zice_mask, time_dependent=time_dependent)


# Mask land and open ocean out of a 2D field, just leaving ice shelves. It can be time-dependent (i.e. 3D) with the optional keyword argument.
def mask_except_zice (data, grid, time_dependent=False):

    return apply_mask(data, np.invert(grid.zice_mask), time_dependent=time_dependent)


# Mask everything except FRIS out of a 2D field. It can be time-dependent (i.e. 3D) with the optional keyword argument.
def mask_except_fris (data, grid, time_dependent=False):

    return apply_mask(data, np.invert(grid.fris_mask), time_dependent=time_dependent)


# Apply the 3D hfac mask. It can be time-dependent (i.e. 4D) with the optional keyword argument.
def mask_3d (data, grid, time_dependent=False):

    return apply_mask(data, grid.hfac==0, time_dependent=time_dependent)
