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
# The only assumption about the input array is that the third last dimension is the vertical dimension. So it can be depth x lat x lon, or time x depth x lat x lon, or even something like experiment x time x depth x lat x lon.
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


    

    
