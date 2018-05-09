import numpy as np

# Given an array containing longitude, make sure it's in the range (-180, 180) as opposed to (0, 360).

# Arguments:
# lon: numpy array of any dimension, containing longitude values in the range (-360, 360)

# Output: numpy array containing longitude values in the range (-180, 180)

def fix_lon_range (lon):

    index = lon > 180
    lon[index] = lon[index] + 360
    index = lon < -180
    lon[index] = lon[index] - 360

    return lon
