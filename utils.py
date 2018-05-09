import numpy as np

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
