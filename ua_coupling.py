#######################################################
# File transfer between MITgcm and Ua
#######################################################

import numpy as np
from scipy.io import savemat

from MITgcmutils import rdmds

from utils import real_dir, convert_ismr


# Pass data from MITgcm to Ua (melt rates).
# All we need to do is extract melt rates from MITgcm's output, convert the units, and write to a .mat file. We don't even need to interpolate them to the Ua nodes, as Ua does this at runtime.

# Arguments:
# mit_dir: path to directory containing MITgcm binary grid files and ice shelf melt rate output (SHIfwFlx)
# ismr_name: beginning of filenames of binary files containing SHIfwFlx data, i.e. <ismr_name>.xxxxx.data and <ismr_name>.xxxxx.meta where xxxxx is the timestep
# ua_out_file: desired path to .mat file for Ua to read melt rates from.

def couple_ocn2ice (mit_dir, ismr_name, ua_out_file):

    # Make sure directory ends in /
    mit_dir = real_dir(mit_dir)

    # Read MITgcm grid
    lon = rdmds(mit_dir+'XC')
    lat = rdmds(mit_dir+'YC')

    # Read the most recent file containing ice shelf melt rate
    data, its, meta = rdmds(mit_dir+ismr_name, itrs=np.Inf, returnmeta=True)
    # Figure out which index contains SHIfwflx
    i = meta['fldlist'].index('SHIfwFlx')
    # Extract the ice shelf melt rate and convert to m/y
    ismr = convert_ismr(data[i,:,:])

    # Put everything in exactly the format that Ua wants: long 1D arrays with an empty second dimension, and double precision
    lon_points = np.ravel(lon)[:,None].astype('float64')
    lat_points = np.ravel(lat)[:,None].astype('float64')
    ismr_points = np.ravel(ismr)[:,None].astype('float64')

    # Write to Matlab file for Ua, as long 1D arrays
    savemat(ua_out_file, {'meltrate':ismr_points, 'x':lon_points, 'y':lat_points})
    

    
    
    
