#######################################################
# Generate a new model domain.
#######################################################

import numpy as np

from constants import deg2rad
from io import write_binary
from utils import factors

def latlon_points (xmin, xmax, ymin, ymax, res, dlat_file, prec=64):

    # Number of iterations for latitude convergence
    num_lat_iter = 10    

    # Build longitude values
    lon = np.arange(xmin, xmax+res, res)
    # Update xmax if the range doesn't evenly divide by res
    if xmax != lon[-1]:
        xmax = lon[-1]
        print 'Eastern boundary moved to ' + str(xmax)
    # Put xmin in the range (0, 360) for namelist
    if xmin < 0:
        xmin += 360

    # First guess for latitude: resolution scaled by latitude of southern edge
    lat = [ymin]
    while lat[-1] < ymax:
        lat.append(lat[-1] + res*np.cos(lat[-1]*deg2rad))
    lat = np.array(lat)
    # Now iterate to converge on resolution scaled by latitude of centres
    for iter in range(num_lat_iter):
        lat_old = np.copy(lat)
        # Latitude at centres    
        lat_c = 0.5*(lat[:-1] + lat[1:])
        j = 0
        lat = [ymin]
        while lat[-1] < ymax and j < lat_c.size:
            lat.append(lat[-1] + res*np.cos(lat_c[j]*deg2rad))
            j += 1
        lat = np.array(lat)
    # Update ymax
    ymax = lat[-1]
    print 'Northern boundary moved to ' + str(ymax)

    # Write latitude resolutions to file
    dlat = lat[1:] - lat[:-1]
    write_binary(dlat, dlat_file, prec=prec)

    # Remind the user what to do in their namelist
    print '\nChanges to make to input/data:'
    print 'xgOrigin=' + str(xmin)
    print 'ygOrigin=' + str(ymin)
    print 'dxSpacing=' + str(res)
    print "delYfile='" + dlat_file + "' (and copy this file into input/)"

    # Find dimensions of tracer grid
    Nx = lon.size-1
    Ny = lat.size-1
    # Find all the factors
    factors_x = factors(Nx)
    factors_y = factors(Ny)
    print '\nNx = ' + str(Nx) + ' which has the factors ' + str(factors_x)
    print 'Ny = ' + str(Ny) + ' which has the factors ' + str(factors_y)
    print 'If you are happy with this, choose your tile size based on the factors and update code/SIZE.h.'
    print 'Otherwise, tweak the boundaries and try again.'
