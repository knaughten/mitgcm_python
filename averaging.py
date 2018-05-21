from numpy import *

def vertical_average (data, grid, gtype='t', time_dependent=False):

    # Choose the correct integrand of depth
    if gtype == 'w':
        dz = grid.dz_t
    else:
        dz = grid.dz
    # Make it 3D
    dz = np.tile(np.expand_dims(np.expand_dims(dz,1),2), (1, grid.ny, grid.nx))
    # Get the correct hFac
    hfac = grid.get_hfac(gtype=gtype)
    if time_dependent:
        # There's also a time dimension
        num_time = data.shape[0]
        dz = np.tile(dz, (num_time, 1, 1, 1))
        hfac = np.tile(hfac, (num_time, 1, 1, 1))
    # Vertically average    
    return sum(data*dz*hfac, axis=-3)/sum(dz*hfac, axis=-3)

    
        
