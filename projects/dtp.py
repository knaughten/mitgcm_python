##################################################################
# DTP proposal to INSPIRE
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from MITgcmutils import rdmds

from ..grid import Grid
from ..plot_utils.latlon import cell_boundaries, shade_land, contour_iceshelf_front
from ..plot_utils.windows import finished_plot


# Make a cool figure showing eddying SST and sea ice in the Amundsen Sea
def eddy_ice_plot (output_dir='PAS_ERA5/output/201901/MITgcm/', grid_dir='PAS_grid/'):

    output_dir = real_dir(output_dir)
    grid = Grid(grid_dir)
    nz = grid.nz

    # Inner function to extract variable from the latest pickup
    def read_var_from_pickup (pickup_head, var_name):
        data, its, meta = rdmds(output_dir+pickup_head, itrs=np.Inf, returnmeta=True)
        data_unpick = []
        for var in meta['fldlist']:
            if var in ['Uvel', 'Vvel', 'Theta', 'Salt', 'GuNm1', 'GvNm1', 'PhiHyd', 'siTICES', 'pTr01']:
                data_unpick.append(data[:nz,:])
                data = data[nz:,:]
            elif var in ['EtaN', 'dEtaHdt', 'EtaH', 'siAREA', 'siHEFF', 'siHSNOW', 'siUICE', 'siVICE', 'siSigm1', 'siSigm2', 'siSigm12']:
                data_unpick.append(data[0,:])
                data = data[1:,:]
            else:
                print 'Error (read_var_from_pickup): unknown variable '+var
                sys.exit()
        i = meta['fldlist'].index(var)
        return data_unpick[i]

    sst = mask_land_ice(read_var_from_pickup('pickup', 'THETA')[0,:])
    hice = mask_land_ice(read_var_from_pickup('pickup_seaice', 'siHEFF'))

    x, y, sst_plot = cell_boundaries(sst, grid, pster=True)
    x, y, hice_plot = cell_boundaries(hice, grid, pster=True)
    fig, ax = plt.subplots(figsize=(10,6))
    shade_land(ax, grid, pster=True)
    # Try cmaps: plasma, inferno, magma
    img = ax.pcolormesh(x, y, sst_plot, cmap='plasma')
    # TODO: shade sea ice thickness on the top
    # Labels? Colourbars?
    # Cutout showing where we are in Antarctica?
    contour_iceshelf_front(ax, grid, pster=True)

    finished_plot(fig)
            
        

    

    
