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
from ..utils import real_dir, mask_land_ice


# Make a cool figure showing eddying SST and sea ice in the Amundsen Sea
def eddy_ice_plot (output_dir='PAS_ERA5/output/201901/MITgcm/', grid_dir='PAS_grid/'):

    output_dir = real_dir(output_dir)
    grid = Grid(grid_dir)
    seaice_nz = 7
    pster = False

    # Inner function to extract variable from the latest pickup
    def read_var_from_pickup (pickup_head, var0, nz):
        data, its, meta = rdmds(output_dir+pickup_head, itrs=np.Inf, returnmeta=True)
        data_unpick = []
        for var in meta['fldlist']:
            if var in ['Uvel', 'Vvel', 'Theta', 'Salt', 'GuNm1', 'GvNm1', 'PhiHyd', 'siTICES', 'pTr01', 'AddMass']:
                data_unpick.append(data[:nz,:])
                data = data[nz:,:]
            elif var in ['EtaN', 'dEtaHdt', 'EtaH', 'siAREA', 'siHEFF', 'siHSNOW', 'siUICE', 'siVICE', 'siSigm1', 'siSigm2', 'siSigm12']:
                data_unpick.append(data[0,:])
                data = data[1:,:]
            else:
                print 'Error (read_var_name_from_pickup): unknown var_nameiable '+var
                sys.exit()
        i = meta['fldlist'].index(var0)
        return data_unpick[i]

    sst = mask_land_ice(read_var_from_pickup('pickup', 'Theta', grid.nz)[0,:], grid)
    hice = mask_land_ice(read_var_from_pickup('pickup_seaice', 'siHEFF', seaice_nz), grid)
    x, y, sst_plot = cell_boundaries(sst, grid, pster=pster)
    x, y, hice_plot = cell_boundaries(hice, grid, pster=pster)

    fig, ax = plt.subplots(figsize=(8,6))
    shade_land(ax, grid)
    img1 = ax.pcolormesh(x, y, sst_plot, cmap='inferno')
    img2 = ax.pcolormesh(x, y, hice_plot, cmap='Blues_r')
    # Labels? Colourbars?
    # Cutout showing where we are in Antarctica?
    contour_iceshelf_front(ax, grid)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    finished_plot(fig)
            
        

    

    
