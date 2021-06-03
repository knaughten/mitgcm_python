##################################################################
# DTP proposal to INSPIRE
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pthe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import numpy as np
import sys

from MITgcmutils import rdmds

from ..grid import Grid
from ..plot_utils.latlon import cell_boundaries
from ..plot_utils.windows import finished_plot
from ..plot_utils.labels import latlon_axes
from ..utils import real_dir, mask_land_ice
from ..constants import deg_string


# Make a cool figure showing eddying SST and sea ice in the Amundsen Sea
def eddy_ice_plot (output_dir='PAS_ERA5/output/201901/MITgcm/', grid_dir='PAS_grid/'):

    output_dir = real_dir(output_dir)
    grid = Grid(grid_dir)
    seaice_nz = 7

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
    aice = mask_land_ice(read_var_from_pickup('pickup_seaice', 'siAREA', seaice_nz), grid)
    aice = np.ma.masked_where(aice==0, aice)
    aice = np.ma.masked_where(sst>-1.5, aice)
    x, y, sst_plot = cell_boundaries(sst, grid)
    x, y, aice_plot = cell_boundaries(aice, grid)
    mask = grid.ice_mask+grid.land_mask
    x0 = grid.lon_1d[0]
    x1 = grid.lon_1d[-1]
    y0 = grid.lat_1d[0]
    y1 = grid.lat_1d[-1]

    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax = plt.subplot(gs[0,0])
    img1 = ax.pcolormesh(x, y, sst_plot, cmap='inferno')
    plt.text(0.63, 0.9, 'warm ocean', fontsize=18, transform=fig.transFigure)
    img2 = ax.pcolormesh(x, y, aice_plot, cmap='Blues_r')
    plt.text(0.1, 0.3, 'sea ice', fontsize=18, transform=fig.transFigure)
    ax.contour(grid.lon_2d, grid.lat_2d, mask, levels=[0.5], colors=('Grey'), linestyles='solid', linewidths=1)
    plt.text(0.68, 0.15, 'melting ice sheet', fontsize=18, transform=fig.transFigure)
    latlon_axes(ax, x, y)
    ax2 = inset_axes(ax, "30%", "30%", loc='upper left')
    map = Basemap()
    map.drawmapboundary(fill_color='MidnightBlue')
    map.fillcontinents(color='white', lake_color='white')
    map.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='red', latlon=True)
    txt = plt.text(0.07, 0.9, 'Amundsen Sea', fontsize=20, color='white', transform=fig.transFigure)
    txt.set_path_effects([pthe.withStroke(linewidth=2, foreground='black')])
    finished_plot(fig, fig_name='dtp_eddies.png')

        

    

    
