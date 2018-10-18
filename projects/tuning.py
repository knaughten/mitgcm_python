##################################################################
# Special plots to look at things for tuning
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..grid import Grid
from ..file_io import read_netcdf
from ..utils import real_dir, select_bottom, mask_3d, var_min_max
from ..constants import deg_string
from ..plot_latlon import latlon_plot
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.latlon import prepare_vel, overlay_vectors

# Make 3 large multi-panelled plots showing interannual variability in (1) bottom water salinity, (2) bottom water temperature, and (3) vertically averaged velocity. Each plot has one panel per year, showing the conditions averaged over that year.
# Also make one three-panelled plot for each year, showing the conditions that year.
def peryear_plots (output_dir='./annual_averages/', grid_dir='../grid/', fig_dir='./'):

    # Set up file paths etc.
    output_dir = real_dir(output_dir)
    grid_dir = real_dir(grid_dir)
    fig_dir = real_dir(fig_dir)
    file_tail = '_avg.nc'
    start_year = 1979
    end_year = 2016
    var_names = ['bwtemp', 'bwsalt', 'vel']
    ctype = ['basic', 'basic', 'vel']
    title = ['Bottom water temperature ('+deg_string+'C)', 'Bottom water salinity (psu)', 'Barotropic velocity (m/s)']

    print 'Building grid'
    grid = Grid(grid_dir)

    all_data = []
    all_vmin = []
    all_vmax = []
    all_extend = []

    # Loop over variables
    for j in range(len(var_names)):
        var = var_names[j]
        print 'Processing ' + var

        # Initialise data arrays and min/max values
        data = []
        if var == 'vel':
            u = []
            v = []
        vmin = 999
        vmax = -999

        # Loop over years
        for year in range(start_year, end_year+1):
            print '...reading ' + str(year)
            i = year-start_year

            # Read data
            file_path = output_dir + str(year) + file_tail
            if var == 'bwsalt':
                data.append(select_bottom(mask_3d(read_netcdf(file_path, 'SALT', time_index=0), grid)))
            elif var == 'bwtemp':
                data.append(select_bottom(mask_3d(read_netcdf(file_path, 'THETA', time_index=0), grid)))
            elif var == 'vel':
                u_tmp = mask_3d(read_netcdf(file_path, 'UVEL', time_index=0), grid, gtype='u')
                v_tmp = mask_3d(read_netcdf(file_path, 'VVEL', time_index=0), grid, gtype='u')
                data_tmp, u_tmp, v_tmp = prepare_vel(u_tmp, v_tmp, grid)
                data.append(data_tmp)
                u.append(u_tmp)
                v.append(v_tmp)
            # Get min and max values and update global min/max as needed
            vmin_tmp, vmax_tmp = var_min_max(data[i], grid, zoom_fris=True)
            vmin = min(vmin, vmin_tmp)
            vmax = max(vmax, vmax_tmp)

        extend = 'neither'
        if var == 'bwsalt':
            # Impose minimum of 34.3 psu if needed
            vmin = max(vmin, 34.3)
            if vmin == 34.3:
                extend = 'min'
        elif var == 'bwtemp':
            # Impose minimum of -2.5 C and maximum of -1.5 C if needed
            vmin = max(vmin, -2.5)
            vmax = min(vmax, -1.5)
            if vmin == -2.5 and vmax == -1.5:
                extend = 'both'
            elif vmin == -2.5 and vmax != -1.5:
                extend = 'min'
            elif vmin != -2.5 and vmax == -1.5:
                extend = 'max'

        # Initialise the plot
        fig, gs, cax = set_panels('5x8C1')

        # Loop over years again
        for year in range(start_year, end_year+1):
            print '...plotting ' + str(year)
            i = year-start_year

            # Draw this panel            
            ax = plt.subplot(gs[i/8, i%8])
            img = latlon_plot(data[i], grid, ax=ax, make_cbar=False, ctype=ctype[j], vmin=vmin, vmax=vmax, zoom_fris=True, title=year)
            if var == 'vel':
                # Add velocity vectors
                overlay_vectors(ax, u[i], v[i], grid, chunk=6, scale=0.8)
            if i%8 != 0:
                # Remove latitude labels
                ax.set_yticklabels([])
            if i/8 != 4:
                # Remove longitude labels
                ax.set_xticklabels([])

        # Colourbar
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend=extend)
        # Main title
        plt.suptitle(title[j], fontsize=30)
        finished_plot(fig, fig_name=fig_dir+var+'_peryear.png')

        # Save the data and bounds for individual year plots
        all_data.append(data)
        all_vmin.append(vmin)
        all_vmax.append(vmax)
        all_extend.append(extend)

    print 'Plotting conditions for each year'
    for year in range(start_year, end_year+1):
        print '...' + str(year)
        i = year-start_year
        fig, gs, cax1, cax2, cax3 = set_panels('1x3C3', figsize=(12, 5))
        cax = [cax1, cax2, cax3]

        for j in range(len(var_names)):
            var = var_names[j]
            data = all_data[j][i]
            ax = plt.subplot(gs[0,j])            
            img = latlon_plot(data, grid, ax=ax, make_cbar=False, ctype=ctype[j], vmin=all_vmin[j], vmax=all_vmax[j], zoom_fris=True, title=title[j])
            if var == 'vel':
                overlay_vectors(ax, u[i], v[i], grid, chunk=6, scale=0.8)
            if j != 0:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            cbar = plt.colorbar(img, cax=cax[j], orientation='horizontal', extend=all_extend[j])
            # Remove every second label so they're not squashed
            for label in cbar.ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

        plt.suptitle(str(year), fontsize=24)
        finished_plot(fig, fig_name=fig_dir+str(year)+'_conditions.png')
    
            
                
                
                

    

    
    

    
