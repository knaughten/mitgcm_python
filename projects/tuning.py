##################################################################
# Special plots to look at things for tuning
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import sys

from ..grid import Grid
from ..file_io import read_netcdf, netcdf_time
from ..utils import real_dir, select_bottom, mask_3d, var_min_max
from ..constants import deg_string
from ..plot_latlon import latlon_plot, plot_empty
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.latlon import prepare_vel, overlay_vectors
from ..plot_utils.labels import parse_date
from ..postprocess import build_file_list
from ..interpolation import interp_bilinear

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


        
def compare_keith_ctd (output_dir, ctd_dir, fig_dir='./'):

    # Site names
    sites = ['S1', 'S2', 'S3', 'S4', 'S5', 'F1', 'F2', 'F3', 'F4']
    num_sites = len(sites)

    output_dir = real_dir(output_dir)
    ctd_dir = real_dir(ctd_dir)
    fig_dir = real_dir(fig_dir)

    # Make one figure for each site
    for i in range(num_sites):
        
        # Construct the filename for this site
        site_file = ctd_dir + sites[i] + '.nc'
        # Read arrays
        ctd_press = read_netcdf(site_file, 'pressure')
        ctd_temp = read_netcdf(site_file, 'theta')
        ctd_salt = read_netcdf(site_file, 'salt')
        # Read global attributes
        id = nc.Dataset(site_file, 'r')
        ctd_lat = id.lat
        ctd_lon = id.lon
        ctd_year = int(id.year)
        ctd_month = int(id.month)
        id.close()

        # Get a list of all the output files from MITgcm
        output_files = build_file_list(output_dir)
        # Find the right file and time index corresponding to the given year and month
        file0 = None
        for file in output_files:
            time = netcdf_time(file)
            for t in range(time.size):
                if time[t].year == ctd_year and time[t].month == ctd_month:
                    file0 = file
                    t0 = t
                    date_string = parse_date(time[t])
                    break
            if file0 is not None:
                break
        if file0 is None:
            print "Error (compare_keith_ctd): couldn't find " + str(ctd_month) + '/' + str(ctd_year) + ' in model output'
            sys.exit()

        # Build Grid object from that file
        grid = Grid(file0)
        # Read variables
        mit_press = read_netcdf(file0, 'PHrefC')  # Note this is 1D with depth
        mit_temp_3d = read_netcdf(file0, 'THETA', time_index=t0)
        mit_salt_3d = read_netcdf(file0, 'SALT', time_index=t0)
        # Interpolate T and S to given point, as well as land mask
        mit_temp, hfac = interp_bilinear(mit_temp_3d, ctd_lon, ctd_lat, grid, return_hfac=True)
        mit_salt = interp_bilinear(mit_salt_3d, ctd_lon, ctd_lat, grid)
        # Mask with hfac
        mit_temp = np.ma.masked_where(hfac==0, mit_temp)
        mit_salt = np.ma.masked_where(hfac==0, mit_salt)
        mit_press = np.ma.masked_where(hfac==0, mit_press)

        # Find the bounds on pressure, adding 5% for white space
        press_min = 0.95*min(np.amin(ctd_press), np.amin(mit_press))
        press_max = 1.05*max(np.amax(ctd_press), np.amax(mit_press))

        # Set up the plot
        fig, gs = set_panels('1x3C1')[:2]  # Discard colourbar axis

        # Plot temperature and salinity
        # First wrap some T/S parameters up in arrays so we can iterate
        ctd_data = [ctd_temp, ctd_salt]
        mit_data = [mit_temp, mit_salt]
        var_name = ['Temperature ('+deg_string+'C)', 'Salinity (psu)']
        for j in range(2):
            ax = plt.subplot(gs[0,j])
            ax.plot(ctd_data[j], ctd_press, color='blue', label='CTD')
            ax.plot(mit_data[j], mit_press, color='red', label='Model')
            ax.grid(True)
            plt.title(var_name[j])
            if j==0:
                plt.ylabel('Pressure (dbar)')
            if j==1:
                # Legend below
                ax.legend(loc=(0.1, 0.1))

        # Plot map
        ax = plt.subplot(gs[0,2])
        plot_empty(grid, ax=ax, zoom_fris=True)
        ax.plot(ctd_lon, ctd_lat, 'o', color='red')

        # Main title with month and year
        plt.suptitle(sites[i] + ', ' + date_string, fontsize=22)        

        finished_plot(fig) #, fig_name=fig_dir+sites[i]+'.png')
            
                        

    

    
    

    
