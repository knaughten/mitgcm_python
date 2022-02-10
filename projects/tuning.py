##################################################################
# Special plots to look at things for tuning
##################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import sys
import datetime

from MITgcmutils.mdjwf import densmdjwf

from ..grid import Grid, ERA5Grid, UKESMGrid
from ..file_io import read_netcdf, netcdf_time, read_binary, NCfile
from ..utils import real_dir, select_bottom, mask_3d, var_min_max, convert_ismr, days_per_month, add_time_dim, xy_to_xyz, z_to_xyz, mask_land_ice, fix_lon_range, split_longitude
from ..constants import deg_string, gravity, sec_per_day, rho_fw, sec_per_year
from ..plot_latlon import latlon_plot, plot_empty
from ..plot_utils.windows import set_panels, finished_plot
from ..plot_utils.latlon import prepare_vel, overlay_vectors
from ..plot_utils.labels import parse_date, reduce_cbar_labels
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

    print('Building grid')
    grid = Grid(grid_dir)

    all_data = []
    all_vmin = []
    all_vmax = []
    all_extend = []

    # Loop over variables
    for j in range(len(var_names)):
        var = var_names[j]
        print(('Processing ' + var))

        # Initialise data arrays and min/max values
        data = []
        if var == 'vel':
            u = []
            v = []
        vmin = 999
        vmax = -999

        # Loop over years
        for year in range(start_year, end_year+1):
            print(('...reading ' + str(year)))
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
            print(('...plotting ' + str(year)))
            i = year-start_year

            # Draw this panel            
            ax = plt.subplot(gs[i//8, i%8])
            img = latlon_plot(data[i], grid, ax=ax, make_cbar=False, ctype=ctype[j], vmin=vmin, vmax=vmax, zoom_fris=True, title=year)
            if var == 'vel':
                # Add velocity vectors
                overlay_vectors(ax, u[i], v[i], grid, chunk=6, scale=0.8)
            if i%8 != 0:
                # Remove latitude labels
                ax.set_yticklabels([])
            if i//8 != 4:
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

    print('Plotting conditions for each year')
    for year in range(start_year, end_year+1):
        print(('...' + str(year)))
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
            reduce_cbar_labels(cbar)

        plt.suptitle(str(year), fontsize=24)
        finished_plot(fig, fig_name=fig_dir+str(year)+'_conditions.png')


# Create a number of 3-panelled figures for each CTD site given by Keith. Compare (1) temperature profiles and (2) salinity profiles between the CTD and the model, interpolated to the correct location. Also show (3) a map of the site location.

def compare_keith_ctd (ctd_dir='/work/n02/n02/kaight/raw_input_data/ctd_data/', output_dir='./', grid_dir='../grid/', pload_file='/work/n02/n02/shared/baspog/MITgcm/WS/WSK/pload_WSK', fig_dir='./', rhoConst=1035., prec=64):

    # Site names
    sites = ['S1', 'S2', 'S3', 'S4', 'S5', 'F1', 'F2', 'F3', 'F4']
    num_sites = len(sites)

    ctd_dir = real_dir(ctd_dir)
    output_dir = real_dir(output_dir)
    grid_dir = real_dir(grid_dir)
    fig_dir = real_dir(fig_dir)

    # Build Grid object
    grid = Grid(grid_dir)
    # Read pressure load anomaly
    pload_anom_2d = read_binary(pload_file, [grid.nx, grid.ny], 'xy', prec=prec)
    
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
            print(("Error (compare_keith_ctd): couldn't find " + str(ctd_month) + '/' + str(ctd_year) + ' in model output'))
            sys.exit()

        # Read variables
        mit_temp_3d = read_netcdf(file0, 'THETA', time_index=t0)
        mit_salt_3d = read_netcdf(file0, 'SALT', time_index=t0)

        # Interpolate to given point
        # Temperature and land mask
        mit_temp, hfac = interp_bilinear(mit_temp_3d, ctd_lon, ctd_lat, grid, return_hfac=True)
        # Salinity
        mit_salt = interp_bilinear(mit_salt_3d, ctd_lon, ctd_lat, grid)
        # Pressure loading anomaly
        pload_anom = interp_bilinear(pload_anom_2d, ctd_lon, ctd_lat, grid)
        # Ice shelf draft
        draft = interp_bilinear(grid.draft, ctd_lon, ctd_lat, grid)

        # Calculate density, assuming pressure in dbar equals depth in m (this term is small)
        rho = densmdjwf(mit_salt, mit_temp, abs(grid.dz))
        # Now calculate pressure in dbar
        mit_press = (pload_anom + rhoConst*gravity*abs(draft) + np.cumsum(rho*gravity*grid.dz*hfac))*1e-4

        # Mask all arrays with hfac
        mit_temp = np.ma.masked_where(hfac==0, mit_temp)
        mit_salt = np.ma.masked_where(hfac==0, mit_salt)
        mit_press = np.ma.masked_where(hfac==0, mit_press)

        # Find the bounds on pressure, adding 5% for white space in the plot
        press_min = 0.95*min(np.amin(ctd_press), np.amin(mit_press))
        press_max = 1.05*max(np.amax(ctd_press), np.amax(mit_press))

        # Set up the plot
        fig, gs_1, gs_2 = set_panels('CTD')

        # Plot temperature and salinity
        # First wrap some T/S parameters up in arrays so we can iterate
        ctd_data = [ctd_temp, ctd_salt]
        mit_data = [mit_temp, mit_salt]
        var_name = ['Temperature ('+deg_string+'C)', 'Salinity (psu)']
        for j in range(2):
            ax = plt.subplot(gs_1[0,j])
            ax.plot(ctd_data[j], ctd_press, color='blue', label='CTD')
            ax.plot(mit_data[j], mit_press, color='red', label='Model')
            ax.set_ylim([press_max, press_min])
            ax.grid(True)
            plt.title(var_name[j], fontsize=16)
            if j==0:
                plt.ylabel('Pressure (dbar)', fontsize=14)
            if j==1:
                # No need for pressure labels
                ax.set_yticklabels([])
                # Legend in bottom right, below the map
                ax.legend(loc=(1.2, 0), ncol=2)

        # Plot map
        ax = plt.subplot(gs_2[0,0])
        plot_empty(grid, ax=ax, zoom_fris=True)
        ax.plot(ctd_lon, ctd_lat, '*', color='red', markersize=15)

        # Main title with month and year
        plt.suptitle(sites[i] + ', ' + date_string, fontsize=22)        

        finished_plot(fig, fig_name=fig_dir+sites[i]+'.png')


# Extract annually averaged melt rates for Sebastian.
def extract_ismr (out_file, start_year=1994, end_year=2016, output_dir='./annual_averages/', grid_dir='../grid/'):

    output_dir = real_dir(output_dir)
    grid_dir = real_dir(grid_dir)
    file_tail = '_avg.nc'

    print('Building grid')
    grid = Grid(grid_dir)

    # Set up array to hold melt rates
    ismr = np.empty([end_year-start_year+1, grid.ny, grid.nx])
    for year in range(start_year, end_year+1):
        print(('Processing ' + str(year)))
        file_path = output_dir + str(year) + file_tail
        ismr_tmp = convert_ismr(read_netcdf(file_path, 'SHIfwFlx', time_index=0))
        ismr[year-start_year,:] = ismr_tmp
    
    print(('Writing ' + out_file))
    ncfile = NCfile(out_file, grid, 'xyt')
    ncfile.add_time(list(range(start_year,end_year+1)), units='year')
    ncfile.add_variable('melt_rate', ismr, 'xyt', units='m/y')
    ncfile.close()


# Calculate components of volume change in the domain, and save to NetCDF timeseries file.
def calc_volume_components (file_path, nc_out):

    grid = Grid(file_path)

    # Calculate the dt of each output step
    # First get the month and year of each 
    time, time_units = netcdf_time(file_path, return_units=True)
    num_time = time.size
    dt = np.empty(num_time)
    for t in range(num_time):
        dt[t] = days_per_month(time[t].month, time[t].year)*sec_per_day
        
    # Expand area of each cell to have a time dimension
    dA = add_time_dim(grid.dA, num_time)

    # Calculate volume changes in m^3 from different components:

    # As seen by free surface
    eta = read_netcdf(file_path, 'ETAN')  # m
    vol_eta = np.sum(eta*dA, axis=(1,2))  # m^3

    # Ice shelf melting (integrated over time)
    shiflx = -read_netcdf(file_path, 'SHIfwFlx')  # kg/m^2/s
    dV_shi = np.sum(shiflx/rho_fw*dA, axis=(1,2)) # m^3/s
    vol_shi = np.cumsum(dV_shi*dt)  # m^3

    # Precipitation minus evaporation (integrated over time)
    pminuse = read_netcdf(file_path, 'EXFpreci') - read_netcdf(file_path, 'EXFevap')  # m/s
    dV_pme = np.sum(pminuse*dA, axis=(1,2))  # m^3/s
    vol_pme = np.cumsum(dV_pme*dt)  # m^3

    # Iceberg melting (integrated over time)
    icebergs = read_netcdf(file_path, 'EXFroff')  # m/s
    dV_ib = np.sum(icebergs*dA, axis=(1,2))  # m^3/s
    vol_ib = np.cumsum(dV_ib*dt)  # m^3

    # Sea ice melting minus freezing (integrated over time)
    siflx = -(read_netcdf(file_path, 'SIdHbATC') + read_netcdf(file_path, 'SIdHbATO') + read_netcdf(file_path, 'SIdHbOCN') + read_netcdf(file_path, 'SIdHbFLO'))  # m/s
    dV_si = np.sum(siflx*dA, axis=(1,2))  # m^3/s
    vol_si = np.cumsum(dV_si*dt)  # m^3

    # Net inflow on eastern boundary (integrated over time)
    vnorm_e = -read_netcdf(file_path, 'UVEL')[:,:,-1]  # m/s
    # Calculate area of boundary face: dy*dz*hfac
    area_e = add_time_dim((xy_to_xyz(grid.dy_w, grid)*z_to_xyz(grid.dz, grid)*grid.hfac)[:,:,-1], num_time)  # m^2
    dV_ebdry = np.sum(vnorm_e*area_e, axis=(1,2))  # m^3/s
    vol_ebdry = np.cumsum(dV_ebdry*dt)  # m^3

    # Net inflow on northern boundary (integrated over time)
    vnorm_n = -read_netcdf(file_path, 'VVEL')[:,-1,:]  # m/s
    area_n = add_time_dim((xy_to_xyz(grid.dx_s, grid)*z_to_xyz(grid.dz, grid)*grid.hfac)[:,-1,:], num_time)  # m^2
    dV_nbdry = np.sum(vnorm_n*area_n, axis=(1,2))  # m^3/s
    vol_nbdry = np.cumsum(dV_nbdry*dt)  # m^3

    # Also calculate total volume components, to compare to eta changes
    vol_total = vol_shi + vol_pme + vol_ib + vol_si + vol_ebdry + vol_nbdry

    # Write to file
    ncfile = NCfile(nc_out, grid, 't')
    ncfile.add_time(time, units=time_units)
    vars = [vol_eta, vol_shi, vol_pme, vol_ib, vol_si, vol_ebdry, vol_nbdry, vol_total]
    var_names = ['vol_eta', 'vol_shi', 'vol_pme', 'vol_ib', 'vol_si', 'vol_ebdry', 'vol_nbdry', 'vol_total']
    titles = ['Volume change seen by free surface', 'Volume change from ice shelf basal melting', 'Volume change from precipitation minus evaporation', 'Volume change from iceberg melting', 'Volume change from sea ice melting minus freezing', 'Volume change from eastern boundary flow', 'Volume change from northern boundary flow', 'Total of volume change components']
    for i in range(len(vars)):
        ncfile.add_variable(var_names[i], vars[i], 't', long_name=titles[i], units='m^3')
    ncfile.close()


# Calculate transports into domain across the east and north boundaries, as seen by the OBCS file, the model output on the boundaries, and the model output on the inner side of the sponge.
def bdry_transports (obcs_e_u, obcs_n_v, file_path, nc_out, sponge=8):

    grid = Grid(file_path)

    # Calculate areas of faces on boundaries
    area_e = (xy_to_xyz(grid.dy_w, grid)*z_to_xyz(grid.dz, grid)*grid.hfac)[:,:,-1]
    area_n = (xy_to_xyz(grid.dx_s, grid)*z_to_xyz(grid.dz, grid)*grid.hfac)[:,-1,:]

    ncfile = NCfile(nc_out, grid, 't')

    def calc_save_transport (vnorm, direction, title):
        if direction == 'E':
            area = area_e
        elif direction == 'N':
            area = area_n
        else:
            print(('Error (calc_save_transport): invalid direction ' + direction))
            sys.exit()
        area = add_time_dim(area, vnorm.shape[0])
        trans = np.sum(vnorm*area, axis=(1,2))*1e-6
        ncfile.add_variable(title, trans, 't', units='Sv')

    vnorms = [-1*read_binary(obcs_e_u, [grid.nx, grid.ny, grid.nz], 'yzt'), -1*read_binary(obcs_n_v, [grid.nx, grid.ny, grid.nz], 'xzt'), -1*read_netcdf(file_path, 'UVEL')[:,:,:,-1], -1*read_netcdf(file_path, 'VVEL')[:,:,-1,:], -1*read_netcdf(file_path, 'UVEL')[:,:,:,-1*sponge], -1*read_netcdf(file_path, 'VVEL')[:,:,-1*sponge,:]]
    directions = ['E', 'N', 'E', 'N', 'E', 'N']
    titles = ['obcs_e', 'obcs_n', 'outer_e', 'outer_n', 'inner_e', 'inner_n']
    
    ntime = 0
    for data in vnorms:
        ntime = max(ntime, data.shape[0])
    ncfile.add_time(np.arange(ntime)+1, units='months')
    
    for i in range(len(titles)):
        calc_save_transport(vnorms[i], directions[i], titles[i])

    ncfile.close()


# Compare evaporation values from ERA-Interim to two MITgcm simulations: one with the original humidity formulation (from Nico, which we now know is correct) and one with the new formulation (from Kaitlin's screw-up).
def evap_compare (file_path_1, file_path_2, file_path_era, month=None, vmin=None, vmax=None, fig_name=None):

    # Read grids
    # Convert ERA-Interim to longitude range -180 to 180, and split and rearrange so strictly ascending
    lon_era = fix_lon_range(read_netcdf(file_path_era, 'longitude'))
    i_split = np.nonzero(lon_era < 0)[0][0]
    lon_era = split_longitude(lon_era, i_split)
    lat_era = read_netcdf(file_path_era, 'latitude')
    grid = Grid(file_path_1)

    # Read ERA-Interim evaporation, split and rearrange, and change sign convention
    evap_era = split_longitude(-1*read_netcdf(file_path_era, 'e'), i_split)
    # Read MITgcm evaporation, mask, and convert to m/12h
    def mit_evap (file_path):
        #evap = read_netcdf(file_path, 'EXFevap')
        #sublim = read_netcdf(file_path, 'SIacSubl')/rho_fw
        #evap = (evap + sublim)*12*60*60
        fwflx = read_netcdf(file_path, 'SIatmFW')/rho_fw
        precip = read_netcdf(file_path, 'EXFpreci')
        roff = read_netcdf(file_path, 'EXFroff')
        evap = -(fwflx - precip - roff)*12*60*60
        return mask_land_ice(evap, grid, time_dependent=True)
    evap_1 = mit_evap(file_path_1)
    evap_2 = mit_evap(file_path_2)

    # Inner function to annually average or select month
    def choose_time (evap):
        # Select the first 12 months
        evap = evap[:12]
        if month is None:
            # Annually average
            return np.mean(evap, axis=0)
        else:
            return evap[month-1,:]

    evap_era = choose_time(evap_era)
    evap_1 = choose_time(evap_1)
    evap_2 = choose_time(evap_2)

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1

    xmin = grid.lon_1d[0]
    xmax = grid.lon_1d[-1]
    ymin = grid.lat_1d[0]
    ymax = grid.lat_1d[-1]

    data = [evap_era, evap_1, evap_2]
    lon = [lon_era, grid.lon_1d, grid.lon_1d]
    lat = [lat_era, grid.lat_1d, grid.lat_1d]
    title = ['ERA-Interim', 'MITgcm (old humidity)', 'MITgcm (new humidity)']
    if month is None:
        suffix = ' average'
    else:
        suffix = '-'+str(month)

    fig, gs, cax = set_panels('1x3C1')
    for i in range(3):
        ax = plt.subplot(gs[0,i])
        img = ax.pcolormesh(lon[i], lat[i], data[i]*1e3, vmin=vmin, vmax=vmax)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.title(title[i], fontsize=18)
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
    plt.suptitle('Evaporation (mm/12h), 1979'+suffix, fontsize=24)
    finished_plot(fig, fig_name)


# Calculate timeseries of atmospheric forcing variables (ERA5 vs UKESM) averaged/integrated/maximised over the southern Weddell Sea continental shelf. Plot, and also save to two NetCDF files.
def forcing_sws_timeseries (era5_dir, ukesm_dir, era5_out_file, ukesm_out_file):

    # Lat-lon boundaries on SWS continental shelf region
    # Lon is in range (0, 360)
    [xmin, xmax, ymin, ymax] = [-65+360, -25+360, -78.5, -71]
    # Beginning of file names for atmospheric forcing files
    file_heads = ['ERA5_', 'piControl_']
    # Final year of output for each (start year is already saved in grid objects)
    end_years = [2018, 2710]
    # Output variable names
    var_names_out = ['max_wind', 'avg_atemp', 'total_precip', 'avg_shum', 'avg_swrad', 'avg_lwrad']
    # and units
    units = ['m/s', 'degC', 'm^3/s', '1', 'W/m^2', 'W/m^2']
    num_vars = len(var_names_out)
    # Corresponding input variable names for each data source (stamped in file names)
    var_names_in = [['uwind', 'atemp', 'precip', 'aqh', 'swdown', 'lwdown'], ['uas', 'tas', 'pr', 'huss', 'rsds', 'rlds']]

    directories = [real_dir(era5_dir), real_dir(ukesm_dir)]
    # Build forcing grids
    grids = [ERA5Grid(), UKESMGrid()]
    start_years = [grid.start_year for grid in grids]        
    # Build masks for the SWS continental shelf
    masks = [(grid.lon >= xmin)*(grid.lon <= xmax)*(grid.lat >= ymin)*(grid.lat <= ymax) for grid in grids]

    # Build time axes (years since start date)
    # ERA5 is on the "real" calendar, so use datetime objects
    # Build this up with a loop
    era5_date = datetime.datetime(start_years[0], 1, 1)
    era5_dt = datetime.timedelta(seconds=grids[0].period)
    era5_time = [0]
    while (era5_date + era5_dt).year <= end_years[0]:
        era5_time.append(era5_time[-1]+grids[0].period/sec_per_year)
        era5_date += era5_dt
    era5_time = np.array(era5_time)
    # UKESM has 30-day months, so it's easier
    ukesm_time = np.arange(0, end_years[1]-start_years[1]+1, grids[1].period/sec_per_year)
    times = [era5_time, ukesm_time]

    # Set up NetCDF files
    ncfiles = [NCfile(era5_out_file, None, 't'), NCfile(ukesm_out_file, None, 't')]
    for i in range(2):
        ncfiles[i].add_time(times[i], units='years')

    # Inner function to read and process forcing file for a single year, forcing product, and variable
    def process_file (directory, file_head, var_name_in, year, var_name_out, grid, mask):
        # Construct file name
        file_path = directory + file_head + var_name_in + '_' + str(year)
        # Read all the data
        data = read_binary(file_path, [grid.nx, grid.ny], 'xyt')
        if var_name_in in ['uwind', 'uas']:
            is_ukesm = var_name_in=='uas'
            if is_ukesm:
                # Interpolate to the t-grid (first need to wrap the periodic boundary)
                data_u = np.concatenate((data, np.expand_dims(data[:,:,0],2)), axis=2)
                data_u = 0.5*(data_u[:,:,:-1] + data_u[:,:,1:])
            else:
                data_u = data
            # Need to read the v-component too
            file_path_v = directory + file_head + var_name_in.replace('u', 'v') + '_' + str(year)
            if is_ukesm:
                data_v = read_binary(file_path_v, [grid.nx, grid.ny_v], 'xyt')
                data_v = 0.5*(data_v[:,:-1,:] + data_v[:,1:,:])
            else:
                data_v = read_binary(file_path_v, [grid.nx, grid.ny], 'xyt')
            # Calculate speed
            data = np.sqrt(data_u**2 + data_v**2)
        # Loop over time indices to save memory
        num_time = data.shape[0]
        timeseries = np.empty(num_time)
        for t in range(num_time):
            # Do the correct transformation
            if var_name_out.startswith('max'):
                timeseries[t] = np.max(data[t]*mask)
            elif var_name_out.startswith('avg'):
                timeseries[t] = np.sum(data[t]*grid.dA*mask)/np.sum(grid.dA*mask)
            elif var_name_out.startswith('total'):
                timeseries[t] = np.sum(data[t]*grid.dA*mask)
        return timeseries

    # Now loop over all the variables and years
    for n in range(num_vars):
        print(('Processing variable ' + var_names_out[n]))
        for i in range(2):
            data = None
            for year in range(start_years[i], end_years[i]+1):
                data_tmp = process_file(directories[i], file_heads[i], var_names_in[i][n], year, var_names_out[n], grids[i], masks[i])
                if data is None:
                    data = data_tmp
                else:
                    data = np.concatenate((data, data_tmp))
            ncfiles[i].add_variable(var_names_out[n], data, 't', units=units[n])

    for i in range(2):
        ncfiles[i].close()















    

    
    

    
