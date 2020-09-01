# Figures for IRF 2020 application

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl

from ..file_io import read_netcdf, netcdf_time
from ..timeseries import calc_annual_averages
from ..utils import moving_average, polar_stereo, select_bottom, bdry_from_hfac, z_to_xyz
from ..constants import deg_string
from ..plot_utils.windows import finished_plot, set_panels
from ..plot_utils.colours import set_colours
from ..plot_utils.latlon import shade_background
from ..interpolation import interp_nonreg_xy

def extract_geomip_westerlies ():

    directories = ['member1/', 'member4/', 'member8/']
    in_files = ['ssp245.nc', 'ssp585.nc', 'g6sulfur.nc', 'g6solar.nc']
    labels = ['low emissions', 'high emissions', 'high emissions + aerosol SRM', 'high emissions + space-based SRM']
    colours = ['black', 'red', 'blue', 'green']
    num_ens = len(directories)
    num_sim = len(in_files)

    times = []
    jet_lat_min = []
    jet_lat_max = []
    jet_lat_mean = []
    for fname in in_files:
        jet_lat_range = None
        for ens in range(num_ens):
            file_path = directories[ens] + fname
            time = netcdf_time(file_path)
            lat = read_netcdf(file_path, 'lat')        
            uas = np.mean(read_netcdf(file_path, 'uas'), axis=2)
            jet_jmax = np.argmax(uas, axis=1)
            jet_lat = np.empty(jet_jmax.shape)  
            for t in range(time.size):
                jet_lat[t] = lat[jet_jmax[t]]
            time, jet_lat = calc_annual_averages(time, jet_lat)
            jet_lat, time = moving_average(jet_lat, 5, time=time)
            if jet_lat_range is None:
                jet_lat_range = np.empty([num_ens, time.size])
            jet_lat_range[ens,:] = jet_lat
        times.append(np.array([t.year for t in time]))
        jet_lat_min.append(np.amin(jet_lat_range, axis=0))
        jet_lat_max.append(np.amax(jet_lat_range, axis=0))
        jet_lat_mean.append(np.mean(jet_lat_range, axis=0))

    fig, ax = plt.subplots(figsize=(8,5))
    for n in range(num_sim):
        ax.fill_between(times[n], jet_lat_min[n], jet_lat_max[n], color=colours[n], alpha=0.15)
        ax.plot(times[n], jet_lat_mean[n], color=colours[n], label=labels[n], linewidth=1.5)
    plt.title('Impact of solar radiation management\non Southern Hemisphere westerly winds', fontsize=18)
    plt.xlabel('year', fontsize=14)
    plt.ylabel('jet latitude', fontsize=14)
    yticks = np.arange(-53, -50, 1)
    yticklabels = [np.str(-y)+deg_string+'S' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim([times[-1][0], times[-1][-1]])
    ax.set_xticks(np.arange(2030, 2100, 20))
    #ax.legend()
    finished_plot(fig, fig_name='geo_winds.png', dpi=300)


# Plot Paul's SO bottom temps versus observations
def bottom_temp_vs_obs (model_file='stateBtemp_avg.nc', model_grid='grid.glob.nc', obs_file='schmidtko_data.txt', woa_file='woa18_decav_t00_04.nc'):

    # Set spatial bounds
    corner_lon = np.array([-45, 135])
    corner_lat = np.array([-59, -59])
    corner_x, corner_y = polar_stereo(corner_lon, corner_lat)
    [xmin, xmax, ymin, ymax] = [corner_x[0], corner_x[1], corner_y[1], corner_y[0]]
    # Set colour bounds
    vmin = -2.5
    vmax = 1.5
    lev = np.linspace(vmin, vmax, num=30)

    # Read model data
    model_lon = read_netcdf(model_file, 'LONGITUDE')
    model_lat = read_netcdf(model_file, 'LATITUDE')
    model_temp = read_netcdf(model_file, 'BTEMP')
    # Mask land (zeros)
    model_temp = np.ma.masked_where(model_temp==0, model_temp)
    # Read other grid variables
    hfac = read_netcdf(model_grid, 'HFacC')
    z_edges = read_netcdf(model_grid, 'RF')
    bathy = bdry_from_hfac('bathy', hfac, z_edges)
    draft = bdry_from_hfac('draft', hfac, z_edges)
    ocean_mask = bathy!=0
    ocean_mask = np.ma.masked_where(np.invert(ocean_mask), ocean_mask)
    draft = np.ma.masked_where(bathy==0, draft)
    # Convert coordinates to polar stereographic
    model_x, model_y = polar_stereo(model_lon, model_lat)

    # Read Schmidtko data on continental shelf
    obs = np.loadtxt(obs_file, dtype=np.str)
    obs_lon_vals = obs[:,0].astype(float)
    obs_lat_vals = obs[:,1].astype(float)
    obs_depth_vals = obs[:,2].astype(float)
    obs_temp_vals = obs[:,3].astype(float)
    num_obs = obs_temp_vals.size
    # Grid it
    obs_lon = np.unique(obs_lon_vals)
    obs_lat = np.unique(obs_lat_vals)
    obs_temp = np.zeros([obs_lat.size, obs_lon.size]) - 999
    for n in range(num_obs):
        j = np.argwhere(obs_lat==obs_lat_vals[n])[0][0]
        i = np.argwhere(obs_lon==obs_lon_vals[n])[0][0]
        obs_temp[j,i] = obs_temp_vals[n]
    obs_temp = np.ma.masked_where(obs_temp==-999, obs_temp)
    obs_lon, obs_lat = np.meshgrid(obs_lon, obs_lat)
    obs_x, obs_y = polar_stereo(obs_lon, obs_lat)

    # Read WOA data for deep ocean
    woa_lon = read_netcdf(woa_file, 'lon')
    woa_lat = read_netcdf(woa_file, 'lat')
    woa_depth = read_netcdf(woa_file, 'depth')
    woa_temp_3d = np.squeeze(read_netcdf(woa_file, 't_an'))
    # Extract bottom values
    woa_temp = select_bottom(woa_temp_3d)
    # Extract bathymetry
    woa_depth = z_to_xyz(woa_depth, [woa_lon.size, woa_lat.size])
    woa_depth = np.ma.masked_where(np.ma.getmask(woa_temp_3d), woa_depth)
    woa_bathy = -1*select_bottom(woa_depth)
    # Now mask shallow regions in the Amundsen and Bellingshausen Sea where weird things happen
    woa_temp = np.ma.masked_where((woa_lon>=-130)*(woa_lon<=-60)*(woa_bathy>=-500), woa_temp)
    woa_lon, woa_lat = np.meshgrid(woa_lon, woa_lat)
    woa_x, woa_y = polar_stereo(woa_lon, woa_lat)

    # Plot
    cmap = set_colours(model_temp, ctype='plusminus', vmin=vmin, vmax=vmax)[0]
    titles = ['a) Existing model', 'b) Observations']
    fig, gs = set_panels('1x2C0', figsize=(8,4))
    for n in range(2):
        ax = plt.subplot(gs[0,n])
        # Shade land in grey
        shade_background(ax)
        ax.contourf(model_x, model_y, ocean_mask, cmap=cl.ListedColormap([(1,1,1)]))
        if n == 0:
            img = ax.contourf(model_x, model_y, model_temp, lev, cmap=cmap, extend='both')
        elif n == 1:
            img = ax.contourf(woa_x, woa_y, woa_temp, lev, cmap=cmap, extend='both')
            img = ax.contourf(obs_x, obs_y, obs_temp, lev, cmap=cmap, extend='both')
        # Contour ice shelf fronts
        ax.contour(model_x, model_y, draft, levels=[np.amax(draft[draft!=0])], colors=('black'), linewidths=0.5, linestyles='solid')
        ax.set_title(titles[n], fontsize=16)
        ax.axis('equal')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_axes([0.01, 0.3, 0.02, 0.4])
    cax.yaxis.set_label_position('left')
    cax.yaxis.set_ticks_position('left')
    cbar = plt.colorbar(img, cax=cax,ticks=np.arange(-2, 2, 1))
    cax.tick_params(length=2)
    plt.suptitle('Bottom temperatures ('+deg_string+'C)', fontsize=18)
    finished_plot(fig, fig_name='bwtemp_compare.png', dpi=300)
