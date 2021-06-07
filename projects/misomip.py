# Type "conda activate animations" before running the animation functions so you can access ffmpeg.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import datetime

from ..grid import Grid
from ..plot_latlon import latlon_plot
from ..plot_1d import timeseries_multi_plot, make_timeseries_plot
from ..utils import real_dir, convert_ismr, mask_3d, mask_except_ice, mask_land, select_top, select_bottom, axis_edges
from ..constants import deg_string, sec_per_year
from ..file_io import read_netcdf
from ..plot_utils.windows import set_panels
from ..plot_utils.colours import get_extend, set_colours
from ..plot_utils.labels import reduce_cbar_labels
from ..postprocess import precompute_timeseries, get_segment_dir


# Helper function to create the MISOMIP time array: first of each month for the given number of years (default 100).
def misomip_time (num_years=100):
    time = []
    for year in range(num_years):
        for month in range(12):
            time.append(datetime.date(year+1, month+1, 1))
    return np.array(time)


# Make animations of lat-lon variables (ismr, bwtemp, bwsalt, bdry_temp, bdry_salt, draft).
def animate_latlon (var, output_dir='./', file_name='output.nc', vmin=None, vmax=None, change_points=None, mov_name=None):

    output_dir = real_dir(output_dir)
    # Get all the directories, one per segment
    segment_dir = get_segment_dir(output_dir)

    # Inner function to read and process data from a single file
    def read_process_data (file_path, var_name, grid, mask_option='3d', gtype='t', lev_option=None, ismr=False):
        data = read_netcdf(file_path, var_name)
        if mask_option == '3d':
            data = mask_3d(data, grid, gtype=gtype, time_dependent=True)
        elif mask_option == 'except_ice':
            data = mask_except_ice(data, grid, gtype=gtype, time_dependent=True)
        elif mask_option == 'land':
            data = mask_land(data, grid, gtype=gtype, time_dependent=True)
        else:
            print(('Error (read_process_data): invalid mask_option ' + mask_option))
            sys.exit()
        if lev_option is not None:
            if lev_option == 'top':
                data = select_top(data)
            elif lev_option == 'bottom':
                data = select_bottom(data)
            else:
                print(('Error (read_process_data): invalid lev_option ' + lev_option))
                sys.exit()
        if ismr:
            data = convert_ismr(data)
        return data

    all_data = []
    all_grids = []
    # Loop over segments
    for sdir in segment_dir:
        # Construct the file name
        file_path = output_dir + sdir + '/MITgcm/' + file_name
        print(('Processing ' + file_path))
        # Build the grid
        grid = Grid(file_path)
        # Read and process the variable we need
        ctype = 'basic'
        gtype = 't'
        if var == 'ismr':
            data = read_process_data(file_path, 'SHIfwFlx', grid, mask_option='except_ice', ismr=True)
            title = 'Ice shelf melt rate (m/y)'
            ctype = 'ismr'
        elif var == 'bwtemp':
            data = read_process_data(file_path, 'THETA', grid, lev_option='bottom')
            title = 'Bottom water temperature ('+deg_string+'C)'
        elif var == 'bwsalt':
            data = read_process_data(file_path, 'SALT', grid, lev_option='bottom')
            title = 'Bottom water salinity (psu)'
        elif var == 'bdry_temp':
            data = read_process_data(file_path, 'THETA', grid, lev_option='top')
            data = mask_except_ice(data, grid, time_dependent=True)
            title = 'Boundary layer temperature ('+deg_string+'C)'
        elif var == 'bdry_salt':
            data = read_process_data(file_path, 'SALT', grid, lev_option='top')
            data = mask_except_ice(data, grid, time_dependent=True)
            title = 'Boundary layer salinity (psu)'
        elif var == 'draft':
            data = mask_except_ice(grid.draft, grid)
            title = 'Ice shelf draft (m)'
        else:
            print(('Error (animate_latlon): invalid var ' + var))
            sys.exit()
        # Loop over timesteps
        if var == 'draft':
            # Just one timestep
            all_data.append(data)
            all_grids.append(grid)
        else:
            for t in range(data.shape[0]):
                # Extract the data from this timestep
                # Save it and the grid to the long lists
                all_data.append(data[t,:])
                all_grids.append(grid)

    extend = get_extend(vmin=vmin, vmax=vmax)
    if vmin is None:
        vmin = np.amax(data)
        for elm in all_data:
            vmin = min(vmin, np.amin(elm))
    if vmax is None:
        vmax = np.amin(data)
        for elm in all_data:
            vmax = max(vmax, np.amax(elm))

    num_frames = len(all_data)

    # Make the initial figure
    fig, gs, cax = set_panels('MISO_C1')
    ax = plt.subplot(gs[0,0])
    img = latlon_plot(all_data[0], all_grids[0], ax=ax, gtype=gtype, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points, title=title+', 1/'+str(num_frames), label_latlon=False, make_cbar=False)
    plt.colorbar(img, cax=cax, extend=extend)

    # Function to update figure with the given frame
    def animate(i):
        ax.cla()
        latlon_plot(all_data[i], all_grids[i], ax=ax, gtype=gtype, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points, title=title+', '+str(i+1)+'/'+str(num_frames), label_latlon=False, make_cbar=False)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=list(range(num_frames)))
    writer = animation.FFMpegWriter(bitrate=500, fps=10)
    anim.save(mov_name, writer=writer)
    if mov_name is not None:
        print(('Saving ' + mov_name))
        anim.save(mov_name, writer=writer)
    else:
        plt.show()


# Precompute timeseries for the given experiment.
def precompute_misomip_timeseries (output_dir='./', file_name='output.nc', timeseries_file='timeseries.nc', segment_dir=None, discard_spinup=True, num_spinup_dir=1):

    timeseries_types = ['all_melting', 'all_massloss', 'ocean_vol', 'avg_temp', 'avg_salt']

    output_dir = real_dir(output_dir)
    if segment_dir is not None:
        # segment_dir is preset
        if isinstance(segment_dir, str):
            # Just one directory, so make it a list
            segment_dir = [segment_dir]
    else:
        # Get all the directories, one per segment
        segment_dir = get_segment_dir(output_dir)
        if discard_spinup:
            # Throw away the spinup directory(s)
            segment_dir = segment_dir[num_spinup_dir:]    

    for sdir in segment_dir:
        file_path = output_dir+sdir+'/MITgcm/'+file_name
        print(('Processing ' + file_path))
        precompute_timeseries(file_path, timeseries_file, timeseries_types=timeseries_types, monthly=False)


# Plot each timeseries on the same axes as Jan's output from the old coupling setup.
def compare_timeseries_jan (timeseries_file='timeseries.nc', jan_file='/work/n02/n02/kaight/jan_output/IceOcean1r_COM_ocean_UaMITgcm.nc', fig_dir='./'):

    fig_dir = real_dir(fig_dir)

    # Variable names in our files and in Jan's old file, plus titles, units, and conversion factors to apply to Jan's file to get the units we want
    var_names = ['all_melting', 'all_massloss', 'ocean_vol', 'avg_temp', 'avg_salt']
    jan_names = ['meanMeltRate', 'totalMeltFlux', 'totalOceanVolume', 'meanTemperature', 'meanSalinity']
    titles = ['Mean ice shelf melt rate', 'Basal mass loss from all ice shelves', 'Volume of ocean in domain', 'Volume-averaged temperature', 'Volume-averaged salinity']
    units = ['m/y', 'Gt/y', r'm$^3$', deg_string+'C', 'psu']
    conversion = [sec_per_year, 1e-12*sec_per_year, 1, 1, 1]

    # Create the time array: first of each month for 100 years
    time = misomip_time()

    # Loop over variables
    for i in range(len(var_names)):
        data_new = read_netcdf(timeseries_file, var_names[i])
        data_old = read_netcdf(jan_file, jan_names[i])*conversion[i]
        timeseries_multi_plot(time, [data_old, data_new], ['MISOMIP_1r, old', 'MISOMIP_1r, new'], ['black', 'blue'], title=titles[i], units=units[i], fig_name=fig_dir+'jan_compare_'+var_names[i]+'.png')


# Plot each timeseries from several different simulations on the same axes.
def compare_timeseries_multi (base_dir='./', simulations=['MISOMIP_1r','MISOMIP_1rv','MISOMIP_1rvb','MISOMIP_1rp'], timeseries_file='/output/timeseries.nc', fig_dir='./'):

    base_dir = real_dir(base_dir)
    fig_dir = real_dir(fig_dir)

    var_names = ['all_melting', 'all_massloss', 'ocean_vol', 'avg_temp', 'avg_salt']
    titles = ['Mean ice shelf melt rate', 'Basal mass loss from all ice shelves', 'Volume of ocean in domain', 'Volume-averaged temperature', 'Volume-averaged salinity']
    units = ['m/y', 'Gt/y', r'm$^3$', deg_string+'C', 'psu']

    colours = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
    num_sim = len(simulations)
    if num_sim > len(colours):
        print('Error (compare_timeseries_multi): not enough colours defined.')
        sys.exit()
    colours = colours[:num_sim]

    time = misomip_time()

    for i in range(len(var_names)):
        data = []
        times = []
        for j in range(num_sim):
            data_tmp = read_netcdf(simulations[j]+timeseries_file, var_names[i])
            data.append(data_tmp)
            times.append(time[:data_tmp.size])
        timeseries_multi_plot(times, data, simulations, colours, title=titles[i], units=units[i], fig_name=fig_dir+'multi_compare_'+var_names[i]+'.png')


# The following functions compare the MISOMIP NetCDF files from two different simulations.

# Compare a timeseries variable. Make one plot with both timeseries on the same axes, and one plot with the difference timeseries (2 minus 1).
def compare_timeseries_netcdf (var_name, file_path_1, file_path_2, name_1, name_2, fig_dir='./', num_years=100):

    fig_dir = real_dir(fig_dir)
    # Read the data
    data_1, title, units = read_netcdf(file_path_1, var_name, return_info=True)
    data_2 = read_netcdf(file_path_2, var_name)
    time = misomip_time(num_years=num_years)
    # Plot timeseries on the same axes
    timeseries_multi_plot(time, [data_1, data_2], [name_1, name_2], ['black', 'blue'], title=title, units=units, fig_name=fig_dir+var_name+'.png')
    # Plot the difference timeseries
    make_timeseries_plot(time, data_2-data_1, title=title+'\n'+name_2+' minus '+name_1, units=units, fig_name=fig_dir+var_name+'_diff.png')


# Compare a 2D variable (lat-lon or slice; just pass x and y as appropriate). Make a 3-panelled animation with data from each simulation, and the difference (2 minus 1).
def compare_2d_anim_netcdf (var_name, file_path_1, file_path_2, name_1, name_2, x, y, fig_dir='./'):

    fig_dir = real_dir(fig_dir)
    # Read the data
    data_1, title, units = read_netcdf(file_path_1, var_name, return_info=True)
    data_2 = read_netcdf(file_path_2, var_name)
    if var_name == 'iceDraft':
        # Mask the open ocean and grounded ice
        # Use -1e-3 not 0 to account for interpolation artifacts
        data_1 = np.ma.masked_where(data_1>=-1e-3, data_1)
        data_2 = np.ma.masked_where(data_2>=-1e-3, data_2)
    data_diff = data_2-data_1
    # Get cell boundaries
    x_bound = axis_edges(x)
    y_bound = axis_edges(y)
    # Get bounds
    vmin = min(np.amin(data_1), np.amin(data_2))
    vmax = max(np.amax(data_1), np.amax(data_2))
    if var_name == 'bottomSalinity':
        vmax = 35
    vmin_diff = np.amin(data_diff)
    vmax_diff = np.amax(data_diff)
    # Set colourmaps
    if var_name == 'meltRate':
        ctype = 'ismr'
    elif var_name in ['uBoundaryLayer', 'vBoundaryLayer', 'barotropicStreamfunction', 'uBase', 'vBase', 'uSurface', 'vSurface', 'uMean', 'vMean', 'overturningStreamfunction']:
        ctype = 'plusminus'
    else:
        ctype = 'basic'
    cmap, vmin, vmax = set_colours(data_1, ctype=ctype, vmin=vmin, vmax=vmax)
    cmap, vmin, vmax = set_colours(data_2, ctype=ctype, vmin=vmin, vmax=vmax)
    cmap_diff, vmin_diff, vmax_diff = set_colours(data_diff, ctype='plusminus', vmin=vmin_diff, vmax=vmax_diff)
    num_frames = data_1.shape[0]

    # Set up the figure
    fig, gs, cax1, cax2 = set_panels('MISO_3_C2')
    ax = [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[1,0])]
    data = [data_1, data_2, data_diff]
    cax = [cax1, None, cax2]
    cmaps = [cmap, cmap, cmap_diff]
    vmins = [vmin, vmin, vmin_diff]
    vmaxs = [vmax, vmax, vmax_diff]
    titles = [name_1, name_2, 'Difference']

    # Function to update figure with the given frame
    def animate(t):
        if (t+1) % 10 == 0:
            print(('Frame ' + str(t+1) + ' of ' + str(num_frames)))
        for i in range(3):
            ax[i].cla()
            img = ax[i].pcolormesh(x_bound, y_bound, data[i][t,:], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_title(titles[i], fontsize=18)
            if cax[i] is not None and t==0:
                cbar = plt.colorbar(img, cax=cax[i], orientation='horizontal')
                reduce_cbar_labels(cbar)
        plt.suptitle(title+' ('+units+'), '+str(t+1)+'/'+str(num_frames), fontsize=24)

    # Call it for each frame and save as animation
    anim = animation.FuncAnimation(fig, func=animate, frames=list(range(num_frames)))
    writer = animation.FFMpegWriter(bitrate=500, fps=10)
    mov_name = fig_dir + var_name + '.mp4'
    print(('Saving ' + mov_name))
    anim.save(mov_name, writer=writer)


# Animate the grounding line locations over time, for two different simulations on the same axes.
def compare_gl_anim_netcdf (file_path_1, file_path_2, name_1, name_2, fig_dir='./'):

    fig_dir = real_dir(fig_dir)
    # Read grounding line locations
    x_1 = read_netcdf(file_path_1, 'xGL')
    y_1 = read_netcdf(file_path_1, 'yGL')
    x_2 = read_netcdf(file_path_2, 'xGL')
    y_2 = read_netcdf(file_path_2, 'yGL')
    xmin = min(np.amin(x_1), np.amin(x_2))
    xmax = max(np.amax(x_1), np.amax(x_2))
    ymin = min(np.amin(y_1), np.amin(y_2))
    ymax = max(np.amax(y_1), np.amax(y_2))
    num_frames = x_1.shape[0]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10,6))

    # Function to update figure with the given frame
    def animate(t):
        if (t+1) % 10 == 0:
            print(('Frame ' + str(t+1) + ' of ' + str(num_frames)))
        ax.cla()
        ax.plot(x_1[t,:], y_1[t,:], '-', color='black', label=name_1)
        ax.plot(x_2[t,:], y_2[t,:], '-', color='blue', label=name_2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('Grounding line position, '+str(t+1)+'/'+str(num_frames), fontsize=18)
        if t==0:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

    # Call it for each frame and save as animation
    anim = animation.FuncAnimation(fig, func=animate, frames=list(range(num_frames)))
    writer = animation.FFMpegWriter(bitrate=500, fps=10)
    mov_name = fig_dir + 'GL.mp4'
    print(('Saving ' + mov_name))
    anim.save(mov_name, writer=writer)  
    

# Call the above functions for all possible variables.
def compare_everything_netcdf (file_path_1_ocean, file_path_1_ice, name_1, file_path_2_ocean, file_path_2_ice, name_2, fig_dir='./', num_years=100):

    # Read grid variables needed later
    xo = read_netcdf(file_path_1_ocean, 'x')
    yo = read_netcdf(file_path_1_ocean, 'y')
    zo = read_netcdf(file_path_1_ocean, 'z')
    xi = read_netcdf(file_path_1_ice, 'x')
    yi = read_netcdf(file_path_1_ice, 'y')

    # Timeseries
    timeseries_var_ocean = ['meanMeltRate', 'totalMeltFlux', 'totalOceanVolume', 'meanTemperature', 'meanSalinity']
    timeseries_var_ice = ['iceVolume', 'iceVAF', 'groundedArea']
    for var in timeseries_var_ocean:
        print(('Processing ' + var))
        compare_timeseries_netcdf(var, file_path_1_ocean, file_path_2_ocean, name_1, name_2, fig_dir=fig_dir, num_years=num_years)
    for var in timeseries_var_ice:
        print(('Processing ' + var))
        compare_timeseries_netcdf(var, file_path_1_ice, file_path_2_ice, name_1, name_2, fig_dir=fig_dir, num_years=num_years)

    # Lat-lon animations
    latlon_var_ocean = ['iceDraft', 'meltRate', 'barotropicStreamfunction', 'bottomTemperature', 'bottomSalinity'] #['iceDraft', 'bathymetry', 'meltRate', 'frictionVelocity', 'thermalDriving', 'halineDriving', 'uBoundaryLayer', 'vBoundaryLayer', 'barotropicStreamfunction', 'bottomTemperature', 'bottomSalinity']
    latlon_var_ice = ['iceThickness'] #['iceThickness', 'upperSurface', 'lowerSurface', 'basalMassBalance', 'groundedMask', 'floatingMask', 'basalTractionMagnitude', 'uBase', 'vBase', 'uSurface', 'vSurface', 'uMean', 'vMean']
    for var in latlon_var_ocean:
        print(('Processing ' + var))
        compare_2d_anim_netcdf(var, file_path_1_ocean, file_path_2_ocean, name_1, name_2, xo, yo, fig_dir=fig_dir)
    for var in latlon_var_ice:
        print(('Processing ' + var))
        compare_2d_anim_netcdf(var, file_path_1_ice, file_path_2_ice, name_1, name_2, xi, yi, fig_dir=fig_dir)

    # Slice animations
    slice_var_xz = ['overturningStreamfunction', 'temperatureXZ', 'salinityXZ']
    slice_var_yz = ['temperatureYZ', 'salinityYZ']
    for var in slice_var_xz:
        print(('Processing ' + var))
        compare_2d_anim_netcdf(var, file_path_1_ocean, file_path_2_ocean, name_1, name_2, xo, zo, fig_dir=fig_dir)
    for var in slice_var_yz:
        print(('Processing ' + var))
        compare_2d_anim_netcdf(var, file_path_1_ocean, file_path_2_ocean, name_1, name_2, yo, zo, fig_dir=fig_dir)

    # Grounding line animation
    print('Processing grounding line')
    compare_gl_anim_netcdf(file_path_1_ice, file_path_2_ice, name_1, name_2, fig_dir=fig_dir)

    
