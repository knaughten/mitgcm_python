#######################################################
# Plots from Ua output in coupled Ua/MITgcm simulations
#######################################################

import sys
import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from plot_utils.colours import set_colours, get_extend
from plot_utils.labels import latlon_axes
from plot_utils.windows import finished_plot
from file_io import read_netcdf
from utils import var_min_max, choose_range
from interpolation import interp_nonreg_xy
from constants import ua_titles

# Plot a 2D variable on either the Ua triangular mesh or interpolated to a regular grid.
# Arguments:
# option: 'tri' or 'reg'
# data: Ua variable at each node
# x, y: locations of each node (polar stereographic)

# Optional keyword arguments: mostly as in function latlon_plot
# connectivity: if option=tri', connectivity from the MUA object
# xGL, yGL: grounding line coordinates to overlay

def ua_plot (option, data, x, y, connectivity=None, xGL=None, yGL=None, ax=None, make_cbar=True, ctype='basic', vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, zoom_fris=False, title=None, titlesize=18, return_fig=False, fig_name=None, extend=None, figsize=None, dpi=None):
    
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    if option == 'tri' and connectivity is None:
        print 'Error (ua_plot): Need to provide connectivity'
        sys.exit()

    if figsize is None:
        if zoom_fris:
            figsize = (8,6)
        else:
            figsize = (10,6)

    # Choose what the endpoints of the colourbar should do
    if extend is None:
        extend = get_extend(vmin=vmin, vmax=vmax)
    # If we're zooming, choose the correct colour bounds
    zoom = zoom_fris or any([xmin, xmax, ymin, ymax])
    if zoom:
        vmin_tmp, vmax_tmp = var_min_max(data, [x,y], pster=True, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ua=True)
        if vmin is None:
            vmin = vmin_tmp
        if vmax is None:
            vmax = vmax_tmp
    # Get colourmap
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, num=26)
    # Make the figure and axes, if needed
    existing_ax = ax is not None
    if not existing_ax:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('equal')
    # Plot the data
    if option == 'tri':
        img = ax.tricontourf(x, y, connectivity, data, levels, cmap=cmap, vmin=vmin, vmax=vmax, extend=extend)
    elif option == 'reg':
        img = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)
    if make_cbar:
        # Add a colourbar
        if option == 'tri':
            plt.colorbar(img)
        elif option == 'reg':
            plt.colorbar(img, extend=extend)
    if xGL is not None and yGL is not None:
        ax.plot(xGL, yGL, color='black')
    # Set axes limits etc.
    latlon_axes(ax, x, y, zoom_fris=zoom_fris, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, pster=True)
    if title is not None:
        # Add a title
        plt.title(title, fontsize=titlesize)

    if return_fig:
        return fig, ax
    elif existing_ax:
        return img
    else:
        finished_plot(fig, fig_name=fig_name, dpi=dpi)
        

# Read x, y, and connectivity from the MUA object within an Ua output file.
# If the file has already been loaded, pass the loadmat object.
# Otherwise, pass the file name.
def read_ua_mesh (f):

    # Check if f is a file name that still needs to be read
    if isinstance(f, str):
        f = loadmat(f)
    x = f['MUA']['coordinates'][0][0][:,0]
    y = f['MUA']['coordinates'][0][0][:,1]
    connectivity = f['MUA']['connectivity'][0][0]-1
    return x, y, connectivity


# Helper function to check and read the grounding line data. It must be precomputed and saved in a NetCDF file.
def check_read_gl (gl_file, gl_time_index):

    if gl_file is not None:
        xGL = read_netcdf(gl_file, 'xGL', time_index=gl_time_index)
        yGL = read_netcdf(gl_file, 'yGL', time_index=gl_time_index)
    else:
        xGL = None
        yGL = None
    return xGL, yGL


# Read a variable from an Ua output file and plot it.
def read_plot_ua_tri (var, file_path, gl_file=None, gl_time_index=-1, title=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, zoom_fris=False, fig_name=None, figsize=None, dpi=None):

    # Read the file
    f = loadmat(file_path)
    x, y, connectivity = read_ua_mesh(f)
    def read_data (var_name):
        return f[var_name][:,0]
    if var == 'velb':
        u = read_data('ub')
        v = read_data('vb')
        data = np.sqrt(u**2 + v**2)
    else:
        data = read_data(var)
    xGL, yGL = check_read_gl(gl_file, gl_time_index)

    if title is None:
        # Choose title
        try:
            title = ua_titles[var]
        except(KeyError):
            title = var
    # Choose colourmap
    ctype = 'basic'
    if var in ['dhdt', 'ub', 'vb']:
        ctype = 'plusminus'
    
    ua_plot('tri', data, x, y, connectivity=connectivity, xGL=xGL, yGL=yGL, ctype=ctype, vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zoom_fris=zoom_fris, title=title, fig_name=fig_name, figsize=figsize, dpi=dpi)


# Helper function to plot the grounding line at the beginning of the simulation, and at the current frame.
def gl_frame (xGL, yGL, t, ax=None, title='Grounding line position', label='Current', xmin=None, xmax=None, ymin=None, ymax=None):

    return_fig = ax is None
    if return_fig:
        # Set up the plot
        fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(xGL[0,:], yGL[0,:], '-', color='red', label='Initial')
    ax.plot(xGL[t,:], yGL[t,:], '-', color='black', label=label)
    # Choose bounds
    xmin, xmax = choose_range(xGL[0,:], x2=xGL[t,:], xmin=xmin, xmax=xmax)
    ymin, ymax = choose_range(yGL[0,:], x2=yGL[t,:], xmin=ymin, xmax=ymax)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=20)
    ax.legend(loc='center')
    if return_fig:
        return fig, ax


# Animate the grounding line position over time, with the original grounding line for comparison. You must have a NetCDF file with the x and y positions of nodes over time (use the ua_postprocess utility within UaMITgcm).
# Type "conda activate animations" before running this, so you can access ffmpeg.
def gl_animation (file_path, mov_name=None):
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Read grounding line locations
    xGL = read_netcdf(file_path, 'xGL')
    yGL = read_netcdf(file_path, 'yGL')
    xmin = np.amin(xGL)
    xmax = np.amax(xGL)
    ymin = np.amin(yGL)
    ymax = np.amax(yGL)
    num_frames = xGL.shape[0]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(7,6))

    # Function to update figure with the given frame
    def animate(t):
        ax.cla()
        gl_frame(xGL, yGL, t, ax=ax, title='Grounding line position, '+str(t+1)+'/'+str(num_frames), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, move_box=t==0)

    # Call it for each frame and save as an animation
    anim = animation.FuncAnimation(fig, func=animate, frames=range(num_frames))
    writer = animation.FFMpegWriter(bitrate=500, fps=10)
    if mov_name is None:
        plt.show()
    else:
        anim.save(mov_name, writer=writer)


# As above, but just plot the last frame.
def gl_final (file_path, fig_name=None, dpi=None):

    xGL = read_netcdf(file_path, 'xGL')
    yGL = read_netcdf(file_path, 'yGL')

    fig, ax = gl_frame(xGL, yGL, -1, label='Final')
    finished_plot(fig, fig_name, dpi=dpi)


# Plot the difference between two Ua output steps for the given variable. This involves interpolating to a common grid.
def read_plot_ua_difference (var, file_path_1, file_path_2, gl_file=None, gl_time_index=-1, title=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, zoom_fris=False, fig_name=None, figsize=None, dpi=None):

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # Read the data for each output step
    def read_data (file_path):
        f = loadmat(file_path)
        x, y, connectivity = read_ua_mesh(f)
        if var == 'velb':
            data = np.sqrt(f['ub'][:,0]**2 + f['vb'][:,0]**2)
        else:
            data = f[var][:,0]
        return x, y, connectivity, data

    x_1, y_1, connectivity_1, data_1 = read_data(file_path_1)
    x, y, connectivity, data_2 = read_data(file_path_2)
    # Interpolate the first grid to the second
    data_1_interp = interp_nonreg_xy(x_1, y_1, data_1, x, y)
    data_diff = data_2 - data_1_interp
    xGL, yGL = check_read_gl(gl_file, gl_time_index)

    if title is None:
        try:
            title = ua_titles[var]
        except(KeyError):
            title = var
        title = 'Change in ' + title[0].lower() + title[1:]

    ua_plot('tri', data_diff, x, y, connectivity=connectivity, xGL=xGL, yGL=yGL, ctype='plusminus', vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zoom_fris=zoom_fris, title=title, fig_name=fig_name, figsize=figsize, dpi=dpi)
    
        
        
