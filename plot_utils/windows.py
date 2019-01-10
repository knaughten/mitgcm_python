#######################################################
# Figure windows and placement of objects within them.
#######################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# If a figure name is defined, save the figure to that file. Otherwise, display the figure on screen.
def finished_plot (fig, fig_name=None):

    if fig_name is not None:
        print 'Saving ' + fig_name
        fig.savefig(fig_name)
    else:
        fig.show()

        
# Set things up for complicated multi-panelled plots. Initialise a figure window of the correct size and set the locations of panels and colourbar(s). The exact output depends on the single argument, which is a string containing the key for the type of plot you want. Read the comments to choose one.
def set_panels (key, figsize=None):

    # Choose figure size
    if figsize is None:
        if key in ['1x2C1', '1x2C2']:
            figsize = (12, 6)
        elif key == '2x2C1':
            figsize = (10, 7.5)
        elif key == '2x2C0':
            figsize = (10, 6.5)
        elif key in ['1x3C1', '1x3C3']:
            figsize = (16, 5)
        elif key in ['5C1', '5C2']:
            figsize = (12, 7)
        elif key == '5C0':
            figsize = (14, 7.5)
        elif key == '5x8C1':
            figsize = (30, 20)
        elif key == 'CTD':
            figsize = (15, 6)

    fig = plt.figure(figsize=figsize)
    
    if key == '1x2C1':        
        # Two side-by-side plots with one colourbar below
        gs = plt.GridSpec(1,2)
        gs.update(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.05)
        cax = fig.add_axes([0.3, 0.05, 0.4, 0.04])
    elif key == '1x2C2':
        # Two side-by-side plots with two colourbars below
        gs = plt.GridSpec(1,2)
        gs.update(left=0.07, right=0.97, bottom=0.15, top=0.85, wspace=0.05)
        cax1 = fig.add_axes([0.12, 0.05, 0.325, 0.04])
        cax2 = fig.add_axes([0.595, 0.05, 0.325, 0.04])
    elif key == '2x2C1':
        # Four plots arranged into two rows and two columns, with one colourbar below
        gs = plt.GridSpec(2,2)
        gs.update(left=0.05, right=0.97, bottom=0.12, top=0.88, wspace=0.05, hspace=0.15)
        cax = fig.add_axes([0.3, 0.03, 0.4, 0.03])
    elif key == '2x2C0':
        # Like 2x2C1 but no colourbar
        gs = plt.GridSpec(2,2)
        gs.update(left=0.05, right=0.97, bottom=0.05, top=0.88, wspace=0.05, hspace=0.15)
    elif key == '1x3C1':
        # Three side-by-side plots with one colourbar below
        gs = plt.GridSpec(1,3)
        gs.update(left=0.05, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
        cax = fig.add_axes([0.3, 0.05, 0.4, 0.04])
    elif key == '1x3C3':
        # Three side-by-side plots with three colourbars below
        gs = plt.GridSpec(1,3)
        gs.update(left=0.05, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
        cax1 = fig.add_axes([0.08, 0.05, 0.25, 0.04])
        cax2 = fig.add_axes([0.395, 0.05, 0.25, 0.04])
        cax3 = fig.add_axes([0.71, 0.05, 0.25, 0.04])
    elif key in ['5C1', '5C2']:
        # Five plots arranged into two rows and three columns, with the empty space in the bottom left filled with either one or two colourbars.
        gs = plt.GridSpec(2,3)
        gs.update(left=0.05, right=0.98, bottom=0.05, top=0.88, wspace=0.05, hspace=0.12)
        if key == '5C1':
            cax = fig.add_axes([0.07, 0.25, 0.25, 0.05])
        elif key == '5C2':
            cax1 = fig.add_axes([0.07, 0.3, 0.25, 0.05])
            cax2 = fig.add_axes([0.07, 0.15, 0.25, 0.05])
    elif key == '5C0':
        # Five plots arranged into two rows and three columns, with no special colourbars (each subplot will get its own automatically), and the empty space in the top left.
        gs = plt.GridSpec(2,3)
        gs.update(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.07, hspace=0.15)
    elif key == '5x8C1':
        # 38 plots (one per year of observational period) arranged into 5 rows and 8 columns, with one colourbar in the empty space of the last 2 panels
        gs = plt.GridSpec(5,8)
        gs.update(left=0.025, right=0.99, bottom=0.03, top=0.93, wspace=0.03, hspace=0.12)
        cax = fig.add_axes([0.77, 0.1, 0.2, 0.025])
    elif key == 'CTD':
        # Special case for compare_keith_ctd in projects/tuning.py
        gs_1 = plt.GridSpec(1,2)
        gs_1.update(left=0.05, right=0.7, bottom=0.1, top=0.85, wspace=0.12)
        gs_2 = plt.GridSpec(1,1)
        gs_2.update(left=0.75, right=0.95, bottom=0.3, top=0.8)

    if key == 'CTD':
        return fig, gs_1, gs_2
    elif key[-1] == '0':
        return fig, gs
    elif key[-1] == '1':        
        return fig, gs, cax
    elif key[-1] == '2':
        return fig, gs, cax1, cax2
    elif key[-1] == '3':
        return fig, gs, cax1, cax2, cax3

