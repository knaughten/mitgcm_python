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
        elif key == '1x3C1':
            figsize = (15, 5)

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
        gs.update(left=0.07, right=0.97, bottom=0.12, top=0.88, wspace=0.05, hspace=0.15)
        cax = fig.add_axes([0.3, 0.03, 0.4, 0.03])
    elif key == '1x3C1':
        # Three side-by-side plots with one colourbar below
        gs = plt.GridSpec(1,3)
        gs.update(left=0.05, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
        cax = fig.add_axes([0.3, 0.05, 0.4, 0.04])

    if key[-1] == '1':        
        return fig, gs, cax
    elif key[-1] == '2':
        return fig, gs, cax1, cax2

