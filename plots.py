import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def quick_plot (grid, var):

    fig, ax = plt.subplots()
    img = ax.contourf(grid.lon_2d, grid.lat_2d, var, 50);
    plt.colorbar(img)
    fig.show()
