import matplotlib.pyplot as plt

def quick_plot (grid, var):

    plt.clf()
    plt.contourf(grid.lon_2d, grid.lat_2d, var, 50);
    plt.colorbar()
    plt.show()
