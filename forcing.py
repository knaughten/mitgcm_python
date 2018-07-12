import numpy as np

from grid import Grid
from file_io import read_netcdf, write_binary, NCfile
from utils import real_dir, fix_lon_range
from interpolation import interp_reg_xy
from plot_latlon import latlon_plot

def icebergs_ma2010 (grid_file, input_dir, output_file, nc_out=None, prec=32):

    input_dir = real_dir(input_dir)    

    file_head = 'icebergs.1861-1960.'
    file_tail = '.melt.nc'
    max_lon = 80  # Iceberg data is weirdly in the range (-280, 80)

    # Read the grid from the first file
    file_path = input_dir + file_head + '01' + file_tail
    lon = read_netcdf(file_path, 'xt')
    lat = read_netcdf(file_path, 'yt')

    # Build the model grid
    model_grid = Grid(grid_file, max_lon=max_lon)

    # Interpolate each month
    icebergs_interp = np.zeros([12, model_grid.ny, model_grid.nx])
    for month in range(12):
        # Read the data
        file_path = input_dir + file_head + '{0:02d}'.format(month+1) + file_tail
        icebergs = read_netcdf(file_path, 'melt')
        # Anything outside the bounds of the source data (i.e. south of ~77S) will be 0
        icebergs_interp_tmp = interp_reg_xy(lon, lat, icebergs, model_grid.lon, model_grid.lat, fill_value=0)
        # Make sure the ice shelf cavities don't get any iceberg melt
        icebergs_interp_tmp[model_grid.zice_mask] = 0
        # Save to the master array
        icebergs_interp[month,:] = icebergs_interp_tmp    
        
    # Write to binary
    write_binary(icebergs_interp, output_file, prec=prec)

    # Now remake the model grid with normal longitude
    model_grid = Grid(grid_file)
    # Make a nice plot of the annual mean
    latlon_plot(np.mean(icebergs_interp, axis=0), grid, include_shelf=False, vmin=0, title=r'Annual mean iceberg melt (kg/m$^2$/s)')                
    if nc_out is not None:
        # Also write to NetCDF file
        ncfile = NCfile(nc_out, model_grid, 'xy')
        ncfile.add_time(np.arange(12)+1, units='months')
        ncfile.add_variable('iceberg_melt', icebergs_interp, 'xyt', units='kg/m^2/s')
        ncfile.close()
    

    

    

    

    
