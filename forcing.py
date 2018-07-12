import numpy as np

from grid import Grid
from file_io import read_netcdf, write_binary, NCfile
from utils import real_dir, fix_lon_range, mask_land_zice
from interpolation import interp_reg_xy
from plot_latlon import latlon_plot

# Interpolate the freshwater flux from iceberg melting (Martin & Adcroft, 2010, doi:doi:10.1016/j.ocemod.2010.05.001) to the model grid so it can be used for runoff forcing.

# Arguments:
# grid_file: path to MITgcm NetCDF grid file
# input_dir: path to directory with Martin & Adcroft data
# output_file: desired path to binary output file which MITgcm will read

# Optional keyword arguments:
# nc_out: path to a NetCDF file to save the interpolated data in, so you can easily check that it looks okay. (The annual mean will also be plotted and shown on screen whether or not you define nc_out.)
# prec: precision to write output_file. Must match exf_iprec in the "data.exf" namelist (default 32)
def icebergs_ma2010 (grid_file, input_dir, output_file, nc_out=None, prec=32):

    input_dir = real_dir(input_dir)    

    file_head = 'icebergs.1861-1960.'
    file_tail = '.melt.nc'
    max_lon = 80  # Iceberg data is weirdly in the range (-280, 80)

    print 'Building grids'
    # Read the grid from the first file
    file_path = input_dir + file_head + '01' + file_tail
    lon = read_netcdf(file_path, 'xt')
    lat = read_netcdf(file_path, 'yt')
    # Build the model grid
    model_grid = Grid(grid_file, max_lon=max_lon)

    print 'Interpolating'
    icebergs_interp = np.zeros([12, model_grid.ny, model_grid.nx])    
    for month in range(12):
        print '...month ' + str(month+1)
        # Read the data
        file_path = input_dir + file_head + '{0:02d}'.format(month+1) + file_tail
        icebergs = read_netcdf(file_path, 'melt')
        # Anything outside the bounds of the source data (i.e. south of ~77S) will be 0
        icebergs_interp_tmp = interp_reg_xy(lon, lat, icebergs, model_grid.lon_1d, model_grid.lat_1d, fill_value=0)
        # Make sure the ice shelf cavities don't get any iceberg melt
        icebergs_interp_tmp[model_grid.zice_mask] = 0
        # Save to the master array
        icebergs_interp[month,:] = icebergs_interp_tmp    

    print 'Writing ' + output_file
    write_binary(icebergs_interp, output_file, prec=prec)

    print 'Plotting'
    # Remake the model grid with normal longitude
    model_grid = Grid(grid_file)
    # Make a nice plot of the annual mean
    latlon_plot(mask_land_zice(np.mean(icebergs_interp, axis=0), model_grid), model_grid, include_shelf=False, vmin=0, title=r'Annual mean iceberg melt (kg/m$^2$/s)', fig_name='icebergs_MA2010.png')                
    if nc_out is not None:
        # Also write to NetCDF file
        print 'Writing ' + nc_out
        ncfile = NCfile(nc_out, model_grid, 'xyt')
        ncfile.add_time(np.arange(12)+1, units='months')
        ncfile.add_variable('iceberg_melt', icebergs_interp, 'xyt', units='kg/m^2/s')
        ncfile.close()
    

    

    

    

    
