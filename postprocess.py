#######################################################
# Files to create when the simulation is done
#######################################################

import os

from grid import Grid
from io import NCfile, read_binary


# When the model crashes, convert its crash-dump to a NetCDF file.
# Arguments:
# crash_dir: directory including all the state*crash.*.data files. The NetCDF file will be saved here too, with the name crash.nc.
# grid_path: path to NetCDF grid file.
def crash_to_netcdf (crash_dir, grid_path):

    # Make sure crash_dir is a proper directory
    if not crash_dir.endswith('/'):
        crash_dir += '/'

    # Read the grid
    grid = Grid(grid_path)
    # Initialise the NetCDF file
    ncfile = NCfile(crash_dir+'crash.nc', grid, 'xyz')

    # Find all the crash files
    for file in os.listdir(crash_dir):
        if file.startswith('stateThetacrash') and file.endswith('.data'):
            # Found temperature
            # Read it from binary
            temp = read_binary(crash_dir + file, grid, 'xyz')
            # Write it to NetCDF
            ncfile.add_variable('THETA', temp, 'xyz', units='C')
        if file.startswith('stateSaltcrash') and file.endswith('.data'):
            salt = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('SALT', salt, 'xyz', units='psu')
        if file.startswith('stateUvelcrash') and file.endswith('.data'):
            u = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('UVEL', u, 'xyz', gtype='u', units='m/s')
        if file.startswith('stateVvelcrash') and file.endswith('.data'):
            v = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('VVEL', v, 'xyz', gtype='v', units='m/s')
        if file.startswith('stateWvelcrash') and file.endswith('.data'):
            w = read_binary(crash_dir + file, grid, 'xyz')
            ncfile.add_variable('WVEL', w, 'xyz', gtype='w', units='m/s')
        if file.startswith('stateEtacrash') and file.endswith('.data'):
            eta = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('ETAN', eta, 'xy', units='m')
        if file.startswith('stateAreacrash') and file.endswith('.data'):
            area = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIarea', area, 'xy', units='fraction')
        if file.startswith('stateHeffcrash') and file.endswith('.data'):
            heff = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIheff', heff, 'xy', units='m')
        if file.startswith('stateUicecrash') and file.endswith('.data'):
            uice = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIuice', uice, 'xy', gtype='u', units='m/s')
        if file.startswith('stateVicecrash') and file.endswith('.data'):
            vice = read_binary(crash_dir + file, grid, 'xy')
            ncfile.add_variable('SIvice', vice, 'xy', gtype='v', units='m/s')

    ncfile.finished()
