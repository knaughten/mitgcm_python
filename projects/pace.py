##################################################################
# Weddell Sea threshold paper
##################################################################

from grid import ERA5Grid, PACEGrid

def calc_climatologies (era5_dir, pace_dir):

    var_era5 = ['atemp', 'aqh', 'apressure', 'uwind', 'vwind', 'precip', 'swdown', 'lwdown']
    var_pace = ['TREFHT', 'QBOT', 'PSL', 'UBOT', 'VBOT', 'PRECT', 'FLDS', 'FSDS']

    era5_grid = ERA5Grid()
    pace_grid = PACEGrid()
    
