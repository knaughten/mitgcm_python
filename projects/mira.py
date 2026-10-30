# Extract data for Mira: ensemble mean T and S Hovmollers for all four scenarios, all available years from 1995

import os
from ..postprocess import precompute_hovmoller
from ..grid import Grid

def pitt_hovmollers (out_dir='./'):

    in_dir = '/gws/ssde/j25a/bas_pog/sg240/kaight_for_ceda/scenarios_ensemble_means/'
    expt_head = 'PAS_'
    expts = ['LENS', 'MENS', 'LW2.0', 'LW1.5']
    hovmoller_file_tail = '_PIT_trough_TS_profiles.nc'
    grid = Grid('/gws/ssde/j25a/bas_pog/kaight/PAS_grid/')

    for expt in expts:
        expt_dir = in_dir + expt_head + expt + '/'
        fnames = os.listdir(expt_dir)
        fnames = [f for f in fnames if expt in f and f.endswith('.nc')]
        fnames.sort()
        for f in fnames:
            file_path = expt_dir + f
            print('Processing '+file_path)
            precompute_hovmoller(file_path, out_dir+expt+hovmoller_file_tail, loc=['PIT_trough'], grid=grid)
        

