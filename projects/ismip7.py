import xarray as xr
import numpy as np

from ..grid import ISMIP7Grid, Grid
from ..interpolation import interp_reg_xy
from ..utils import convert_ismr

# Interpolate the output of WSFRIS 2021 paper (two_timescale.py) and PAS 2023 paper (scenarios.py) to the ISMIP7 grid for sharing

def interp_year (file_path):

    var_in = ['THETA', 'SALT', 'SHIfwFlx']
    var_out = ['temperature', 'salinity', 'basal_melt']
    units = ['degC', 'psu', 'm/y']
    long_name = ['potential temperature (EOS80)', 'practical salinity (EOS80)', 'ice shelf basal melt rate, positive means melting']
    grid_out = ISMIP7Grid()

    ds = xr.open_dataset(file_path)
    ds_out = None
    for v in range(len(var_in)):
        # Annual average
        data_in = ds[var_in].mean('time')
        if var_in['v'] == 'SHIfwFlx':
            # Unit conversion for ice shelf melting
            var_in = convert_ismr(var_in)
        if 'Z' in data_in.coords:
            pass
        else:
            # 2D variable - interpolate once
            data_out = interp_reg_xy(ds['XC'], ds['YC'], data_in, grid_out.lon, grid_out.lat, fill_value=np.nan)
        # Mask land - TO DO
        # Attach coordinates etc
        data_out = xr.DataArray(data_out, coords={'y':grid_out.y, 'x':grid_out.x}, attrs={'description':long_name[v], 'units':units[v]})
        # Save variable to the Dataset
        if ds_out is None:
            ds_out = xr.Dataset({var_out[v]:data_out})
        else:
            ds_out = ds_out.assign({var_out[v]:data_out})
    ds.close()
    return ds_out
            
        

    

