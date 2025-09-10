import xarray as xr
import numpy as np
import os
import gc

from ..grid import ISMIP7Grid, Grid
from ..interpolation import interp_reg_xy, extend_into_mask
from ..utils import convert_ismr, days_per_month
from ..constants import months_per_year
from ..file_io import read_netcdf

# Interpolate the output of WSFRIS 2021 paper (two_timescale.py) and PAS 2023 paper (scenarios.py) to the ISMIP7 grid for sharing

def interp_year (file_path, calendar='noleap'):

    var_in = ['THETA', 'SALT', 'SHIfwFlx']
    var_out = ['temperature', 'salinity', 'basal_melt']
    units = ['degC', 'psu', 'm/y']
    long_name = ['potential temperature (EOS80)', 'practical salinity (EOS80)', 'ice shelf basal melt rate, positive means melting']
    fill_value = 9999
    
    grid_in = Grid(file_path)
    grid_out = ISMIP7Grid()

    # Prepare for annual averaging
    if calendar == '360-day':
        ndays = [30]*months_per_year
    elif calendar == 'noleap':
        ndays = [days_per_month(month+1, 1979, allow_leap=False) for month in range(months_per_year)]
    else:
        raise Exception('Unsupported calendar '+calendar)

    # Inner function to interpolate a variable to the ISMIP7 grid
    def interp_var(data_in, is_3d=False):
        if is_3d:
            # 3D variable - interpolate once for each depth index
            data_out = np.empty([grid_in.nz, grid_out.ny, grid_out.nx])
            for k in range(grid_in.nz):
                data_out[k,:] = interp_reg_xy(grid_in.lon_1d, grid_in.lat_1d, data_in[k,:], grid_out.lon, grid_out.lat, fill_value=fill_value)
            data_out = xr.DataArray(data_out, coords={'z':grid_in.z, 'y':grid_out.y, 'x':grid_out.x})
        else:
            # 2D variable - interpolate once
            data_out = interp_reg_xy(grid_in.lon_1d, grid_in.lat_1d, data_in, grid_out.lon, grid_out.lat, fill_value=fill_value)
            data_out = xr.DataArray(data_out, coords={'y':grid_out.y, 'x':grid_out.x})
        data_out = data_out.where(data_out != fill_value)
        return data_out

    # Interpolate masks
    land_mask = np.round(interp_var(grid_in.land_mask))
    ice_mask = np.round(interp_var(grid_in.ice_mask))
    mask_3d = np.round(interp_var(grid_in.hfac!=0, is_3d=True))

    ds_out = None
    for v in range(len(var_in)):
        data_in = read_netcdf(file_path, var_in[v])
        # Annual average
        data_in = np.average(data_in, axis=0, weights=ndays)
        is_3d = len(data_in.shape)==3
        if var_in[v] == 'SHIfwFlx':
            # Unit conversion for ice shelf melting
            data_in = convert_ismr(data_in)
        # Extend into mask a few times to prevent interpolation artifacts near coast
        if is_3d:
            data_in = np.ma.masked_where(grid_in.hfac==0, data_in)
        elif var_in[v] == 'SHIfwFlx':
            data_in = np.ma.masked_where(grid_in.ice_mask==0, data_in)
        else:
            data_in = np.ma.masked_where(grid_in.land_mask, data_in)
        data_in = extend_into_mask(data_in, masked=True, use_3d=is_3d, num_iters=3)
        # Interpolate to model grid
        data_out = interp_var(data_in, is_3d=is_3d)
        # Attach attributes
        data_out = data_out.assign_attrs(description=long_name[v], units=units[v])
        # Mask as needed
        if is_3d:
            data_out = data_out.where(mask_3d>0)
        elif var_out[v] == 'basal_melt':
            data_out = data_out.where(ice_mask>0)
        else:
            data_out = data_out.where(land_mask<0)
        # Save variable to the Dataset
        if ds_out is None:
            ds_out = xr.Dataset({var_out[v]:data_out})
        else:
            ds_out = ds_out.assign({var_out[v]:data_out})
        del data_in
    del grid_in
    gc.collect()
    return ds_out


# Process one ensemble member of 2023 Amundsen Sea MITgcm simulations (10 ensemble members)
def process_PAS (ens, out_dir='./'):

    in_dir = '/gws/nopw/j04/bas_pog/kaight/CESM_scenarios/'
    dir_head = 'PAS_LENS'
    dir_mid = '_O/output/'
    file_tail = '01/MITgcm/output.nc'
    calendar = 'noleap'
    ens = int(ens)
    start_year = 2006
    end_year = 2100

    out_file = out_dir + 'MITgcm_ASE_RCP85_ens' + str(ens).zfill(2) + '.nc'
    for year in range(start_year, end_year+1):
        in_file = in_dir + dir_head + str(ens).zfill(3) + dir_mid + str(year) + file_tail
        ds = interp_year(in_file, calendar=calendar).expand_dims({'time':[year]})
        if os.path.isfile(out_file):
            ds_old = xr.open_dataset(out_file)
            ds = xr.concat([ds_old, ds], dim='time')
            ds_old.close()
        ds.to_netcdf(out_file, mode='w')
        ds.close()


# Process one experiment of 2021 Weddell Sea MITgcm simulations ('abrupt-4xCO2' or '1pctCO2').
def process_WSFRIS (expt, out_dir='./'):

    if expt == 'abrupt-4xCO2':
        sim_name = 'WSFRIS_abIO'
    elif expt == '1pctCO2':
        sim_name = 'WSFRIS_1pIO'
    else:
        raise Exception('Invalid experiment '+expt)
    in_dir = '/gws/nopw/j04/bas_pog/kaight/2timescale/'
    dir_head = sim_name + '/output/'
    file_tail = '01/MITgcm/output.nc'
    calendar = '360-day'
    start_year = 1850
    end_year = 2049

    out_file = out_dir + 'MITgcm_WS_'+expt+'.nc'
    for year in range(start_year, end_year+1):
        in_file = in_dir + dir_head + str(year) + file_tail
        ds = interp_year(in_file, calendar=calendar).expand_dims({'time':[year]})
        if os.path.isfile(out_file):
            ds_old = xr.open_dataset(out_file)
            ds = xr.concat([ds_old, ds], dim='time')
            ds_old.close()
        ds.to_netcdf(out_file, mode='w')
        ds.close()

                
    
            
        

    

