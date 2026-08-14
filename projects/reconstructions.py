import xesmf as xe
import xarray as xr
import numpy as np
import warnings
warnings.filterwarnings('ignore', module='xesmf')

from ..grid import Grid, CAMGrid
from ..file_io import read_binary

# Read one LENS historical ensemble member and save monthly averages of the forcing to NetCDF (corrected and uncorrected; MITgcm grid)
# Before running this on BAS workstations, do "conda activate xe"
def forcing_to_netcdf (ens, out_dir='./'):

    in_dir = '/data/oceans_input/processed_input_data/CESM/LENS/'
    grid_dir = '/data/oceans_output/shelf/kaight/mitgcm/PAS_grid/'
    start_year = 1920
    end_year = 2005
    var_names = ['TREFHT', 'QBOT', 'PRECT', 'PSL', 'UBOT', 'VBOT', 'FSDS', 'FLDS']
    bias_dir = '/data/oceans_output/shelf/kaight/ics_obcs/PAS/'
    var_names_bias = ['atemp', 'aqh', 'precip', None, None, None, 'swdown', 'lwdown']
    file_tail_bias = '_offset_PAS'
    var_names_wind_bias = ['rotate', 'scale']
    file_head_wind_bias = 'katabatic_'
    file_tail_wind_bias = '_PAS_90W'
    units = ['degC', '1', 'm/s', 'Pa', 'm/s', 'm/s', 'W/m^2', 'W/m^2']
    long_names = ['Near-surface air temperature', 'Near-surface specific humidity', 'Precipitation', 'Pressure at sea level', 'Near-surface u-wind', 'Near-surface v-wind', 'Surface downwelling shortwave', 'Surface downwelling longwave']

    # Read grids and compute regridding weights
    cesm_grid = CAMGrid()
    cesm_lon, cesm_lat = cesm_grid.get_lon_lat(dim=1)
    ds_cesm_grid = xr.Dataset({'lat':(['lat'],cesm_lat), 'lon':(['lon'],cesm_lon)})
    mit_grid = Grid(grid_dir)
    ds_mit_grid = xr.Dataset({'lat':(['lat'],mit_grid.lat_1d), 'lon':(['lon'],mit_grid.lon_1d)})
    regridder = xe.Regridder(ds_cesm_grid, ds_mit_grid, method='bilinear', periodic=True)
    land_mask = xr.DataArray(mit_grid.land_mask+mit_grid.ice_mask, dims=ds_mit_grid.dims, coords=ds_mit_grid.coords)

    # Read bias correction fields
    ds_bias = None
    def read_add_bias (ds_bias, var, file_path, factor=1):
        data = read_binary(file_path, [mit_grid.nx, mit_grid.ny], 'xy', prec=64)*factor
        data = xr.DataArray(data, coords={'lat':mit_grid.lat_1d, 'lon':mit_grid.lon_1d})
        if ds_bias is None:
            ds_bias = xr.Dataset({var:data})
        else:
            ds_bias = ds_bias.assign({var:data})
        return ds_bias
    for var in var_names_bias:
        if var is not None:
            file_path = bias_dir+var+file_tail_bias
            if var in ['swdown', 'lwdown']:
                # Swap sign on radiation fluxes for MITgcm
                ds_bias = read_add_bias(ds_bias, var, file_path, factor=-1)
            else:
                ds_bias = read_add_bias(ds_bias, var, file_path)                        
    for var in var_names_wind_bias:
        file_path = bias_dir+file_head_wind_bias+var+file_tail_wind_bias
        ds_bias = read_add_bias(ds_bias, var, file_path)

    # Read all the LENS files
    ds_out = None
    ds_out_corr = None
    for year in range(start_year, end_year):
        print('Processing '+str(year))
        # Create daily time axis
        time_daily = xr.date_range(start=str(year)+'-01-01', end=str(year)+'-12-31', freq='D', calendar='noleap', use_cftime=True)
        # Also need monthly time axis for radiation
        time_monthly = xr.date_range(start=str(year)+'-01', end=str(year)+'-12', freq='MS', calendar='noleap', use_cftime=True)
        ds_year = None
        ds_year_corr = None
        for n in range(len(var_names)):
            file_path = in_dir+'LENS_ens'+str(ens).zfill(3)+'_'+var_names[n]+'_'+str(year)
            data = read_binary(file_path, [cesm_grid.nx, cesm_grid.ny], 'xyt')
            if var_names[n] in ['FSDS', 'FLDS']:
                # Monthly data
                data = xr.DataArray(data, coords={'time':time_monthly, 'lat':cesm_lat, 'lon':cesm_lon}).assign_attrs(long_name=long_names[n], units=units[n])
            else:
                # Daily data
                data = xr.DataArray(data, coords={'time':time_daily, 'lat':cesm_lat, 'lon':cesm_lon}).assign_attrs(long_name=long_names[n], units=units[n])
            # Interpolate to MITgcm grid
            data = regridder(data)
            # Mask land
            data = data.where(~land_mask)
            if var_names[n] == 'UBOT':
                # Save for VBOT
                data_u = data
            elif var_names[n] == 'VBOT':
                # Bias-correct winds before time-averaging
                data_v = data
                magnitude = np.sqrt(data_u**2 + data_v**2)
                angle = np.arctan2(data_v, data_u)
                data_u_corr = magnitude*ds_bias['scale']*np.cos(angle + ds_bias['rotate'])
                data_v_corr = magnitude*ds_bias['scale']*np.sin(angle + ds_bias['rotate'])
            if var_names[n] not in ['FSDS', 'FLDS']:
                # Monthly average
                data = data.resample(time='1MS').mean()
            # Save to dataset for this year
            if ds_year is None:
                ds_year = xr.Dataset({var_names[n]:data})
            else:
                ds_year = ds_year.assign({var_names[n]:data})
            # Now bias-correct
            if var_names[n] == 'UBOT':
                # Processed in VBOT
                pass
            elif var_names[n] == 'VBOT':
                # Already corrected; time-average both u and v then save
                data_u_corr = data_u_corr.resample(time='1MS').mean()
                data_v_corr = data_v_corr.resample(time='1MS').mean()
                # ds_year_corr will already exist from a previous variable
                ds_year_corr = ds_year_corr.assign({'UBOT':data_u_corr, 'VBOT':data_v_corr})
            else:
                # Constant correction
                if var_names_bias[n] is None:
                    # No bias correction for this variable
                    data_corr = data
                else:
                    data_corr = data + ds_bias[var_names_bias[n]]
                if ds_year_corr is None:
                    ds_year_corr = xr.Dataset({var_names[n]:data_corr})
                else:
                    ds_year_corr = ds_year_corr.assign({var_names[n]:data_corr})
        # Concatenate with main datasets
        if ds_out is None:
            ds_out = ds_year
            ds_out_corr = ds_year_corr
        else:
            ds_out = xr.concat([ds_out, ds_year], dim='time')
            ds_out_corr = xr.concat([ds_out_corr, ds_year_corr], dim='time')
    # Save to file
    ds_out.to_netcdf(out_dir+'LENS_ens'+str(ens).zfill(3)+'_AmundsenSea_'+str(start_year)+'-'+str(end_year)+'_monthly.nc', unlimited_dims='time')
    ds_out_corr.to_netcdf(out_dir+'LENS_ens'+str(ens).zfill(3)+'_AmundsenSea_'+str(start_year)+'-'+str(end_year)+'_monthly_corrected.nc', unlimited_dims='time')
    
    

    

