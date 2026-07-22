import xarray as xr
import cftime
import numpy as np
from ..grid import Grid

def process_expt (expt_dir, out_dir='output/', historical=False):

    missval = np.float32(1e20)
    # Dictionary of dictionaries of standard variable attributes
    var_attrs = {'lon': {'standard_name':'longitude',
                         'long_name':'longitude',
                         'units':'degrees_east'},
                 'lat': {'standard_name':'latitude',
                         'long_name':'latitude',
                         'units':'degrees_north'},
                 'zos': {'standard_name':'sea_surface_height_above_geoid',
                         'long_name':'Sea Surface Height Above Geoid',
                         'units':'m',
                         'cell_methods':'area: mean where sea time: mean',
                         'missing_value':missval,
                         'coordinates':'lat lon'},
                 'tos': {'standard_name':'sea_surface_temperature',
                         'long_name':'Sea Surface Temperature',
                         'units':'degC',
                         'cell_methods':'area: mean where sea time: mean',
                         'missing_value':missval,
                         'coordinates':'lat lon'},
                 'sos': {'standard_name':'sea_surface_salinity',
                         'long_name':'Sea Surface Salinity',
                         'units':'0.001',
                         'cell_methods':'area: mean where sea time: mean',
                         'missing_value':missval,
                         'coordinates':'lat lon'},
                 'siconc': {'standard_name':'sea_ice_area_fraction',
                            'long_name':'Sea-Ice Area Percentage (Ocean Grid)',
                            'units':'%',
                            'cell_methods':'area: mean where sea time: mean',
                            'missing_value':missval,
                            'coordinates':'lat lon'},
                 'areacello': {'standard_name':'cell_area',
                               'long_name':'Grid-Cell Area for Ocean Variables',
                               'units':'m^2',
                               'cell_methods':'area: sum',
                               'coordinates':'lat lon'},
                 'deptho': {'standard_name':'sea_floor_depth_below_geoid',
                           'long_name':'Sea Floor Depth Below Geoid',
                           'units':'m',
                           'cell_methods':'area: mean where sea',
                           'missing_value':missval,
                           'coordinates':'lat lon'}
                 }
    

    # Set file naming conventions
    domain_id = 'ANT_04'  # Part of Antarctica, approx 4km
    driving_source_id = 'CESM1'
    if 'LW1.5' in expt_dir:
        expt_name = 'LW1.5_' # Just for extracting ensemble member later
        driving_experiment_id = 'stab-1-5-deg'
    elif 'LW2.0' in expt_dir:
        expt_name = 'LW2.0_'
        driving_experiment_id = 'stab-2deg'
    elif 'MENS' in expt_dir:
        expt_name = 'MENS_'
        driving_experiment_id = 'RCP45'
    elif 'LENS' in expt_dir:
        expt_name = 'LENS'
        if historical:
            driving_experiment_id = 'historical'
        else:
            driving_experiment_id = 'RCP85'
    i = expt_dir.index(expt_name)+len(expt_name)
    driving_variant_label = 'ens'+expt_dir[i:i+3]
    institution_id = 'BAS'
    source_id = 'BAS-MITgcm-AS'
    version_realization = 'v1-r1'
    frequency = 'mon'

    bathy = None  # Will calculate once later

    # Get range of years to process
    if expt_name == 'LENS' and historical:
        start_year = 1920
    else:
        start_year = 2006
    if 'MENS' in expt_dir:
        end_year = 2080
    else:
        end_year = 2100

    # Loop over years
    for year in range(start_year, end_year+1):
        file_path = expt_dir+'/output/'+str(year)+'01/MITgcm/output.nc'
        print('Processing '+file_path)
        ds = xr.open_dataset(file_path)
        if bathy is None:
            grid = Grid(file_path)
            bathy = -1*grid.bathy
            # Make it a DataArray
            bathy = xr.DataArray(data=bathy, dims=ds['rA'].dims, coords=ds['rA'].coords)
            # Also prepare 2D lon and lat arrays
            lat, lon = xr.broadcast(ds['YC'], ds['XC'])
        # Select the variables we want
        ds_out = xr.Dataset({'lon':lon, 'lat':lat, 'zos':ds['ETAN'], 'tos':ds['THETA'].isel(Z=0), 'sos':ds['SALT'].isel(Z=0), 'siconc':ds['SIarea'], 'areacello':ds['rA'], 'deptho':bathy})
        # Unit conversions
        ds_out['siconc'] = ds_out['siconc']*1e2
        # Fill land mask (identically zero) with missing value
        for var in ds_out:
            if var in ['lon', 'lat', 'areacello']:
                continue
            ds_out[var] = xr.where(ds['SALT'].isel(Z=0,time=0)==0, missval, ds_out[var])
        # Now fix the dimensions
        ds_out = ds_out.rename({'XC':'x', 'YC':'y'})
        # Time should be midpoints of each month
        time_new = []
        for month in range(1, 12+1):
            time_start = cftime.DatetimeNoLeap(year, month, 1)
            if month == 12:
                time_end = cftime.DatetimeNoLeap(year+1, 1, 1)
            else:
                time_end = cftime.DatetimeNoLeap(year, month+1, 1)
            time_new.append(time_start + (time_end - time_start)/2)
        if expt_name=='LENS' and historical:
            time_units = 'days since 1850-01-01'
        else:
            time_units = 'days since 1950-01-01'
        time_new = xr.DataArray(cftime.date2num(time_new, units=time_units, calendar='365_day'), dims=['time'], attrs={'units':time_units, 'calendar':'365_day', 'long_name':'time'})
        ds_out = ds_out.assign_coords(time=time_new)        
        # Loop over the other variables and overwrite their attributes
        for var in var_attrs:
            ds_out[var].attrs = var_attrs[var]            
        # Now loop over data variables and write to file
        for var in ds_out:
            if var in ['lon', 'lat', 'time']:
                continue
            data = ds_out[var].assign_coords({'lon':ds_out['lon'], 'lat':ds_out['lat']})
            for coord in data.coords:
                if coord not in ['time', 'y', 'x', 'lat', 'lon']:
                    data = data.drop_vars(coord)
            encoding = {var:{'_FillValue':missval}}
            for coord_name in data.coords:
                encoding[coord_name] = {'_FillValue':None}
            # Construct standard file name
            file_path = out_dir+var+'_'+domain_id+'_'+driving_source_id+'_'+driving_experiment_id+'_'+driving_variant_label+'_'+institution_id+'_'+source_id+'_'+version_realization+'_'+frequency+'_'+str(year)+'01-'+str(year)+'12.nc'
            print('Writing '+file_path)            
            data.to_netcdf(file_path, encoding=encoding, unlimited_dims=('time' if 'time' in data.coords else None))
            
            
        
        
        
        
        
