import xarray as xr
from ..grid import Grid

def process_expt (expt_dir, historical=False):

    # Set file naming conventions
    domain_id = 'AMU_04'  # Amundsen Sea approx 4km
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
        file_path = expt_dir+'/output/MITgcm/'+str(year)+'01/output.nc'
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
        # TODO: confirm if EOS80 is ok or if I need to convert to TEOS10
        ds_out = xr.Dataset({'lon':lon, 'lat':lat, 'zos':ds['ETAN'], 'tos':ds['THETA'].isel(Z=0), 'sos':ds['SALT'].isel(Z=0), 'siconc':ds['SIarea'], 'areacello':ds['rA'], 'deptho':bathy})
        # Now fix the dimensions
        ds_out = ds_out.rename({'XC':'x', 'YC':'y'})
        time_new = 
        
        
        
