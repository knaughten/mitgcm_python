from ..file_io import netcdf_time

bash_script = 'extract_pdc.sh'
main_outdir = 'archive_for_grl/'
timeseries_dir = 'timeseries/'
hovmoller_dir = 'hovmollers/'
latlon_dir = 'latlon/'
trend_dir = 'ensemble_trends/'
num_ens = 20
pace_indir = ['PAS_PACE'+str(n+1).zfill(2)+'/output/' for n in range(num_ens)]
pace_outdir = ['PACE'+str(n+1).zfill(2)+'/' for n in range(num_ens)]
era5_indir = 'PAS_ERA5/output/'
era5_outdir = 'ERA5/'
trend_indir = 'precomputed_trends/'
pace_spinup_year0 = 1890
pace_start_year = 1920
pace_end_year = 2013
months_per_year = 12
era5_spinup_year0 = 1947
era5_start_year = 1979
era5_seaice_start_year = 1988
era5_end_year = 2019
dir_used = []

f = open(bash_script, 'w')
f.write('#!/bin/bash\n')
f.write('set -ex\n')
f.write('export HDF5_USE_FILE_LOCKING=FALSE\n')
f.write('mkdir '+main_outdir+'\n')
dir_used.append(main_outdir)

def get_region (var_name):
    if var_name.startswith('amundsen_shelf_break_'):
        return 'AmundsenSeaContinentalShelfBreak', -115, -102, -71.8, -70.2
    elif var_name.startswith('amundsen_shelf_'):
        return 'AmundsenSeaContinentalShelf', -115, -100, -75.5, -70
    elif var_name.startswith('amundsen_west_shelf_break_'):
        return 'PineIslandThwaitesWestTrough', -115, -112, -72, -71
    elif var_name.startswith('pine_island_bay_'):
        return 'PineIslandBay', -104, -100.5, -75.2, -74.2
    elif var_name.startswith('dotson_bay_'):
        return 'InFrontOfDotsonIceShelf', -114, -110.5, -74.3, -73.5
    elif var_name.startswith('dotson_to_cosgrove_'):
        return 'IceShelvesFromDotsonToCosgrove', -114.6, -98.5, -75.5, -73.1
    elif var_name.startswith('pig_'):
        return 'PineIslandIceShelf', -103, -99, -75.4, -74
    elif var_name.startswith('dotson_'):
        return 'DotsonIceShelf', -114.6, -111.2, -75.3, -74.1
    elif var_name.startswith('getz_'):
        return 'GetzIceShelf', -135, -114.7, -75.2, -73.5
    elif var_name.startswith('crosson_'):
        return 'CrossonIceShelf', -111.2, -109, -75.4, -74.1
    elif var_name.startswith('thwaites_'):
        return 'ThwaitesIceShelf', -109, -103, -75.4, -74.6
    elif var_name.startswith('cosgrove_'):
        return 'CosgroveIceShelf', -102, -98.5, -73.8, -73.2
    elif var_name.startswith('abbot_'):
        return 'AbbotIceShelf', -104, -88.9, -73.4, -71.5
    elif var_name.startswith('venable_'):
        return 'VenableIceShelf', -88.9, -85.6, -73.4, -72.7
    else:
        return 'AmundsenSea', -140, -80, -76, -62

def add_attributes (fname, sim, lon_min, lon_max, lat_min, lat_max, zmin, zmax):
    if sim == 'pace':
        f.write('ncatted -a title,global,o,c,"Amundsen Sea MITgcm model output forced with Pacific Pacemaker Ensemble, 1920-2013" '+fname+'\n')
    elif sim == 'era5':
        f.write('ncatted -a title,global,o,c,"Amundsen Sea MITgcm model output forced with ERA5 reanalysis, 1979-2019" '+fname+'\n')
    f.write('ncatted -a summary,global,o,c,"This dataset provides model output for 20th-century ice-ocean simulations in the Amundsen Sea. The simulations are performed with the MITgcm model at 1/10 degree resolution, including components for the ocean, sea ice, and ice shelf thermodynamics. Atmospheric forcing is provided by the CESM Pacific Pacemaker Ensemble, using 20 members from 1920-2013. An additional simulation is forced with the ERA5 atmospheric reanalysis from 1979-2019. The simulations were completed in 2021 by Kaitlin Naughten at the British Antarctic Survey (Polar Oceans team)." '+fname+'\n')
    f.write('ncatted -a keywords,global,o,c,"Antarctica,Amundsen Sea,ocean,ice shelves,sea ice,modelling" '+fname+'\n')
    f.write('ncatted -a Conventions,global,o,c,"ACDD-1.3" '+fname+'\n')
    f.write('ncatted -a source,global,o,c,"MITgcm version 67s" '+fname+'\n')
    f.write('ncatted -a creator_name,global,o,c,"Kaitlin Naughten" '+fname+'\n')
    f.write('ncatted -a creator_email,global,o,c,"kaight@bas.ac.uk" '+fname+'\n')
    f.write('ncatted -a institution,global,o,c,"British Antarctic Survey" '+fname+'\n')
    f.write('ncatted -a project,global,o,c,"UKRI Fund for International Collaboration NE/S011994/1" '+fname+'\n')
    f.write('ncatted -a geospatial_lat_min,global,o,f,'+str(lat_min)+' '+fname+'\n')
    f.write('ncatted -a geospatial_lat_max,global,o,f,'+str(lat_max)+' '+fname+'\n')
    f.write('ncatted -a geospatial_lon_min,global,o,f,'+str(lon_min)+' '+fname+'\n')
    f.write('ncatted -a geospatial_lon_max,global,o,f,'+str(lon_max)+' '+fname+'\n')
    if zmin is not None and zmax is not None:
        f.write('ncatted -a geospatial_vertical_positive,global,o,c,"down" '+fname+'\n')
        f.write('ncatted -a geospatial_vertical_units,global,o,c,"m" '+fname+'\n')
        if zmin is not None:
            f.write('ncatted -a geospatial_vertical_min,global,o,f,'+str(zmin)+' '+fname+'\n')
        if zmax is not None:
            f.write('ncatted -a geospatial_vertical_max,global,o,f,'+str(zmax)+' '+fname+'\n')
    if sim == 'pace':
        f.write('ncatted -a time_coverage_start,global,o,c,1920-01-01 '+fname+'\n')
        f.write('ncatted -a time_coverage_end,global,o,c,2013-12-31 '+fname+'\n')
    elif sim == 'era5':
        f.write('ncatted -a time_coverage_start,global,o,c,1979-01-01 '+fname+'\n')
        f.write('ncatted -a time_coverage_end,global,o,c,2019-12-31 '+fname+'\n')    

# Timeseries
f.write('mkdir '+main_outdir+timeseries_dir+'\n')
dir_used.append(main_outdir+timeseries_dir)
def process_timeseries_var (var_name, sim='pace', ts_file='timeseries_final.nc'):
    print('Processing '+var_name+' for '+sim)
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    dir_tmp = main_outdir+timeseries_dir+region+'/'
    if dir_tmp not in dir_used:
        f.write('mkdir '+dir_tmp+'\n')
        dir_used.append(dir_tmp)
    if var_name.endswith('below_100m'):
        depth = 'below_100m'
        zmin = None
        zmax = -100
    elif var_name.endswith('below_200m'):
        depth = 'below_200m'
        zmin = None
        zmax = -200
    elif var_name.endswith('200_700m'):
        depth = '200-700m'
        zmin = -700
        zmax = -200
    elif var_name.endswith('below_700m'):
        depth = 'below_700m'
        zmin = None
        zmax = -700
    else:
        depth = '2D'
        zmin = None
        zmax = None
    if depth in ['below_200m', '200-700m', 'below_700m']:
        depth_str = '_'+depth+'_'
        dir_tmp += depth+'/'
        if dir_tmp not in dir_used:
            f.write('mkdir '+dir_tmp+'\n')
            dir_used.append(dir_tmp)
    else:
        depth_str = '_'
    if 'uwind_avg' in var_name:
        new_var = 'eastward_wind'
    elif 'temp' in var_name:
        new_var = 'sea_water_potential_temperature'
    elif 'massloss' in var_name:
        new_var = 'ice_shelf_basal_melt_flux'
    elif 'salt' in var_name:
        new_var = 'sea_water_salinity'
    elif 'advection_heat_xy' in var_name:
        new_var = 'ocean_heat_convergence_due_to_horizontal_advection'
    elif 'advection_heat_z' in var_name:
        new_var = 'ocean_heat_convergence_due_to_vertical_advection'
    elif 'diffusion_heat_implicit_z' in var_name:
        new_var = 'ocean_heat_convergence_due_to_vertical_diffusion'
    elif 'kpp_heat_z' in var_name:
        new_var = 'ocean_heat_convergence_due_to_parameterised_vertical_mixing'
    elif 'shortwave_penetration' in var_name:
        new_var = 'ocean_heat_convergence_due_to_shortwave_flux'
    elif 'isotherm_0.5C' in var_name:
        new_var = 'depth_of_0.5C_isotherm'
    elif 'isotherm_0C' in var_name:
        new_var = 'depth_of_0C_isotherm'
    elif 'isotherm_-1C' in var_name:
        new_var = 'depth_of_-1C_isotherm'
    if sim == 'pace':
        for n in range(num_ens):
            dir_tmp_pace = dir_tmp+pace_outdir[n]
            if dir_tmp_pace not in dir_used:
                f.write('mkdir '+dir_tmp_pace+'\n')
                dir_used.append(dir_tmp_pace)
            fname_in = pace_indir[n]+ts_file
            fname_out = dir_tmp_pace+'MITgcm_'+region+'_1920-2013_'+new_var+depth_str+'PACE'+str(n+1).zfill(2)+'.nc'
            # Extract the years 1920-2013 into the new file
            first_year = netcdf_time(fname_in, monthly=False)[0].year
            if first_year == pace_start_year:
                f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            elif first_year == pace_spinup_year0:
                # Need to trim the spinup
                t_start = (pace_start_year-pace_spinup_year0)*months_per_year
                t_end = (pace_end_year-pace_spinup_year0+1)*months_per_year-1
                f.write('ncks -d time,'+str(t_start)+','+str(t_end)+' -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            # Now rename the variable
            f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
            # Add some attributes
            add_attributes(fname_out, 'pace', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
            f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')            
    if sim == 'era5':
        dir_tmp_era5 = dir_tmp+era5_outdir
        if dir_tmp_era5 not in dir_used:
            f.write('mkdir '+dir_tmp_era5+'\n')
            dir_used.append(dir_tmp_era5)
        fname_in = era5_indir+ts_file
        fname_out = dir_tmp_era5+'MITgcm_'+region+'_1979-2019_'+new_var+depth_str+'ERA5.nc'
        # Extract the years 1979-2019 into the new file
        first_year = netcdf_time(fname_in, monthly=False)[0].year
        if first_year == era5_start_year:
            f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
        elif first_year == era5_spinup_year0:
            t_start = (era5_start_year-era5_spinup_year0)*months_per_year
            t_end = (era5_end_year-era5_spinup_year0+1)*months_per_year-1
            f.write('ncks -d time,'+str(t_start)+','+str(t_end)+' -v '+var_name+' '+fname_in+' '+fname_out+'\n')
        # Rename the variable
        f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
        # Add some attributes
        add_attributes(fname_out, 'era5', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
        f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

# PACE ensemble
for var in ['amundsen_shelf_break_uwind_avg', 'amundsen_shelf_temp_btw_200_700m', 'dotson_to_cosgrove_massloss', 'amundsen_shelf_salt_btw_200_700m']:
    process_timeseries_var(var)
for var in ['amundsen_shelf_advection_heat_xy_below_200m', 'amundsen_shelf_advection_heat_z_below_200m', 'amundsen_shelf_diffusion_heat_implicit_z_below_200m', 'amundsen_shelf_kpp_heat_z_below_200m', 'amundsen_shelf_shortwave_penetration_below_200m']:
    process_timeseries_var(var, ts_file='timeseries_heat_budget.nc')
for var in ['amundsen_shelf_isotherm_0.5C_below_100m', 'pine_island_bay_isotherm_0C_below_100m', 'dotson_bay_isotherm_-1C_below_100m']:
    process_timeseries_var(var, ts_file='timeseries_isotherm.nc')
# ERA5
for var in ['amundsen_shelf_break_uwind_avg', 'amundsen_shelf_temp_btw_200_700m', 'dotson_to_cosgrove_massloss', 'pine_island_bay_isotherm_0C_below_100m', 'dotson_bay_isotherm_-1C_below_100m', 'pine_island_bay_temp_below_700m', 'dotson_bay_temp_below_700m', 'pig_massloss', 'dotson_massloss', 'getz_massloss', 'crosson_massloss', 'thwaites_massloss', 'cosgrove_massloss', 'abbot_massloss', 'venable_massloss']:
    process_timeseries_var(var, sim='era5')

# Hovmollers
f.write('mkdir '+main_outdir+hovmoller_dir+'\n')
dir_used.append(main_outdir+hovmoller_dir)
def process_hovmoller_var (var_name, sim='pace', hov_file='hovmoller.nc'):
    print('Processing '+var_name+' for '+sim)
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    zmin = None
    zmax = None
    dir_tmp = main_outdir+hovmoller_dir+region+'/'
    if dir_tmp not in dir_used:
        f.write('mkdir '+dir_tmp+'\n')
        dir_used.append(dir_tmp)
    if 'temp' in var_name:
        new_var = 'sea_water_potential_temperature'
    elif 'salt' in var_name:
        new_var = 'sea_water_salinity'
    if sim == 'pace':
        for n in range(num_ens):
            dir_tmp_pace = dir_tmp+pace_outdir[n]
            if dir_tmp_pace not in dir_used:
                f.write('mkdir '+dir_tmp_pace+'\n')
                dir_used.append(dir_tmp_pace)
            fname_in = pace_indir[n]+hov_file
            fname_out = dir_tmp_pace+'MITgcm_'+region+'_1920-2013_'+new_var+'_PACE'+str(n+1).zfill(2)+'.nc'
            first_year = netcdf_time(fname_in, monthly=False)[0].year
            if first_year == pace_start_year:
                f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            elif first_year == pace_spinup_year0:
                # Need to trim the spinup
                t_start = (pace_start_year-pace_spinup_year0)*months_per_year
                t_end = (pace_end_year-pace_spinup_year0+1)*months_per_year-1
                f.write('ncks -d time,'+str(t_start)+','+str(t_end)+' -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            # Now rename the variable
            f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
            # Add some attributes
            add_attributes(fname_out, 'pace', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
            f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')
    elif sim == 'era5':
        dir_tmp_era5 = dir_tmp+era5_outdir
        if dir_tmp_era5 not in dir_used:
            f.write('mkdir '+dir_tmp_era5+'\n')
            dir_used.append(dir_tmp_era5)
        fname_in = era5_indir+hov_file
        fname_out = dir_tmp_era5+'MITgcm_'+region+'_1979-2019_'+new_var+'_ERA5.nc'
        # Extract the years 1979-2019 into the new file
        first_year = netcdf_time(fname_in, monthly=False)[0].year
        if first_year == era5_start_year:
            f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
        elif first_year == era5_spinup_year0:
            t_start = (era5_start_year-era5_spinup_year0)*months_per_year
            t_end = (era5_end_year-era5_spinup_year0+1)*months_per_year-1
            f.write('ncks -d time,'+str(t_start)+','+str(t_end)+' -v '+var_name+' '+fname_in+' '+fname_out+'\n')
        # Rename the variable
        f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
        # Add some attributes
        add_attributes(fname_out, 'era5', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
        f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

# PACE ensemble
hov_var = ['amundsen_shelf_temp', 'amundsen_shelf_break_temp', 'pine_island_bay_temp', 'dotson_bay_temp']
hov_fname = ['hovmoller3.nc', 'hovmoller3.nc', 'hovmoller1.nc', 'hovmoller2.nc']
for n in range(len(hov_var)):
    process_hovmoller_var(hov_var[n], hov_file=hov_fname[n])
# ERA5
for var in ['amundsen_west_shelf_break_temp', 'amundsen_west_shelf_break_salt', 'pine_island_bay_temp', 'pine_island_bay_salt', 'dotson_bay_temp', 'dotson_bay_salt']:
    process_hovmoller_var(var, sim='era5')

# Lat-lon fields for ERA5 (1988-2019)
f.write('mkdir '+main_outdir+latlon_dir+'\n')
dir_used.append(main_outdir+latlon_dir)
def process_latlon_var_era5 (var_name):
    print('Processing '+var_name+' for ERA5')
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    zmin = None
    zmax = None
    if var_name == 'SIarea':
        new_var = 'sea_ice_area_fraction'
    elif var_name == 'SIheff':
        new_var = 'sea_ice_thickness'
    dir_tmp = main_outdir+latlon_dir+'ERA5/'
    if dir_tmp not in dir_used:
        f.write('mkdir '+dir_tmp+'\n')
        dir_used.append(dir_tmp)
    # Process one year at a time
    for year in range(era5_seaice_start_year, era5_end_year+1):
        fname_in = era5_indir+str(year)+'01/MITgcm/output.nc'
        fname_out = dir_tmp+'MITgcm_'+region+'_'+str(year)+'_'+new_var+'_ERA5.nc'
        # Extract the variable
        f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
        # Rename it
        f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
        # Add some attributes
        add_attributes(fname_out, 'era5', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
        f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

for var in ['SIarea', 'SIheff']:
    process_latlon_var_era5(var)

# Trends for ensemble, precomputed over 1920-2013
f.write('mkdir '+main_outdir+trend_dir+'\n')
dir_used.append(main_outdir+trend_dir)
def process_trend_var (var_name):
    print('Processing '+var_name)
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    zmin = None
    zmax = None
    dir_tmp = main_outdir+trend_dir
    if var_name == 'ADVx_TH':
        new_var = 'eastward_ocean_heat_transport_due_to_advection'
    elif var_name == 'ADVy_TH':
        new_var = 'northward_ocean_heat_transport_due_to_advection'
    elif var_name == 'advection_3d':
        new_var = 'ocean_heat_convergence_due_to_advection'
    elif var_name == 'diffusion_kpp':
        new_var = 'ocean_heat_convergence_due_to_diffusion_and_parameterised_vertical_mixing'
    elif var_name == 'shortwave_pen':
        new_var = 'ocean_heat_convergence_due_to_shortwave_flux'
    elif var_name == 'hb_total':
        new_var = 'ocean_heat_convergence_in_interior'
    elif var_name == 'EXFuwind':
        new_var = 'eastward_wind'
    elif var_name == 'EXFvwind':
        new_var = 'northward_wind'
    elif var_name == 'EXFatemp':
        new_var = 'air_temperature'
    elif var_name == 'EXFaqh':
        new_var = 'specific_humidity'
    elif var_name == 'EXFpreci':
        new_var = 'precipitation_flux'
    elif var_name == 'SIfwfrz':
        new_var = 'freshwater_flux_into_sea_water_due_to_sea_ice_freezing'
    elif var_name == 'SIfwmelt':
        new_var = 'freshwater_flux_into_sea_water_due_to_sea_ice_melting'
    elif var_name == 'SIarea':
        new_var = 'sea_ice_area_fraction'
    elif var_name == 'SIheff':
        new_var = 'sea_ice_thickness'
    elif var_name == 'oceFWflx':
        new_var = 'freshwater_flux_into_sea_water'
    elif var_name == 'sst':
        new_var = 'sea_surface_temperature'
    elif var_name == 'sss':
        new_var = 'sea_surface_salinity'
    new_var = 'trend_in_'+new_var
    if var_name in ['advection_3d', 'diffusion_kpp', 'hb_total', 'shortwave_pen']:
        fname_in = trend_indir+var_name+'_trends.nc'
    elif var_name in ['ADVx_TH', 'ADVy_TH', 'EXFaqh', 'EXFatemp', 'EXFpreci', 'EXFuwind', 'EXFvwind', 'oceFWflx', 'SIarea', 'SIfwfrz', 'SIfwmelt', 'SIheff', 'sss', 'sst']:
        fname_in = trend_indir+var_name+'_trend.nc'
    fname_out = dir_tmp+'MITgcm_'+region+'_1920-2013_'+new_var+'_PACE.nc'
    # Copy the file
    f.write('cp '+fname_in+' '+fname_out+'\n')
    # Rename the variable
    f.write('ncrename -v '+var_name+'_trend,'+new_var+' '+fname_out+'\n')
    # Rename the time axis and variable (fudge from before)
    f.write('ncrename -d time,ensemble_member '+fname_out+'\n')
    f.write('ncrename -v time,ensemble_member '+fname_out+'\n')
    # Add some attributes
    add_attributes(fname_out, 'pace', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
    f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

for var in ['ADVx_TH', 'ADVy_TH', 'advection_3d', 'diffusion_kpp', 'shortwave_pen', 'hb_total', 'EXFuwind', 'EXFvwind', 'EXFatemp', 'EXFaqh', 'EXFpreci', 'SIfwfrz', 'SIfwmelt', 'SIarea', 'SIheff', 'oceFWflx', 'sst', 'sss']:
    process_trend_var (var)

f.close()
