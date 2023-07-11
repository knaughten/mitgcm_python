from ..file_io import netcdf_time

bash_script = 'extract_pdc_scenarios.sh'
main_outdir = 'archive_for_natcli/'
timeseries_dir = 'timeseries/'
hovmoller_dir = 'hovmollers/'
trend_dir = 'trend_maps/'
latlon_dir = 'latlon/'
expt_names = ['Historical', 'Historical_FixedBCs', 'Paris1.5C', 'Paris2C', 'RCP4.5', 'RCP8.5', 'RCP8.5_FixedBCs']
num_expt = len(expt_names)
num_ens = [10, 5, 5, 10, 10, 10, 5]
start_years = [1920, 1920, 2006, 2006, 2006, 2006, 2006]
end_years = [2005, 2005, 2100, 2100, 2080, 2100, 2100]
spinup_year0 = 1890
months_per_year = 12
expt_dir_1 = '/data/oceans_output/shelf/kaight/archer2_mitgcm/PAS_'
expt_dir_2 = ['LENS', 'LENS', 'LW_1.5', 'LW2.0', 'MENS', 'LENS', 'LENS']
expt_dir_3 = ['', '', '_', '_', '_', '', '']
expt_dir_4 = ['_O', '_noOBC', '_O', '_O', '_O', '_O', '_noOBC']
expt_dir_5 = '/output/'
expt_dir_head = [expt_dir_1 + expt_dir_2[n] + expt_dir_3[n] for n in range(num_expt)]
expt_dir_tail = [expt_dir_4[n] + expt_dir_5 for n in range(num_expt)]
expt_cesm_descriptions = ['historical Large Ensemble', 'historical Large Ensemble', 'Low Warming 1.5C scenario', 'Low Warming 2C scenario', 'RCP 4.5 Medium Ensemble', 'RCP 8.5 Large Ensemble', 'RCP 8.5 Large Ensemble']
obcs_descriptions = ['', ' and climatological boundary conditions ', '', '', '', '', ' and climatological boundary conditions ']
trend_indir = 'precomputed_trends/'
hovmoller_file = 'hovmoller_shelf.nc'
buttressing_file = '/data/oceans_output/shelf/kaight/BFRN/AMUND_BFRN_Bedmachinev2_mainGL_withLatLon.mat'
dir_used = []

f = open(bash_script, 'w')
f.write('#!/bin/bash\n')
f.write('set -ex\n')
f.write('export HDF5_USE_FILE_LOCKING=FALSE\n')
f.write('mkdir '+main_outdir+'\n')
dir_used.append(main_outdir)

def get_region (var_name):
    if var_name.startswith('amundsen_shelf_'):
        return 'AmundsenSeaContinentalShelf', -115, -100, -75.5, -70
    elif var_name.startswith('dotson_to_cosgrove_'):
        return 'IceShelvesFromDotsonToCosgrove', -114.6, -98.5, -75.5, -73.1
    elif var_name.startswith('PITE_'):
        return 'PineIslandThwaitesEastTrough', -107, -105, -73, -73
    else:
        return 'AmundsenSea', -140, -80, -76, -62

def add_attributes (fname, expt_index, lon_min, lon_max, lat_min, lat_max, zmin, zmax):
    if expt_index == 'Ua':
        f.write('ncatted -a title,global,o,c,"Amundsen Sea Ua model output for diagnostic buttressing simulations." '+fname+'\n')
    elif expt_index == 'topo':
        f.write('ncatted -a title,global,o,c,"Amundsen Sea MITgcm model topography." '+fname+'\n')
    else:
        f.write('ncatted -a title,global,o,c,"Amundsen Sea MITgcm model output forced with CESM1 '+expt_cesm_descriptions[expt_index]+obcs_descriptions[expt_index]+', '+start_years[expt_index]+'-'+end_years[expt_index]+'" '+fname+'\n')
    f.write('ncatted -a summary,global,o,c,"This dataset provides model output for 20th and 21st-century ice-ocean simulations in the Amundsen Sea. The simulations are performed with the MITgcm model at 1/10 degree resolution, including components for the ocean, sea ice, and ice shelf thermodynamics. Atmospheric forcing is provided by the CESM1 climate model for the historical period (1920-2005) and four future scenarios (2006-2100), using 5-10 ensemble members each. The open ocean boundaries are forced by either the corresponding CESM1 simulation or a present-day climatology. The simulations were completed in 2022 by Kaitlin Naughten at the British Antarctic Survey (Polar Oceans team)." '+fname+'\n')
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
    if expt_names[expt_index].startswith('Historical'):
        f.write('ncatted -a time_coverage_start,global,o,c,1920-01-01 '+fname+'\n')
        f.write('ncatted -a time_coverage_end,global,o,c,2005-12-31 '+fname+'\n')
    else:
        f.write('ncatted -a time_coverage_start,global,o,c,2006-01-01 '+fname+'\n')
        f.write('ncatted -a time_coverage_end,global,o,c,2100-12-31 '+fname+'\n')    


# Timeseries
f.write('mkdir '+main_outdir+timeseries_dir+'\n')
dir_used.append(main_outdir+timeseries_dir)
def process_timeseries_var (var_name):
    print('Processing '+var_name+' timeseries')
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    dir_tmp = main_outdir+timeseries_dir+region+'/'
    if dir_tmp not in dir_used:
        f.write('mkdir '+dir_tmp+'\n')
        dir_used.append(dir_tmp)
    if var_name.endswith('200_700m'):
        depth = '200-700m'
        zmin = -700
        zmax = -200
        depth_str = '_'+depth+'_'
        dir_tmp += depth+'/'
        if dir_tmp not in dir_used:
            f.write('mkdir '+dir_tmp+'\n')
            dir_used.append(dir_tmp)
    else:
        depth = '2D'
        zmin = None
        zmax = None
        depth_str = '_'
    if 'temp' in var_name:
        new_var = 'sea_water_potential_temperature'
    elif 'massloss' in var_name:
        new_var = 'ice_shelf_basal_melt_flux'
    elif 'trans' in var_name:
        new_var = 'ocean_volume_y_transport'
    for n in range(num_expt):
        print('...'+expt_names[n])
        dir_tmp_expt = dir_tmp + expt_names[n]
        if dir_tmp_expt not in dir_used:
            f.write('mkdir '+dir_tmp_expt+'\n')
            dir_used.append(dir_tmp_expt)
        if expt_names[n].endswith('FixedBCs') and var_name == 'PITE_trans':
            ts_file = 'timeseries_PITE.nc'
        else:
            ts_file = 'timeseries.nc'
        for e in range(num_ens[n]):
            fname_in = expt_dir_head[n] + str(e+1).zfill(3) + expt_dir_tail[n] + ts_file
            fname_out = dir_tmp_expt+'MITgcm_'+region+'_'+start_years[n]+'-'+end_years[n]+'_'+new_var+depth_str+expt_names[n]+'_ens'+str(e+1).zfill(3)+'.nc'
            # Extract the right years into the new file
            first_year = netcdf_time(fname_in, monthly=False)[0].year
            if first_year == start_years[n]:
                f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            elif first_year == spinup_year0:
                # Need to trim the spinup
                t_start = (start_years[n]-spinup_year0)*months_per_year
                t_end = (end_years[n]-spinup_year0+1)*months_per_year-1
                f.write('ncks -d time,'+str(t_start)+','+str(t_end)+' -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            # Now rename the variable
            f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
            if var_name == 'PITE_trans':
                f.write('ncatted -c scale_factor,'+new_var+',o,f,-1 '+fname+'\n')
            # Add some attributes
            add_attributes(fname_out, n, lon_min, lon_max, lat_min, lat_max, zmin, zmax)
            f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

for var in ['amundsen_shelf_temp_btw_200_700m', 'dotson_to_cosgrove_massloss', 'PITE_trans']:
    process_timeseries_var(var)
    

# Hovmollers
f.write('mkdir '+main_outdir+hovmoller_dir+'\n')
dir_used.append(main_outdir+hovmoller_dir)
def process_hovmoller_var (var_name, hov_file='hovmoller_shelf.nc'):
    print('Processing '+var_name+' Hovmoller')
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    zmin = None
    zmax = None
    dir_tmp = main_outdir+hovmoller_dir+region+'/'
    if dir_tmp not in dir_used:
        f.write('mkdir '+dir_tmp+'\n')
        dir_used.append(dir_tmp)
    if 'temp' in var_name:
        new_var = 'sea_water_potential_temperature'
    for n in range(num_expt):
        print('...'+expt_names[n])
        dir_tmp_expt = dir_tmp + expt_names[n]
        if dir_tmp_expt not in dir_used:
            f.write('mkdir '+dir_tmp_expt+'\n')
            dir_used.append(dir_tmp_expt)
        for e in range(num_ens[n]):
            fname_in = expt_dir_head[n] + str(e+1).zfill(3) + expt_dir_tail[n] + hov_file
            fname_out = dir_tmp_expt+'MITgcm_'+region+'_'+start_years[n]+'-'+end_years[n]+'_'+new_var+'_'+expt_names[n]+'_ens'+str(e+1).zfill(3)+'.nc'
            first_year = netcdf_time(fname_in, monthly=False)[0].year
            if first_year == start_years[n]:
                f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            elif first_year == spinup_year0:
                # Need to trim the spinup
                t_start = (start_years[n]-spinup_year0)*months_per_year
                t_end = (end_years[n]-spinup_year0+1)*months_per_year-1
                f.write('ncks -d time,'+str(t_start)+','+str(t_end)+' -v '+var_name+' '+fname_in+' '+fname_out+'\n')
            # Now rename the variable
            f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
            # Add some attributes
            add_attributes(fname_out, n, lon_min, lon_max, lat_min, lat_max, zmin, zmax)
            f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

process_hovmoller_var('amundsen_shelf_temp')

# Trend maps (note special naming conventions including no_OBCS)
f.write('mkdir '+main_outdir+trend_dir+'\n')
dir_used.append(main_outdir+trend_dir)
def process_trend_var (var_name):
    print('Processing '+var_name)
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    dir_tmp = main_outdir+trend_dir
    if var_name.endswith('200_700m'):
        depth = '200-700m'
        zmin = -700
        zmax = -200
    elif var_name.endswith('bottom100m'):
        depth = 'bottom_100m'
        depth_str = '_'+depth+'_'
    else:
        depth = '2D'
        zmin = None
        zmax = None
        depth_str = '_'
    if depth != '2D':
        depth_str = '_'+depth+'_'
        dir_tmp += depth+'/'
        if dir_tmp not in dir_used:
            f.write('mkdir '+dir_tmp+'\n')
            dir_used.append(dir_tmp)
    if var_name.startswith('temp'):
        new_var = 'sea_water_potential_temperature'
    elif var_name == 'ismr':
        new_var = 'ice_shelf_basal_melt_flux'
    elif var_name in ['UVEL', 'u_bottom100m']:
        new_var = 'eastward_sea_water_velocity'
    elif var_name in ['VVEL', 'v_bottom100m']:
        new_var = 'northward_sea_water_velocity'
    elif var_name == 'EXFuwind':
        new_var = 'eastward_wind'
    elif var_name == 'EXFvwind':
        new_var = 'northward_wind'
    elif var_name == 'EXFatemp':
        new_var = 'air_temperature'
    elif var_name == 'EXFpreci':
        new_var = 'precipitation_flux'
    for n in range(num_expt):
        print('...'+expt_names[n])
        dir_tmp_expt = dir_tmp + expt_names[n]
        if dir_tmp_expt not in dir_used:
            f.write('mkdir '+dir_tmp_expt+'\n')
            dir_used.append(dir_tmp_expt)
        in_dir = trend_indir
        if expt_names[n].endswith('FixedBCs'):
            in_dir += 'no_obcs/'
        fname_in = in_dir + var_name + '_trend_'
        if expt_names[n].startswith('Historical'):
            fname_in += 'historical.nc'
        else:
            fname_in += expt_dir_2[n] + '.nc'
        fname_out = dir_tmp_expt+'MITgcm_'+region+'_'+start_years[n]+'-'+end_years[n]+'_'+new_var+depth_str+expt_names[n]+'.nc'
        # Copy the file
    f.write('cp '+fname_in+' '+fname_out+'\n')
    # Rename the variable
    f.write('ncrename -v '+var_name+'_trend,'+new_var+' '+fname_out+'\n')
    # Rename the time axis and variable (fudge from before)
    f.write('ncks -3 -O '+fname_out+' tmp.nc\n')
    f.write('ncrename -d time,ensemble_member '+fname_out+'\n')
    f.write('ncks -4 -O tmp.nc '+fname_out+'\n')
    f.write('ncrename -v time,ensemble_member '+fname_out+'\n')
    # Add some attributes
    add_attributes(fname_out, n, lon_min, lon_max, lat_min, lat_max, zmin, zmax)
    f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

for var in ['temp_btw_200_700m', 'ismr', 'UVEL', 'VVEL', 'u_bottom100m', 'v_bottom100m', 'EXFuwind', 'EXFvwind', 'EXFatemp', 'EXFpreci']:
    process_trend_var(var)

    
# Lat-lon fields (constant in time)
f.write('mkdir '+main_outdir+latlon_dir+'\n')
dir_used.append(main_outdir+latlon_dir)
def process_latlon_var (var_name):
    print('Processing '+var_name+' lat-lon')
    region, lon_min, lon_max, lat_min, lat_max = get_region(var_name)
    zmin = None
    zmax = None
    if var_name == 'bathy':
        new_var = 'seafloor_depth'
    elif var_name == 'draft':
        new_var = 'ice_shelf_draft'
    dir_tmp = main_outdir+latlon_dir
    if dir_tmp not in dir_used:
        f.write('mkdir '+dir_tmp+'\n')
        dir_used.append(dir_tmp)
    fname_in = 'PAS_'+var_name+'.nc'
    fname_out = dir_tmp+'MITgcm_'+region+'_'+new_var+'.nc'
    # Extract the variable
    f.write('ncks -v '+var_name+' '+fname_in+' '+fname_out+'\n')
    # Rename it
    f.write('ncrename -v '+var_name+','+new_var+' '+fname_out+'\n')
    # Add some attributes
    add_attributes(fname_out, 'topo', lon_min, lon_max, lat_min, lat_max, zmin, zmax)
    f.write('ncatted -a standard_name,'+new_var+',o,c,'+new_var+' '+fname_out+'\n')

for var in ['bathy', 'draft']:
    process_latlon_var(var)

# BFRN field


f.close()
