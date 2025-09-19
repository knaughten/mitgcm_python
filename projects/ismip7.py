import xarray as xr
import numpy as np
import os

from ..grid import ISMIP7Grid, Grid
from ..interpolation import interp_reg_xy, extend_into_mask, interp_reg_xyz
from ..utils import convert_ismr, days_per_month
from ..constants import months_per_year, sec_per_year
from ..file_io import read_netcdf

from fesomtools.fesom_grid import *
from fesomtools.in_triangle import *
from fesomtools.triangle_area import *

grid_path = '/gws/nopw/j04/bas_pog/kaight/ismip7_interp/ismip_8km_60m_grid.nc'
grid_out = ISMIP7Grid(grid_path)

# Interpolate the output of WSFRIS 2021 paper (two_timescale.py) and PAS 2023 paper (scenarios.py) to the ISMIP7 grid for sharing

def interp_year (file_path, calendar='noleap', interpolant=None):

    var_in = ['THETA', 'SALT', 'SHIfwFlx']
    var_out = ['temperature', 'salinity', 'basal_melt']
    units = ['degC', 'psu', 'm/y']
    long_name = ['potential temperature (EOS80)', 'practical salinity (EOS80)', 'ice shelf basal melt rate, positive means melting']
    fill_value = 9999
    
    grid_in = Grid(file_path)

    # Prepare for annual averaging
    if calendar == '360-day':
        ndays = [30]*months_per_year
    elif calendar == 'noleap':
        ndays = [days_per_month(month+1, 1979, allow_leap=False) for month in range(months_per_year)]
    else:
        raise Exception('Unsupported calendar '+calendar)

    # Inner function to interpolate a variable to the ISMIP7 grid
    def interp_var(data_in, is_3d=False, interpolant=None, return_interpolant=False):
        if is_3d:
            data_out, interpolant = interp_reg_xyz(grid_in.lon_1d, grid_in.lat_1d, grid_in.z, data_in, grid_out.lon, grid_out.lat, grid_out.z, fill_value=fill_value, interpolant=interpolant, return_interpolant=True)
            data_out = xr.DataArray(data_out, coords={'z':grid_out.z, 'y':grid_out.y, 'x':grid_out.x})
        else:
            # 2D variable - interpolate once
            data_out = interp_reg_xy(grid_in.lon_1d, grid_in.lat_1d, data_in, grid_out.lon, grid_out.lat, fill_value=fill_value)
            data_out = xr.DataArray(data_out, coords={'y':grid_out.y, 'x':grid_out.x})
        data_out = data_out.where(data_out != fill_value)
        if return_interpolant:
            return data_out, interpolant
        else:
            return data_out

    # Interpolate masks
    land_mask = np.round(interp_var(grid_in.land_mask))
    ice_mask = np.round(interp_var(grid_in.ice_mask))
    mask_3d, interpolant = interp_var(grid_in.hfac!=0, is_3d=True, interpolant=interpolant, return_interpolant=True)
    mask_3d = np.round(mask_3d)

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
        data_out = interp_var(data_in, is_3d=is_3d, interpolant=interpolant)
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
    return ds_out, interpolant


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

    out_subdir = out_dir + 'RCP85_ens' + str(ens).zfill(2) + '/'
    if not os.path.isdir(out_subdir):
        os.mkdir(out_subdir)

    interpolant = None
    for year in range(start_year, end_year+1):
        in_file = in_dir + dir_head + str(ens).zfill(3) + dir_mid + str(year) + file_tail
        ds, interpolant = interp_year(in_file, calendar=calendar, interpolant=interpolant).expand_dims({'time':[year]})
        out_file = out_subdir + 'MITgcm_ASE_RCP85_ens' + str(ens).zfill(2) + '_' + str(year) + '.nc'
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

    out_subdir = out_dir + expt + '/'
    if not os.path.isdir(out_subdir):
        os.mkdir(out_subdir)

    interpolant = None
    for year in range(start_year, end_year+1):
        in_file = in_dir + dir_head + str(year) + file_tail
        ds, interpolant = interp_year(in_file, calendar=calendar, interpolant=interpolant).expand_dims({'time':[year]})
        out_file = out_subdir + 'MITgcm_WS_' + expt + '_' + str(year) + '.nc'
        ds.to_netcdf(out_file, mode='w')
        ds.close()


# Interpolate the output of FESOM 2018 paper.
def interp_year_fesom (file_head, nodes, elements, n2d, cavity):
 
    var_in = ['temp', 'salt', 'wnet']
    file_tail = ['.oce.mean.nc', '.oce.mean.nc', '.forcing.diag.nc']
    var_out = ['temperature', 'salinity', 'basal_melt']
    units = ['degC', 'psu', 'm/y']
    long_name = ['potential temperature (EOS80)', 'practical salinity (EOS80)', 'ice shelf basal melt rate, positive means melting']
    num_var = len(var_in)
    xmin = np.amin(grid_out.x)
    xmax = np.amax(grid_out.x)
    ymin = np.amin(grid_out.y)
    ymax = np.amax(grid_out.y)

    # Get the variables we want into a Dataset, and set up empty Dataset for output
    ds_in = None
    ds_out = None
    for v in range(num_var):
        file_path = file_head + file_tail[v]
        ds = xr.open_dataset(file_path)
        data_in = ds[var_in[v]]
        is_3d = 'nodes_3d' in data_in.dims
        # Annually average - equal 5-day intervals
        data_in = data_in.mean(dim='T')
        if var_in[v] == 'wnet':
            # Unit conversion (m/s to m/y)
            data_in *= sec_per_year
        if is_3d:
            data_out = xr.DataArray(np.zeros([grid_out.nz, grid_out.ny, grid_out.nx]), coords={'z':grid_out.z, 'y':grid_out.y, 'x':grid_out.x})
        else:
            data_out = xr.DataArray(np.zeros([grid_out.ny, grid_out.nx]), coords={'y':grid_out.y, 'x':grid_out.x})
        if ds_in is None:
            ds_in = xr.Dataset({var_in[v]:data_in})
            ds_out = xr.Dataset({var_out[v]:data_out})
        else:
            ds_in = ds_in.assign({var_in[v]:data_in})
            ds_out = ds_out.assign({var_out[v]:data_out})
        ds.close()

    # Interpolate all at once
    valid_mask = xr.DataArray(np.zeros([grid_out.ny, grid_out.nx]), coords={'y':grid_out.y, 'x':grid_out.x})
    for elm in elements:        
        # Check if we are within domain of regular grid (just check northern boundary)
        if np.amin(elm.lat) > np.amax(grid_out.lat):
            continue
        # Convert element coordinates to polar stereo
        elm_x, elm_y = polar_stereo(elm.lon, elm.lat)
        # Check if we are within domain of regular grid
        if np.amax(elm_x) < xmin or np.amin(elm_x) > xmax or np.amax(elm.y) < ymin or np.amin(elm.y) > ymax:
            continue
        # Find bounds on ISMIP7 coordinates around element
        tmp = np.nonzero(grid_out.x > np.amin(elm.x))[0]
        if len(tmp) == 0:
            i0 = 0
        else:
            i0 = tmp[0] - 1
        tmp = np.nonzero(grid_out.x > np.amax(elm.x))[0]
        if len(tmp) == 0:
            i1 = grid_out.nx
        else:
            i1 = tmp[0]
        tmp = np.nonzero(grid_out.y > np.amin(elm.y))[0]
        if len(tmp) == 0:
            j0 = 0
        else:
            j0 = tmp[0] - 1
        tmp = np.nonzero(grid_out.y > np.amax(elm.y))[0]
        if len(tmp) == 0:
            j1 = grid_out.ny
        else:
            j1 = tmp[0]
        for i in range(i0+1, i1):
            for j in range(j0+1, j1):
                # There is a chance that the ISMIP7 gridpoint at (i,j) lies within this element
                x0 = grid_out.x[i]
                y0 = grid_out.y[j]
                if in_triangle(elm, x0, y0):
                    # Get area of entire triangle
                    area = triangle_area(elm.x, elm.y)
                    # Get area of each sub-triangle formed by (x0, y0)
                    area0 = triangle_area([x0, elm.x[1], elm.x[2]], [y0, elm.y[1], elm.y[2]])
                    area1 = triangle_area([x0, elm.x[0], elm.x[2]], [y0, elm.y[0], elm.y[2]])
                    area2 = triangle_area([x0, elm.x[0], elm.x[1]], [y0, elm.y[0], elm.y[1]])
                    # Find fractional area of each
                    cff = np.array([area0/area, area1/area, area2/area])
                    for v in range(num_var):
                        is_3d = 'nodes_3d' in ds_in[var_in[v]].dims
                        if is_3d:
                            # Interpolate to each depth value
                            for k in range(grid_out.nz):
                                # Find each corner of the triangular element, interpolated to this depth
                                corners = []
                                for n in range(3):
                                    id1, id2, coeff1, coeff2 = elm.nodes[n].find_depth(grid.z[k])
                                    if any(np.isnan([id1, id2, coeff1, coeff2])):
                                        # Seafloor or ice shelf
                                        corners.append(np.nan)
                                    else:
                                        corners.append(coeff1*ds_in[var_in[v]].isel(nodes_3d=id1) + coeff2*ds_in[var_in[v]].isel(nodes_3d=id2))
                                if any(np.isnan(corners)):
                                    pass
                                else:
                                    # Barycentric interpolation to (x0, y0)
                                    ds_out[var_out[v]] = xr.where((ds.coords['x']==i)*(ds.coords['y']==j)*(ds_coords['z']==k), np.sum(cff*corners), ds_out[var_out[v]])
                        else:
                            corners = [ds_in[var_in[v]].isel(nodes_2d=elm.nodes[n].id) for n in range(3)]
                            ds_out[var_out[v]] = xr.where((ds.coords['x']==i)*(ds.coords['y']==j), np.sum(cff*corners), ds_out[var_out[v]])
                    valid_mask = xr.where((ds.coords['x']==i)*(ds.coords['y']==j), 1, valid_mask)
    # Mask out anywhere that had nothing to interpolate to
    ds_out = ds_out.where(valid_mask > 0)
    return ds_out            
                    

# Process one experiment of 2018 FESOM simulations ('RCP8.5_MMM' or 'RCP8.5_ACCESS')
def process_FESOM (expt, out_dir='./'):

    in_dir = '/gws/nopw/j04/bas_pog/kaight/PhD/future_projections/'+expt+'/'
    mesh_dir = '/gws/nopw/j04/bas_pog/kaight/PhD/FESOM_mesh/high_res/'
    start_year = 2006
    end_year = 2100

    # Build FESOM mesh
    nodes, elements = fesom_grid(mesh_dir, return_nodes=True)
    # Read the cavity mask
    cavity = np.fromfile(mesh_dir+'cavity_flag_nod2d.out', dtype=int)

    out_subdir = out_dir + expt + '/'
    if not os.path.isdir(out_subdir):
        os.mkdir(out_subdir)

    for year in range(start_year, end_year+1):
        in_file_head = in_dir + 'MK44005.' + str(year)
        ds = interp_year_fesom(in_file_head, nodes, elements, cavity).expand_dims({'time':[year]})
        out_file = out_subdir + 'FESOM_' + expt + '_' + str(year) + '.nc'
        ds.to_netcdf(out_file, mode='w')
        ds.close()

    

                
    
            
        

    

