#######################################################
# Calculation of integral timeseries
#######################################################

import numpy as np
import sys
import datetime

from .grid import choose_grid, Grid
from .file_io import read_netcdf, netcdf_time
from .utils import convert_ismr, var_min_max, mask_land_ice, days_per_month, apply_mask, mask_3d, xy_to_xyz, select_top, select_bottom, add_time_dim, z_to_xyz, mask_2d_to_3d, mask_land, depth_of_isoline
from .diagnostics import total_melt, wed_gyre_trans, transport_transect, density, in_situ_temp, tfreeze, adv_heat_wrt_freezing, thermocline
from .calculus import over_area, area_integral, over_volume, vertical_average_column, area_average, volume_average, volume_integral
from .interpolation import interp_bilinear, neighbours, interp_to_depth, interp_grid
from .constants import deg_string, region_names, temp_C2K, sec_per_year, sec_per_day, rhoConst, Cp_sw


# Calculate total mass loss or area-averaged melt rate from ice shelves in the given NetCDF file. You can specify specific ice shelves (as specified in region_names in constants.py). The default behaviour is to calculate the melt at each time index in the file, but you can also select a subset of time indices, and/or time-average - see optional keyword arguments. You can also split into positive (melting) and negative (freezing) components.

# Arguments:
# file_path: path to NetCDF file containing 'SHIfwFlx' variable
# grid = Grid object

# Optional keyword arguments:
# shelf: 'fris' (default) restricts the calculation to FRIS. 'ewed' restricts the calculation to ice shelves between the Eastern Weddell bounds given in constants.py. 'all' considers all ice shelves.
# result: 'massloss' (default) calculates the total mass loss in Gt/y. 'melting' calculates the area-averaged melt rate in m/y.
# time_index, t_start, t_end, time_average: as in function read_netcdf
# mass_balance: if True, split into positive (melting) and negative (freezing) terms. Default False.
# z0: optional list of length 2 containing deep and shallow bounds for depth ranges of ice shelf base to consider (negative, in metres).

# Output:
# If time_index is set, or time_average=True: single value containing mass loss or average melt rate
# Otherwise: 1D array containing timeseries of mass loss or average melt rate
# If mass_balance=True: two values/arrays will be returned, with the positive and negative components.

def timeseries_ismr (file_path, grid, shelf='fris', result='massloss', time_index=None, t_start=None, t_end=None, time_average=False, mass_balance=False, z0=None):

    # Choose the appropriate mask
    mask = grid.get_ice_mask(shelf=shelf)
    if z0 is not None:
        [z_deep, z_shallow] = z0
        # Mask out regions where the ice base is outside this depth range
        mask[grid.draft <= z_deep] = False
        mask[grid.draft > z_shallow] = False

    # Read ice shelf melt rate and convert to m/y
    ismr = convert_ismr(read_netcdf(file_path, 'SHIfwFlx', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average))
    if len(ismr.shape)==2:
        # Just one timestep; add a dummy time dimension
        ismr = np.expand_dims(ismr,0)

    if mass_balance:
        # Split into melting and freezing
        ismr_positive = np.maximum(ismr, 0)
        ismr_negative = np.minimum(ismr, 0)
        
    # Loop over timesteps
    num_time = ismr.shape[0]
    if mass_balance:
        melt = np.zeros(num_time)
        freeze = np.zeros(num_time)
        for t in range(num_time):
            melt[t] = total_melt(ismr_positive[t,:], mask, grid, result=result)
            freeze[t] = total_melt(ismr_negative[t,:], mask, grid, result=result)
        return melt, freeze
    else:
        melt = np.zeros(num_time)
        for t in range(num_time):
            melt[t] = total_melt(ismr[t,:], mask, grid, result=result)
        # Mask out any NaNs (can happen when no cells fall within the given depth range during a coupled run)
        melt = np.ma.masked_where(np.isnan(melt), melt)
        return melt


# Read the given lat x lon variable from the given NetCDF file, and calculate timeseries of its maximum value in the given region.
def timeseries_max (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, xmin=None, xmax=None, ymin=None, ymax=None, mask=None):

    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if var_name == 'PsiVEL':
        # Special case to get absolute value of vertically integrated streamfunction
        data = np.abs(np.sum(data, axis=-3))
    if len(data.shape)==2:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)

    num_time = data.shape[0]
    max_data = np.zeros(num_time)
    for t in range(num_time):
        # Mask
        if mask is None:
            data_tmp = mask_land(data[t,:], grid, gtype=gtype)
        else:
            data_tmp = apply_mask(data[t,:], np.invert(mask))
        max_data[t] = var_min_max(data_tmp, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)[1]
    return max_data


# Helper function for timeseries_avg_sfc and timeseries_int_sfc.
def timeseries_area_sfc (option, file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, operator='add'):
    
    # Read the data
    if isinstance(var_name, str):
        # Just one variable
        # Make it a list
        var_name = [var_name]
    # Now we have multiple variables to add or subtract together.
    data = None
    for var in var_name:
        if var == 'EXFwind':
            # Special case to get wind speed
            u = read_netcdf(file_path, 'EXFuwind', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
            v = read_netcdf(file_path, 'EXFvwind', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
            data_tmp = np.sqrt(u**2 + v**2)
        elif var == 'TminusTf':
            # Special case to get thermal driving
            temp_3d = read_netcdf(file_path, 'THETA', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
            salt_3d = read_netcdf(file_path, 'SALT', time_index, t_start=t_start, t_end=t_end, time_average=time_average)
            time_dependent = len(temp_3d.shape)==4
            temp = select_top(temp_3d, masked=False, grid=grid, time_dependent=time_dependent)
            salt = select_top(salt_3d, masked=False, grid=grid, time_dependent=time_dependent)
            z = select_top(z_to_xyz(grid.z, grid), masked=False, grid=grid)
            if time_dependent:
                z = add_time_dim(z, temp.shape[0])
            data_tmp = in_situ_temp(temp, salt, z) - tfreeze(salt, z)
        else:
            data_tmp = read_netcdf(file_path, var, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        if var in ['THETA', 'SALT', 'WSLTMASS']:
            # 3D variable; have to take surface layer
            data_tmp = select_top(data_tmp, masked=False, grid=grid, time_dependent=len(data_tmp.shape)==4)
        if var == 'PsiVEL':
            # Special case to get absolute value of vertically integrated streamfunction
            data_tmp = np.abs(np.sum(data_tmp, axis=-3))        
        if data is None:
            data = data_tmp
        else:
            if operator == 'add':
                data += data_tmp
            elif operator == 'subtract':
                data -= data_tmp
            else:
                print(('Error (timeseries_area_sfc): invalid operator ' + operator))
                sys.exit()
    if len(data.shape)==2:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)
    
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data.shape[0]):
        # Mask
        if mask is None:
            data_tmp = mask_land_ice(data[t,:], grid, gtype=gtype)
        else:
            data_tmp = apply_mask(data[t,:], np.invert(mask))
        # Area-average or integrate
        timeseries.append(over_area(option, data_tmp, grid, gtype=gtype))
    return np.array(timeseries)


# Read the given lat x lon variable from the given NetCDF file, and calculate timeseries of its area-averaged value over the sea surface.
def timeseries_avg_sfc (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, operator='add'):
    return timeseries_area_sfc('average', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, operator=operator)


# Like timeseries_avg_sfc, but for area-integrals over the sea surface.
def timeseries_int_sfc (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, operator='add'):
    return timeseries_area_sfc('integrate', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, operator=operator)


# Integrate the area of the sea surface where the given variable exceeds the given threshold.
def timeseries_area_threshold (file_path, var_name, val0, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False):

    # Read the data
    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==2:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)
    # Convert to array of 1s and 0s based on threshold
    data = (data >= val0).astype(float)
    # Now build the timeseries
    timeseries = []
    for t in range(data.shape[0]):
        timeseries.append(area_integral(data[t,:], grid, gtype=gtype))
    return np.array(timeseries)


# Helper function for timeseries_avg_3d, timeseries_int_3d, timeseries_avg_bottom, timeseries_avg_z0, timeseries_avg_btw_z0, timeseries_int_btw_z0.
def timeseries_vol_3d (option, file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None, z0=None):
        
    if var_name == 'RHO':
        if rho is None:
            print('Error (timeseries_avg_3d): must precompute density')
            sys.exit()
        data = rho
    elif var_name == 'TMINUSTF':
        # For now, use surface freezing point
        temp = read_netcdf(file_path, 'THETA', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        salt = read_netcdf(file_path, 'SALT', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        data = temp - tfreeze(salt, 0)
    elif var_name == 'shortwave_penetration':
        # Get some variables we'll need
        z_edges_3d = z_to_xyz(grid.z_edges, grid)
        dA_3d = xy_to_xyz(grid.dA, grid)
        swfrac = 0.62*np.exp(z_edges_3d[:-1,:]/0.6) + (1-0.62)*np.exp(z_edges_3d[:-1,:]/20.)
        swfrac1 = 0.62*np.exp(z_edges_3d[1:,:]/0.6) + (1-0.62)*np.exp(z_edges_3d[1:,:]/20.)
        # Read shortwave flux at surface
        data_xy = read_netcdf(file_path, 'oceQsw', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        if len(data_xy.shape)==2:
            data_xy = np.expand_dims(data_xy,0)
        # Loop over timesteps to calculate 3D penetration
        data = np.ma.empty([data_xy.shape[0], grid.nz, grid.ny, grid.nx])
        for t in range(data.shape[0]):
            data[t,:] = xy_to_xyz(data_xy[t,:], grid)*(swfrac-swfrac1)*dA_3d/(rhoConst*Cp_sw)
    else:
        data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if var_name == 'THETA' and option in ['integrate', 'int_btw_z0']:
        # Convert to Kelvin
        data += temp_C2K
    if len(data.shape)==3:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)
    if option in ['avg_btw_z0', 'int_btw_z0']:
        # Need to make mask 3D
        if mask is None:
            # Dummy mask
            mask = np.ones([grid.ny, grid.nx]).astype(bool)
        mask = mask_2d_to_3d(mask, grid, zmin=z0[0], zmax=z0[1])
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data.shape[0]):
        # First mask the land and ice shelves
        data_tmp = mask_3d(data[t,:], grid, gtype=gtype)
        if option in ['average', 'integrate']:
            # 3D volume average
            if mask is not None:
                # Also mask outside the given region
                data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
            # Volume average or integrate
            timeseries.append(over_volume(option, data_tmp, grid, gtype=gtype))
        elif option == 'avg_btw_z0':
            # 3D volume average between the given depths
            data_tmp = apply_mask(data_tmp, np.invert(mask))
            timeseries.append(volume_average(data_tmp, grid, gtype=gtype))
        elif option == 'int_btw_z0':
            # 3D volume integral between the given depths
            data_tmp = apply_mask(data_tmp, np.invert(mask))
            if var_name == 'shortwave_penetration':
                # Already mass-weighted, so just do a regular sum
                timeseries.append(np.sum(data_tmp))
            else:
                timeseries.append(volume_integral(data_tmp, grid, gtype=gtype))
        elif option in ['avg_bottom', 'avg_z0']:
            # 2D area-average
            if option == 'avg_bottom':
                # Select the bottom layer
                data_tmp = select_bottom(data_tmp)
            elif option == 'avg_z0':
                # Interpolate to the given depth
                data_tmp = interp_to_depth(data_tmp, z0, grid, gtype=gtype)
            if mask is not None:
                # Mask outside the given region
                data_tmp = apply_mask(data_tmp, np.invert(mask))
            # Area-average
            timeseries.append(area_average(data_tmp, grid, gtype=gtype))
    return np.array(timeseries)


# Read the given 3D variable from the given NetCDF file, and calculate timeseries of its volume-averaged value. Restrict it to the given mask (default just mask out the land).
def timeseries_avg_3d (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    return timeseries_vol_3d('average', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho)


# Same but volume-integrate.
def timeseries_int_3d (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    return timeseries_vol_3d('integrate', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho)


# Same but the area-averaged value over the bottom layer.
def timeseries_avg_bottom (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    return timeseries_vol_3d('avg_bottom', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho)


# Same but area-averaged value over the given depth.
def timeseries_avg_z0 (file_path, var_name, z0, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    return timeseries_vol_3d('avg_z0', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho, z0=z0)


# Same but volume-averaged value between the given depths (where z0=[z_deep, z_shallow]).
def timeseries_avg_btw_z0 (file_path, var_name, z0, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    if not isinstance(z0, list) or len(z0) != 2:
        print('Error (timeseries_avg_btw_z0): z0 must be a list of length 2: [z_deep, z_shallow]')
    return timeseries_vol_3d('avg_btw_z0', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho, z0=z0)


# Same but volume-integrated value between the given depths.
def timeseries_int_btw_z0 (file_path, var_name, z0, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    if not isinstance(z0, list) or len(z0) != 2:
        print('Error (timeseries_int_btw_z0): z0 must be a list of length 2: [z_deep, z_shallow]')
    return timeseries_vol_3d('int_btw_z0', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho, z0=z0)


def timeseries_thermocline (file_path, grid, mask=None, time_index=None, t_start=None, t_end=None, time_average=False):

    data = read_netcdf(file_path, 'THETA', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==3:
        data = np.expand_dims(data,0)
    timeseries = []
    for t in range(data.shape[0]):
        # Calculate the thermocline at every point - this will mask the land
        data_tmp = thermocline(data[t,:], grid)
        # Apply mask
        if mask is not None:
            data_tmp = apply_mask(data_tmp, np.invert(mask))
        timeseries.append(area_average(data_tmp, grid))
    return np.array(timeseries)


# Find the depth of the shallowest given isotherm, below the given depth z0.
def timeseries_iso_depth (file_path, var_name, val0, grid, z0=None, mask=None, time_index=None, t_start=None, t_end=None, time_average=False):

    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==3:
        data = np.expand_dims(data,0)
    timeseries = []
    for t in range(data.shape[0]):
        data_tmp = mask_3d(data[t,:], grid)
        if mask is not None:
            data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
        iso_depth_tmp = depth_of_isoline(data_tmp, grid.z, val0, z0=z0)
        timeseries.append(area_average(iso_depth_tmp, grid))
    return np.array(timeseries)


# Read the given 3D variable from the given NetCDF file, and calculate timeseries of its depth-averaged value over a given latitude and longitude.
def timeseries_point_vavg (file_path, var_name, lon0, lat0, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False):

    # Read the data
    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==3:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)
    # Interpolate to the point, and get hfac too
    data_point, hfac_point = interp_bilinear(data, lon0, lat0, grid, gtype=gtype, return_hfac=True)
    # Vertically average to get timeseries
    return vertical_average_column(data_point, hfac_point, grid, gtype=gtype, time_dependent=True)


# Calculate timeseries of the Weddell Gyre transport in the given NetCDF file. Assumes the Weddell Gyre is actually in your domain.
def timeseries_wed_gyre (file_path, grid, time_index=None, t_start=None, t_end=None, time_average=False):

    # Read u
    u = read_netcdf(file_path, 'UVEL', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(u.shape)==3:
        # Just one timestep; add a dummy time dimension
        u = np.expand_dims(u,0)
    # Build the timeseries
    timeseries = []
    for t in range(u.shape[0]):
        timeseries.append(wed_gyre_trans(u[t,:], grid))
    return np.array(timeseries)


# Calculate timeseries of the volume (as a percentage of the entire domain, neglecting free surface changes) of the water mass between the given temperature and salinity bounds.
def timeseries_watermass_volume (file_path, grid, tmin=None, tmax=None, smin=None, smax=None, time_index=None, t_start=None, t_end=None, time_average=False):

    # Read T and S
    temp = read_netcdf(file_path, 'THETA', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    salt = read_netcdf(file_path, 'SALT', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(temp.shape)==3:
        # Just one timestep; add a dummy time dimension
        temp = np.expand_dims(temp,0)
        salt = np.expand_dims(salt,0)
    # Set any unset bounds
    if tmin is None:
        tmin = -9999
    if tmax is None:
        tmax = 9999
    if smin is None:
        smin = -9999
    if smax is None:
        smax = 9999
    # Build the timeseries
    timeseries = []
    for t in range(temp.shape[0]):
        # Find points within these bounds
        index = (temp[t,:] >= tmin)*(temp[t,:] <= tmax)*(salt[t,:] >= smin)*(salt[t,:] <= smax)*(grid.hfac > 0)
        # Integrate volume of those cells, and get percent of total volume
        timeseries.append(np.sum(grid.dV[index])/np.sum(grid.dV)*100)
    return np.array(timeseries)


# Calculate timeseries of the volume of the entire domain, including free surface changes.
def timeseries_domain_volume (file_path, grid, time_index=None, t_start=None, t_end=None, time_average=False):

    # Read free surface
    eta = read_netcdf(file_path, 'ETAN', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(eta.shape)==2:
        # Just one timestep; add a dummy time dimension
        eta = np.expand_dims(eta,0)
    # Calculate volume without free surface changes
    volume = np.sum(grid.dV)
    # Build the timeseries
    timeseries = []
    for t in range(eta.shape[0]):
        # Get volume change in top layer due to free surface
        volume_top = np.sum(eta[t,:]*grid.dA)
        timeseries.append(volume+volume_top)
    return np.array(timeseries)


# Calculate timeseries of the transport across the transect given by the two points. The sign convention is to assume point0 is "west" and point1 is "east", returning the "meridional" transport in the local coordinate system based on whether you want the net northward transport (direction='N') or southward (direction='S').
def timeseries_transport_transect (file_path, grid, point0, point1, direction='N', time_index=None, t_start=None, t_end=None, time_average=False):

    # Read u and v
    u = mask_3d(read_netcdf(file_path, 'UVEL', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype='u', time_dependent=True)
    v = mask_3d(read_netcdf(file_path, 'VVEL', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average), grid, gtype='v', time_dependent=True)
    if len(u.shape)==3:
        # Just one timestep; add a dummy time dimension
        u = np.expand_dims(u,0)
        v = np.expand_dims(v,0)
    # Build the timeseries
    timeseries = []
    for t in range(u.shape[0]):
        # Get the "southward" and "northward" components
        trans_S, trans_N =  transport_transect(u[t,:], v[t,:], grid, point0, point1)
        # Combine them
        if direction == 'N':
            trans = trans_N - trans_S
        elif direction == 'S':
            trans = trans_S - trans_N
        else:
            print(('Error (timeseries_transport_transect): invalid direction ' + direction))
            sys.exit()
        timeseries.append(trans)
    return np.array(timeseries)


# Helper function for timeseries_adv_dif and timeseries_adv_dif_bdry: read the x and y components of the data
def read_data_xy (file_path, var_name, time_index=None, t_start=None, t_end=None, time_average=False):

    # We were given the variable name for the x-component, now get the y-component
    var_x = var_name
    var_y = var_name.replace('x', 'y')
    data_x = read_netcdf(file_path, var_x, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    data_y = read_netcdf(file_path, var_y, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    # Don't actually need to do this bit because it will become a convergence of fluxes
    #if var_name == 'ADVx_TH':
        # Need to convert to heat advection relative to freezing point
        #u = read_netcdf(file_path, 'UVEL',  time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        #v = read_netcdf(file_path, 'VVEL',  time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        #grid = Grid(file_path)
        #[data_x, data_y] = adv_heat_wrt_freezing([data_x, data_y], [u, v], grid)
    if len(data_x.shape)==3:
        # Just one timestep; add a dummy time dimension
        data_x = np.expand_dims(data_x,0)
        data_y = np.expand_dims(data_y,0)
    return data_x, data_y


# Calculate the net horizontal advection or diffusion into the given 3D region.
def timeseries_adv_dif (file_path, var_name, grid, z0, time_index=None, t_start=None, t_end=None, time_average=False, mask=None):

    data_x, data_y = read_data_xy(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if z0 is not None:
        # Mask out bounds
        if mask is None:
            mask = np.ones([grid.ny, grid.nx]).astype(bool)
        mask = mask_2d_to_3d(mask, grid, zmin=z0[0], zmax=z0[1])
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data_x.shape[0]):
        # Sum the fluxes across each face, padding with zeros at the eastern and northern boundaries of the domain
        data_tmp = np.ma.zeros(data_x.shape[1:])
        data_tmp[:,:-1,:-1] = data_x[t,:,:-1,:-1] - data_x[t,:,:-1,1:] + data_y[t,:,:-1,:-1] - data_y[t,:,1:,:-1]
        # Sum over the given region
        data_tmp = mask_3d(data_tmp, grid)
        if mask is not None:
            data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
        timeseries.append(np.sum(data_tmp))
    return np.array(timeseries)


# Calculate the net vertical advection or diffusion into the given 3D region.
def timeseries_adv_dif_z (file_path, var_name, grid, z0, time_index=None, t_start=None, t_end=None, time_average=False, mask=None):

    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==3:
        data = np.expand_dims(data,0)
    if z0 is not None:
        # Mask out bounds
        if mask is None:
            mask = np.ones([grid.ny, grid.nx]).astype(bool)
        mask = mask_2d_to_3d(mask, grid, zmin=z0[0], zmax=z0[1])
    timeseries = []
    for t in range(data.shape[0]):
        data_tmp = np.ma.zeros(data.shape[1:])
        data_tmp[:-1,:] = data[t,1:,:] - data[t,:-1,:]
        data_tmp = mask_3d(data_tmp, grid)
        if mask is not None:
            data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
        timeseries.append(np.sum(data_tmp))
    return np.array(timeseries)


# Calculate the net horizontal advection or diffusion across the given boundary into the given region.
def timeseries_adv_dif_bdry (file_path, var_name, grid, region_mask, bdry_mask, time_index=None, t_start=None, t_end=None, time_average=False):

    data_x, data_y = read_data_xy(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    # Now get data_x and data_y shifted one index to the east and north respectively
    data_x_plus1 = neighbours(data_x)[1]
    data_y_plus1 = neighbours(data_y)[3]
    # Find which points have neighbours outside the region, in each direction
    outside_w, outside_e, outside_s, outside_n = neighbours(region_mask.astype(float), missing_val=1)[4:8]
    def face_bdry_mask (outside):
        return xy_to_xyz(bdry_mask*outside.astype(bool), grid)*(grid.hfac != 0)
    index_w = face_bdry_mask(outside_w)
    index_e = face_bdry_mask(outside_e)
    index_s = face_bdry_mask(outside_s)
    index_n = face_bdry_mask(outside_n)
    # Process one time index at a time
    timeseries = []
    for t in range(data_x.shape[0]):
        net_flux = 0
        # Sum the flux across the western faces of all cells whose western faces are on the boundary of the region
        if np.count_nonzero(index_w) > 0:
            net_flux += np.sum(data_x[t,index_w])
        # Similarly for the other boundaries
        if np.count_nonzero(index_e) > 0:
            net_flux += np.sum(-1*data_x_plus1[t,index_e])
        if np.count_nonzero(index_s) > 0:
            net_flux += np.sum(data_y[t,index_s])
        if np.count_nonzero(index_n) > 0:
            net_flux += np.sum(-1*data_y_plus1[t,index_n])
        timeseries.append(net_flux)
    return np.array(timeseries)


# Calculate the mean residence time of the given cavity from the barotropic streamfunction.
def timeseries_cavity_res_time (file_path, grid, shelf, time_index=None, t_start=None, t_end=None, time_average=False):

    # Get the 2D mask for this ice shelf
    shelf_mask = grid.get_ice_mask(shelf=shelf)
    # Calculate volume of cavity
    cavity_vol = np.sum(grid.dV*xy_to_xyz(shelf_mask, grid))
    # Read streamfunction, vertically integrate, and mask to given region
    psi = read_netcdf(file_path, 'PsiVEL', time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    psi = np.sum(psi, axis=-3)
    if len(psi.shape)==2:
        # Just one timestep; add a dummy time dimension
        psi = np.expand_dims(psi,0)
    psi = apply_mask(psi, np.invert(shelf_mask), time_dependent=True)    
    # Loop over timesteps
    timeseries = []
    for t in range(psi.shape[0]):
        # Area-average absolute value of streamfunction
        psi_mean = area_average(np.mean(psi[t,:]), grid)
        # Divide volume by this value to get mean residence time, convert to years
        res_time = cavity_vol/psi_mean/sec_per_year
        timeseries.append(res_time)
    return np.array(timeseries)


# Calculate the difference in density between the two points.
def timeseries_delta_rho (file_path, grid, point0, point1, z0, time_index=None, t_start=None, t_end=None, time_average=False, eosType='MDJWF'):

    # Inner function to read a variable (temperature or salinity) and interpolate it to the given point
    def read_interp_var (var_name, point):
        data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        if len(data.shape)==3:
            # Add a time dimension
            data = np.expand_dims(data,0)
        # Interpolate to the given depth
        data_xy = interp_to_depth(data, z0, grid, time_dependent=True)
        # Interpolate to the given point
        return interp_bilinear(data_xy, point[0], point[1], grid)
    
    # Inner function to do this for both temperature and salinity, and then calculate the timeseries of density at that point
    def density_point (point):
        salt0 = read_interp_var('SALT', point)
        temp0 = read_interp_var('THETA', point)
        return density(eosType, salt0, temp0, z0)

    # Return the difference between the two points
    return density_point(point0) - density_point(point1)


# Calculate the maximum value of the given variable along the given ice shelf front.
def timeseries_icefront_max (file_path, var_name, grid, shelf, time_index=None, t_start=None, t_end=None, time_average=False):

    mask = grid.get_icefront_mask(shelf=shelf)
    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    is_3d = var_name in ['THETA', 'SALT']  # Update this as needed when more variables are used
    if is_3d:
        mask = xy_to_xyz(mask, grid)*(grid.hfac!=0)
    if (is_3d and len(data.shape)==3) or len(data.shape)==2:
        # Just one timestep
        data = np.expand_dims(data,0)
    timeseries = []
    for t in range(data.shape[0]):
        data_tmp = data[t,:]
        timeseries.append(np.amax(data_tmp[mask]))
    return np.array(timeseries)


# Arguments:
# file_path: either a single filename or a list of filenames

# Optional keyword arguments:
# option: 'ismr': calculates net melting OR total melting and freezing beneath given ice shelf; must specify shelf and mass_balance
#          'max': calculates maximum value of variable in region; must specify var_name and possibly xmin etc.
#          'avg_sfc': calculates area-averaged value over the sea surface, i.e. not counting cavities
#          'int_sfc': calculates area-integrated value over the sea surface
#          'area_threshold': calculates area of sea surface where the variable exceeds the given threshold
#          'avg_3d': calculates volume-averaged value over the given region.
#          'int_3d': calculates volume-integrated value over the given region.
#          'point_vavg': calculates depth-averaged value interpolated to a specific lat-lon point
#          'wed_gyre_trans': calculates Weddell Gyre transport.
#          'watermass': calculates percentage volume of the water mass defined by any of tmin, tmax, smin, smax.
#          'transport_transect': calculates net transport across a given transect
#          'iceprod': calculates total sea ice production over the given region
#          'pmepr': calculates total precipitation minus evaporation plus runoff over the given region
#          'res_time': calculates mean cavity residence time over the given ice shelf cavity
#          'icefront_max': calculates maximum value over the given ice shelf front (2D or 3D variable)
#          'time': just returns the time array
# grid: as in function read_plot_latlon
# gtype: as in function read_plot_latlon
# region: ice shelf (for option='ismr') or region (for option='avg_3d').
# bdry: boundary key (for option='adv_dif_bdry')
# mass_balance, result: as in function timeseries_ismr. Only matters for 'ismr'.
# var_name: variable name to process. Doesn't matter for 'ismr' or 'wed_gyre_trans'.
# xmin, xmax, ymin, ymax: as in function var_min_max. Only matters for 'max'.
# val0: as in function timeseries_area_threshold. Only matters for 'area_threshold'.
# lon0, lat0: point to interpolate to. Only matters for 'point_vavg'.
# tmin, tmax, smin, smax: as in function timeseries_watermass_volume. Only matters for 'watermass'.
# point0, point1: endpoints of transect, each in form (lon, lat). Only matters for 'transport_transect' or 'delta_rho'.
# z0: specific depth for some timeseries types (negative, in metres)
# direction: 'N' or 'S', as in function timeseries_transport_transect. Only matters for 'transport_transect'.
# monthly: as in function netcdf_time
# rho: precomputed density field
# factor: constant value to multiply the timeseries by (default 1)
# offset: constant value to add to the timeseries (default 0)

# Output:
# if option='ismr' and mass_balance=True, returns three 1D arrays of time, melting, and freezing.
# if option='time', just returns the time array.
# Otherwise, returns two 1D arrays of time and the relevant timeseries.


def calc_timeseries (file_path, option=None, grid=None, gtype='t', var_name=None, region='fris', bdry=None, mass_balance=False, result='massloss', xmin=None, xmax=None, ymin=None, ymax=None, val0=None, lon0=None, lat0=None, tmin=None, tmax=None, smin=None, smax=None, point0=None, point1=None, z0=None, direction='N', monthly=True, rho=None, time_average=False, factor=1, offset=0):

    if option not in ['time', 'ismr', 'wed_gyre_trans', 'watermass', 'volume', 'transport_transect', 'iceprod', 'pmepr', 'res_time', 'delta_rho', 'thermocline'] and var_name is None:
        print('Error (calc_timeseries): must specify var_name')
        sys.exit()
    if option == 'point_vavg' and (lon0 is None or lat0 is None):
        print('Error (calc_timeseries): must specify lon0 and lat0')
        sys.exit()
    if option in ['area_threshold', 'iso_depth'] and val0 is None:
        print('Error (calc_timeseries): must specify val0')
        sys.exit()
    if option in ['transport_transect', 'delta_rho'] and (point0 is None or point1 is None):
        print('Error (calc_timeseries): must specify point0 and point1')
        sys.exit()
    elif option in ['delta_rho', 'avg_z0', 'avg_btw_z0', 'int_btw_z0'] and z0 is None:
        print('Error (calc_timeseries): must specify z0')
        sys.exit()
    if var_name == 'RHO' and rho is None:
        print('Error (calc_timeseries): must precompute density')
        sys.exit()
    if option == 'adv_dif_bdry' and bdry is None:
        print('Error (calc_timeseries): must specify bdry')
        sys.exit()

    if isinstance(file_path, str):
        # Just one file - make it a list of length 1
        file_path = [file_path]
    # Build the grid if needed
    if option != 'time':
        grid = choose_grid(grid, file_path[0])

    # Set region mask, if needed
    if option in ['avg_3d', 'int_3d', 'iceprod', 'avg_sfc', 'int_sfc', 'pmepr', 'adv_dif', 'adv_dif_z', 'adv_dif_bdry', 'avg_bottom', 'avg_z0', 'avg_btw_z0', 'int_btw_z0', 'thermocline', 'iso_depth', 'max']:
        if region == 'all' or region is None:
            mask = None
        elif region == 'fris':
            mask = grid.get_ice_mask(shelf=region)
        elif region.endswith('_front'):
            mask = grid.get_icefront_mask(shelf=region[:region.index('_front')])
        elif region.endswith('icefront'):  # I realise this is confusing
            mask = grid.get_region_bdry_mask(region[:region.index('_icefront')], 'icefront')
        elif region.endswith('openocean'):
            mask = grid.get_region_bdry_mask(region[:region.index('_openocean')], 'openocean')
        elif region.endswith('upstream'):
            mask = grid.get_region_bdry_mask(region[:region.index('_upstream')], 'upstream')
        elif region.endswith('downstream'):
            mask = grid.get_region_bdry_mask(region[:region.index('_downstream')], 'downstream')
        elif region == 'wdw_core':
            mask = grid.get_region_mask(region, is_3d=True)
        else:
            mask = grid.get_region_mask(region)
    if option == 'adv_dif_bdry':
        bdry_mask = grid.get_region_bdry_mask(region, bdry)
    
    melt = None
    freeze = None
    values = None
    time = None
    for fname in file_path:
        if option == 'ismr':
            if mass_balance:
                melt_tmp, freeze_tmp = timeseries_ismr(fname, grid, shelf=region, mass_balance=mass_balance, result=result, time_average=time_average, z0=z0)
            else:
                values_tmp = timeseries_ismr(fname, grid, shelf=region, mass_balance=mass_balance, result=result, time_average=time_average, z0=z0)
        elif option == 'max':
            values_tmp = timeseries_max(fname, var_name, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mask=mask, time_average=time_average)
        elif option == 'avg_sfc':
            values_tmp = timeseries_avg_sfc(fname, var_name, grid, gtype=gtype, mask=mask, time_average=time_average)
        elif option == 'int_sfc':
            values_tmp = timeseries_int_sfc(fname, var_name, grid, gtype=gtype, mask=mask, time_average=time_average)
        elif option == 'area_threshold':
            values_tmp = timeseries_area_threshold(fname, var_name, val0, grid, gtype=gtype, time_average=time_average)
        elif option == 'avg_3d':
            values_tmp = timeseries_avg_3d(fname, var_name, grid, gtype=gtype, mask=mask, rho=rho, time_average=time_average)
        elif option == 'int_3d':
            values_tmp = timeseries_int_3d(fname, var_name, grid, gtype=gtype, mask=mask, time_average=time_average)
        elif option == 'point_vavg':
            values_tmp = timeseries_point_vavg(fname, var_name, lon0, lat0, grid, gtype=gtype, time_average=time_average)
        elif option == 'wed_gyre_trans':
            values_tmp = timeseries_wed_gyre(fname, grid, time_average=time_average)
        elif option == 'watermass':
            values_tmp = timeseries_watermass_volume(fname, grid, tmin=tmin, tmax=tmax, smin=smin, smax=smax, time_average=time_average)
        elif option == 'volume':
            values_tmp = timeseries_domain_volume(fname, grid, time_average=time_average)
        elif option == 'transport_transect':
            values_tmp = timeseries_transport_transect(fname, grid, point0, point1, direction=direction, time_average=time_average)
        elif option == 'iceprod':
            values_tmp = timeseries_int_sfc(fname, ['SIdHbOCN', 'SIdHbATC', 'SIdHbATO', 'SIdHbFLO'], grid, mask=mask, time_average=time_average)
        elif option == 'pmepr':
            values_tmp = timeseries_int_sfc(fname, ['oceFWflx', 'SIfwmelt', 'SIfwfrz'], grid, mask=mask, time_average=time_average, operator='subtract')
        elif option == 'adv_dif':
            values_tmp = timeseries_adv_dif(fname, var_name, grid, z0, mask=mask, time_average=time_average)
        elif option == 'adv_dif_z':
            values_tmp = timeseries_adv_dif_z(fname, var_name, grid, z0, mask=mask, time_average=time_average)
        elif option == 'adv_dif_bdry':
            values_tmp = timeseries_adv_dif_bdry(fname, var_name, grid, mask, bdry_mask, time_average=time_average)
        elif option == 'res_time':
            values_tmp = timeseries_cavity_res_time(fname, grid, region, time_average=time_average)
        elif option == 'delta_rho':
            values_tmp = timeseries_delta_rho(fname, grid, point0, point1, z0, time_average=time_average)
        elif option == 'icefront_max':
            values_tmp = timeseries_icefront_max(fname, var_name, grid, region, time_average=time_average)
        elif option == 'avg_bottom':
            values_tmp = timeseries_avg_bottom(fname, var_name, grid, gtype=gtype, mask=mask, rho=rho, time_average=time_average)
        elif option == 'avg_z0':
            values_tmp = timeseries_avg_z0(fname, var_name, z0, grid, gtype=gtype, mask=mask, rho=rho, time_average=time_average)
        elif option == 'avg_btw_z0':
            values_tmp = timeseries_avg_btw_z0(fname, var_name, z0, grid, gtype=gtype, mask=mask, rho=rho, time_average=time_average)
        elif option == 'int_btw_z0':
            values_tmp = timeseries_int_btw_z0(fname, var_name, z0, grid, gtype=gtype, mask=mask, rho=rho, time_average=time_average)
        elif option == 'thermocline':
            values_tmp = timeseries_thermocline(fname, grid, mask=mask, time_average=time_average)
        elif option == 'iso_depth':
            values_tmp = timeseries_iso_depth(fname, var_name, val0, grid, z0=z0, mask=mask, time_average=time_average)
        if not (option == 'ismr' and mass_balance):
            values_tmp = values_tmp*factor + offset
        time_tmp = netcdf_time(fname, monthly=monthly)
        if time_average:
            # Just save the first time index
            time_tmp = np.array([time_tmp[0]])
        if time is None:
            # Initialise the arrays
            if option == 'ismr' and mass_balance:
                melt = melt_tmp
                freeze = freeze_tmp
            elif option != 'time':
                values = values_tmp
            time = time_tmp
        else:
            # Concatenate the arrays
            if option == 'ismr' and mass_balance:
                melt = np.concatenate((melt, melt_tmp))
                freeze = np.concatenate((freeze, freeze_tmp))
            elif option != 'time':
                values = np.concatenate((values, values_tmp))
            time = np.concatenate((time, time_tmp))

    if option == 'ismr' and mass_balance:
        return time, melt, freeze
    elif option == 'time':
        return time
    else:
        return time, values


# Helper function to calculate difference timeseries, trimming if needed.

# Arguments:
# time_1, time_2: 1D arrays containing time values for the two simulations (assumed to start at the same time, but might not be the same length)
# data_1, data_2: Arrays containing timeseries for the two simulations. Can be any dimension as long as time is the first one.

# Output:
# time: 1D array containing time values for the overlapping period of simulation
# data_diff: Array containing differences (data_2 - data_1) at these times
def trim_and_diff (time_1, time_2, data_1, data_2):

    num_time = min(time_1.size, time_2.size)
    time = time_1[:num_time]
    data_diff = data_2[:num_time,...] - data_1[:num_time,...]
    return time, data_diff


# Call calc_timeseries twice, for two simulations, and calculate the difference in the timeseries. Doesn't work for the complicated case of timeseries_ismr with mass_balance=True.
def calc_timeseries_diff (file_path_1, file_path_2, option=None, region='fris', bdry=None, mass_balance=False, result='massloss', var_name=None, grid=None, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, val0=None, lon0=None, lat0=None, tmin=None, tmax=None, smin=None, smax=None, point0=None, point1=None, z0=None, direction='N', monthly=True, rho=None, factor=1, offset=0):

    if option == 'ismr' and mass_balance:
        print("Error (calc_timeseries_diff): this function can't be used for ice shelf mass balance")
        sys.exit()

    # Calculate timeseries for each
    time_1, values_1 = calc_timeseries(file_path_1, option=option, var_name=var_name, grid=grid, gtype=gtype, region=region, bdry=bdry, mass_balance=mass_balance, result=result, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, val0=val0, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, z0=z0, direction=direction, monthly=monthly, rho=rho, factor=factor, offset=offset)
    time_2, values_2 = calc_timeseries(file_path_2, option=option, var_name=var_name, grid=grid, gtype=gtype, region=region, bdry=bdry, mass_balance=mass_balance, result=result, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, val0=val0, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, z0=z0, direction=direction, monthly=monthly, rho=rho, factor=factor, offset=offset)
    # Find the difference, trimming if needed
    time, values_diff = trim_and_diff(time_1, time_2, values_1, values_2)
    return time, values_diff


# Set a bunch of parameters corresponding to a given timeseries variable:
#      '*_mass_balance': melting, freezing, and net melting beneath the given ice shelf (including 'all')
#      '*_massloss': net mass loss beneath the given ice shelf
#      '*_melting': average melt rate beneath the given ice shelf
#      '*_temp', '*_salt', '*_age', '*_density':
#                volume-averaged temperature, salinity, age tracer, or potential density in the given region (defined in constants.py)
#      '*_temp_bottom': area-averaged bottom temperature over the given region
#      '*_salt_bottom': area-averaged bottom salinity over the given region
#      '*_density_bottom': area-averaged bottom density over the given region
#      '*_density_*m': area-averaged density at the given depth (positive, in metres) over the given region - eg offshore_filchner_density_600m
#      'hice_corner': maximum sea ice thickness in the southwest corner of the Weddell Sea, between the Ronne and the peninsula
#      'hice_max': maximum sea ice thickness in whole domain
#      'mld_ewed': maximum mixed layer depth in the open Eastern Weddell Sea
#      'eta_avg': area-averaged sea surface height
#      'seaice_area': total sea ice area
#      'temp_polynya': depth-averaged temperature through the centre of a polynya
#      'salt_polynya': depth-averaged salinity through the centre of a polynya
#      'conv_area': total area of convection (MLD > 2000 m)
#      'wed_gyre_trans': Weddell Gyre transport
#      'isw_vol': volume of ISW (<-1.9 C)
#      'hssw_vol': volume of HSSW (-1.8 to -1.9 C, >34.55 psu)
#      'wdw_vol': volume of WDW (>0 C)
#      'mwdw_vol': volume of MWDW (-1.5 to 0 C, 34.25-34.55 psu)
#      'ocean_vol': volume of entire ocean domain (m^3)
#      'filchner_trans': transport across the Filchner Ice Shelf front into the cavity (Sv)
#      '*_atemp_avg': surface air temperature averaged over the given region (or entire domain if just atemp_avg) (C)
#      '*_wind_avg': wind speed averaged over the given region (C)
#      '*_sst_avg': sea surface temperature averaged over the given region (C)
#      '*_sss_avg': sea surface salinity averaged over the given region (psu)
#      '*_iceprod': total sea ice production over the given region (10^3 m^3/s)
#      '*_seaice_melt': total freshwater flux from sea ice melting over the given region (10^3 m^3/s)
#      '*_seaice_freeze': total freshwater flux from sea ice freezing over the given region (10^3 m^3/s)
#      '*_pmepr': total precipitation minus evaporation plus runoff over the given region, not counting precipitation or evaporation over sea ice (10^3 m^3/s)
#      '*_salt_adv': horizontal advection of salt integrated over the given region (psu m^3/s)
#      '*_salt_dif': horizontal diffusion of salt integrated over the given region (psu m^3/s)
#      '*_salt_adv_*', '*_salt_dif_*': as above but integrated over the given boundary (psu m^3/s)
#      '*_salt_sfc': surface salt flux integrated over the given region (psu m^3/s)
#      '*_salt_sfc_corr': surface salt correction term (from linear free surface) integrated over the given region (psu m^3/s) - assumes linFSConserve=false
#      '*_salt_tend': total salt tendency integrated over the given region (psu m^3/s)
#      '*_res_time': mean cavity residence time for the given ice shelf (years)
#      '*_mean_psi': mean absolute value of the barotropic streamfunction for the given region (Sv)
#      '*_max_psi': maximum absolute value of the barotropic streamfunction for the given region (Sv)
#      '*_ustar': area-averaged friction velocity under the given ice shelf (m/s)
#      '*_thermal_driving': area-averaged ice-ocean boundary temperature minus in-situ freezing point under the given ice shelf (C)
#      'ronne_delta_rho': difference in density between Ronne Depression and Ronne cavity
#      'ft_sill_delta_rho': difference in density between the onshore and offshore side of the Filchner Trough sill
#      '*_front_tmax': maximum temperature at the ice shelf front of the given ice shelf
#      '*_uwind_avg': zonal wind averaged over the given region
#      '*_temp_below_*': temperature volume-averaged below the given depth of the given region, eg pine_island_bay_temp_below_500m
#      '*_salt_below_*': similar for salinity
def set_parameters (var):

    var_name = None
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    region = None
    bdry = None
    mass_balance = None
    result = None
    val0 = None
    tmin = None
    tmax = None
    smin = None
    smax = None
    point0 = None
    point1 = None
    z0 = None
    direction = None
    factor = 1
    offset = 0

    if var.endswith('mass_balance') or var.endswith('massloss') or var.endswith('melting'):
        option = 'ismr'
        var_name = 'SHIfwFlx'
        mass_balance = var.endswith('mass_balance')
        if var.endswith('melting'):
            result = 'melting'
            units = 'm/y'
        else:
            result = 'massloss'
            units = 'Gt/y'
        # Extract name of ice shelf
        if var.endswith('mass_balance'):
            region = var[:var.index('_mass_balance')]
            title = 'Basal mass balance of '
        elif var.endswith('massloss'):
            region = var[:var.index('_massloss')]
            title = 'Basal mass loss from '
        elif var.endswith('melting'):
            region = var[:var.index('_melting')]
            title = 'Mean melt rate of '
        if region == 'all':
            title += 'all ice shelves in domain'
        else:
            title += region_names[region]
    elif 'mass_balance_btw' in var or 'massloss_btw' in var or 'melting_btw' in var:
        option = 'ismr'
        var_name = 'SHIfwFlx'
        mass_balance = 'mass_balance_btw' in var
        if 'melting_btw' in var:
            result = 'melting'
            units = 'm/y'
        else:
            result = 'massloss'
            units = 'Gt/y'
            if mass_balance:
                # Temporarily overwrite result for string convenience
                result = 'mass_balance'
        # Extract name of ice shelf
        region = var[:var.index('_'+result+'_btw')]
        if 'mass_balance_btw' in var:
            title = 'Basal mass balance of '
        elif 'melting_btw' in var:
            title = 'Mean melt rate of '
        else:
            title = 'Basal mass loss from '
        if region == 'all':
            title += 'all ice shelves in domain'
        else:
            title += region_names[region]
        # Extract depth range
        z_vals = var[len(region+'_'+result+'_btw_'):-1]
        z_shallow = -1*int(z_vals[:z_vals.index('_')])
        z_deep = -1*int(z_vals[z_vals.index('_')+1:])
        z0 = [z_deep, z_shallow]
        title += ' between '+str(-z_shallow)+'-'+str(-z_deep)+'m'
        if mass_balance:
            result = 'massloss'
    elif var.endswith('_temp') or var.endswith('_salt') or var.endswith('_density') or var.endswith('_age') or var.endswith('_tminustf'):
        option = 'avg_3d'
        title = 'Volume-averaged '
        if var.endswith('_temp'):
            region = var[:var.index('_temp')]
            var_name = 'THETA'
            title += 'temperature '
            units = deg_string+'C'
        elif var.endswith('_salt'):
            region = var[:var.index('_salt')]
            var_name = 'SALT'
            title += 'salinity '
            units = 'psu'
        elif var.endswith('_density'):
            region = var[:var.index('_density')]
            var_name = 'RHO'
            title += 'potential density '
            units = r'kg/m$^3$'
        elif var.endswith('_age'):
            region = var[:var.index('_age')]
            var_name = 'TRAC01'
            title += 'age tracer'
            units = 'years'
        elif var.endswith('_tminustf'):
            region = var[:var.index('_tminustf')]
            var_name = 'TMINUSTF'  # Special case in timeseries_vol_3d
            title += 'depression from surface freezing point'
            units = deg_string+'C'
        if region == 'avg':
            region = 'all'
        elif region == 'fris':
            title += 'in FRIS cavity'
        elif region.endswith('icefront'):
            title += 'in '+region_names[region[:region.index('_icefront')]]+'\n'+region_names['icefront']
        elif region.endswith('openocean'):
            title += 'in '+region_names[region[:region.index('_openocean')]]+'\n'+region_names['openocean']
        elif region.endswith('upstream'):
            title += 'in '+region_names[region[:region.index('_upstream')]]+'\n'+region_names['upstream']
        elif region.endswith('downstream'):
            title += 'in '+region_names[region[:region.index('_downstream')]]+'\n'+region_names['downstream']
        else:
            title += 'in '+region_names[region]
    elif var.endswith('_temp_bottom'):
        option = 'avg_bottom'
        region = var[:var.index('_temp_bottom')]
        var_name = 'THETA'
        title = 'Bottom temperature in '+region_names[region]
        units = deg_string+'C'
    elif var.endswith('_salt_bottom'):
        option = 'avg_bottom'
        region = var[:var.index('_salt_bottom')]
        var_name = 'SALT'
        title = 'Bottom salinity in '+region_names[region]
        units = 'psu'
    elif var.endswith('_density_bottom'):
        option = 'avg_bottom'
        region = var[:var.index('_density_bottom')]
        var_name = 'RHO'
        title = 'Bottom density in '+region_names[region]
        units = r'kg/m^$3$'
    elif '_density_' in var and var.endswith('m'):
        option = 'avg_z0'
        region = var[:var.index('_density_')]
        z0 = -1*int(var[len(region+'_density_'):-1])
        var_name = 'RHO'
        title = str(abs(z0))+'m density in '+region_names[region]
        units = r'kg/m^$3$'
    elif var in ['hice_corner', 'mld_ewed', 'hice_max']:
        # Maximum between spatial bounds
        option = 'max'
        if var == 'hice_corner':
            var_name = 'SIheff'
            xmin = -62
            xmax = -59.5
            ymin = -75.5
            ymax = -74
            title = 'Maximum sea ice thickness in problematic corner'
            units = 'm'
        elif var == 'hice_max':
            var_name = 'SIheff'
            xmin = None
            xmax = None
            ymin = None
            ymax = None
            title = 'Maximum sea ice thickness in domain'
            units = 'm'
        elif var == 'mld_ewed':
            var_name = 'MXLDEPTH'
            xmin = -30
            xmax = 30
            ymin = -69
            ymax = -60
            title = 'Maximum mixed layer depth in Eastern Weddell'
            units = 'm'
    elif var == 'eta_avg':
        option = 'avg_sfc'
        var_name = 'ETAN'
        title = 'Area-averaged sea surface height'
        units = 'm'
    elif var == 'seaice_area':
        option = 'int_sfc'
        var_name = 'SIarea'
        title = 'Total sea ice area'
        units = r'million km$^2$'
    elif var in ['temp_polynya', 'salt_polynya']:
        option = 'point_vavg'
        if var == 'temp_polynya':
            var_name = 'THETA'
            title = 'Depth-averaged temperature in polynya'
            units = deg_string+'C'
        elif var == 'salt_polynya':
            var_name = 'SALT'
            title = 'Depth-averaged salinity in polynya'
            units = 'psu'
    elif var == 'conv_area':
        option = 'area_threshold'
        var_name = 'MXLDEPTH'
        val0 = 2000.
        title = 'Convective area'
        units = r'million km$^2$'
    elif var == 'wed_gyre_trans':
        option = 'wed_gyre_trans'
        title = 'Weddell Gyre transport'
        units = 'Sv'
    elif var in ['isw_vol', 'hssw_vol', 'wdw_vol', 'mwdw_vol']:
        option = 'watermass'
        units = '% of domain'
        if var == 'isw_vol':
            tmax = -1.9
            title ='Volume of ISW'
        elif var == 'hssw_vol':
            tmin = -1.9
            tmax = -1.8
            smin = 34.55
            title = 'Volume of HSSW'
        elif var == 'wdw_vol':
            tmin = 0
            title = 'Volume of WDW'
        elif var == 'mwdw_vol':
            tmin = -1.5
            tmax = 0
            smin = 34.25
            smax = 34.55
            title = 'Volume of MWDW'
    elif var == 'ocean_vol':
        option = 'volume'
        units = r'm$^3$'
        title = 'Volume of ocean in domain'
    elif var == 'filchner_trans':
        option = 'transport_transect'
        units = 'Sv'
        title = 'Transport across Filchner Ice Shelf front'
        point0 = (-45, -78.05)
        point1 = (-34.8, -78.2)
        direction = 'S'
    elif var.endswith('atemp_avg'):
        option = 'avg_sfc'
        var_name = 'EXFatemp'
        title = 'Surface air temperature'
        units = deg_string+'C'
        if var == 'atemp_avg':
            region = 'all'
        else:
            region = var[:var.index('_atemp_avg')]
            title += ' over ' + region_names[region]
        offset = -1*temp_C2K
    elif var.endswith('_wind_avg'):
        option = 'avg_sfc'
        var_name = 'EXFwind'
        units = 'm/s'
        region = var[:var.index('_wind_avg')]
        title = 'Wind speed over ' + region_names[region]
    elif var.endswith('uwind_avg'):
        option = 'avg_sfc'
        var_name = 'EXFuwind'
        units = 'm/s'
        region = var[:var.index('_uwind_avg')]
        title = 'Zonal wind averaged over ' + region_names[region]
    elif var.endswith('aqh_avg'):
        option = 'avg_sfc'
        var_name = 'EXFaqh'
        region = var[:var.index('_aqh_avg')]
        title = 'Specific humidity over '+region_names[region]
        units = r'10$^{-3}$ kg/kg'
        factor = 1e3
    elif var.endswith('precip_avg'):
        option = 'avg_sfc'
        var_name = 'EXFpreci'
        region = var[:var.index('_precip_avg')]
        title = 'Precipitation averaged over '+region_names[region]
        units = r'10$^{-9}$ m/s'
        factor = 1e9
    elif var.endswith('sst_avg'):
        option = 'avg_sfc'
        var_name = 'THETA'
        units = deg_string+'C'
        region = var[:var.index('_sst_avg')]
        title = 'Sea surface temperature over ' + region_names[region]
    elif var.endswith('sss_avg'):
        option = 'avg_sfc'
        var_name = 'SALT'
        units = 'psu'
        region = var[:var.index('_sss_avg')]
        title = 'Sea surface salinity over ' + region_names[region]
    elif var.endswith('hice_avg'):
        option = 'avg_sfc'
        var_name = 'SIheff'
        units = 'm'
        region = var[:var.index('_hice_avg')]
        title = 'Average sea ice thickness over ' + region_names[region]
    elif var.endswith('iceprod'):
        option = 'iceprod'
        region = var[:var.index('_iceprod')]
        title = 'Total sea ice production over ' + region_names[region]
        units = r'10$^3$ m$^3$/s'
        factor = 1e-3
    elif var.endswith('seaice_melt'):
        option = 'int_sfc'
        var_name = 'SIfwmelt'
        region = var[:var.index('_seaice_melt')]
        title = 'Total freshwater flux from sea ice melting over ' + region_names[region]
        units = r'10$^3$ m$^3$/s'
        factor = 1e-6
    elif var.endswith('seaice_freeze'):
        option = 'int_sfc'
        var_name = 'SIfwfrz'
        region = var[:var.index('_seaice_freeze')]
        title = 'Total freshwater flux from sea ice freezing over ' + region_names[region]
        units = r'10$^3$ m$^3$/s'
        factor = 1e-6
    elif var.endswith('pmepr'):
        option = 'pmepr'
        region = var[:var.index('_pmepr')]
        title = 'Total freshwater flux from precipitation, evaporation, and runoff over ' + region_names[region]
        units = r'10$^3$ m$^3$/s'
        factor = 1e-6
    elif 'salt_adv' in var:
        var_name = 'ADVx_SLT'
        region = var[:var.index('_salt_adv')]
        units = r'psu m$^3$/s'
        title = 'Net horizontal advection of salt into ' + region_names[region]
        if var.endswith('salt_adv'):
            option = 'adv_dif'
        else:
            option = 'adv_dif_bdry'
            bdry = var[var.index('salt_adv_')+len('salt_adv_'):]
            title += ' from ' + region_names[bdry]
    elif 'salt_dif' in var:
        var_name = 'DFxE_SLT'
        region = var[:var.index('_salt_dif')]
        units = r'psu m$^3$/s'
        title = 'Net horizontal diffusion of salt into ' + region_names[region]
        if var.endswith('salt_dif'):
            option = 'adv_dif'
        else:
            option = 'adv_dif_bdry'
            bdry = var[var.index('salt_dif_')+len('salt_dif_'):]
            title += ' from ' + region_names[bdry]
    elif var.endswith('salt_sfc'):
        option = 'int_sfc'
        var_name = 'SFLUX'
        region = var[:var.index('_salt_sfc')]
        title = 'Total surface salt flux over ' + region_names[region]
        units = r'psu m$^3$/s'
        factor = 1./rhoConst
    elif var.endswith('salt_sfc_corr'):
        option = 'int_sfc'
        var_name = 'WSMsfc'
        region = var[:var.index('_salt_sfc_corr')]
        title = 'Total linear free surface salt correction over ' + region_names[region]
        units = r'psu m$^3/s'
        factor = -1
    elif var.endswith('salt_tend'):
        option = 'int_3d'
        var_name = 'TOTSTEND'
        region = var[:var.index('_salt_tend')]
        title = 'Total tendency of salinity over ' + region_names[region]
        units = r'psu m$^3/s'
        factor = 1./sec_per_day
    elif var.endswith('res_time'):
        option = 'res_time'
        region = var[:var.index('_res_time')]
        title = 'Mean residence time of ' + region_names[region] + ' cavity'
        units = 'y'
    elif var.endswith('mean_psi'):
        option = 'avg_sfc'
        var_name = 'PsiVEL'
        region = var[:var.index('_mean_psi')]
        title = 'Mean absolute barotropic streamfunction\nfor ' + region_names[region]
        units = 'Sv'
        factor = 1e-6
    elif var.endswith('max_psi'):
        option = 'max'
        var_name = 'PsiVEL'
        region = var[:var.index('_max_psi')]
        title = 'Maximum absolute barotropic streamfunction\nfor ' + region_names[region]
        units = 'Sv'
        factor = 1e-6
    elif var.endswith('ustar'):
        option = 'avg_sfc'
        var_name = 'SHIuStar'
        region = var[:var.index('_ustar')]
        title = 'Mean friction velocity beneath ' + region_names[region]
        units = 'm/s'
    elif var.endswith('thermal_driving'):
        option = 'avg_sfc'
        var_name = 'TminusTf'
        region = var[:var.index('_thermal_driving')]
        title = 'Mean thermal driving beneath ' + region_names[region]
        units = deg_string+'C'
    elif var == 'ft_sill_delta_rho':
        option = 'delta_rho'
        point0 = (-32, -75)
        point1 = (-32, -74)
        z0 = -600
        title = 'Difference in density at 600 m across the Filchner Trough sill'
        units = r'kg/m$^3$'
    elif var == 'ronne_delta_rho':
        option = 'delta_rho'
        point0 = (-60, -74.75)
        point1 = (-70, -78)
        z0 = -600
        title = 'Difference in density at 600 m across Ronne Depression'
        units = r'kg/m$^3$'
    elif var.endswith('front_tmax'):
        option = 'icefront_max'
        var_name = 'THETA'
        region = var[:var.index('_front_tmax')]
        title = 'Maximum temperature at the ' + region_names[region] + ' front'
        units = deg_string+'C'
    elif '_temp_below_' in var:
        option = 'avg_btw_z0'
        var_name = 'THETA'
        region = var[:var.index('_temp_below')]
        z_shallow = -1*int(var[len(region+'_temp_below_'):-1])
        z_deep = None
        z0 = [z_deep, z_shallow]
        title = 'Average temperature below '+str(-z_shallow)+'m in '+region_names[region]
        units = deg_string+'C'
    elif '_temp_btw_' in var:
        option = 'avg_btw_z0'
        var_name = 'THETA'
        region = var[:var.index('_temp_btw')]
        z_vals = var[len(region+'_temp_btw_'):-1]
        z_shallow = -1*int(z_vals[:z_vals.index('_')])
        z_deep = -1*int(z_vals[z_vals.index('_')+1:])
        z0 = [z_deep, z_shallow]
        title = 'Average temperature between '+str(-z_shallow)+'-'+str(-z_deep)+'m in '+region_names[region]
        units = deg_string+'C'        
    elif '_salt_below_' in var:
        option = 'avg_btw_z0'
        var_name = 'SALT'
        region = var[:var.index('_salt_below')]
        z_shallow = -1*int(var[len(region+'_salt_below_'):-1])
        z_deep = None
        z0 = [z_deep, z_shallow]
        title = 'Average salinity below '+str(-z_shallow)+'m in '+region_names[region]
        units = 'psu'
    elif '_salt_btw_' in var:
        option = 'avg_btw_z0'
        var_name = 'SALT'
        region = var[:var.index('_salt_btw')]
        z_vals = var[len(region+'_salt_btw_'):-1]
        z_shallow = -1*int(z_vals[:z_vals.index('_')])
        z_deep = -1*int(z_vals[z_vals.index('_')+1:])
        z0 = [z_deep, z_shallow]
        title = 'Average salinity between '+str(-z_shallow)+'-'+str(-z_deep)+'m in '+region_names[region]
        units = 'psu'
    elif var.endswith('_thermocline'):
        option = 'thermocline'
        var_name = 'THETA'
        region = var[:var.index('_thermocline')]
        title = 'Average thermocline depth in '+region_names[region]
        units = 'm'
    elif 'isotherm' in var:
        option = 'iso_depth'
        var_name = 'THETA'
        region = var[:var.index('_isotherm')]
        var_tail = var[len(region+'_isotherm_'):]
        if 'below' in var_tail:
            val0 = var_tail[:var_tail.index('C_below')]
            z0 = -1*int(var_tail[len(val0+'C_below_'):-1])
            val0 = float(val0)
        else:
            val0 = float(var_tail[:-1])
            z0 = None
        title = 'Average depth of '+str(val0)+deg_string+'C isotherm in '+region_names[region]
        units = 'm'            
    elif '_adv_heat_ns' in var:
        option = 'int_btw_z0'
        var_name = 'ADVy_TH'
        region = var[:var.index('_adv_heat_ns')]
        # Parse depth range
        z_vals = var[len(region+'_adv_heat_ns_'):-1]
        z_shallow = -1*int(z_vals[:z_vals.index('_')])
        z_deep = -1*int(z_vals[z_vals.index('_')+1:])
        z0 = [z_deep, z_shallow]
        title = 'Meridional advection of heat in '+region_names[region]+', '+str(-z_shallow)+'-'+str(-z_deep)+'m'
        units = deg_string+r'C m$^3$/s'
    elif 'dohc_adv_below' in var:
        option = 'adv_dif'
        var_name = 'ADVx_TH'
        region = var[:var.index('_dohc_adv_below')]
        z_shallow = -1*int(var[len(region+'_dohc_adv_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Change in ocean heat content from horizontal advection in '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ/s'
    elif 'ohc_below' in var:
        option = 'int_btw_z0'
        var_name = 'THETA'
        region = var[:var.index('_ohc_below')]
        z_shallow = -1*int(var[len(region+'_ohc_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Ocean heat content in '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ'
    elif 'advection_heat_xy_below' in var:
        option = 'adv_dif'
        var_name = 'ADVx_TH'
        region = var[:var.index('_advection_heat_xy_below')]
        z_shallow = -1*int(var[len(region+'_advection_heat_xy_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Net horizontal advection of heat into '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ/s'
    elif 'advection_heat_z_below' in var:
        option = 'adv_dif_z'
        var_name = 'ADVr_TH'
        region = var[:var.index('_advection_heat_z_below')]
        z_shallow = -1*int(var[len(region+'_advection_heat_z_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Net vertical advection of heat into '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ/s'
    elif 'diffusion_heat_implicit_z_below' in var:
        option = 'adv_dif_z'
        var_name = 'DFrI_TH'
        region = var[:var.index('_diffusion_heat_implicit_z_below')]
        z_shallow = -1*int(var[len(region+'_diffusion_heat_implicit_z_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Net vertical implicit diffusion of heat into '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ/s'
    elif 'kpp_heat_z_below' in var:
        option = 'adv_dif_z'
        var_name = 'KPPg_TH'
        region = var[:var.index('_kpp_heat_z_below')]
        z_shallow = -1*int(var[len(region+'_kpp_heat_z_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Net vertical KPP transport of heat into '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ/s'
    elif 'shortwave_penetration_below' in var:
        option = 'int_btw_z0'
        var_name = 'shortwave_penetration'
        region = var[:var.index('_shortwave_penetration_below')]
        z_shallow = -1*int(var[len(region+'_shortwave_penetration_below_'):-1])
        z0 = [None, z_shallow]
        factor = 1e-9*Cp_sw*rhoConst
        title = 'Shortwave penetration of heat into '+region_names[region]+' below '+str(-z_shallow)+'m'
        units = 'GJ/s'
    else:
        print(('Error (set_parameters): invalid variable ' + var))
        sys.exit()

    return option, var_name, title, units, xmin, xmax, ymin, ymax, region, bdry, mass_balance, result, val0, tmin, tmax, smin, smax, point0, point1, z0, direction, factor, offset


# Interface to calc_timeseries for particular timeseries variables, defined in set_parameters.
def calc_special_timeseries (var, file_path, grid=None, lon0=None, lat0=None, monthly=True, rho=None, time_average=False):

    # Set parameters (don't care about title or units)
    option, var_name, title, units, xmin, xmax, ymin, ymax, region, bdry, mass_balance, result, val0, tmin, tmax, smin, smax, point0, point1, z0, direction, factor, offset = set_parameters(var)

    # Calculate timeseries
    if option == 'ismr' and mass_balance:
        # Special case for calc_timeseries, with extra output argument
        time, melt, freeze = calc_timeseries(file_path, option=option, region=region, mass_balance=mass_balance, grid=grid, monthly=monthly, time_average=time_average)
        return time, melt, freeze
    else:
        time, data = calc_timeseries(file_path, option=option, region=region, bdry=bdry, mass_balance=mass_balance, result=result, var_name=var_name, grid=grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, val0=val0, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, z0=z0, direction=direction, monthly=monthly, rho=rho, time_average=time_average, factor=factor, offset=offset)
        if var in ['seaice_area', 'conv_area']:
            # Convert from m^2 to million km^2
            data *= 1e-12
        return time, data


# Interface to calc_timeseries_diff for particular timeseries variables, defined in set_parameters.
def calc_special_timeseries_diff (var, file_path_1, file_path_2, grid=None, lon0=None, lat0=None, monthly=True, rho=None, time_average=False):

    # Set parameters (don't care about title or units)
    option, var_name, title, units, xmin, xmax, ymin, ymax, region, bdry, mass_balance, result, val0, tmin, tmax, smin, smax, point0, point1, z0, direction, factor, offset = set_parameters(var)

    # Calculate difference timeseries
    if option == 'ismr' and mass_balance:
        # Special case; calculate each timeseries separately because there are extra output arguments
        time_1, melt_1, freeze_1 = calc_timeseries(file_path_1, option=option, region=region, mass_balance=mass_balance, grid=grid, monthly=monthly, time_average=time_average)
        time_2, melt_2, freeze_2 = calc_timeseries(file_path_2, option=option, region=region, mass_balance=mass_balance, grid=grid, monthly=monthly, time_average=time_average)
        time, melt_diff = trim_and_diff(time_1, time_2, melt_1, melt_2)
        freeze_diff = trim_and_diff(time_1, time_2, freeze_1, freeze_2)[1]
        return time, melt_diff, freeze_diff
    else:
        time, data_diff = calc_timeseries_diff(file_path_1, file_path_2, option=option, var_name=var_name, region=region, bdry=bdry, mass_balance=mass_balance, result=result, grid=grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, val0=val0, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, z0=z0, direction=direction, monthly=monthly, rho=rho, time_average=time_average, factor=factor, offset=offset)
        if var in ['seaice_area', 'conv_area']:
            # Convert from m^2 to million km^2
            data_diff *= 1e-12
        return time, data_diff


# Given a monthly timeseries (and corresponding array of Date objects), calculate the annually-averaged timeseries. Return it as well as a new Date array with dates at the beginning of each year.
def monthly_to_annual (data, time):

    # Make sure we start at the beginning of a year
    if time[0].month != 1:
        print('Error (monthly_to_annual): timeseries must start with January.')
        sys.exit()

    # Weighted average of each year, taking days per month into account
    new_data = []
    new_time = []
    data_accum = 0
    ndays = 0
    for t in range(data.size):
        ndays_curr = days_per_month(time[t].month, time[t].year)
        data_accum += data[t]*ndays_curr
        ndays += ndays_curr
        if time[t].month == 12:
            # End of the year
            # Convert from integral to average
            new_data.append(data_accum/ndays)
            # Save the date at the beginning of the year
            new_time.append(datetime.date(time[t].year, 1, 1))
            # Reset the accumulation arrays
            data_accum = 0
            ndays = 0

    return np.array(new_data), np.array(new_time)


# Calculate annual averages from monthly data.
# This only works properly if it's a 360-day calendar with full years.
# Input arguments:
# times: either an array of time values, or a list of several arrays of time values
# datas: either an array of 1D data, or a list of several arrays of 1D data.
# Returns the annually-averaged version of each.
def calc_annual_averages (times, datas):

    time_single = not isinstance(times, list)
    data_single = not isinstance(datas, list)

    # Make each a list even if they're just singular
    if time_single:
        times = [times]
    if data_single:
        datas = [datas]

    # Get midpoint of each year
    for n in range(len(times)):
        times[n] = np.array([times[n][i] for i in range(6, times[n].size, 12)])
    # Average in blocks of 12
    for n in range(len(datas)):
        datas[n] = np.mean(datas[n].reshape(datas[n].shape[0]//12, 12), axis=-1)

    if time_single:
        times = times[0]
    if data_single:
        datas = datas[0]

    return times, datas

    
        
        

    

    


    
