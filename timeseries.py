#######################################################
# Calculation of integral timeseries
#######################################################

import numpy as np
import sys
import datetime

from grid import choose_grid
from file_io import read_netcdf, netcdf_time
from utils import convert_ismr, var_min_max, mask_land_ice, days_per_month, apply_mask, mask_3d
from diagnostics import total_melt, wed_gyre_trans, transport_transect
from calculus import over_area, area_integral, over_volume, vertical_average_column
from interpolation import interp_bilinear
from constants import deg_string, region_names, temp_C2K, sec_per_year, sec_per_day


# Calculate total mass loss or area-averaged melt rate from ice shelves in the given NetCDF file. You can specify specific ice shelves (as specified in region_names in constants.py). The default behaviour is to calculate the melt at each time index in the file, but you can also select a subset of time indices, and/or time-average - see optional keyword arguments. You can also split into positive (melting) and negative (freezing) components.

# Arguments:
# file_path: path to NetCDF file containing 'SHIfwFlx' variable
# grid = Grid object

# Optional keyword arguments:
# shelf: 'fris' (default) restricts the calculation to FRIS. 'ewed' restricts the calculation to ice shelves between the Eastern Weddell bounds given in constants.py. 'all' considers all ice shelves.
# result: 'massloss' (default) calculates the total mass loss in Gt/y. 'melting' calculates the area-averaged melt rate in m/y.
# time_index, t_start, t_end, time_average: as in function read_netcdf
# mass_balance: if True, split into positive (melting) and negative (freezing) terms. Default False.

# Output:
# If time_index is set, or time_average=True: single value containing mass loss or average melt rate
# Otherwise: 1D array containing timeseries of mass loss or average melt rate
# If mass_balance=True: two values/arrays will be returned, with the positive and negative components.

def timeseries_ismr (file_path, grid, shelf='fris', result='massloss', time_index=None, t_start=None, t_end=None, time_average=False, mass_balance=False):

    # Choose the appropriate mask
    mask = grid.get_ice_mask(shelf=shelf)

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
        return melt


# Read the given lat x lon variable from the given NetCDF file, and calculate timeseries of its maximum value in the given region.
def timeseries_max (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, xmin=None, xmax=None, ymin=None, ymax=None):

    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==2:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)

    num_time = data.shape[0]
    max_data = np.zeros(num_time)
    for t in range(num_time):
        max_data[t] = var_min_max(data[t,:], grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)[1]
    return max_data


# Helper function for timeseries_avg_sfc and timeseries_int_sfc.
def timeseries_area_sfc (option, file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, operator='add', rhoConst=1035.):
    
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
        else:
            data_tmp = read_netcdf(file_path, var, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
            if var == 'EXFatemp':
                # Convert from K to C
                data_tmp -= temp_C2K
            if var == 'SFLUX':
                # Divide by density
                data_tmp /= rhoConst
            if var == 'WSLTMASS':
                # Swap sign
                data_tmp *= -1
        if var in ['THETA', 'SALT', 'WSLTMASS']:
            # 3D variable; have to take surface
            if len(data_tmp.shape)==3:
                # Just one timestep
                data_tmp = data_tmp[0,:,:]
            else:
                data_tmp = data_tmp[:,0,:,:]                
        if data is None:
            data = data_tmp
        else:
            if operator == 'add':
                data += data_tmp
            elif operator == 'subtract':
                data -= data_tmp
            else:
                print 'Error (timeseries_area_sfc): invalid operator ' + operator
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
def timeseries_area_threshold (file_path, var_name, threshold, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False):

    # Read the data
    data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data.shape)==2:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)
    # Convert to array of 1s and 0s based on threshold
    data = (data >= threshold).astype(float)
    # Now build the timeseries
    timeseries = []
    for t in range(data.shape[0]):
        timeseries.append(area_integral(data[t,:], grid, gtype=gtype))
    return np.array(timeseries)


# Helper function for timeseries_avg_3d and timeseries_int_3d.
def timeseries_vol_3d (option, file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):

    if var_name == 'RHO':
        if rho is None:
            print 'Error (timeseries_avg_3d): must precompute density'
            sys.exit()
        data = rho
    else:
        data = read_netcdf(file_path, var_name, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
        if var_name == 'TOTSTEND':
            # Convert units
            data /= sec_per_day
    if len(data.shape)==3:
        # Just one timestep; add a dummy time dimension
        data = np.expand_dims(data,0)
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data.shape[0]):
        if mask is None:
            data_tmp = mask_3d(data[t,:], grid, gtype=gtype)
        else:
            data_tmp = apply_mask(data[t,:], np.invert(mask), depth_dependent=True)
        # Volume average or integrate
        timeseries.append(over_volume(option, data_tmp, grid, gtype=gtype))
    return np.array(timeseries)


# Read the given 3D variable from the given NetCDF file, and calculate timeseries of its volume-averaged value. Restrict it to the given mask (default just mask out the land).
def timeseries_avg_3d (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    return timeseries_vol_3d('average', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho)


# Same but volume-integrate.
def timeseries_int_3d (file_path, var_name, grid, gtype='t', time_index=None, t_start=None, t_end=None, time_average=False, mask=None, rho=None):
    return timeseries_vol_3d('integrate', file_path, var_name, grid, gtype=gtype, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average, mask=mask, rho=rho)


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
            print 'Error (timeseries_transport_transect): invalid direction ' + direction
            sys.exit()
        timeseries.append(trans)
    return np.array(timeseries)    


# Calculate the net horizontal advection or diffusion into the given 3D region.
def timeseries_adv_dif (file_path, var_name, grid, time_index=None, t_start=None, t_end=None, time_average=False, mask=None):

    # We were given the variable name for the x-component, now get the y-component
    var_x = var_name
    var_y = var_name.replace('x', 'y')
    data_x = read_netcdf(file_path, var_x, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    data_y = read_netcdf(file_path, var_y, time_index=time_index, t_start=t_start, t_end=t_end, time_average=time_average)
    if len(data_x.shape)==3:
        # Just one timestep; add a dummy time dimension
        data_x = np.expand_dims(data_x,0)
        data_y = np.expand_dims(data_y,0)
    # Process one time index at a time to save memory
    timeseries = []
    for t in range(data_x.shape[0]):
        # Sum the fluxes across each face, padding with zeros at the eastern and northern boundaries
        data_tmp = np.ma.zeros(data_x.shape[1:])
        data_tmp[:,:-1,:-1] = data_x[t,:,:-1,:-1] - data_x[t,:,:-1,1:] + data_y[t,:,:-1,:-1] - data_y[t,:,1:,:-1]
        # Sum over the given region
        if mask is None:
            data_tmp = mask_3d(data_tmp, grid)
        else:
            data_tmp = apply_mask(data_tmp, np.invert(mask), depth_dependent=True)
        timeseries.append(np.sum(data_tmp))
    return np.array(timeseries)            


# Calculate timeseries from one or more files.

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
#          'pminuse': calculates total precipitation minus evaporation over the given region
#          'time': just returns the time array
# grid: as in function read_plot_latlon
# gtype: as in function read_plot_latlon
# region: ice shelf (for option='ismr') or region (for option='avg_3d').
# mass_balance, result: as in function timeseries_ismr. Only matters for 'ismr'.
# var_name: variable name to process. Doesn't matter for 'ismr' or 'wed_gyre_trans'.
# xmin, xmax, ymin, ymax: as in function var_min_max. Only matters for 'max'.
# threshold: as in function timeseries_area_threshold. Only matters for 'area_threshold'.
# lon0, lat0: point to interpolate to. Only matters for 'point_vavg'.
# tmin, tmax, smin, smax: as in function timeseries_watermass_volume. Only matters for 'watermass'.
# point0, point1: endpoints of transect, each in form (lon, lat). Only matters for 'transport_transect'.
# direction: 'N' or 'S', as in function timeseries_transport_transect. Only matters for 'transport_transect'.
# monthly: as in function netcdf_time

# Output:
# if option='ismr' and mass_balance=True, returns three 1D arrays of time, melting, and freezing.
# if option='time', just returns the time array.
# Otherwise, returns two 1D arrays of time and the relevant timeseries.


def calc_timeseries (file_path, option=None, grid=None, gtype='t', var_name=None, region='fris', mass_balance=False, result='massloss', xmin=None, xmax=None, ymin=None, ymax=None, threshold=None, lon0=None, lat0=None, tmin=None, tmax=None, smin=None, smax=None, point0=None, point1=None, direction='N', monthly=True, rho=None, time_average=False):

    if option not in ['time', 'ismr', 'wed_gyre_trans', 'watermass', 'volume', 'transport_transect', 'iceprod', 'pminuse'] and var_name is None:
        print 'Error (calc_timeseries): must specify var_name'
        sys.exit()
    if option == 'point_vavg' and (lon0 is None or lat0 is None):
        print 'Error (calc_timeseries): must specify lon0 and lat0'
        sys.exit()
    if option == 'area_threshold' and threshold is None:
        print 'Error (calc_timeseries): must specify threshold'
        sys.exit()
    if option == 'transport_transect' and (point0 is None or point1 is None):
        print 'Error (calc_timeseries): must specify point0 and point1'
        sys.exit()
    if var_name == 'RHO' and rho is None:
        print 'Error (calc_timeseries): must precompute density'
        sys.exit()

    if isinstance(file_path, str):
        # Just one file - make it a list of length 1
        file_path = [file_path]
    # Build the grid if needed
    if option != 'time':
        grid = choose_grid(grid, file_path[0])

    # Set region mask, if needed
    if option in ['avg_3d', 'int_3d', 'iceprod', 'avg_sfc', 'int_sfc', 'pminuse', 'adv_dif']:
        if region == 'fris':
            mask = grid.get_ice_mask(shelf=region)
        elif region in ['sws_shelf', 'filchner_trough', 'ronne_depression']:
            mask = grid.get_region_mask(region)
        elif region == 'wdw_core':
            mask = grid.get_region_mask(region, is_3d=True)
        elif region == 'all':
            mask = None
    
    melt = None
    freeze = None
    values = None
    time = None
    for fname in file_path:
        if option == 'ismr':
            if mass_balance:
                melt_tmp, freeze_tmp = timeseries_ismr(fname, grid, shelf=region, mass_balance=mass_balance, result=result, time_average=time_average)
            else:
                values_tmp = timeseries_ismr(fname, grid, shelf=region, mass_balance=mass_balance, result=result, time_average=time_average)
        elif option == 'max':
            values_tmp = timeseries_max(fname, var_name, grid, gtype=gtype, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, time_average=time_average)
        elif option == 'avg_sfc':
            values_tmp = timeseries_avg_sfc(fname, var_name, grid, gtype=gtype, mask=mask, time_average=time_average)
        elif option == 'int_sfc':
            values_tmp = timeseries_int_sfc(fname, var_name, grid, gtype=gtype, time_average=time_average)
        elif option == 'area_threshold':
            values_tmp = timeseries_area_threshold(fname, var_name, threshold, grid, gtype=gtype, time_average=time_average)
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
            values_tmp = timeseries_int_sfc(fname, ['SIdHbOCN', 'SIdHbATC', 'SIdHbATO', 'SIdHbFLO'], grid, mask=mask, time_average=time_average)*1e-3
        elif option == 'pminuse':
            values_tmp = timeseries_int_sfc(fname, ['EXFpreci', 'EXFevap'], grid, mask=mask, time_average=time_average, operator='subtract')*1e-3
        elif option == 'adv_dif':
            values_tmp = timeseries_adv_dif(fname, var_name, grid, mask=mask, time_average=time_average)
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
def calc_timeseries_diff (file_path_1, file_path_2, option=None, region='fris', mass_balance=False, result='massloss', var_name=None, grid=None, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, threshold=None, lon0=None, lat0=None, tmin=None, tmax=None, smin=None, smax=None, point0=None, point1=None, direction='N', monthly=True, rho=None):

    if option == 'ismr' and mass_balance:
        print "Error (calc_timeseries_diff): this function can't be used for ice shelf mass balance"
        sys.exit()

    # Calculate timeseries for each
    time_1, values_1 = calc_timeseries(file_path_1, option=option, var_name=var_name, grid=grid, gtype=gtype, region=region, mass_balance=mass_balance, result=result, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, threshold=threshold, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, direction=direction, monthly=monthly, rho=rho)
    time_2, values_2 = calc_timeseries(file_path_2, option=option, var_name=var_name, grid=grid, gtype=gtype, region=region, mass_balance=mass_balance, result=result, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, threshold=threshold, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, direction=direction, monthly=monthly, rho=rho)
    # Find the difference, trimming if needed
    time, values_diff = trim_and_diff(time_1, time_2, values_1, values_2)
    return time, values_diff


# Set a bunch of parameters corresponding to a given timeseries variable:
#      '*_mass_balance': melting, freezing, and net melting beneath the given ice shelf (including 'all')
#      '*_massloss': net mass loss beneath the given ice shelf
#      '*_melting': average melt rate beneath the given ice shelf
#      '*_temp', '*_salt', '*_age', '*_density':
#                volume-averaged temperature, salinity, age tracer, or potential density in the given region:
#                'fris' (FRIS cavity),
#                'sws_shelf' (Southern Weddell Sea continental shelf),
#                'filchner_trough' (Filchner Trough),
#                'wdw_core' (WDW core region offshore),
#                'avg' (entire domain)
#      'hice_corner': maximum sea ice thickness in the southwest corner of the Weddell Sea, between the Ronne and the peninsula
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
#      '*_pminuse': total precipitation minus evaporation over the given region (10^3 m^3/s)
#      '*_salt_adv': horizontal advection of salt integrated over the given region (psu m^3/s)
#      '*_salt_dif': horizontal diffusion of salt integrated over the given region (psu m^3/s)
#      '*_salt_sfc': surface salt flux integrated over the given region (psu m^3/s)
#      '*_salt_sfc_corr': surface salt correction term (from linear free surface) integrated over the given region (psu m^3/s) - assumes linFSConserve=false
#      '*_salt_tend': total salt tendency integrated over the given region (psu m^3/s)
def set_parameters (var):

    var_name = None
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    region = None
    mass_balance = None
    result = None
    threshold = None
    tmin = None
    tmax = None
    smin = None
    smax = None
    point0 = None
    point1 = None
    direction = None

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
        title += region_names[region]
    elif var.endswith('_temp') or var.endswith('_salt') or var.endswith('_density') or var.endswith('_age'):
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
        if region == 'avg':
            region = 'all'
        elif region == 'fris':
            title += 'in FRIS cavity'
        else:
            title += 'in '+region_names[region]
    elif var in ['hice_corner', 'mld_ewed']:
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
        threshold = 2000.
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
    elif var.endswith('wind_avg'):
        option = 'avg_sfc'
        var_name = 'EXFwind'
        units = 'm/s'
        region = var[:var.index('_wind_avg')]
        title = 'Wind speed over ' + region_names[region]
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
    elif var.endswith('iceprod'):
        option = 'iceprod'
        region = var[:var.index('_iceprod')]
        title = 'Total sea ice production over ' + region_names[region]
        units = r'10$^3$ m$^3$/s'
    elif var.endswith('pminuse'):
        option = 'pminuse'
        region = var[:var.index('_pminuse')]
        title = 'Total precipitation minus evaporation over ' + region_names[region]
        units = r'10$^3$ m$^3$/s'
    elif var.endswith('salt_adv'):
        option = 'adv_dif'
        var_name = 'ADVx_SLT'
        region = var[:var.index('_salt_adv')]
        title = 'Total horizontal advection of salt into ' + region_names[region]
        units = r'psu m$^3$/s'
    elif var.endswith('salt_dif'):
        option = 'adv_dif'
        var_name = 'DFxE_SLT'
        region = var[:var.index('_salt_dif')]
        title = 'Total horizontal diffusion of salt into ' + region_names[region]
        units = r'psu m$^3$/s'
    elif var.endswith('salt_sfc'):
        option = 'int_sfc'
        var_name = 'SFLUX'
        region = var[:var.index('_salt_sfc')]
        title = 'Total surface salt flux over ' + region_names[region]
        units = r'psu m$^3$/s'
    elif var.endswith('salt_sfc_corr'):
        option = 'int_sfc'
        var_name = 'WSLTMASS'
        region = var[:var.index('salt_sfc_corr')]
        title = 'Total linear free surface salt correction over ' + region_names[region]
        units = r'psu m$^3/s'
    elif var.endswith('salt_tend'):
        option = 'int_3d'
        var_name = 'TOTSTEND'
        region = var[:var.index('salt_tend')]
        title = 'Total tendency of salinity over ' + region_names[region]
        units = r'psu m$^3/s'        
    else:
        print 'Error (set_parameters): invalid variable ' + var
        sys.exit()

    return option, var_name, title, units, xmin, xmax, ymin, ymax, region, mass_balance, result, threshold, tmin, tmax, smin, smax, point0, point1, direction


# Interface to calc_timeseries for particular timeseries variables, defined in set_parameters.
def calc_special_timeseries (var, file_path, grid=None, lon0=None, lat0=None, monthly=True, rho=None, time_average=False):

    # Set parameters (don't care about title or units)
    option, var_name, title, units, xmin, xmax, ymin, ymax, region, mass_balance, result, threshold, tmin, tmax, smin, smax, point0, point1, direction = set_parameters(var)

    # Calculate timeseries
    if option == 'ismr' and mass_balance:
        # Special case for calc_timeseries, with extra output argument
        time, melt, freeze = calc_timeseries(file_path, option=option, region=region, mass_balance=mass_balance, grid=grid, monthly=monthly, time_average=time_average)
        return time, melt, freeze
    else:
        time, data = calc_timeseries(file_path, option=option, region=region, mass_balance=mass_balance, result=result, var_name=var_name, grid=grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, threshold=threshold, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, direction=direction, monthly=monthly, rho=rho, time_average=time_average)
        if var in ['seaice_area', 'conv_area']:
            # Convert from m^2 to million km^2
            data *= 1e-12
        return time, data


# Interface to calc_timeseries_diff for particular timeseries variables, defined in set_parameters.
def calc_special_timeseries_diff (var, file_path_1, file_path_2, grid=None, lon0=None, lat0=None, monthly=True, rho=None, time_average=False):

    # Set parameters (don't care about title or units)
    option, var_name, title, units, xmin, xmax, ymin, ymax, region, mass_balance, result, threshold, tmin, tmax, smin, smax, point0, point1, direction = set_parameters(var)

    # Calculate difference timeseries
    if option == 'ismr' and mass_balance:
        # Special case; calculate each timeseries separately because there are extra output arguments
        time_1, melt_1, freeze_1 = calc_timeseries(file_path_1, option=option, region=region, mass_balance=mass_balance, grid=grid, monthly=monthly, time_average=time_average)
        time_2, melt_2, freeze_2 = calc_timeseries(file_path_2, option=option, region=region, mass_balance=mass_balance, grid=grid, monthly=monthly, time_average=time_average)
        time, melt_diff = trim_and_diff(time_1, time_2, melt_1, melt_2)
        freeze_diff = trim_and_diff(time_1, time_2, freeze_1, freeze_2)[1]
        return time, melt_diff, freeze_diff
    else:
        time, data_diff = calc_timeseries_diff(file_path_1, file_path_2, option=option, var_name=var_name, region=region, mass_balance=mass_balance, result=result, grid=grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, threshold=threshold, lon0=lon0, lat0=lat0, tmin=tmin, tmax=tmax, smin=smin, smax=smax, point0=point0, point1=point1, direction=direction, monthly=monthly, rho=rho, time_average=time_average)
        if var in ['seaice_area', 'conv_area']:
            # Convert from m^2 to million km^2
            data_diff *= 1e-12
        return time, data_diff


# Given a monthly timeseries (and corresponding array of Date objects), calculate the annually-averaged timeseries. Return it as well as a new Date array with dates at the beginning of each year.
def monthly_to_annual (data, time):

    # Make sure we start at the beginning of a year
    if time[0].month != 1:
        print 'Error (monthly_to_annual): timeseries must start with January.'
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
# datas: either an array of 1D data, or a list of several arrays of 1D data
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
        datas[n] = np.mean(datas[n].reshape(datas[n].size/12, 12), axis=-1)

    if time_single:
        times = times[0]
    if data_single:
        datas = datas[0]

    return times, datas

    
        
        

    

    


    
