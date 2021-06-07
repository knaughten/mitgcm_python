#######################################################
# Offline calculation of diagnostic variables
#######################################################

import numpy as np
import sys

from .constants import rho_ice, region_bounds, Cp_sw, Tf_ref
from .utils import z_to_xyz, add_time_dim, xy_to_xyz, var_min_max, check_time_dependent, mask_land, depth_of_max, mask_3d
from .calculus import area_integral, vertical_integral, indefinite_ns_integral
from .plot_utils.slices import get_transect
from .interpolation import interp_grid


# Calculate the adiabatic temperature gradient exactly like MITgcm does. This originates from section 7 of "Algorithms for computation of fundamental properties of seawater", UNESCO technical papers in marine science 44, 1983.
# Helper function for in_situ_temp.

# Arguments:
# temp: temperature (degC)
# salt: salinity (psu)
# press: pressure (dbar ~= absolute value of depth in m)
# Note that temp, salt, and press can be of any dimension, as long as they all match.

# Output: adiabatic temperature gradient in degC/dbar, same dimension as input arguments.

def ad_temp_grad (temp, salt, press):

    s_ref = 35.
    a0 = 3.5803e-5
    a1 = 8.5258e-6
    a2 = -6.836e-8
    a3 = 6.6228e-10
    b0 = 1.8932e-6
    b1 = -4.2393e-8
    c0 = 1.8741e-8
    c1 = -6.7795e-10
    c2 = 8.733e-12
    c3 = -5.4481e-14
    d0 = -1.1351e-10
    d1 = 2.7759e-12
    e0 = -4.6206e-13
    e1 = 1.8676e-14
    e2 = -2.1687e-16

    return a0 + a1*temp + a2*temp**2 + a3*temp**3 + (b0 + b1*temp)*(salt - s_ref) + (c0 + c1*temp + c2*temp**2 + c3*temp**3 + (d0 + d1*temp)*(salt - s_ref))*press + (e0 + e1*temp + e2*temp**2)*press**2


# Calculate in-situ temperature from potential temperature, exactly like MITgcm does.

# Arguments:
# temp: potential temperature (degC)
# salt: salinity (psu)
# z: depth (m, sign doesn't matter)

# Output: in-situ temperature (degC), same dimension as input arguments

def in_situ_temp (temp, salt, z):

    press_ref = 0.
    sqrt_2 = np.sqrt(2)

    # Step 1
    dpress = np.abs(z) - press_ref
    dtemp = dpress*ad_temp_grad(temp, salt, press_ref)
    temp_new = temp + 0.5*dtemp
    q = dtemp

    # Step 2
    dtemp = dpress*ad_temp_grad(temp_new, salt, press_ref + 0.5*dpress)
    temp_new += (1 - 1/sqrt_2)*(dtemp - q)
    q = (2 - sqrt_2)*dtemp + (-2 + 3/sqrt_2)*q

    # Step 3
    dtemp = dpress*ad_temp_grad(temp_new, salt, press_ref + 0.5*dpress)
    temp_new += (1 + 1/sqrt_2)*(dtemp - q)
    q = (2 + sqrt_2)*dtemp + (-2 - 3/sqrt_2)*q

    # Step 4
    dtemp = dpress*ad_temp_grad(temp_new, salt, press_ref + dpress)
    temp_new += (dtemp - 2*q)/6

    return temp_new


# Calculate the in-situ freezing point (helper function for t_minus_tf)

# Arguments:
# salt, z: arrays of any dimension (but both the same dimension, or else one is a scalar) containing salinity (psu) and depth (m, sign doesn't matter).

# Output: array of the same dimension as salt and z, containing the in-situ freezing point in degC.

def tfreeze (salt, z):

    a0 = -0.0575
    b = -7.61e-4
    c0 = 0.0901

    return a0*salt + b*abs(z) + c0


# Calculate the temperature difference from the in-situ freezing point.

# Arguments:
# temp, salt: arrays of temperature and salinity. They can be 3D (depth x lat x lon) or 4D (time x depth x lat x lon), in which case you need time_dependent=True.
# grid = Grid object

# Optional keyword arguments:
# time_dependent: boolean indicating that temp and salt are 4D, with a time dimension. Default False.

# Output: array of the same dimensions as temp and salt, containing the difference from the in-situ freezing point.

def t_minus_tf (temp, salt, grid, time_dependent=False):

    # Tile the z coordinates to be the same size as temp and salt
    # First assume 3D arrays
    z = z_to_xyz(grid.z, grid)
    if time_dependent:
        # 4D arrays
        z = add_time_dim(z, temp.shape[0])

    return in_situ_temp(temp, salt, z) - tfreeze(salt, z)


# Calculate the total mass loss or area-averaged melt rate.

# Arguments:
# ismr: 2D (lat x lon) array of ice shelf melt rate in m/y
# mask: boolean array which is True in the points to be included in the calculation (such as grid.ice_mask)
# grid: Grid object

# Optional keyword argument:
# result: 'massloss' (default) calculates the total mass loss in Gt/y. 'melting' calculates the area-averaged melt rate in m/y.

# Output: float containing mass loss or average melt rate

def total_melt (ismr, mask, grid, result='massloss'):

    if result == 'melting':
        # Area-averaged melt rate
        return np.sum(ismr*grid.dA*mask)/np.sum(grid.dA*mask)
    elif result == 'massloss':
        # Total mass loss
        return np.sum(ismr*grid.dA*mask)*rho_ice*1e-12


# Find the time indices of minimum and maximum sea ice area.

# Arguments:
# aice: 3D (time x lat x lon) array of sea ice area at each time index
# grid: Grid object

# Output: two integers containing the time indices (0-indexed) of minimum and maximum sea ice area, respectively

def find_aice_min_max (aice, grid):

    total_aice = area_integral(aice, grid, time_dependent=True)
    return np.argmin(total_aice), np.argmax(total_aice)


# Calculate the barotropic transport streamfunction. u is assumed not to be time-dependent.
def barotropic_streamfunction (u, grid):

    check_time_dependent(u)
        
    # Vertically integrate
    udz_int = vertical_integral(u, grid, gtype='u')
    # Indefinite integral from south to north
    strf = indefinite_ns_integral(udz_int, grid, gtype='u')
    # Convert to Sv
    return strf*1e-6


# Calculate the Weddell Gyre transport: absolute value of the most negative streamfunction within the Weddell Gyre bounds.
def wed_gyre_trans (u, grid):

    strf = barotropic_streamfunction(u, grid)
    [xmin, xmax, ymin, ymax] = region_bounds['wed_gyre']
    vmin, vmax = var_min_max(strf, grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, gtype='u')
    return -1*vmin


# Calculate seawater density for a linear equation of state.
def dens_linear (salt, temp, rhoConst, Tref, Sref, tAlpha, sBeta):

    return rhoConst*(1 - tAlpha*(temp-Tref) + sBeta*(salt-Sref))


# Calculate density for the given equation of state. Pressure can be a constant scalar if you want a reference pressure.
def density (eosType, salt, temp, press, rhoConst=None, Tref=None, Sref=None, tAlpha=None, sBeta=None):

    # Check if pressure is a constant value
    if isinstance(press, float) or isinstance(press, int):
        # Make it an array
        press = np.zeros(temp.shape) + press

    if eosType == 'MDJWF':
        from MITgcmutils.mdjwf import densmdjwf
        return densmdjwf(salt, temp, press)
    elif eosType == 'JMD95':
        from MITgcmutils.jmd95 import densjmd95
        return densjmd95(salt, temp, press)
    elif eosType == 'LINEAR':
        if None in [rhoConst, Tref, Sref, tAlpha, sBeta]:
            print('Error (density): for eosType LINEAR, you must set rhoConst, Tref, Sref, tAlpha, sBeta')
            sys.exit()
        return dens_linear(salt, temp, rhoConst, Tref, Sref, tAlpha, sBeta)
    else:
        print(('Error (density): invalid eosType ' + eosType))
        sys.exit()
        

# Wrapper for potential density.
def potential_density (eosType, salt, temp, rhoConst=None, Tref=None, Sref=None, tAlpha=None, sBeta=None):

    return density(eosType, salt, temp, 0, rhoConst=rhoConst, Tref=Tref, Sref=Sref, tAlpha=tAlpha, sBeta=sBeta)


# Calculate heat content relative to the in-situ freezing point. Just use potential temperature and density.
def heat_content_freezing (temp, salt, grid, eosType='MDJWF', rhoConst=None, Tref=None, Sref=None, tAlpha=None, sBeta=None, time_dependent=False):

    dV = grid.dV
    # Get 3D z (for freezing point)
    z = z_to_xyz(grid.z, grid)
    # Add time dimensions if needed
    if time_dependent:
        num_time = temp.shape[0]
        dV = add_time_dim(dV, num_time)
        z = add_time_dim(z, num_time)
    # Calculate freezing temperature
    Tf = tfreeze(salt, z)
    # Calculate potential density
    rho = potential_density(eosType, salt, temp, rhoConst=rhoConst, Tref=Tref, Sref=Sref, tAlpha=tAlpha, sBeta=sBeta)        
    # Now calculate heat content relative to Tf, in J
    return (temp-Tf)*rho*Cp_sw*dV


# Helper function for normal_vector and parallel_vector.
def rotate_vector (u, v, grid, point0, point1, option='both', time_dependent=False):
    
    # Find angle between east and the transect (intersecting at point0)
    [lon0, lat0] = point0
    [lon1, lat1] = point1
    x = np.cos(lat1)*np.sin(lon1-lon0)
    y = np.cos(lat0)*np.sin(lat1) - np.sin(lat0)*np.cos(lat1)*np.cos(lon1-lon0)
    angle = np.arctan2(y, x)

    # Interpolate u and v to the tracer grid
    u_t = interp_grid(u, grid, 'u', 't', time_dependent=time_dependent)
    v_t = interp_grid(v, grid, 'v', 't', time_dependent=time_dependent)

    u_new = u_t*np.cos(-angle) - v_t*np.sin(-angle)
    v_new = u_t*np.sin(-angle) + v_t*np.cos(-angle)

    if option == 'normal':
        return v_new
    elif option == 'parallel':
        return u_new
    elif option == 'both':
        return u_new, v_new
    else:
        print(('Error (rotate_vector): invalid option ' + option))
        sys.exit()


# Calculate the normal component of the vector field with respect to the angle angle given by the transect between the 2 points. Assumes u and v only include one time record (i.e. 3D fields). Does not extract the transect itself.
def normal_vector (u, v, grid, point0, point1, time_dependent=False):
    return rotate_vector(u, v, grid, point0, point1, option='normal', time_dependent=time_dependent)


# Calculate the parallel component of the vector.
def parallel_vector (u, v, grid, point0, point1, time_dependent=False):
    return rotate_vector(u, v, grid, point0, point1, option='parallel', time_dependent=time_dependent)


# Calculate the total onshore and offshore transport with respect to the given transect. Default is for the shore to be to the "south" of the line from point0 ("west") to point1 ("east").
def transport_transect (u, v, grid, point0, point1, shore='S', time_dependent=False):

    # Calculate normal velocity
    u_norm = normal_vector(u, v, grid, point0, point1, time_dependent=time_dependent)
    # Extract the transect
    u_norm_trans, left, right, below, above = get_transect(u_norm, grid, point0, point1, time_dependent=time_dependent)
    # Calculate integrands
    dh = (right - left)*1e3  # Convert from km to m
    dz = above - below
    if time_dependent:
        # Make them 3D
        num_time = u.shape[0]
        dh = add_time_dim(dh, num_time)
        dz = add_time_dim(dh, num_time)
    # Integrate and convert to Sv
    trans_S = np.sum(np.minimum(u_norm_trans,0)*dh*dz*1e-6, axis=(-2,-1))
    trans_N = np.sum(np.maximum(u_norm_trans,0)*dh*dz*1e-6, axis=(-2,-1))
    # Retrn onshore, then offshore transport
    if shore == 'S':
        return trans_S, trans_N
    elif shore == 'N':
        return trans_N, trans_S
    else:
        print(('Error (transport_transect): invalid shore ' + shore))
        sys.exit()


# Convert the heat advection terms from MITgcm (with respect to 0C) to be with respect to the surface freezing point (assuming constant salinity for simplicity).
# Input arguments:
# adv: list of length 2 or 3, containing the x, y, and maybe z components of advection. If you only want some components, set the others to be None.
# vel: list of length 2 or 3, containing the u, v, and maybe w arrays (same shape as adv arrays). One of them can be None as for adv.
# grid: Grid object
def adv_heat_wrt_freezing (adv, vel, grid):

    dim = len(adv)
    calc = [adv[n] is not None for n in range(dim)] # Which dimensions we need to calculate
    result = [None for n in range(dim)]
    dz = z_to_xyz(grid.dz, grid)
    dA = [xy_to_xyz(grid.dy_w, grid)*dz, xy_to_xyz(grid.dx_s, grid)*dz, xy_to_xyz(grid.dA, grid)]  # Product of two faces from other dimensions

    time_dependent = False
    for n in range(dim):
        if adv[n] is not None and len(adv[n].shape)==4:
            time_dependent=True
    
    if time_dependent:
        # Add time dimension to dA
        num_time = None
        for n in range(dim):
            if calc[n]:
                num_time = adv[n].shape[0]
                break
        for n in range(dim):
            dA[n] = add_time_dim(dA[n], num_time)

    # Now calculate the result
    for n in range(dim):
        if calc[n]:
            result[n] = adv[n] - Tf_ref*vel[n]*dA[n]

    return result


# Calculate the thermocline given a 3D temperature field.
def thermocline (temp, grid):

    if len(temp.shape)==4:
        print('Error (thermocline): have not written time-dependent case yet')
        sys.exit()
    temp = mask_3d(temp, grid)
    dtemp_dz = (temp[1:,:,:]-temp[:-1,:,:])/np.abs(grid.z[1:,None,None]-grid.z[:-1,None,None])
    sfc_mask = np.ma.masked_where(True, dtemp_dz[0,:,:])
    dtemp_dz = np.ma.concatenate((sfc_mask[None,:,:], dtemp_dz), axis=0)
    return depth_of_max(dtemp_dz, grid)
    
    
    
    



    
    

    

        

    
    





