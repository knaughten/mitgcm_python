#######################################################
# Scalar constants
#######################################################

import numpy as np

# Acceleration due to gravity (m/s^2)
gravity = 9.81
# Density of freshwater (kg/m^3)
rho_fw = 1e3
# Density of ice (kg/m^3)
rho_ice = 917.
# Seconds per year
sec_per_year = 365.25*24*60*60
# Degrees to radians conversion factor
deg2rad = np.pi/180.0
# Celsius to Kelvins intercept
temp_C2K = 273.15
# Latent heat of vapourisation (J/kg)
Lv = 2.5e6
# Ideal gas constant for water vapour (J/K/kg)
Rv = 461.5
# Reference saturation vapour pressure (Pa)
es0 = 611
# Coefficient for specific humidity calculation
sh_coeff = 0.62197
# Specific heat of seawater (J/K/kg)
Cp_sw = 4180.

# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'

# Bounds on FRIS (a few bits of Eastern Weddell ice shelves are included too - use Grid.fris_mask to get just FRIS. These bounds are for plotting.)
# lon_min, lon_max, lat_min, lat_max
fris_bounds = [-85, -29, -84, -74]
# Bounds on Eastern Weddell ice shelves.
ewed_bounds = [-30, 40, -77, -65]
# Bounds on Weddell Gyre.
wed_gyre_bounds = [-60, 30, -90, -50]
# Bounds on location to search for grounded iceberg A-23A
a23a_bounds = [-47, -38, -77, -75]

# Resolution of SOSE grid in degrees
sose_res = 1/6.

# BEDMAP2 grid parameters
bedmap_dim = 6667    # Dimension
bedmap_bdry = 3333000    # Polar stereographic coordinate (m) on boundary
bedmap_res = 1000    # Resolution (m)
bedmap_missing_val = -9999    # Missing value for bathymetry north of 60S


