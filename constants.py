#######################################################
# Scalar constants
#######################################################

import numpy as np

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

# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'

# Bounds on FRIS (a few bits of Eastern Weddell ice shelves are included too - use Grid.fris_mask to get just FRIS. These bounds are for plotting.)
# lon_min, lon_max, lat_min, lat_max
fris_bounds = [-85, -29, -84, -74]

# Resolution of SOSE grid in degrees
sose_res = 1/6.

# BEDMAP2 grid parameters
bedmap_dim = 6667    # Dimension
bedmap_bdry = 3333000    # Polar stereographic coordinate (m) on boundary
bedmap_res = 1000    # Resolution (m)
bedmap_missing_val = -9999    # Missing value for bathymetry north of 60S
