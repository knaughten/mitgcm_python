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
# Reference density of seawater (kg/m^3)
rhoConst = 1028.5
# Seconds per day
sec_per_day = 24*60*60.
# Seconds per year
sec_per_year = 365.25*sec_per_day
# Months per year
months_per_year = 12
# Degrees to radians conversion factor
deg2rad = np.pi/180.0
rad2deg = 1./deg2rad
# Radius of Earth
rEarth = 6.371e6
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
# Conversion from ice VAF change to global mean sea level contribution (m^-2) following Goelzer 2020 
vaf_to_gmslr = -910./1028./3.625e14
# kg per Gt
kg_per_Gt = 1e12
# Reference surface freezing point (degC)
Tf_ref = -1.9

# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'

# Dictionary of bounds on different regions - some in 2 parts
# lon_min, lon_max, lat_min, lat_max
region_bounds = {
    'fris_plot': [-85., -24., -84., -74.],
    'fris_pster_plot': [-1.5e6, -4.95e5, 1.1e5, 1.5e6],
    'fris_ua_plot': [-1.6e6, -4e5, 9e4, 1.2e6],
    'fris1': [-85., -45., -84., -74.4],
    'fris2': [-45., -24., -84., -77.85],
    'ewed': [-30., 40., -77., -65.],
    'wed_gyre': [-60., 30., -90., -50.],
    'sws_shelf': [-70., -30., -79., -72.],
    'a23a': [-47., -38., -77., -75.],
    'berkner_island': [-55, -41, -81, -77],
    'pine_island_bay': [-104, -100.5, -75.2, -74.2],
    'dotson_bay': [-114, -110.5, -74.3, -73.5],
    'getz': [-135., -114.7, -75.2, -73.5],
    'dotson': [-114.6, -111.2, -75.3, -74.1],
    'crosson': [-111.2, -109., -75.4, -74.1],
    'dotson_crosson': [-114.7, -109., -75.4, -74.1],
    'thwaites': [-109., -103., -75.4, -74.6],
    'pig': [-103., -99., -75.4, -74.],
    'cosgrove': [-102., -98.5, -73.8, -73.2],
    'abbot1': [-104., -99., -73.2, -71.5],
    'abbot2': [-99., -88.9, -73.4, -71.5],
    'venable': [-88.9, -85.6, -73.4, -72.7],
    'filchner_trough': [-45, -30, -79, -75],
    'ronne_depression': [-70, -55, -76, -73],
    'berkner_bank': [-50, -44, -78, -77],
    'wdw_core': [-60, -20, -75, -65],
    'filchner_front': [-45, -36, -79, -77],
    'deep_ronne_cavity': [-85, -66, -85, -75.5],
    'offshore_filchner': [-40, -25, -75, -72],
    'crosson_thwaites': [-110, -106.7, -75.5, -74.75],
    'thwaites_pig': [-106, -100, -75.5, -74.75],
    'inner_amundsen_shelf': [-112, -100, -75.5, -74.25],
    'amundsen_shelf_break': [-115, -102, -71.8, -70.2],
    'amundsen_west_shelf_break': [-115, -112, -72, -71],
    'brunt_riiser_larsen': [-28, -10, -76.5, -71.6],
    'ekstrom_jelbart_fimbul': [-10, 7.7, -72, -69],
    'filchner_sill': [-36, -30, -74.7, -74],
    'filchner_trough_cavity': [-44, -38, -81, -76],
    'amundsen_shelf': [-115, -100, -75.5, -70],
    'dotson_to_cosgrove': [-114.6, -98.5, -75.5, -73.1]
}
# Regions that are in two parts
region_split = ['fris', 'abbot']
# Isobaths restricting some regions
region_bathy_bounds = {
    'sws_shelf': [-1250, None],
    'filchner_trough': [-1250, -650],
    'ronne_depression': [None, -550],
    'berkner_bank': [-325, None],
    'deep_ronne_cavity': [None, -1000],
    'wdw_core': [None, -2000],
    'offshore_filchner': [None, -2000],
    'filchner_sill': [-1500, -600],
    'filchner_trough_cavity': [None, -1000],
    'amundsen_shelf': [-1750, None]
}
# Depth bounds for 3D regions
region_depth_bounds = {
    'wdw_core': [-1000, -250]
}
# Names corresponding to some keys (used for plotting)
region_names = {
    'fris': 'Filchner-Ronne Ice Shelf',
    'ewed': 'Eastern Wedddell ice shelves',
    'getz': 'Getz Ice Shelf',
    'dotson': 'Dotson Ice Shelf',
    'dotson_cavity': 'Dotson Ice Shelf cavity',
    'crosson': 'Crosson Ice Shelf',
    'dotson_crosson': 'Dotson and Crosson Ice Shelves',
    'thwaites': 'Thwaites Ice Shelf',
    'pig': 'Pine Island Glacier Ice Shelf',
    'pig_cavity': 'Pine Island Glacier Ice Shelf cavity',
    'cosgrove': 'Cosgrove Ice Shelf',
    'abbot': 'Abbot Ice Shelf',
    'venable': 'Venable Ice Shelf',
    'all': 'all ice shelves',
    'filchner_trough': 'Filchner Trough',
    'ronne_depression': 'Ronne Depression',
    'sws_shelf': 'Southern Weddell Sea shelf',
    'wdw_core': 'Warm Deep Water core',
    'pine_island_bay': 'Pine Island Bay',
    'dotson_bay': 'front of Dotson',
    'icefront': 'ice shelf fronts',
    'upstream': 'upstream continental shelf',
    'downstream': 'downstream continental shelf',
    'openocean': 'open ocean',
    'filchner': 'Filchner Ice Shelf',
    'berkner_bank': 'Berkner Bank',
    'deep_ronne_cavity': 'Deep Ronne Ice Shelf cavity',
    'offshore_filchner': 'Offshore of Filchner Trough sill',
    'filchner_front': 'Filchner Ice Shelf front',
    'filchner_trough_front': 'Ice shelf front at Filchner Trough',
    'ronne_depression_front': 'Ice shelf front at Ronne Depression',
    'crosson_thwaites': 'Continental shelf between Crosson and Thwaites',
    'thwaites_pig': 'Continental shelf between Thwaites and PIG',
    'inner_amundsen_shelf': 'Inner Amundsen Sea continental shelf',
    'amundsen_shelf_break': 'Amundsen Sea shelf break',
    'amundsen_west_shelf_break': 'Western Amundsen Sea shelf break',
    'brunt_riiser_larsen': 'Brunt and Riiser-Larsen Ice Shelves',
    'ekstrom_jelbart_fimbul': 'Ekstrom, Jelbart, and Fimbul Ice Shelves',
    'filchner_sill': 'Filchner Trough Sill',
    'filchner_trough_cavity': 'Filchner Trough in cavity',
    'amundsen_shelf': 'Amundsen Sea continental shelf',
    'dotson_to_cosgrove': 'Ice shelves between Dotson and Cosgrove'
}    

# Resolution of SOSE grid in degrees
sose_res = 1/6.

# BEDMAP2 grid parameters
bedmap_dim = 6667    # Dimension
bedmap_bdry = 3333000    # Polar stereographic coordinate (m) on boundary
bedmap_res = 1000    # Resolution (m)
bedmap_missing_val = -9999    # Missing value for bathymetry north of 60S

# Rignot 2013 estimates of ice shelf melting
# [mass loss in Gt/y, standard deviation in mass loss, average melt rate in m/y, standard deviation in melt rate]
# Dotson and Crosson are combined.
rignot_melt = {
    'getz': [144.9, 14, 4.3, 0.4],
    'dotson_crosson': [83.7, 8, 9.3, 0.74],
    'thwaites': [97.5, 7, 17.7, 1],
    'pig': [101.2, 8, 16.2, 1],
    'cosgrove': [8.5, 2, 2.8, 0.7],
    'abbot': [51.8, 19, 1.7, 0.6],
    'venable': [19.4, 2, 6.1, 0.7]
}
# Adusumilli 2020 (for time period 1994-2018)
adusumilli_melt = {
    'venable': [14.3, 5.5, 5.1, 2.0],
    'abbot': [37.1, 38.1, 1.5, 1.5],
    'cosgrove': [2.7, 4.1, 1, 1.5],
    'pig': [76.6, 8.6, 14.0, 1.6],
    'thwaites': [81.9, 7.4, 26.7, 2.4],
    'crosson': [20.9, 4.9, 7.8, 1.8],
    'dotson': [28.2, 8.5, 5.4, 1.6],
    'getz': [124.1, 40.9, 4.2, 1.4]
}

# Jenkins 2018 estimates of Dotson melting (Gt/y)
dotson_melt_years = {
    'year': [2000, 2006, 2007, 2009, 2011, 2012, 2014, 2016],
    'melt': [25.0, 55.7, 44.4, 91.6, 53.8, 20.3, 20.9, 19.5],
    'err': [9.1, 15.3, 35.2, 31.6, 12.3, 6.1, 8.0, 15.4]
}
# Jacobs 2013 estimates of Getz melting (Gt/y)
getz_melt_years = {
    'year': [2000, 2007],
    'melt': np.array([37, 137])*rho_ice/rho_fw,
    'err': np.array([13, 16])*rho_ice/rho_fw
}
# Dutrieux 2014 + Heywood 2016 estimates of PIG melting (Gt/y)
dutrieux_sw = np.array([51.3, 79.7, 75.2, 37.3])*rho_ice/rho_fw
dutrieux_mw = np.array([49.1, 79.4, 69.2, 34.7])*rho_ice/rho_fw
dutrieux_melt = 0.5*(dutrieux_sw+dutrieux_mw)
dutrieux_err_fac = 0.1
heywood_melt = np.array([40])*rho_ice/rho_fw
heywood_err_fac = 0.4
pig_melt_years = {
    'year': [1994, 2009, 2010, 2012, 2014],
    'melt': np.concatenate((dutrieux_melt, heywood_melt)),
    'err': np.concatenate((dutrieux_melt*dutrieux_err_fac, heywood_melt*heywood_err_fac))
}

# Titles for Ua variables
ua_titles = {
    'b': 'Ice base elevation (m)',
    'ab': 'Basal mass balance (m/y)',
    'B': 'Bedrock elevation (m)',
    'dhdt': 'Ice thickness rate of change (m/y)',
    'h': 'Ice thickness (m)',
    'rho': r'Ice density (kg/m$^3$)',
    's': 'Ice surface elevation (m)',
    'ub': 'Basal x-velocity (m/y)',
    'vb': 'Basal y-velocity (m/y)',
    'velb': 'Basal velocity (m/y)'
}


