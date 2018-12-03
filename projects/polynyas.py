##################################################################
# Weddell Sea polynya project
##################################################################

from ..postprocess import precompute_timeseries


# Get longitude and latitude at the centre of the polynya
def get_polynya_loc (polynya):
    
    if polynya.startswith('maud_rise'):
        lon0 = 0
        lat0 = -65
    elif polynya == 'near_shelf':
        lon0 = -30
        lat0 = -70
    else:
        print 'Error (get_polynya_loc): please specify a valid polynya.'
        sys.exit()
    return lon0, lat0


# Precompute timeseries for analysis
def precompute_polynya_timeseries (mit_file, timeseries_file, polynya=None):

    if polynya is None:
        # Baseline simulation; skip temp_polynya and salt_polynya options
        lon0 = None
        lat0 = None
    else:
        lon0, lat0 = get_polynya_loc(polynya)
    precompute_timeseries(mit_file, timeseries_file, polynya=True, lon0=lon0, lat0=lat0)



# Function to precompute timeseries
#   Convective area (MLD > 2000 m)
#   Net ismr of FRIS
#   Net ismr of E Weddell ice shelves
#   Weddell Gyre transport
#   T & S depth-averaged through centre of polynya
#   T & S volume-averaged in FRIS cavity


# Function to preliminarily analyse everything
#   All timeseries on same axes
#   Difference timeseries (all except polynya T & S) on same axes
#   Lat-lon plots, averaged over entire period (all except 5-year)
#     Sea ice area (absolute)
#     Baseline absolute and others differenced, zoomed in and out:
#       BW temp and salt
#       ismr
#       Barotropic velocity
