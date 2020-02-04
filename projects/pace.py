##################################################################
# Weddell Sea threshold paper
##################################################################

from ..grid import ERA5Grid, PACEGrid
from ..file_io import read_binary, write_binary
from ..utils import real_dir

def calc_climatologies (era5_dir, pace_dir, out_dir):

    var_era5 = ['atemp', 'aqh', 'apressure', 'uwind', 'vwind', 'precip', 'swdown', 'lwdown']
    var_pace = ['TREFHT', 'QBOT', 'PSL', 'UBOT', 'VBOT', 'PRECT', 'FLDS', 'FSDS']
    file_head_era5 = 'ERA5_'
    file_head_pace = 'PACE_ens'
    num_vars = len(var_era5)
    days_per_year = 365
    # Day of the year that's 29 Feb (0-indexed)
    leap_day = 31+28
    # Climatology over the years that both products have data (not counting the RCP8.5 extension)
    start_year = 1979
    end_year = 2005
    num_years = end_year-start_year+1
    # Number of PACE ensemble members
    num_ens = 20

    era5_grid = ERA5Grid()
    pace_grid = PACEGrid()

    # Get right edges of PACE grid (for binning)
    pace_lon, pace_lat = pace_grid.get_lon_lat(dim=1)
    def right_edges (A):
        edges_mid = 0.5*(A[:-1] + A[1:])
        edges_end = 2*edges_mid[-1] - edges_mid[-2]
        return np.concatenate((edges_mid, [edges_end]), axis=0)
    pace_lon_bins = right_edges(pace_lon)
    pace_lat_bins = right_edges(pace_lat)
    # Figure out i and j indices for ERA5 to PACE binning
    i_bins = np.digitize(era_grid.lon, pace_lon_bins)
    # Wrap the periodic boundary
    i_bins[i_bins==pace_grid.nx] = 0
    j_bins = np.digitize(era_grid.lat, pace_lat_bins)

    # Loop over variables
    era5_clim = np.empty([num_vars, days_per_year, era5_grid.ny, era5_grid.nx])
    print 'Processing ERA5'
    for n in range(num_vars):
        print 'Processing ' + var_pace[n]
        
        # Accumulate data over each year
        data_accum = np.zeros([days_per_year, era5_grid.ny, era5_grid.nx])
        for year in range(start_year, end_year+1):
            file_path = real_dir(era5_dir) + file_head_era5 + var_era5[n] + '_' + str(year)
            data = read_binary(file_path, [era5_grid.nx, era5_grid.ny], 'xyt')
            if data.shape[0] == days_per_year+1:
                # Remove leap day
                data = np.concatenate((data[:leap_day-1,:], data[leap_day+1:,:]), axis=0)
            data_accum += data
        # Convert from integral to average
        data_clim = data_accum/num_years
        era5_clim[n,:] = data_clim

    # Now do all the binning at once to save memory
    era5_clim_regrid = np.zeros([num_vars, days_per_year, pace_grid.ny, pace_grid.nx])
    print 'Regridding from ERA5 to PACE grid'
    for j in range(pace_grid.ny):
        for i in range(pace_grid.nx):
            if np.any(i_bins==i) and np.any(j_bins==j):
                # Get an array which is 1 within this bin, 0 elsewhere
                flag = np.nonzero((i_bins==i)*(j_bins==j)).astype(float)
                # Tile it to be 4D
                flag = np.tile(np.tile(flag, (days_per_year, 1, 1)), (num_vars, 1, 1, 1))
                era5_clim_regrid[:,:,j,i] = np.mean(era5_clim*flag, axis=(2,3))
    # Write each variable to binary
    for n in range(num_vars):   
        file_path = real_dir(out_dir) + 'ERA5_' + var_pace[n] + '_clim'
        write_binary(era5_clim_regrid[n,:], file_path)

    print 'Processing PACE'
    for n in range(num_vars):
        print 'Processing ' + var_pace[n]
        for ens in range(1, num_ens+1):
            if ens == 13:
                continue
            ens_str = str(ens).zfill(2)
            print 'Processing PACE ensemble member ' + ens_str
            # As before, but simpler because no leap days and no need to regrid
            data_accum = np.zeros([days_per_year, pace_grid.ny, pace_grid.nx])
            for year in range(start_year, end_year+1):
                file_path = real_dir(pace_dir) + file_head_pace + ens_str + '_' + var_pace[n] + '_' + str(year)
                data = read_binary(file_path, [pace_grid.nx, pace_grid.ny], 'xyt')
                data_accum += data
            data_clim = data_accum/num_years
            file_path = real_dir(out_dir) + 'PACE_ens' + ens_str + '_' + var_pace[n] + '_clim'
            write_binary(data_clim, file_path)


        
    
    
