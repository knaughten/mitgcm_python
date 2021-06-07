#######################################################
# Generate a new model domain.
#######################################################

import numpy as np
from scipy.io import loadmat
import sys
import shutil

from .constants import deg2rad, bedmap_dim, bedmap_bdry, bedmap_res, bedmap_missing_val, region_bounds
from .file_io import write_binary, NCfile_basiclatlon, read_netcdf
from .utils import factors, polar_stereo, mask_box, mask_above_line, mask_iceshelf_box, real_dir, mask_3d, xy_to_xyz, z_to_xyz
from .interpolation import extend_into_mask, interp_topo, neighbours, neighbours_z, remove_isolated_cells 
from .grid import Grid


# Create the 2D grid points (regular lon and lat) based on the boundaries and resolution given by the user. Resolution is constant in longitude, but scaled by cos(latitude) in latitude.
def latlon_points (xmin, xmax, ymin, ymax, res, dlat_file, prec=64):

    # Number of iterations for latitude convergence
    num_lat_iter = 10

    if xmin > xmax:
        print("Error (latlon_points): looks like your domain crosses 180E. Try again with your longitude in the range (0, 360) instead of (-180, 180).")
        sys.exit()

    # Build longitude values
    lon = np.arange(xmin, xmax+res, res)
    # Update xmax if the range doesn't evenly divide by res
    if xmax != lon[-1]:
        xmax = lon[-1]
        print(('Eastern boundary moved to ' + str(xmax)))
    # Put xmin in the range (0, 360) for namelist
    if xmin < 0:
        xmin += 360

    # First guess for latitude: resolution scaled by latitude of southern edge
    lat = [ymin]
    while lat[-1] < ymax:
        lat.append(lat[-1] + res*np.cos(lat[-1]*deg2rad))
    lat = np.array(lat)
    # Now iterate to converge on resolution scaled by latitude of centres
    for iter in range(num_lat_iter):
        lat_old = np.copy(lat)
        # Latitude at centres    
        lat_c = 0.5*(lat[:-1] + lat[1:])
        j = 0
        lat = [ymin]
        while lat[-1] < ymax and j < lat_c.size:
            lat.append(lat[-1] + res*np.cos(lat_c[j]*deg2rad))
            j += 1
        lat = np.array(lat)
    # Update ymax
    ymax = lat[-1]
    print(('Northern boundary moved to ' + str(ymax)))

    # Write latitude resolutions to file
    dlat = lat[1:] - lat[:-1]
    write_binary(dlat, dlat_file, prec=prec)

    # Remind the user what to do in their namelist
    print('\nChanges to make to input/data:')
    print(('xgOrigin=' + str(xmin)))
    print(('ygOrigin=' + str(ymin)))
    print(('dxSpacing=' + str(res)))
    print(("delYfile='" + dlat_file + "' (and copy this file into input/)"))

    # Find dimensions of tracer grid
    Nx = lon.size-1
    Ny = lat.size-1
    # Find all the factors
    factors_x = factors(Nx)
    factors_y = factors(Ny)
    print(('\nNx = ' + str(Nx) + ' which has the factors ' + str(factors_x)))
    print(('Ny = ' + str(Ny) + ' which has the factors ' + str(factors_y)))
    print('If you are happy with this, proceed with interp_bedmap2. At some point, choose your tile size based on the factors and update code/SIZE.h.')
    print('Otherwise, tweak the boundaries and try again.')

    return lon, lat


# Helper function to add the grounded iceberg A23A to the domain.
# lon, lat, bathy, omask are from the domain being built.
def add_grounded_iceberg (rtopo_file, lon, lat, bathy, omask):

    import netCDF4 as nc

    print('Adding grounded iceberg A-23A')
    
    print('Reading data')
    id = nc.Dataset(rtopo_file, 'r')
    lon_rtopo = id.variables['lon'][:]
    lat_rtopo = id.variables['lat'][:]
    [xmin, xmax, ymin, ymax] = region_bounds['a23a']
    # Select the region we care about
    i_start = np.nonzero(lon_rtopo >= xmin)[0][0] - 1
    i_end = np.nonzero(lon_rtopo >= xmax)[0][0]
    j_start = np.nonzero(lat_rtopo >= ymin)[0][0] - 1
    j_end = np.nonzero(lat_rtopo >= ymax)[0][0]
    # Read mask just from this section
    mask_rtopo = id.variables['amask'][j_start:j_end, i_start:i_end]
    id.close()
    # Trim the grid too
    lon_rtopo = lon_rtopo[i_start:i_end]
    lat_rtopo = lat_rtopo[j_start:j_end]

    print('Interpolating mask')
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    mask_rtopo_interp = interp_topo(lon_rtopo, lat_rtopo, mask_rtopo, lon_2d, lat_2d)
    # The mask is 1 in the grounded iceberg region and 0 elsewhere. Take a threshold of 0.5 for interpolation errors. Treat it as land.
    index = mask_rtopo_interp >= 0.5
    bathy[index] = 0
    omask[index] = 0

    return bathy, omask


# Interpolate BEDMAP2 bathymetry, ice shelf draft, and masks to the new grid. Write the results to a NetCDF file so the user can check for any remaining artifacts that need fixing (eg blocking out the little islands near the peninsula).
# You can set an alternate bed file (eg Filchner updates by Sebastian Rosier) with the keyword argument bed_file.
# If you want the A-23A grounded iceberg in your land mask, set grounded_iceberg=True and make sure you have the RTopo-2 aux file (containing the variable "amask") in your topo_dir; set rtopo_file if it has a different path than the default.
def interp_bedmap2 (lon, lat, topo_dir, nc_out, bed_file=None, grounded_iceberg=False, rtopo_file=None):

    import netCDF4 as nc
    from .plot_latlon import plot_tmp_domain

    topo_dir = real_dir(topo_dir)

    # BEDMAP2 file names
    surface_file = topo_dir+'bedmap2_surface.flt'
    thickness_file = topo_dir+'bedmap2_thickness.flt'
    mask_file = topo_dir+'bedmap2_icemask_grounded_and_shelves.flt'
    if bed_file is None:
        bed_file = topo_dir+'bedmap2_bed.flt'
    # GEBCO file name
    gebco_file = topo_dir+'GEBCO_2014_2D.nc'
    if grounded_iceberg and (rtopo_file is None):
        # RTopo-2 file name (auxiliary file including masks)        
        rtopo_file = topo_dir+'RTopo-2.0.1_30sec_aux.nc'

    if np.amin(lat) > -60:
        print("Error (interp_bedmap2): this domain doesn't go south of 60S, so it's not covered by BEDMAP2.")
        sys.exit()
    if np.amax(lat) > -60:
        use_gebco = True
        # Find the first index north of 60S
        j_split = np.nonzero(lat >= -60)[0][0]
        # Split grid into a BEDMAP2 section and a GEBCO section (remembering lat is edges, not centres, so lat[j_split-1] is in both sections)
        lat_b = lat[:j_split]
        lat_g = lat[j_split-1:]
    else:
        use_gebco = False
        lat_b = lat

    # Set up BEDMAP grid (polar stereographic)
    x = np.arange(-bedmap_bdry, bedmap_bdry+bedmap_res, bedmap_res)
    y = np.arange(-bedmap_bdry, bedmap_bdry+bedmap_res, bedmap_res)

    print('Reading data')
    # Have to flip it vertically so lon0=0 in polar stereographic projection
    # Otherwise, lon0=180 which makes x_interp and y_interp strictly decreasing when we call polar_stereo later, and the interpolation chokes
    bathy = np.flipud(np.fromfile(bed_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))
    surf = np.flipud(np.fromfile(surface_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))
    thick = np.flipud(np.fromfile(thickness_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))
    mask = np.flipud(np.fromfile(mask_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))

    if np.amax(lat_b) > -61:
        print('Extending bathymetry past 60S')
        # Bathymetry has missing values north of 60S. Extend into that mask so there are no artifacts in the splines near 60S.
        bathy = extend_into_mask(bathy, missing_val=bedmap_missing_val, num_iters=5)

    print('Calculating ice shelf draft')
    # Calculate ice shelf draft from ice surface and ice thickness
    draft = surf - thick

    print('Calculating ocean and ice masks')
    # Mask: -9999 is open ocean, 0 is grounded ice, 1 is ice shelf
    # Make an ocean mask and an ice mask. Ice shelves are in both.
    omask = (mask!=0).astype(float)
    imask = (mask!=-9999).astype(float)

    # Convert lon and lat to polar stereographic coordinates
    lon_2d, lat_2d = np.meshgrid(lon, lat_b)
    x_interp, y_interp = polar_stereo(lon_2d, lat_2d)

    # Interpolate fields
    print('Interpolating bathymetry')
    bathy_interp = interp_topo(x, y, bathy, x_interp, y_interp)
    print('Interpolating ice shelf draft')
    draft_interp = interp_topo(x, y, draft, x_interp, y_interp)
    print('Interpolating ocean mask')
    omask_interp = interp_topo(x, y, omask, x_interp, y_interp)
    print('Interpolating ice mask')
    imask_interp = interp_topo(x, y, imask, x_interp, y_interp)

    if use_gebco:
        print('Filling in section north of 60S with GEBCO data')

        print('Reading data')
        id = nc.Dataset(gebco_file, 'r')
        lat_gebco_grid = id.variables['lat'][:]
        lon_gebco_grid = id.variables['lon'][:]
        # Figure out which indices we actually care about - buffer zone of 5 cells so the splines have room to breathe
        j_start = max(np.nonzero(lat_gebco_grid >= lat_g[0])[0][0] - 1 - 5, 0)
        j_end = min(np.nonzero(lat_gebco_grid >= lat_g[-1])[0][0] + 5, lat_gebco_grid.size-1)
        i_start = max(np.nonzero(lon_gebco_grid >= lon[0])[0][0] - 1 - 5, 0)
        i_end = min(np.nonzero(lon_gebco_grid >= lon[-1])[0][0] + 5, lon_gebco_grid.size-1)
        # Read GEBCO bathymetry just from this section
        bathy_gebco = id.variables['elevation'][j_start:j_end, i_start:i_end]
        id.close()
        # Trim the grid too
        lat_gebco_grid = lat_gebco_grid[j_start:j_end]
        lon_gebco_grid = lon_gebco_grid[i_start:i_end]

        print('Interpolating bathymetry')
        lon_2d, lat_2d = np.meshgrid(lon, lat_g)
        bathy_gebco_interp = interp_topo(lon_gebco_grid, lat_gebco_grid, bathy_gebco, lon_2d, lat_2d)

        print('Combining BEDMAP2 and GEBCO sections')
        # Deep copy the BEDMAP2 section of each field
        bathy_bedmap_interp = np.copy(bathy_interp)
        draft_bedmap_interp = np.copy(draft_interp)
        omask_bedmap_interp = np.copy(omask_interp)
        imask_bedmap_interp = np.copy(imask_interp)
        # Now combine them (remember we interpolated to the centres of grid cells, but lat and lon arrays define the edges, so minus 1 in each dimension)
        bathy_interp = np.empty([lat.size-1, lon.size-1])
        bathy_interp[:j_split-1,:] = bathy_bedmap_interp
        bathy_interp[j_split-1:,:] = bathy_gebco_interp
        # Ice shelf draft will be 0 in GEBCO region
        draft_interp = np.zeros([lat.size-1, lon.size-1])
        draft_interp[:j_split-1,:] = draft_bedmap_interp
        # Set ocean mask to 1 in GEBCO region; any land points will be updated later based on bathymetry > 0
        omask_interp = np.ones([lat.size-1, lon.size-1])
        omask_interp[:j_split-1,:] = omask_bedmap_interp
        # Ice mask will be 0 in GEBCO region
        imask_interp = np.zeros([lat.size-1, lon.size-1])
        imask_interp[:j_split-1,:] = imask_bedmap_interp

    print('Processing masks')
    # Deal with values interpolated between 0 and 1
    omask_interp[omask_interp < 0.5] = 0
    omask_interp[omask_interp >= 0.5] = 1
    imask_interp[imask_interp < 0.5] = 0
    imask_interp[imask_interp >= 0.5] = 1
    # Zero out bathymetry and ice shelf draft on land    
    bathy_interp[omask_interp==0] = 0
    draft_interp[omask_interp==0] = 0
    # Zero out ice shelf draft in the open ocean
    draft_interp[imask_interp==0] = 0
    
    # Update masks due to interpolation changing their boundaries
    # Anything with positive bathymetry should be land
    index = bathy_interp > 0
    omask_interp[index] = 0
    bathy_interp[index] = 0
    draft_interp[index] = 0
    # Anything with negative or zero water column thickness should be land
    index = draft_interp - bathy_interp <= 0
    omask_interp[index] = 0
    bathy_interp[index] = 0
    draft_interp[index] = 0
    # Anything with positive ice shelf draft should be land
    index = draft_interp > 0
    omask_interp[index] = 0
    bathy_interp[index] = 0
    draft_interp[index] = 0
    # Any points with zero ice shelf draft should not be in the ice mask
    # (This will also remove grounded ice, and ice shelves with total thickness (draft + freeboard) thinner than firn_air)
    index = draft_interp == 0
    imask_interp[index] = 0

    print('Removing isolated ocean cells')
    omask_interp = remove_isolated_cells(omask_interp)
    bathy_interp[omask_interp==0] = 0
    draft_interp[omask_interp==0] = 0
    imask_interp[omask_interp==0] = 0
    print('Removing isolated ice shelf cells')
    imask_interp = remove_isolated_cells(imask_interp)
    draft_interp[imask_interp==0] = 0
    
    if grounded_iceberg:
        bathy_interp, omask_interp = add_grounded_iceberg(rtopo_file, lon, lat, bathy_interp, omask_interp)
        
    print('Plotting')
    if use_gebco:
        # Remesh the grid, using the full latitude array
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    plot_tmp_domain(lon_2d, lat_2d, bathy_interp, title='Bathymetry (m)')
    plot_tmp_domain(lon_2d, lat_2d, draft_interp, title='Ice shelf draft (m)')
    plot_tmp_domain(lon_2d, lat_2d, draft_interp - bathy_interp, title='Water column thickness (m)')
    plot_tmp_domain(lon_2d, lat_2d, omask_interp, title='Ocean mask')
    plot_tmp_domain(lon_2d, lat_2d, imask_interp, title='Ice mask')

    # Write to NetCDF file (at cell centres not edges!)
    ncfile = NCfile_basiclatlon(nc_out, 0.5*(lon[1:] + lon[:-1]), 0.5*(lat[1:] + lat[:-1]))
    ncfile.add_variable('bathy', bathy_interp, units='m')
    ncfile.add_variable('draft', draft_interp, units='m')
    ncfile.add_variable('omask', omask_interp)
    ncfile.add_variable('imask', imask_interp)
    ncfile.close()

    print(('The results have been written into ' + nc_out))
    print('Take a look at this file and make whatever edits you would like to the mask (eg removing everything west of the peninsula; you can use edit_mask if you like)')
    print(("Then set your vertical layer thicknesses in a plain-text file, one value per line (make sure they clear the deepest bathymetry of " + str(abs(np.amin(bathy_interp))) + " m), and run remove_grid_problems"))


# Read topography which has been pre-interpolated to the new grid, from Ua output (to set up the initial domain for coupling). Add the grounded iceberg if needed.
def ua_topo (lon, lat, ua_file, nc_out, grounded_iceberg=True, topo_dir=None, rtopo_file=None):

    from .plot_latlon import plot_tmp_domain

    if grounded_iceberg:
        if topo_dir is None and rtopo_file is None:
            print('Error (ua_topo): must set topo_dir or rtopo_file if grounded_iceberg is True')
            sys.exit()
        if rtopo_file is None:
            rtopo_file = topo_dir+'RTopo-2.0.1_30sec_aux.nc'

    print(('Reading ' + ua_file))
    f = loadmat(ua_file)
    bathy = np.transpose(f['bathy'])
    draft = np.transpose(f['draft'])
    omask = np.transpose(f['omask'])
    imask = np.transpose(f['imask'])
    if (bathy.shape[0] != len(lat)-1) or (bathy.shape[1] != len(lon)-1):
        print(('Error (ua_topo): The fields in ' + ua_file + ' do not agree with the dimensions of your latitude and longitude.'))
        sys.exit()

    print('Removing isolated ocean cells')
    omask = remove_isolated_cells(omask)
    bathy[omask==0] = 0
    draft[omask==0] = 0
    imask[omask==0] = 0
    print('Removing isolated ice shelf cells')
    imask = remove_isolated_cells(imask)
    draft[imask==0] = 0

    if grounded_iceberg:
        bathy, omask = add_grounded_iceberg(rtopo_file, lon, lat, bathy, omask)

    print('Plotting')
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    plot_tmp_domain(lon_2d, lat_2d, bathy, title='Bathymetry (m)')
    plot_tmp_domain(lon_2d, lat_2d, draft, title='Ice shelf draft (m)')
    plot_tmp_domain(lon_2d, lat_2d, draft - bathy, title='Water column thickness (m)')
    plot_tmp_domain(lon_2d, lat_2d, omask, title='Ocean mask')
    plot_tmp_domain(lon_2d, lat_2d, imask, title='Ice mask')

    # Write to NetCDF file (at cell centres not edges!)
    ncfile = NCfile_basiclatlon(nc_out, 0.5*(lon[1:] + lon[:-1]), 0.5*(lat[1:] + lat[:-1]))
    ncfile.add_variable('bathy', bathy, units='m')
    ncfile.add_variable('draft', draft, units='m')
    ncfile.add_variable('omask', omask)
    ncfile.add_variable('imask', imask)
    ncfile.close()

    print(('The results have been written into ' + nc_out))
    print('Take a look at this file and make whatever edits you would like to the mask (eg removing everything west of the peninsula; you can use edit_mask if you like)')
    print(("Then set your vertical layer thicknesses in a plain-text file, one value per line (make sure they clear the deepest bathymetry of " + str(abs(np.amin(bathy))) + " m), and run remove_grid_problems"))


# Helper function to read variables from a temporary NetCDF grid file
def read_nc_grid (nc_file):

    lon = read_netcdf(nc_file, 'lon')
    lat = read_netcdf(nc_file, 'lat')
    bathy = read_netcdf(nc_file, 'bathy')
    draft = read_netcdf(nc_file, 'draft')
    omask = read_netcdf(nc_file, 'omask')
    imask = read_netcdf(nc_file, 'imask')
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    return lon_2d, lat_2d, bathy, draft, omask, imask


# Helper function to update variables in a temporary NetCDF grid file
def update_nc_grid (nc_file, bathy, draft, omask, imask):

    import netCDF4 as nc

    id = nc.Dataset(nc_file, 'a')
    id.variables['bathy'][:] = bathy
    id.variables['draft'][:] = draft
    id.variables['omask'][:] = omask
    id.variables['imask'][:] = imask
    id.close()


# Edit the land mask as desired, to block out sections of a domain. For example, Weddell Sea domains might like to make everything west of the peninsula into land.

# Arguments:
# nc_in: path to the temporary NetCDF grid file created by interp_bedmap2
# nc_out: desired path to the new NetCDF grid file with edits

# Optional keyword argument:
# key: string (default 'WSK') indicating which domain this is. You can make your own and do custom edits.

def edit_mask (nc_in, nc_out, key='WSK'):

    # Read all the variables
    lon_2d, lat_2d, bathy, draft, omask, imask = read_nc_grid(nc_in)

    # Edit the ocean mask based on the domain type
    if key == 'WSK':
        # Big Weddell Sea domain
        # Block out everything west of the peninsula, and extend the peninsula north to 61S
        # First, close a big box
        omask = mask_box(omask, lon_2d, lat_2d, xmax=-66, ymin=-74)
        # Now close everything north of a piecewise line defined by these points
        points = [[-66, -67], [-62, -65], [-60, -64.5], [-52, -61]]
        for i in range(len(points)-1):
            omask = mask_above_line(omask, lon_2d, lat_2d, points[i], points[i+1])
        # Now close a couple of little channels near the peninsula, with a few more boxes defined by [xmin, xmax, ymin, ymax]
        boxes = [[-59, -58, -64.3, -63.6], [-58.5, -57, -63.8, -63.4], [-57, -56.3, -63.4, -63]]
        for box in boxes:
            omask = mask_box(omask, lon_2d, lat_2d, xmin=box[0], xmax=box[1], ymin=box[2], ymax=box[3])
        # Close a disconnected region near Foundation Ice Stream
        omask = mask_box(omask, lon_2d, lat_2d, xmin=-65.5, xmax=-63.5, ymin=-81.8, ymax=-81.6) 
        # Turn the Baudouin Ice Shelf into land so there are no ice shelves on the open boundaries
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=24)
    elif key == 'SO-WISE-GYRE':
        # SO-WISE (gyre configuration)
        # Block out everything west of South America
        omask = mask_box(omask, lon_2d, lat_2d, xmin=-85.0, xmax=-70.0, ymin=-50.0, ymax=-30.0)
        # Close a disconnected region in the Falkland Islands
        omask = mask_box(omask, lon_2d, lat_2d, xmin=-60.4, xmax=-58.9, ymin=-52.1, ymax=-51.2)
        # Close disconnected regions in South America
        boxes = [[-65.1, -63.9, -43.0, -42.3],[-65.6, -64.6, -40.8, -39.9],[-72.4, -68.6, -54.5, -51.8]] 
        for box in boxes:
            omask = mask_box(omask, lon_2d, lat_2d, xmin=box[0], xmax=box[1], ymin=box[2], ymax=box[3])
        # Turn a few small, isolated ice shelves into land, fill a few other small shelves
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmax=-80, ymin=-75)
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=-65.1, xmax=-63.4, ymin=-81.8, ymax=-81.7)
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=-36.0, xmax=-34.0, ymin=-78.2, ymax=-77.4)
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=45.0, xmax=60.0, ymin=-69.0, ymax=-66.0)
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=80.0, xmax=90.0, ymin=-68.0, ymax=-66.0)
        # Fill in everything deeper than 6000 m (e.g. South Sandwich Trench)
        bathy[bathy<-6000] = -6000
    elif key == 'WSS':
        # Small Weddell Sea domain used for coupling
        # Block out everything west of the peninsula
        omask = mask_box(omask, lon_2d, lat_2d, xmax=-65, ymin=-75)
        # Fill all non-FRIS ice shelves with land
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmax=-55, ymin=-74.5)
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=-40, ymin=-78)
        # Also a few 1-cell ocean points surrounded by ice shelf draft. Fill them with the ice shelf draft of their neighbours.
        draft_w, draft_e, draft_s, draft_n = neighbours(draft)[:4]
        imask_w, imask_e, imask_s, imask_n, valid_w, valid_e, valid_s, valid_n, num_valid_neighbours = neighbours(imask, missing_val=0)
        index = (imask==0)*(num_valid_neighbours==4)
        imask[index] = 1
        draft[index] = 0.25*(draft_w+draft_e+draft_s+draft_n)[index]
    elif key == 'WSFRIS':
        # Big Weddell Sea domain used for coupling
        # Similar to WSK
        # Block out everything west of the peninsula, and extend the peninsula north to 61S
        omask = mask_box(omask, lon_2d, lat_2d, xmax=-66, ymin=-74)
        points = [[-66, -67], [-62, -65], [-60, -64.5], [-52, -61]]
        for i in range(len(points)-1):
            omask = mask_above_line(omask, lon_2d, lat_2d, points[i], points[i+1])
        boxes = [[-59, -58, -64.3, -63.6], [-58.5, -57, -63.8, -63.4], [-57, -56.3, -63.4, -63]]
        for box in boxes:
            omask = mask_box(omask, lon_2d, lat_2d, xmin=box[0], xmax=box[1], ymin=box[2], ymax=box[3])
            # Turn the Baudouin Ice Shelf into land so there are no ice shelves on the open boundaries
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=24)
        # Also a few 1-cell ocean points surrounded by ice shelf draft. Fill them with the ice shelf draft of their neighbours.
        draft_w, draft_e, draft_s, draft_n = neighbours(draft)[:4]
        num_valid_neighbours = neighbours(imask, missing_val=0)[-1]
        index = (imask==0)*(num_valid_neighbours==4)
        imask[index] = 1
        draft[index] = 0.25*(draft_w+draft_e+draft_s+draft_n)[index]
        # There is one 1-cell open-ocean ocean point surrounded by ice shelf and land. Fill it with land.
        oomask = omask-imask
        num_valid_neighbours = neighbours(oomask, missing_val=0)[-1]
        index = (oomask==1)*(num_valid_neighbours==0)
        omask[index] = 0
        bathy[index] = 0
    elif key == 'WSS_old_smaller':
        # Small Weddell Sea domain - temporary before coupling      
        # Block out everything west of the peninsula
        omask = mask_box(omask, lon_2d, lat_2d, xmax=-65, ymin=-75)
        # Remove Larsen D which intersects the open boundary
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, ymin=-73)
        # There are a few 1-cell islands; fill them with the bathymetry and ice shelf draft of their neighbours
        omask_w, omask_e, omask_s, omask_n, valid_w, valid_e, valid_s, valid_n, num_valid_neighbours = neighbours(omask, missing_val=0)
        index = (omask==0)*(num_valid_neighbours==4)
        omask[index] = 1
        bathy_w, bathy_e, bathy_s, bathy_n = neighbours(bathy)[:4]
        bathy[index] = 0.25*(bathy_w+bathy_e+bathy_s+bathy_n)[index]
        imask_w, imask_e, imask_s, imask_n = neighbours(imask)[:4]
        imask[index] = np.ceil(0.25*(imask_w+imask_e+imask_s+imask_n))[index]
        draft_w, draft_e, draft_s, draft_n = neighbours(draft)[:4]
        draft[index] = 0.25*(draft_w+draft_e+draft_s+draft_n)[index]
        # Also a few 1-cell ocean points surrounded by ice shelf draft. Fill them with the ice shelf draft of their neighbours.
        imask_w, imask_e, imask_s, imask_n, valid_w, valid_e, valid_s, valid_n, num_valid_neighbours = neighbours(imask, missing_val=0)
        index = (imask==0)*(num_valid_neighbours==4)
        imask[index] = 1
        draft[index] = 0.25*(draft_w+draft_e+draft_s+draft_n)[index]
    else:
        raise Exception("Key not found. No edits have been applied") 
        
    # Make the other fields consistent with this new mask
    index = omask == 0
    bathy[index] = 0
    draft[index] = 0
    imask[index] = 0

    # Copy the NetCDF file to a new name
    shutil.copyfile(nc_in, nc_out)
    # Update the variables
    update_nc_grid(nc_out, bathy, draft, omask, imask)

    print(("Fields updated successfully. The deepest bathymetry is now " + str(abs(np.amin(bathy))) + " m."))

    
# Helper function to read vertical layer thicknesses from an ASCII file, and compute the edges of the z-levels. Returns dz and z_edges.
def vertical_layers (dz_file):

    dz = []
    f = open(dz_file, 'r')
    for line in f:
        dz.append(float(line))
    f.close()
    z_edges = -1*np.cumsum(np.array([0] + dz))
    dz = np.array(dz)
    return dz, z_edges


# Helper function to calculate a few variables about z-levels based on the given depth field A (which could be bathymetry or ice shelf draft):
# (1) Index (0-based) of vertical layer that A falls into
# (2) Depth of the first z-level (edge, not centre) above A
# (3) Depth of the first z-level below A
# (4) Thickness of the vertical layer that A falls into (i.e. difference between 2 and 3)
# (5) Thickness of the vertical layer above that
# (6) Thickness of the vertical layer below that
# The keyword argument include_edge determines what happens to values of A on the edge of a level. If include_edge='top', the top edge is considered to be part of the level, while the bottom edge is considered to be part of the level below. If include_edge='bottom', the bottom edge is considered to be part of the level, while the top edge is considered to be part of the level above.
def level_vars (A, dz, z_edges, include_edge='top'):

    # Prepare to calculate the variables
    layer_number = np.zeros(A.shape)
    level_above = np.zeros(A.shape)
    level_below = np.zeros(A.shape)
    dz_layer = np.zeros(A.shape)
    dz_layer_above = np.zeros(A.shape)
    dz_layer_below = np.zeros(A.shape)
    # Flag to catch undefined variables after the loop
    flag = np.zeros(A.shape)
    # Loop over vertical levels
    for k in range(dz.size):
        # Find points within this vertical layer
        if (include_edge == 'top' and k==dz.size-1) or (include_edge=='bottom' and k==0):
            # Include both edges because there are no more levels
            index = (A <= z_edges[k])*(A >= z_edges[k+1])
        elif include_edge == 'top':
            index = (A <= z_edges[k])*(A > z_edges[k+1])
        elif include_edge == 'bottom':
            index = (A < z_edges[k])*(A >= z_edges[k+1])
        else:
            print(('Error (level_vars): invalid include_edge=' + include_edge))
            sys.exit()
        layer_number[index] = k
        level_above[index] = z_edges[k]
        level_below[index] = z_edges[k+1]
        dz_layer[index] = dz[k]
        if k == 0:  # Points in the top layer will extrapolate dz_layer_above
            dz_layer_above[index] = dz[k]
        else:
            dz_layer_above[index] = dz[k-1]
        if k+1 == dz.size:  # Points in the botttom layer will extrapolate dz_layer_below
            dz_layer_below[index] = dz[k]
        else:
            dz_layer_below[index] = dz[k+1]
        flag[index] = 1
    if (flag==0).any():
        print('Error (level_vars): some values not caught by the loop. This could happen if some of your ice shelf draft points are in the bottommost vertical layer. This will impede digging. Adjust your vertical layer thicknesses and try again.')
        sys.exit()
    return layer_number, level_above, level_below, dz_layer, dz_layer_above, dz_layer_below


# Helper function to calculate the actual bathymetry or ice shelf draft as seen by the model, based on hFac constraints, but WITHOUT considering the case where the water column is so thin that it will be closed. (These cases will be taken care of by digging later.) Instead, calculate the bathymetry as if the ice shelf draft is infinitely thin, and calculate the ice shelf draft as if the bathymetry is infinitely deep.
def single_model_bdry (A, dz, z_edges, option='bathy', hFacMin=0.1, hFacMinDr=20.):

    if option == 'bathy':
        include_edge = 'bottom'
    elif option == 'draft':
        include_edge = 'top'
    else:
        print(('Error (single_model_bdry): invalid option ' + option))
        sys.exit()

    # Get some intermediate variables
    level_above, level_below, dz_layer = level_vars(A, dz, z_edges, include_edge=include_edge)[1:4]
    # Determine which is the open edge and which is the closed ege
    if option == 'bathy':
        open_edge = level_above
        closed_edge = level_below
    elif option == 'draft':
        open_edge = level_below
        closed_edge = level_above
    # Calculate the hFac of the partial cell
    hfac = np.abs(A - open_edge)/dz_layer
    # Find the minimum acceptable hFac
    hfac_limit = np.maximum(hFacMin, np.minimum(hFacMinDr/dz_layer, 1))
    # Update A; start with a deep copy
    model_A = np.copy(A)
    # Find cells which should be fully closed
    index = hfac < hfac_limit/2
    model_A[index] = open_edge[index]
    # Find cells which should be fully open
    index = (hfac < hfac_limit)*(hfac >= hfac_limit/2)
    model_A[index] = closed_edge[index]

    return model_A


# Deal with three problems which can result with z-levels:
# (1) Cells on the bottom can become isolated, with no horizontal neighbours (i.e. adjacent open cells), just one vertical neighbour above. Dense water tends to pool in these cells and can't mix out very easily. Fix this by raising the bathymetry until the bottom cell has at least one horizontal neighbour. We call it "filling".
# (2) Subglacial lakes can form beneath the ice shelves, whereby two cells which should have connected water columns (based on the masks we interpolated from BEDMAP2) are disconnected, i.e. the ice shelf draft at one cell is deeper than the bathymetry at a neighbouring cell (due to interpolation). Fix this by deepening the bathymetry where needed, so there are a minimum of 2 (at least partially) open faces between the neighbouring cells, ensuring that both tracers and velocities are connected. This preserves the BEDMAP2 grounding line locations, even if the bathymetry is somewhat compromised. We call it "digging".
# (3) Ice shelf drafts thin enough to be contained within the surface layer cause problems. There are two cases:
#    (3a) If hFacMinDr is equal to or greater than the depth of the surface layer, and an ice shelf draft is thinner than half the depth of the surface layer, it will be removed by MITgcm. However, older versions of MITgcm have a bug whereby some parts of the code don't remove the ice shelf draft at these points, and they are simultaneously treated as ice shelf and sea ice points. Fix this by removing all such points. We call it "zapping".
#    (3b) If hFacMinDr is less than the depth of the surface layer, and an ice shelf draft is also less than the depth of the surface layer, the MITgcm sea ice code (which masks based on hFacC[surface] > 0) will grow sea ice there anyway. Fix this by zapping ice shelf drafts less than half the depth of the surface layer (as in 3a above) and growing ice shelf drafts between half the depth and the full depth of the surface layer, to the bottom of that layer.

# do_filling, do_digging, and do_zapping are the helper functions; remove_grid_problems is the API.


# Fix problem (1) above.
def do_filling (bathy, dz, z_edges, hFacMin=0.1, hFacMinDr=20.):

    # Find the actual bathymetry as the model will see it (based on hFac constraints)
    model_bathy = single_model_bdry(bathy, dz, z_edges, option='bathy', hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    # Find the depth of the z-level below each bathymetry point
    level_below = level_vars(model_bathy, dz, z_edges, include_edge='bottom')[2]
    # Also find this value at the deepest horizontal neighbour for every point
    level_below_w, level_below_e, level_below_s, level_below_n = neighbours(level_below)[:4]
    level_below_neighbours = np.stack((level_below_w, level_below_e, level_below_s, level_below_n))
    level_below_deepest_neighbour = np.amin(level_below_neighbours, axis=0)
    # Find cells which are in a deeper vertical layer than all their neighbours, and build them up by the minimum amount necessary
    print(('...' + str(np.count_nonzero(bathy < level_below_deepest_neighbour)) + ' cells to fill'))
    bathy = np.maximum(bathy, level_below_deepest_neighbour)

    return bathy


# Fix problem (2) above.
# Default is to dig the bathymetry; another option (for coupled simulations at restart points) is to dig the ice shelf draft. 
def do_digging (bathy, draft, dz, z_edges, hFacMin=0.1, hFacMinDr=20., dig_option='bathy'):

    # Figure out which field will be modified, which the other field is, which edge should be included in call to level_vars, and whether we are making the field deeper (-1) or shallower (1).
    if dig_option == 'bathy':
        field = bathy
        other_option = 'draft'
        other_field = draft
        include_edge = 'top'
        direction_flag = -1
    elif dig_option == 'draft':
        field = draft
        other_option = 'bathy'
        other_field = bathy
        include_edge = 'bottom'
        direction_flag = 1
    else:
        print(('Error (do_digging): invalid dig_option ' + dig_option))
        sys.exit()

    # Find the other field as the model will see it (based on hFac constraints)
    model_other_field = single_model_bdry(other_field, dz, z_edges, option=other_option, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    # Get some variables about the vertical grid
    layer_number, level_above, level_below, dz_layer, dz_layer_above, dz_layer_below = level_vars(model_other_field, dz, z_edges, include_edge=include_edge)
    # Figure out which ones we care about
    if dig_option == 'bathy':
        level_next = level_below  # Depth of the z-level below the draft
        dz_next = dz_layer_below  # Thickness of the layer below that
    elif dig_option == 'draft':
        level_next = level_above  # Depth of the z-level above the bathymetry
        dz_next = dz_layer_above  # Thickness of the layer above that
        # Also make sure the bathymetry itself is deep enough
        if (layer_number == 1).any():
            print("Error (do_digging): some bathymetry points are within the first vertical layer. If this is a coupled simulation, you need to set up the initial domain using bathymetry digging. If this is not a coupled simulation, use dig_option='bathy'.")
            sys.exit()
            
    # Figure out the shallowest acceptable bathymetry OR the deepest acceptable ice shelf draft of each point and its neighbours. We want 2 (at least partially) open cells.
    # The first open cell is between the draft and the z-level below it, OR the bathymetry and the z-level above it.
    limit = level_next
    # The second open cell digs into the layer below OR above that by the minimum amount (based on hFac constraints).
    hfac_limit = np.maximum(hFacMin, np.minimum(hFacMinDr/dz_next, 1))
    limit += direction_flag*dz_next*hfac_limit
    # In the land mask, there is no limit.
    if dig_option == 'bathy':
        # Shallowest acceptable bathymetry is zero
        limit[bathy==0] = 0
    elif dig_option == 'draft':
        # Deepest acceptable ice shelf draft is the bottom of the grid
        limit[bathy==0] = z_edges[-1]
    # Get limit at each point's 4 neighbours
    limit_w, limit_e, limit_s, limit_n = neighbours(limit)[:4]

    # Inner function to apply limits to the field (based on each point itself, or each point's neighbour in a single direction eg. west).
    def dig_one_direction (limit):
        # Mask out the land either way
        if dig_option == 'bathy':
            # Find bathymetry that's too shallow
            index = (bathy != 0)*(field > limit)
        elif dig_option == 'draft':
            # Find ice shelf draft that's too deep
            index = (bathy != 0)*(field < limit)
        print(('...' + str(np.count_nonzero(index)) + ' cells to dig'))
        field[index] = limit[index]
        return field

    field = dig_one_direction(limit)
    limit_neighbours = [limit_w, limit_e, limit_s, limit_n]
    loc_strings = ['west', 'east', 'south', 'north']
    for i in range(len(loc_strings)):
        print(('Digging based on field to ' + loc_strings[i]))
        field = dig_one_direction(limit_neighbours[i])

    # Error checking
    if (dig_option == 'bathy' and (field < z_edges[-1]).any()) or (dig_option == 'draft' and (field > 0).any()):
        print('Error (do_digging): we have dug off the edge of the grid!!')
        sys.exit()

    return field


# Fix problem (3) above.
def do_zapping (draft, imask, dz, z_edges, hFacMinDr=20., only_grow=False):

    if only_grow:
        # Find any points which are less than the depth of the surface layer and grow them
        index = (draft != 0)*(abs(draft) < dz[0])
        print(('...' + str(np.count_nonzero(index)) + ' cells to grow'))
        draft[index] = -1*dz[0]
    else:
        # Find any points which are less than half the depth of the surface layer and remove them
        index = (draft != 0)*(abs(draft) < 0.5*dz[0])
        print(('...' + str(np.count_nonzero(index)) + ' cells to zap'))
        draft[index] = 0
        imask[index] = 0
        if hFacMinDr < dz[0]:
            # Also find any points which are between half the depth and the full depth of the surface layer, and grow them
            index = (abs(draft) >= 0.5*dz[0])*(abs(draft) < dz[0])
            print(('...' + str(np.count_nonzero(index)) + ' cells to grow'))
            draft[index] = -1*dz[0]
        
    return draft, imask
        

# Fix all three problems at once.

# Arguments:
# nc_in: NetCDF temporary grid file (created by edit_mask if you used that function, otherwise created by interp_bedmap2)
# nc_out: desired path to the new NetCDF grid file with edits
# dz_file: path to an ASCII (plain text) file containing your desired vertical layer thicknesses, one per line, positive, in metres

# Optional keyword arguments:
# hFacMin, hFacMinDr: make sure these match the values in your "data" namelist for MITgcm
# coupled: set to True if this is the initial topography for a coupled run. This will only grow ice shelf draft rather than zapping it.

def remove_grid_problems (nc_in, nc_out, dz_file, hFacMin=0.1, hFacMinDr=20., coupled=False):

    from .plot_latlon import plot_tmp_domain

    # Read all the variables
    lon_2d, lat_2d, bathy, draft, omask, imask = read_nc_grid(nc_in)
    # Generate the vertical grid
    dz, z_edges = vertical_layers(dz_file)
    if z_edges[-1] > np.amin(bathy):
        print(('Error (remove_grid_problems): deepest bathymetry is ' + str(abs(np.amin(bathy))) + ' m, but your vertical levels only go down to ' + str(abs(z_edges[-1])) + ' m. Adjust your vertical layer thicknesses and try again.'))
        sys.exit()

    print('Filling isolated bottom cells')
    bathy_orig = np.copy(bathy)
    bathy = do_filling(bathy, dz, z_edges, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    # Plot how the results have changed
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, bathy), title='Bathymetry (m) after filling')
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, bathy-bathy_orig), title='Change in bathymetry (m)\ndue to filling')

    print('Digging subglacial lakes')
    bathy_orig = np.copy(bathy)
    bathy = do_digging(bathy, draft, dz, z_edges, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    # Plot how the results have changed
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, bathy), title='Bathymetry (m) after digging')
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, bathy-bathy_orig), title='Change in bathymetry (m)\ndue to digging')

    print('Zapping thin ice shelf draft')
    draft_orig = np.copy(draft)
    draft, imask = do_zapping(draft, imask, dz, z_edges, hFacMinDr=hFacMinDr, only_grow=coupled)
    # Plot how the results have changed
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, draft-draft_orig), title='Change in ice shelf draft (m)\ndue to zapping')

    # Copy the NetCDF file to a new name
    shutil.copyfile(nc_in, nc_out)
    # Update the variables
    update_nc_grid(nc_out, bathy, draft, omask, imask)

    print(("The updated grid has been written into " + nc_out + ". Take a look and make sure everything looks okay. If you're happy, run write_topo_files to generate the binary files for MITgcm input."))


# Given a precomputed grid (from interp_bedmap2 + edit_mask + remove_grid_problems), swap in topography from an Ua simulation - just like in the UaMITgcm coupler. First remove FRIS from the existing domain so the ice front is identical.
def swap_ua_topo (nc_file, ua_file, dz_file, out_file, hFacMin=0.1, hFacMinDr=20.):

    # Read the input grid
    lon, lat, bathy_old, draft_old, omask_old, imask_old = read_nc_grid(nc_file)
    dz, z_edges = vertical_layers(dz_file)

    # Remove FRIS
    regions = [region_bounds['fris1'], region_bounds['fris2']]
    for bounds in regions:
        imask_old = mask_iceshelf_box(omask_old, imask_old, lon, lat, xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3], option='ocean')
    index = imask_old == 0
    draft_old[index] = 0

    # Read Ua topography
    f = loadmat(ua_file)
    bathy = np.transpose(f['B_forMITgcm'])
    draft = np.transpose(f['b_forMITgcm'])
    mask = np.transpose(f['mask_forMITgcm'])        
    # Mask grounded ice out of both fields
    bathy[mask==0] = 0
    draft[mask==0] = 0
    # Mask out regions with bathymetry greater than zero
    index = bathy > 0
    bathy[index] = 0
    draft[index] = 0

    # Preserve ocean mask
    index = (mask==2)*(bathy_old==0)
    bathy[index] = 0

    # Preserve static ice
    index = (mask==2)*(draft_old<0)
    draft[index] = draft_old[index]

    # Recompute ocean masks
    omask = bathy!=0
    imask = draft!=0

    # Fix grid problems
    bathy = do_filling(bathy, dz, z_edges, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    bathy = do_digging(bathy, draft, dz, z_edges, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    draft = do_zapping(draft, draft!=0, dz, z_edges, hFacMinDr=hFacMinDr, only_grow=True)[0]

    # Write results
    shutil.copyfile(nc_file, out_file)
    update_nc_grid(out_file, bathy, draft, omask, imask)

    print('Finished swapping out Ua geometry. Take a look and then call write_topo_files.')

    
# Write the bathymetry and ice shelf draft fields, currently stored in a NetCDF file, into binary files to be read by MITgcm.
def write_topo_files (nc_grid, bathy_file, draft_file, prec=64):

    bathy = read_netcdf(nc_grid, 'bathy')
    draft = read_netcdf(nc_grid, 'draft')
    write_binary(bathy, bathy_file, prec=prec)
    write_binary(draft, draft_file, prec=prec)
    print('Files written successfully. Now go try them out! Make sure you update all the necessary variables in data, data.shelfice, SIZE.h, job scripts, etc.')


# Helper function to check that neighbouring ocean cells have at least 2 open faces in the given direction.
def check_one_direction (open_cells, open_cells_beside, loc_string, problem):

    # Open faces are equivalent to adjacent open cells
    open_face = open_cells.astype(int)*open_cells_beside.astype(int)
    # Check pairs of points which are both non-land (at least 1 open cell in the water column), to make sure they all have at least 2 open faces between them
    num_pinched = np.count_nonzero((np.sum(open_cells, axis=0) != 0)*(np.sum(open_cells_beside, axis=0) != 0)*(np.sum(open_face, axis=0)<2))
    if num_pinched > 0:
        problem = True
        print(('Problem!! There are ' + str(num_pinched) + ' locations with less than 2 open faces on the ' + loc_string + ' side.'))
    return problem
        

# Given the path to a directory containing the binary grid files produced by MITgcm, make sure that the filling and digging worked and that the 2 open cell rule holds.
def check_final_grid (grid_path):

    grid = Grid(grid_path)
    problem = False

    # Check there are no isolated bottom cells
    # Find points which are open, their neighbour below is closed (i.e. they're at the seafloor), and their horizontal neighoburs are all closed
    hfac = grid.hfac
    num_valid_neighbours = neighbours(hfac, missing_val=0)[-1]
    valid_below = neighbours_z(hfac, missing_val=0)[3]
    num_isolated = np.count_nonzero((hfac!=0)*(valid_below==0)*(num_valid_neighbours==0))
    if num_isolated > 0:
        problem = True
        print(('Problem!! There are ' + str(num_isolated) + ' locations with isolated bottom cells.'))    

    # Check that every water column has at least 2 open cells (partial cells count)
    open_cells = np.ceil(grid.hfac)
    num_pinched = np.count_nonzero(np.sum(open_cells, axis=0)==1)
    if num_pinched > 0:
        problem = True
        print(('Problem!! There are ' + str(num_pinched) + ' locations with only one open cell in the water column.'))

    # Check that neighbouring ocean cells have at least 2 open faces between
    open_cells_w, open_cells_e, open_cells_s, open_cells_n = neighbours(open_cells)[:4]
    open_cells_neighbours = [open_cells_w, open_cells_e, open_cells_s, open_cells_n]
    loc_strings = ['western', 'eastern', 'southern', 'northern']
    for i in range(len(loc_strings)):
        problem = check_one_direction(open_cells, open_cells_neighbours[i], loc_strings[i], problem)
        
    if problem:
        print('Something went wrong with the filling or digging. Are you sure that your values of hFacMin and hFacMinDr are correct? Are you working with a version of MITgcm that calculates Ro_sfc and R_low differently?')
    else:
        print('Everything looks good!')


# Merge updates to a BEDMAP2 field from two or more sources (e.g. Sebastian Rosier's updates to the Filchner and Lianne Harrison's updates to the Larsen). The code makes sure that no updates are contradictory, i.e. trying to change the same points in different ways.

# Arguments:
# orig_file: path to an original BEDMAP2 file
# updated_files: list of paths to altered BEDMAP2 files, of the same format (encoding, size, row vs column major)
# out_file: path to desired merged file

def merge_bedmap2_changes (orig_file, updated_files, out_file):

    # Read all the files
    data_orig = np.fromfile(orig_file, dtype='<f4')
    num_files = len(updated_files)
    data_new = np.empty([num_files, data_orig.size])
    for i in range(num_files):
        data_new[i,:] = np.fromfile(updated_files[i], dtype='<f4')

    # Make sure none of the changes overlap
    changes = (data_new!=data_orig).astype(float)
    if np.amax(np.sum(changes, axis=0)) > 1:
        # Some changes overlap, but maybe they are the same changes.
        stop = False
        for i in range(num_files):
            for j in range(i+1, num_files):
                index = (changes[i,:]==1)*(changes[j,:]==1)
                if (data_new[i,:][index] != data_new[j,:][index]).any():
                    stop = True
                    print((updated_files[i] + ' contradicts ' + updated_files[j]))
        if stop:
            print('Error (merge_bedmap2_changes): some changes are contradictory')
            sys.exit()

    # Apply the changes
    data_final = np.copy(data_orig)
    for i in range(num_files):
        data_tmp = data_new[i,:]
        index = data_tmp != data_orig
        data_final[index] = data_tmp[index]

    # Write to file
    write_binary(data_final, out_file, prec=32, endian='little')



    
    
    
    

    
