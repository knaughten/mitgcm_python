#######################################################
# Generate a new model domain.
#######################################################

import numpy as np
import sys
import netCDF4 as nc
import shutil
import os

from constants import deg2rad
from file_io import write_binary, NCfile_basiclatlon, read_netcdf, read_binary
from utils import factors, polar_stereo, mask_box, mask_above_line, mask_iceshelf_box, real_dir, mask_3d, select_top, z_to_xyz
from interpolation import extend_into_mask, interp_topo, neighbours, remove_isolated_cells 
from plot_latlon import plot_tmp_domain
from grid import Grid

def latlon_points (xmin, xmax, ymin, ymax, res, dlat_file, prec=64):

    # Number of iterations for latitude convergence
    num_lat_iter = 10

    if xmin > xmax:
        print "Error (latlon_points): looks like your domain crosses 180E. Try again with your longitude in the range (0, 360) instead of (-180, 180)."
        sys.exit()

    # Build longitude values
    lon = np.arange(xmin, xmax+res, res)
    # Update xmax if the range doesn't evenly divide by res
    if xmax != lon[-1]:
        xmax = lon[-1]
        print 'Eastern boundary moved to ' + str(xmax)
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
    print 'Northern boundary moved to ' + str(ymax)

    # Write latitude resolutions to file
    dlat = lat[1:] - lat[:-1]
    write_binary(dlat, dlat_file, prec=prec)

    # Remind the user what to do in their namelist
    print '\nChanges to make to input/data:'
    print 'xgOrigin=' + str(xmin)
    print 'ygOrigin=' + str(ymin)
    print 'dxSpacing=' + str(res)
    print "delYfile='" + dlat_file + "' (and copy this file into input/)"

    # Find dimensions of tracer grid
    Nx = lon.size-1
    Ny = lat.size-1
    # Find all the factors
    factors_x = factors(Nx)
    factors_y = factors(Ny)
    print '\nNx = ' + str(Nx) + ' which has the factors ' + str(factors_x)
    print 'Ny = ' + str(Ny) + ' which has the factors ' + str(factors_y)
    print 'If you are happy with this, proceed with interp_bedmap2. At some point, choose your tile size based on the factors and update code/SIZE.h.'
    print 'Otherwise, tweak the boundaries and try again.'

    return lon, lat


# Interpolate BEDMAP2 bathymetry, ice shelf draft, and masks to the new grid. Write the results to a NetCDF file so the user can check for any remaining artifacts that need fixing (eg blocking out the little islands near the peninsula).
def interp_bedmap2 (lon, lat, topo_dir, nc_out, seb_updates=True):

    topo_dir = real_dir(topo_dir)

    # BEDMAP2 file names
    if seb_updates:
        bed_file = 'bedmap2_bed_seb.flt'
    else:
        bed_file = 'bedmap2_bed.flt'
    surface_file = 'bedmap2_surface.flt'
    thickness_file = 'bedmap2_thickness.flt'
    mask_file = 'bedmap2_icemask_grounded_and_shelves.flt'
    # GEBCO file name
    gebco_file = 'GEBCO_2014_2D.nc'

    # BEDMAP2 grid parameters
    bedmap_dim = 6667    # Dimension
    bedmap_bdry = 3333000    # Polar stereographic coordinate (m) on boundary
    bedmap_res = 1000    # Resolution (m)
    missing_val = -9999    # Missing value for bathymetry north of 60S

    if np.amin(lat) > -60:
        print "Error (interp_bedmap2): this domain doesn't go south of 60S, so it's not covered by BEDMAP2."
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

    print 'Reading data'
    # Have to flip it vertically so lon0=0 in polar stereographic projection
    # Otherwise, lon0=180 which makes x_interp and y_interp strictly decreasing when we call polar_stereo later, and the interpolation chokes
    bathy = np.flipud(np.fromfile(topo_dir+bed_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))
    surf = np.flipud(np.fromfile(topo_dir+surface_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))
    thick = np.flipud(np.fromfile(topo_dir+thickness_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))
    mask = np.flipud(np.fromfile(topo_dir+mask_file, dtype='<f4').reshape([bedmap_dim, bedmap_dim]))

    if np.amax(lat_b) > -61:
        print 'Extending bathymetry slightly past 60S'
        # Bathymetry has missing values north of 60S. Extend into that mask so there are no artifacts in the splines near 60S.
        bathy = extend_into_mask(bathy, missing_val=missing_val, num_iters=5)

    print 'Calculating ice shelf draft'
    # Calculate ice shelf draft from ice surface and ice thickness
    draft = surf - thick

    print 'Calculating ocean and ice masks'
    # Mask: -9999 is open ocean, 0 is grounded ice, 1 is ice shelf
    # Make an ocean mask and an ice mask. Ice shelves are in both.
    omask = (mask!=0).astype(float)
    imask = (mask!=-9999).astype(float)

    # Convert lon and lat to polar stereographic coordinates
    lon_2d, lat_2d = np.meshgrid(lon, lat_b)
    x_interp, y_interp = polar_stereo(lon_2d, lat_2d)

    # Interpolate fields
    print 'Interpolating bathymetry'
    bathy_interp = interp_topo(x, y, bathy, x_interp, y_interp)
    print 'Interpolating ice shelf draft'
    draft_interp = interp_topo(x, y, draft, x_interp, y_interp)
    print 'Interpolating ocean mask'
    omask_interp = interp_topo(x, y, omask, x_interp, y_interp)
    print 'Interpolating ice mask'
    imask_interp = interp_topo(x, y, imask, x_interp, y_interp)

    if use_gebco:
        print 'Filling in section north of 60S with GEBCO data'

        print 'Reading data'
        id = nc.Dataset(topo_dir+gebco_file, 'r')
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

        print 'Interpolating bathymetry'
        lon_2d, lat_2d = np.meshgrid(lon, lat_g)
        bathy_gebco_interp = interp_topo(lon_gebco_grid, lat_gebco_grid, bathy_gebco, lon_2d, lat_2d)

        print 'Combining BEDMAP2 and GEBCO sections'
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

    print 'Processing masks'
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
    # Any points with zero ice shelf draft should not be in the ice mask
    # (This will remove grounded ice)
    index = draft_interp == 0
    imask_interp[index] = 0

    print 'Removing isolated ocean cells'
    omask_interp = remove_isolated_cells(omask_interp)
    bathy_interp[omask_interp==0] = 0
    draft_interp[omask_interp==0] = 0
    imask_interp[omask_interp==0] = 0
    print 'Removing isolated ice shelf cells'
    imask_interp = remove_isolated_cells(imask_interp)
    draft_interp[imask_interp==0] = 0
        
    print 'Plotting'
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

    print 'The results have been written into ' + nc_out
    print 'Take a look at this file and make whatever edits you would like to the mask (eg removing everything west of the peninsula; you can use edit_mask if you like)'
    print "Then set your vertical layer thicknesses in a plain-text file, one value per line (make sure they clear the deepest bathymetry of " + str(abs(np.amin(bathy_interp))) + " m), and run remove_grid_problems"


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
# key: string (default 'WSB') indicating which domain this is. You can make your own and do custom edits.

def edit_mask (nc_in, nc_out, key='WSB'):

    # Read all the variables
    lon_2d, lat_2d, bathy, draft, omask, imask = read_nc_grid(nc_in)

    # Edit the ocean mask based on the domain type
    if key == 'WSB':
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
        # Finally, turn the Baudouin Ice Shelf into land so there are no ice shelves on the open boundaries
        omask = mask_iceshelf_box(omask, imask, lon_2d, lat_2d, xmin=24)

    # Make the other fields consistent with this new mask
    index = omask == 0
    bathy[index] = 0
    draft[index] = 0
    imask[index] = 0

    # Copy the NetCDF file to a new name
    shutil.copyfile(nc_in, nc_out)
    # Update the variables
    update_nc_grid(nc_out, bathy, draft, omask, imask)

    print "Fields updated successfully. The deepest bathymetry is now " + str(abs(np.amin(bathy))) + " m."

    
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


# Helper function to calculate a few variables about z-levels based on the given ice shelf draft:
# (1) Depth of the first z-level below the draft (if the draft is exactly at a z-level, it will still go down to the next one)
# (2) Thickness of the z-layer the draft is in (i.e. difference between (1) and the level above that)
# (3) Thickness of the z-level below that
def draft_level_vars (draft, dz, z_edges):

    # Prepare to calculate the 3 variables
    level_below_draft = np.zeros(draft.shape)    
    dz_at_draft = np.zeros(draft.shape)
    dz_below_draft = np.zeros(draft.shape)
    # Flag to catch undefined variables after the loop
    flag = np.zeros(draft.shape)
    # Loop over vertical levels
    for k in range(dz.size-1):
        # Find ice shelf drafts within this vertical layer (not counting the bottom edge)
        index = (draft <= z_edges[k])*(draft > z_edges[k+1])
        level_below_draft[index] = z_edges[k+1]
        dz_at_draft[index] = dz[k]
        dz_below_draft[index] = dz[k+1]
        flag[index] = 1
    if (flag==0).any():
        print 'Error (draft_level_vars): some of your ice shelf draft points are in the bottommost vertical layer. This will impede digging. Adjust your vertical layer thicknesses and try again.'
        sys.exit()
    return level_below_draft, dz_at_draft, dz_below_draft


# Helper function to apply limits to bathymetry (based on each point itself, or each point's neighbour in a single direction eg. west).
def dig_one_direction (bathy, bathy_limit):

    index = (bathy != 0)*(bathy > bathy_limit)
    print '...' + str(np.count_nonzero(index)) + ' cells to dig'
    bathy[index] = bathy_limit[index]
    return bathy    


# Deal with two problems which can result from ice shelves and z-levels:
# (1) Subglacial lakes can form beneath the ice shelves, whereby two cells which should have connected water columns (based on the masks we interpolated from BEDMAP2) are disconnected, i.e. the ice shelf draft at one cell is deeper than the bathymetry at a neighbouring cell (due to interpolation). Fix this by deepening the bathymetry where needed, so there are a minimum of 2 (at least partially) open faces between the neighbouring cells, ensuring that both tracers and velocities are connected. This preserves the BEDMAP2 grounding line locations, even if the bathymetry is somewhat compromised. We call it "digging".
# (2) Very thin ice shelf drafts (less than half the depth of the surface layer) will violate the hFacMin constraints and be removed by MITgcm. However, older versions of MITgcm have a bug whereby some parts of the code don't remove the ice shelf draft at these points, and they are simultaneously treated as ice shelf and sea ice points. Fix this by removing all such points. We call it "zapping".

# Arguments:
# nc_in: NetCDF temporary grid file (created by edit_mask if you used that function, otherwise created by interp_bedmap2)
# nc_out: desired path to the new NetCDF grid file with edits
# dz_file: path to an ASCII (plain text) file containing your desired vertical layer thicknesses, one per line, positive, in metres

# Optional keyword arguments:
# hFacMin, hFacMinDr: make sure these match the values in your "data" namelist for MITgcm

def remove_grid_problems (nc_in, nc_out, dz_file, hFacMin=0.1, hFacMinDr=20.):

    # Read all the variables
    lon_2d, lat_2d, bathy, draft, omask, imask = read_nc_grid(nc_in)
    # Generate the vertical grid
    dz, z_edges = vertical_layers(dz_file)
    if z_edges[-1] > np.amin(bathy):
        print 'Error (remove_grid_problems): deepest bathymetry is ' + str(abs(np.amin(bathy))) + ' m, but your vertical levels only go down to ' + str(abs(z_edges[-1])) + ' m. Adjust your vertical layer thicknesses and try again.'
        sys.exit()

    # Find the actual draft as the model will see it (based on hFac constraints)
    model_draft = np.copy(draft)
    # Get some intermediate variables
    level_below_draft, dz_at_draft = draft_level_vars(draft, dz, z_edges)[:2]
    # Calculate the hFac of the partial cell below the draft
    hfac_below_draft = (draft - level_below_draft)/dz_at_draft
    # Now, modify the draft based on hFac constraints
    hfac_limit = np.maximum(hFacMin, np.minimum(hFacMinDr/dz_at_draft, 1))
    # Find cells which should be fully closed
    index = hfac_below_draft < hfac_limit/2
    model_draft[index] = level_below_draft[index]
    # Find cells which should be fully open
    index = (hfac_below_draft < hfac_limit)*(hfac_below_draft >= hfac_limit/2)
    model_draft[index] = level_below_draft[index] + dz_at_draft[index]
    # Update the intermediate variables (as the layers might have changed now), and also get dz of the layer below the draft
    level_below_draft, dz_at_draft, dz_below_draft = draft_level_vars(model_draft, dz, z_edges)
    
    # Figure out the shallowest acceptable depth of each point and its neighbours, based on the ice shelf draft. We want 2 (at least partially) open cells.
    # The first open cell is between the draft and the z-level below it.
    bathy_limit = level_below_draft
    # The second open cell digs into the level below that by the minimum amount (based on hFac constraints).
    hfac_limit = np.maximum(hFacMin, np.minimum(hFacMinDr/dz_below_draft, 1))
    bathy_limit -= dz_below_draft*hfac_limit
    # Get bathy_limit at each point's 4 neighbours
    bathy_limit_w, bathy_limit_e, bathy_limit_s, bathy_limit_n = neighbours(bathy_limit)[:4]
    # Make a copy of the original bathymetry for comparison later
    bathy_orig = np.copy(bathy)
    
    print 'Digging based on local ice shelf draft'
    bathy = dig_one_direction(bathy, bathy_limit)
    bathy_limit_neighbours = [bathy_limit_w, bathy_limit_e, bathy_limit_s, bathy_limit_n]
    loc_strings = ['west', 'east', 'south', 'north']
    for i in range(len(loc_strings)):
        print 'Digging based on ice shelf draft to ' + loc_strings[i]
        bathy = dig_one_direction(bathy, bathy_limit_neighbours[i])
    
    # Plot how the results have changed
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, bathy), title='Bathymetry (m) after digging')
    plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, bathy-bathy_orig), title='Change in bathymetry (m)\ndue to digging')

    if hFacMinDr >= dz[0]:
        print 'Zapping ice shelf drafts which are too shallow'
        # Find any points which are less than half the depth of the surface layer
        index = (draft != 0)*(abs(draft) < 0.5*dz[0])
        print '...' + str(np.count_nonzero(index)) + ' cells to zap'
        draft[index] = 0
        imask[index] = 0
        # Plot how the results have changed
        plot_tmp_domain(lon_2d, lat_2d, np.ma.masked_where(omask==0, index.astype(int)), title='Ice shelf points which were zapped')

    # Copy the NetCDF file to a new name
    shutil.copyfile(nc_in, nc_out)
    # Update the variables
    update_nc_grid(nc_out, bathy, draft, omask, imask)

    print "The updated grid has been written into " + nc_out + ". Take a look and make sure everything looks okay. If you're happy, run write_topo_files to generate the binary files for MITgcm input."

    
# Write the bathymetry and ice shelf draft fields, currently stored in a NetCDF file, into binary files to be read by MITgcm.
def write_topo_files (nc_grid, bathy_file, draft_file, prec=64):

    bathy = read_netcdf(nc_grid, 'bathy')
    draft = read_netcdf(nc_grid, 'draft')
    write_binary(bathy, bathy_file, prec=prec)
    write_binary(draft, draft_file, prec=prec)
    print 'Files written successfully. Now go try them out! Make sure you update all the necessary variables in data, data.shelfice, SIZE.h, job scripts, etc.'


# Helper function to check that neighbouring ocean cells have at least 2 open faces in the given direction.
def check_one_direction (open_cells, open_cells_beside, loc_string, problem):

    # Open faces are equivalent to adjacent open cells
    open_face = open_cells.astype(int)*open_cells_beside.astype(int)
    # Check pairs of points which are both non-land (at least 1 open cell in the water column), to make sure they all have at least 2 open faces between them
    num_pinched = np.count_nonzero((np.sum(open_cells, axis=0) != 0)*(np.sum(open_cells_beside, axis=0) != 0)*(np.sum(open_face, axis=0)<2))
    if num_pinched > 0:
        problem = True
        print 'Problem!! There are ' + str(num_pinched) + ' locations with less than 2 open faces on the ' + loc_string + ' side.'
    return problem
        

# Given a NetCDF grid file produced by MITgcm (and glued together from all the per-processor files), make sure that the digging worked and that the 2 open cell rule holds.
def check_final_grid (grid_path):

    grid = Grid(grid_path)
    problem = False

    # Check that every water column has at least 2 open cells (partial cells count)
    open_cells = np.ceil(grid.hfac)
    num_pinched = np.count_nonzero(np.sum(open_cells, axis=0)==1)
    if num_pinched > 0:
        problem = True
        print 'Problem!! There are ' + str(num_pinched) + ' locations with only one open cell in the water column.'

    # Check that neighbouring ocean cells have at least 2 open faces between
    open_cells_w, open_cells_e, open_cells_s, open_cells_n = neighbours(open_cells)[:4]
    open_cells_neighbours = [open_cells_w, open_cells_e, open_cells_s, open_cells_n]
    loc_strings = ['western', 'eastern', 'southern', 'northern']
    for i in range(len(loc_strings)):
        problem = check_one_direction(open_cells, open_cells_neighbours[i], loc_strings[i], problem)

    if problem:
        print 'Something went wrong with the digging. Are you sure that your values of hFacMin and hFacMinDr are correct? Are you working with a version of MITgcm that calculates Ro_sfc and R_low differently?'
    else:
        print 'Everything looks good!'


def calc_load_anomaly (grid_path, init_t_path, init_s_path, mitgcm_code_path, out_file, option='constant', rhoConst=1035, prec=64):

    print 'Things to check in your "data" namelist:'
    print "eosType='MDJWF'"
    print 'rhoConst='+str(rhoConst)
    print 'readBinaryPrec=64'

    g = 9.81  # gravity (m/s^2)
    if option == 'gradient':
        drho_dz = 4.78e-3  # vertical density gradient of water displaced by ice (kg/m^4, assumed constant, depths are positive)
    elif option == 'icemass':
        rho_ice = 917  # density of ice shelf (kg/m^3, assumed constant)

    # Build the grid
    grid = Grid(grid_path)
    draft = abs(grid.zice)

    if option in ['gradient', 'constant']:
        
        # Load the MDJWF density function
        mitgcm_utils_path = real_dir(mitgcm_code_path) + 'utils/python/MITgcmutils/MITgcmutils/'
        if not os.path.isfile(mitgcm_utils_path+'mdjwf.py'):
            print 'Error (calc_load_anomaly): ' + mitgcm_utils_path + ' does not contain the script mdjwf.py.'
            sys.exit()    
        sys.path.insert(0, mitgcm_utils_path)
        from mdjwf import densmdjwf
        
        # Calculate the initial potential density in the first layer below the ice shelves
        temp_top = select_top(mask_3d(read_binary(init_t_path, grid, 'xyz', prec=prec), grid))
        salt_top = select_top(mask_3d(read_binary(init_s_path, grid, 'xyz', prec=prec), grid))
        # Fill the land mask with zeros
        temp_top[temp_top.mask] = 0
        salt_top[salt_top.mask] = 0
        rho_top = densmdjwf(salt_top, temp_top, np.zeros(temp_top.shape))

    if option == 'gradient':

        # Find the (positive) depth of this density: halfway between the ice shelf base (as seen by the model after hFac corrections) and the layer below it
        z_top = 0.5*(draft + abs(select_top(mask_3d(z_to_xyz(grid.z_edges[1:], grid), grid))))
        z_top[z_top.mask] = 0

    # Now we can calculate the analytical solution to the integral
    if option == 'constant':
        pload = g*draft*(rho_top - rhoConst)
    elif option == 'gradient':
        pload = g*draft*(rho_top + drho_dz*(0.5*draft - z_top) - rhoConst)
    elif option == 'icemass':
        pload = g*draft*(rho_ice - rhoConst)

    # Write to file
    write_binary(pload, out_file, prec=prec)

    
