# 2021-03 : Initial code [N. Jourdain, IGE-CNRS]
#====================================================================================

import numpy as np
import xarray as xr
import gsw
from pyproj import Proj
from .def_grids import grid_bounds_oce
from datetime import datetime

#====================================================================================
def load_oce_mod_mitgcm(files_T='MITgcm_all.nc',\
                        files_S='dummy',\
                        files_U='dummy',\
                        files_V='dummy',\
                        files_I='dummy',\
                        files_SRF='dummy',\
                        files_M='dummy',\
                        rho0=1026.0, teos10=False, region='Amundsen', parallel=False, projection=None ):
   """ Read MITgcm outputs and define an xarray dataset containing 
       all variables required in MISOMIP2. It automatically detects
       whether coordinates are stereographic or lon-lat.

       Input:

       files_T: file or list of files containing the temperature and related variables [default='MITgcm_all.nc']
       files_S: file or list of files containing the salinity variable [default=files_T]
       files_U: file or list of files containing the x-velocity and related variables [default=files_T]
       files_V: file or list of files containing the y-velocity and related variables [default=files_T]
       files_I: file or list of files containing the sea-ice variables [default=files_T]
       files_SRF: file or list of files containing the surface fluxes variables [default=files_T]
       files_M: file or list of files containing grid/mesh variables [default=files_T]

       rho0: volumic mass of seawater used in ocean model

       teos10=False -> assumes the nemo outputs are in potential temperature & practical salinity (EOS80)

             =True  -> assumes the nemo outputs are in CT and AS and convert to PT and PS
        
       parallel: If True, the open and preprocess steps of this function will be performed in parallel

       projection: 'lat_lon', 'polar_stereo', or None (default; code will guess)

       Output:
          xarray dataset of coordinates ("time", "z", "sxy") (sxy= one-dimensionalized horizontal space)

       Example1:
          ds = load_oce_mod_mitgcm()

       Example2:
          dir= 'datadir/model/'
          ff = [ dir+'MITgcm_y2009.nc', dir+'MITgcm_y2010.nc', dir+'MITgcm_y2011.nc' ]
          ds = load_oce_mod_mitgcm(files_T=ff, rho0=1028.0, region='Weddell')

   """

   startTime = datetime.now()

   # if only files_T are specified, use it for all variables:
   if ( files_S == 'dummy' ):
      files_S = files_T
   if ( files_U == 'dummy' ):
      files_U = files_T
   if ( files_V == 'dummy' ):
      files_V = files_T
   if ( files_I == 'dummy' ):
      files_I = files_T
   if ( files_SRF == 'dummy' ):
      files_SRF = files_T
   if ( files_M == 'dummy' ):
      files_M = files_T

   ncT = xr.open_mfdataset(files_T, decode_coords=False, parallel=parallel)
   ncS = xr.open_mfdataset(files_S, decode_coords=False, parallel=parallel)
   ncU = xr.open_mfdataset(files_U, decode_coords=False, parallel=parallel)
   ncV = xr.open_mfdataset(files_V, decode_coords=False, parallel=parallel)
   ncI = xr.open_mfdataset(files_I, decode_coords=False, parallel=parallel)
   ncSRF = xr.open_mfdataset(files_SRF, decode_coords=False, parallel=parallel)
   ncM = xr.open_mfdataset(files_M, decode_coords=False, parallel=parallel)

   mtime = ncT.time.shape[0]

   if projection is None:
      if ( ( ncM.XC.min() < -180.1 ) | ( ncM.XC.max() > 360.1 ) ):
         print('    !!! Assuming that (XC,YC) are stereographic coordinates (EPSG:3031) !!!')
         projection = 'polar_stereo'
      else:
         print('    !!! Assuming that (XC,YC) are (longitude,latitude) !!!')
         projection = 'lat_lon'
         

   # longitude & latitude on U, V, T grids
   if projection == 'polar_stereo':
      p = Proj('+init=EPSG:3031')
      XC2d, YC2d = np.meshgrid( ncM.XC.values, ncM.YC.values )
      XG2d, YG2d = np.meshgrid( ncM.XG.values, ncM.YG.values )
      lons, lats = p(XC2d, YC2d, inverse=True)
      lonT = xr.DataArray( lons, dims=['YC', 'XC'] )
      latT = xr.DataArray( lats, dims=['YC', 'XC'] )
      lons, lats = p(XG2d, YC2d, inverse=True)
      lonU = xr.DataArray( lons, dims=['YC', 'XG'] )
      latU = xr.DataArray( lats, dims=['YC', 'XG'] )
      lons, lats = p(XC2d, YG2d, inverse=True)
      lonV = xr.DataArray( lons, dims=['YG', 'XC'] )
      latV = xr.DataArray( lats, dims=['YG', 'XC'] )
   elif projection == 'lat_lon':      
      lonT = ncM.XC
      latT = ncM.YC
      lonU = ncM.XG
      latU = ncM.YC
      lonV = ncM.XC
      latV = ncM.YG
      # Put longitude in the range (-180, 180)
      def fix_lon_range(lon):
         lon = xr.where(lon >= 180, lon-360, lon)
         lon = xr.where(lon < -180, lon+360, lon)
         return lon
      lonT = fix_lon_range(lonT)
      lonU = fix_lon_range(lonU)
      lonV = fix_lon_range(lonV)
      if 'YC' not in lonT.dims:
         # 1D arrays: broadcast to 2D
         latT, lonT = xr.broadcast(latT, lonT)
         latU, lonU = xr.broadcast(latU, lonU)
         latV, lonV = xr.broadcast(latV, lonV)
   else:
      raise Exception('Invalid projection '+projection)

   # save original domain boundaries:
   domain_minlat = latT.min().values
   domain_maxlat = latT.max().values
   domain_minlon = lonT.min().values
   domain_maxlon = lonT.max().values

   # grid mesh widths along x and y at C/T, U and V points [m]:
   dxT = xr.DataArray( 0.500000000 * (ncM.dxC.values+ncM.dxC.shift(XG=1).values), dims=['YC', 'XC'] )
   dyT = xr.DataArray( 0.500000000 * (ncM.dyC.values+ncM.dyC.shift(YG=1).values), dims=['YC', 'XC'] )

   dlatTdx = 0.500000000 * ( latT.shift(XC=-1) - latT.shift(XC=1) )
   dlatUdx = 0.500000000 * ( latU.shift(XG=-1) - latU.shift(XG=1) )
   dlatVdx = 0.500000000 * ( latV.shift(XC=-1) - latV.shift(XC=1) )

   dlonTdx = 0.500000000 * ( lonT.shift(XC=-1) - lonT.shift(XC=1) ) 
   dlonUdx = 0.500000000 * ( lonU.shift(XG=-1) - lonU.shift(XG=1) ) 
   dlonVdx = 0.500000000 * ( lonV.shift(XC=-1) - lonV.shift(XC=1) )

   # local C/T, U, V grid rotation angle compared to the (zonal,meridional) direction [rad]
   thetaT = np.arctan2( dlatTdx, dlonTdx*np.cos(latT*np.pi/180.) )
   thetaU = np.arctan2( dlatUdx, dlonUdx*np.cos(latU*np.pi/180.) )
   thetaV = np.arctan2( dlatVdx, dlonVdx*np.cos(latV*np.pi/180.) )
   print('    Minimum local grid angle in degrees w.r.t. (zonal,meridional):',thetaU.min().values*180./np.pi)
   print('    Maximum local grid angle in degrees w.r.t. (zonal,meridional):',thetaU.max().values*180./np.pi)

   # depth of U, V, C/T grids (neglecting the effects of partial steps in the interpolation) [m, positive in the ocean]
   depTUV=ncM.Z*(-1)

   [mz, my, mx] = ncM.hFacC.shape

   # Domain mask (ones with a halo of nans), used not to interpolate beyond the original domain:
   halonan = np.ones((my,mx))
   halonan[0,:] = np.nan ; halonan[-1,:] = np.nan
   halonan[:,0] = np.nan ; halonan[:,-1] = np.nan
   DOMMSKT = xr.DataArray( halonan, dims=['YC', 'XC'] )
   DOMMSKU = xr.DataArray( halonan, dims=['YC', 'XG'] )
   DOMMSKV = xr.DataArray( halonan, dims=['YG', 'XC'] )
   del halonan

   # Ocean fraction at each level on U, V, C/T grids:
   LEVOFT = ncM.hFacC*100.0 # (=100 if ocean, =0 elsewhere)
   LEVOFU = ncM.hFacW*100.0
   LEVOFV = ncM.hFacS*100.0

   # ice-shelf fraction:
   SFTFLF = LEVOFT.max('Z').where( (LEVOFT.max('Z')>1.0) & (LEVOFT.isel(Z=0)<1.0), 0.0 )

   # Bathymetry (including under ice shelves) [m, positive]
   if ( "Depthxx" in ncM.data_vars ):
     DEPTHO = ncM.Depthxx
   else:
     dz=xr.DataArray( ncM.Zl.values-ncM.Zu.values, dims=['Z'] )
     tsp=dz*ncM.hFacC
     trq = tsp.cumsum('Z')
     newhFacC = ncM.hFacC.where( trq > 0.01, 1.e0 ) # =1 in ice-shelf and ocean cells =0 in bedrock cells
     DEPTHO = dz.dot(newhFacC)

   # Depth of ice shelf draft [m]:
   if ( "isfdraft" in ncM.data_vars ):
     DEPFLF = ncM.isfdraft
   elif ( "ice_draft" in ncM.data_vars ):
     DEPFLF = ncM.ice_draft
   elif ( "ice_shelf_draft" in ncM.data_vars ):
     DEPFLF = ncM.ice_shelf_draft
   elif ( "isfdraft" in ncT.data_vars ):
     DEPFLF = ncT.isfdraft
   elif ( "ice_draft" in ncT.data_vars ):
     DEPFLF = ncT.ice_draft
   elif ( "ice_shelf_draft" in ncT.data_vars ):
     DEPFLF = ncT.ice_shelf_draft
   else:
     dz = xr.DataArray( ncM.Zl.values-ncM.Zu.values, dims=['Z'] )
     DEPFLF = DEPTHO - dz.dot(ncM.hFacC)

   # ocean temperature [degC]
   isTT=True
   if ( "toce" in ncT.data_vars ):
     TT = ncT.toce
   elif ( "thetao" in ncT.data_vars ):
     TT = ncT.thetao
   elif ( "THETA" in ncT.data_vars ):
     TT = ncT.THETA
   elif ( "T" in ncT.data_vars ):
     TT = ncT.T
   else:
     isTT=False
     print('    WARNING :   No data found for TT  -->  filled with NaNs')
     TT = xr.DataArray( np.zeros((mtime,mz,my,mx))*np.nan, dims=['time', 'Z', 'YC', 'XC'] )

   # ocean salinity [1.e-3]
   isSS=True
   if ( "soce" in ncS.data_vars ):
     SS = ncS.soce
   elif ( "so" in ncS.data_vars ):
     SS = ncS.so
   elif ( "SALT" in ncS.data_vars ):
     SS = ncS.SALT
   elif ( "S" in ncS.data_vars ):
     SS = ncS.S
   else:
     isSS=False
     print('    WARNING :   No data found for SS  -->  filled with NaNs')
     SS = xr.DataArray( np.zeros((mtime,mz,my,mx))*np.nan, dims=['time', 'Z', 'YC', 'XC'] )

   # sea bottom ocean temperature [degC]
   if ( "sbt" in ncT.data_vars ):
     TTB = ncT.sbt
   elif ( "tob" in ncT.data_vars ):
     TTB = ncT.tob
   elif ( isTT ):
     minval=ncM.hFacC.where( ncM.hFacC != 0. ).min().values # minimum non-zero sea fraction (e.g. 0.05)
     trq = ncM.hFacC[::-1,:,:].where( ncM.hFacC[::-1,:,:] >= minval )*0.+1.
     tsp = trq.cumsum('Z',skipna=True)
     trq = tsp.where( tsp==1. )
     tsp = TT[:,::-1,:,:]*trq
     TTB = tsp.sum('Z',skipna=True)
   else:
     print('    WARNING :   No data found for TTB  -->  filled with NaNs')
     TTB = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # sea bottom ocean salinity [1.e-3]
   if ( "sbs" in ncT.data_vars ):
     SSB = ncT.sbs
   elif ( "sob" in ncT.data_vars ):
     SSB = ncT.sob
   elif (isSS ):
     minval=ncM.hFacC.where( ncM.hFacC != 0. ).min().values # minimum non-zero sea fraction (e.g. 0.05)
     trq = ncM.hFacC[::-1,:,:].where( ncM.hFacC[::-1,:,:] >= minval )*0.+1.
     tsp = trq.cumsum('Z',skipna=True)
     trq = tsp.where( tsp==1. )
     tsp = SS[:,::-1,:,:]*trq
     SSB = tsp.sum('Z',skipna=True)
   else:
     print('    WARNING :   No data found for SSB  -->  filled with NaNs')
     SSB = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # ocean potential temperature and practical salinity :
   if (teos10):
     TOB = xr.apply_ufunc(gsw.pt_from_CT, SSB, TTB, dask = 'allowed')
     SOB = xr.apply_ufunc(gsw.SP_from_SA, SSB, DEPTHO, lonT, latT, dask = 'allowed')
     THETAO = xr.apply_ufunc(gsw.pt_from_CT, SS, TT, dask = 'allowed')
     SO = xr.apply_ufunc(gsw.SP_from_SA, SS, DEPTHO, lonT, latT, dask = 'allowed')
   else: 
     TOB = TTB
     SOB = SSB
     THETAO = TT
     SO = SS

   # ocean x-ward velocity [m s-1]
   if ( "uoce" in ncU.data_vars ):
     UX = ncU.uoce
   elif ( "UVEL" in ncU.data_vars ):
     UX = ncU.UVEL
   elif ( "uo" in ncU.data_vars ):
     UX = ncU.uo
   elif ( "U" in ncU.data_vars ):
     UX = ncU.U
   else:
     print('    WARNING :   No data found for UX  -->  filled with NaNs')
     UX = xr.DataArray( np.zeros((mtime,mz,my,mx))*np.nan, dims=['time', 'Z', 'YC', 'XG'] )

   # ocean y-ward velocity [m s-1]
   if ( "voce" in ncV.data_vars ):
     VY = ncV.voce
   elif ( "VVEL" in ncV.data_vars ):
     VY = ncV.VVEL
   elif ( "vo" in ncV.data_vars ):
     VY = ncV.vo
   elif ( "V" in ncV.data_vars ):
     VY = ncV.V
   else:
     print('    WARNING :   No data found for VY  -->  filled with NaNs')
     VY = xr.DataArray( np.zeros((mtime,mz,my,mx))*np.nan, dims=['time', 'Z', 'YG', 'XC'] )

   # surface stress received by the ocean along x [W m-1]
   if ( "utau" in ncU.data_vars ):
     TAUX = ncU.utau
   elif ( "TAUX" in ncU.data_vars ):
     TAUX = ncU.TAUX
   else:
     print('    WARNING :   No data found for TAUX  -->  filled with NaNs')
     TAUX = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XG'] )

   # surface stress received by the ocean along x [W m-1]
   if ( "vtau" in ncV.data_vars ):
     TAUY = ncV.vtau
   elif ( "TAUY" in ncV.data_vars ):
     TAUY = ncV.TAUY
   else:
     print('    WARNING :   No data found for TAUY  -->  filled with NaNs')
     TAUY = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YG', 'XC'] )

   # Sea surface height [m]
   if ( "ETAN" in ncT.data_vars ):
     ZOS = ncT.ETAN
   elif ( "SSH" in ncT.data_vars ):
     ZOS = ncT.SSH
   elif ( "ssh" in ncT.data_vars ):
     ZOS = ncT.ssh
   elif ( "zos" in ncT.data_vars ):
     ZOS = ncT.zos
   else:
     print('    WARNING :   No data found for ZOS  -->  filled with NaNs')
     ZOS = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # mass barotropic streamfunction
   # see Griffies et al. (2016, section H26): d(psi)/dy=-U (U: x-ward mass transport), d(psi)/dx=V (V: yward mass transport)
   if ( "sobarstf" in ncU.data_vars ):
     MSFTBAROT = ncU.sobarstf * rho0
   else:
     print('    WARNING :   No data found for MSFTBAROT  -->  filled with NaNs')
     MSFTBAROT = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # ice shelf melt [kg m-2 s-1, positive for actual melting] :
   if ( "SHIfwFlx" in ncT.data_vars ):
     FICESHELF = ncT.SHIfwFlx*(-1)
   else:
     print('    WARNING :   No data found for FICESHELF  -->  filled with NaNs')
     FICESHELF = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # ice shelf dynamical driving (heat exchange velocity) [m s-1]:
   if ( "SHIgammT" in ncT.data_vars ):
     DYDRFLF = ncT.SHIgammT
   else:
     print('    WARNING :   No data found for DYDRFLF  -->  filled with NaNs')
     DYDRFLF = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # ice shelf thermal driving [degC]:
   if ( "SHIThDr" in ncT.data_vars ):
     THDRFLF = ncT.SHIThDr
   else:
     print('    WARNING :   No data found for THDRFLF  -->  filled with NaNs')
     THDRFLF = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # ice shelf haline driving [0.001]:
   if ( "SHIHaDR" in ncT.data_vars ):
     HADRFLF = ncT.SHIHaDr
   else:
     print('    WARNING :   No data found for HADRFLF  -->  filled with NaNs')
     HADRFLF = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # sea-ice concentration [0-100]
   if ( "siconc" in ncI.data_vars ):
     SICONC = ncI.siconc*100.0
     SICONC = SICONC.where( (~np.isnan(SICONC)) & (~np.isinf(SICONC)), 0.e0 )
   elif ( "SIarea" in ncI.data_vars ):
      SICONC = ncI.SIarea*100.0
      SICONC = SICONC.where( (~np.isnan(SICONC)) & (~np.isinf(SICONC)), 0.e0 )
   else:
     print('    WARNING :   No data found for SICONC  -->  filled with NaNs')
     SICONC = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )   

   # sea-ice volume per area [m]
   if ( "sivolu" in ncI.data_vars ):
     SIVOL = ncI.sivolu
   elif ( "sivol" in ncI.data_vars ):
     SIVOL = ncI.sivol
   elif ( "SIheff" in ncI.data_vars ):
      SIVOL = ncI.SIheff
   else:
     print('    WARNING :   No data found for SIVOL  -->  filled with NaNs')
     SIVOL = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # sea-ice x-ward velocity [m/s]
   if ( "sivelu" in ncI.data_vars ):
     SIUX = ncI.sivelu
   elif ("siu" in ncI.data_vars ):
     SIUX = ncI.siu
   elif ("SIuice" in ncI.data_vars ):
      SIUX = ncI.SIuice
   else:
     print('    WARNING :   No data found for SIUX  -->  filled with NaNs')
     SIUX = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XG'] )

   # sea-ice y-ward velocity [m/s]
   if ( "sivelv" in ncI.data_vars ):
     SIVY = ncI.sivelv
   elif ("siv" in ncI.data_vars ):
     SIVY = ncI.siv
   elif ("SIvice" in ncI.data_vars ):
      SIVY = ncI.SIvice
   else:
     print('    WARNING :   No data found for SIUY  -->  filled with NaNs')
     SIVY = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YG', 'XC'] )

   # Total heat flux received by the ocean surface (including ice-shelf/ocean interface) [W m-2] 
   # see Griffies et al. (2016, section K4-K5) NB: here, including correction if any unlike Griffies (to avoid 2 variables)
   if ( "qt_oce" in ncSRF.data_vars ):
     HFDS = ncSRF.qt_oce
   elif ("oceQnet" in ncSRF.data_vars ):
      HFDS = ncSRF.oceQnet
   else:
     print('    WARNING :   No data found for HFDS  -->  filled with NaNs')
     HFDS = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # Water flux entering the ocean due to sea-ice (melting-freezing) and surface correction (SSS restoring)
   # (= fsitherm + wfocorr in Griffies 2016 section K2) [kg m-2 s-1]      
   if ( "wfocorr" in ncSRF.data_vars ):
     WFOCORR = - ncSRF.wfocorr
   else:
     WFOCORR = xr.DataArray( np.zeros((mtime,my,mx)), dims=['time', 'YC', 'XC'] )
   if ( "fsitherm" in ncSRF.data_vars ):
     WFOSICOR = WFOCORR - ncSRF.fsitherm
   elif ("SIfwmelt" in ncSRF.data_vars and "SIfwfrz" in ncSRF.data_vars):
      WFOSICOR = WFOCORR + ncSRF.SIfwmelt + ncSRF.SIfwfrz
   elif ("SIdHbOCN" in ncSRF.data_vars and "SIdHbATC" in ncSRF.data_vars and "SIdHbATO" in ncSRF.data_vars and "SIdHbFLO" in ncSRF.data_vars):
      WFOSICOR = WFOCORR - (ncSRF.SIdHbOCN + ncSRF.SIdHbATC + ncSRF.SIdHbATO + ncSRF.SIdHbFLO)*1e3
   else:
     print('    WARNING :   No data found for WFOSICOR  -->  filled with NaNs')
     WFOSICOR = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )

   # Water flux entering the ocean due to rainfall, snowfall, condensation - evap, 
   # river runoff, iceberg and ice-shelf melt [kg m-2 s-1]  
   # (= pr+prs+evs+ficeberg+friver+ficeshelf in Griffies 2016, section K2)
   if ( "empmr" in ncSRF.data_vars ):
     WFOATRLI = - ncSRF.empmr + FICESHELF
   else:
     print('    WARNING :   No data found for WFOATRLI  -->  filled with NaNs')
     WFOATRLI = xr.DataArray( np.zeros((mtime,my,mx))*np.nan, dims=['time', 'YC', 'XC'] )
  
   #----------
   # Reduce the size of ocean dataset
 
   [lonmin,lonmax,latmin,latmax] = grid_bounds_oce(region=region)
   wdeg = 1.5 * np.max([dxT.where(latT <= latmax).max(),dyT.where(latT <= latmax).max()]) / 6.37e6 * 180. / np.pi / np.cos(0.5*(latmin+latmax)*np.pi/180.)
   lonmin=lonmin - wdeg # take a bit more for interpolation
   lonmax=lonmax + wdeg
   latmin=latmin - wdeg
   latmax=latmax + wdeg

   condT2d = ( (latT >= latmin) & (latT <= latmax) & (lonT >= lonmin) & (lonT <= lonmax) )

   for ii in np.arange(latT.shape[1]):
      if ( np.sum(condT2d.isel(XC=ii).values) == 0 ):
        imin=ii
      else:
        imin=ii
        break
   for ii in np.arange(latT.shape[1]-1,0,-1):
      if ( np.sum(condT2d.isel(XC=ii).values) == 0 ):
        imax=ii
      else:
        imax=ii
        break
   for jj in np.arange(latT.shape[0]):
      if ( np.sum(condT2d.isel(YC=jj).values) == 0 ):
        jmin=jj
      else:
        jmin=jj
        break
   for jj in np.arange(latT.shape[0]-1,0,-1):
      if ( np.sum(condT2d.isel(YC=jj).values) == 0 ):
        jmax=jj
      else:
        jmax=jj
        break

   print('    Reducing domain size to useful area, i.e.: ',[imin,imax,jmin,jmax])

   #----------
   # Create new xarray dataset including all useful variables:
   # reshaping (x,y) as 1-dimensional (sxy)

   nxy=(jmax-jmin+1)*(imax-imin+1)

   newdepth=depTUV.values
   newdepth[np.argmin(newdepth)]=0.e0 # so that 1st level is taken at the surface without extrapolation

   time_conv=ncT.time.dtype
   if ( time_conv == 'datetime64[ns]' ):
     time_val = ncT.time.values # standard calendar 
   else:
     time_val = ncT.indexes['time'].to_datetimeindex().values # to enable dealing with non-standard calendar (e.g. noleap)

   ds = xr.Dataset(
      {
       "SO":        (["time", "z", "sxy"], np.reshape( SO.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,mz,nxy)) ),
       "THETAO":    (["time", "z", "sxy"], np.reshape( THETAO.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,mz,nxy)) ),
       "UX":        (["time", "z", "sxy"], np.reshape( UX.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,mz,nxy)) ),
       "VY":        (["time", "z", "sxy"], np.reshape( VY.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, (mtime,mz,nxy)) ),
       "TAUX":      (["time", "sxy"], np.reshape( TAUX.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "TAUY":      (["time", "sxy"], np.reshape( TAUY.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "ZOS":       (["time", "sxy"], np.reshape( ZOS.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "TOB":       (["time", "sxy"], np.reshape( TOB.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "SOB":       (["time", "sxy"], np.reshape( SOB.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "FICESHELF": (["time", "sxy"], np.reshape( FICESHELF.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "DYDRFLF":   (["time", "sxy"], np.reshape( DYDRFLF.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "THDRFLF":   (["time", "sxy"], np.reshape( THDRFLF.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "HADRFLF":   (["time", "sxy"], np.reshape( HADRFLF.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "MSFTBAROT": (["time", "sxy"], np.reshape( MSFTBAROT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "HFDS":      (["time", "sxy"], np.reshape( HFDS.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "WFOATRLI":  (["time", "sxy"], np.reshape( WFOATRLI.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "WFOSICOR":  (["time", "sxy"], np.reshape( WFOSICOR.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "SICONC":    (["time", "sxy"], np.reshape( SICONC.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "SIVOL":     (["time", "sxy"], np.reshape( SIVOL.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "SIUX":      (["time", "sxy"], np.reshape( SIUX.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "SIVY":      (["time", "sxy"], np.reshape( SIVY.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, (mtime,nxy)) ),
       "LEVOFT":    (["z", "sxy"], np.reshape( LEVOFT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mz,nxy)) ),
       "LEVOFU":    (["z", "sxy"], np.reshape( LEVOFU.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, (mz,nxy)) ),
       "LEVOFV":    (["z", "sxy"], np.reshape( LEVOFV.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, (mz,nxy)) ),
       "SFTFLF":    (["sxy"], np.reshape( SFTFLF.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "DEPFLF":    (["sxy"], np.reshape( DEPFLF.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "DEPTHO":    (["sxy"], np.reshape( DEPTHO.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "DOMMSKT":   (["sxy"], np.reshape( DOMMSKT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "DOMMSKU":   (["sxy"], np.reshape( DOMMSKU.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "DOMMSKV":   (["sxy"], np.reshape( DOMMSKV.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, nxy) ),
       "lonT":      (["sxy"], np.reshape( lonT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "lonU":      (["sxy"], np.reshape( lonU.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "lonV":      (["sxy"], np.reshape( lonV.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, nxy) ),
       "latT":      (["sxy"], np.reshape( latT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "latU":      (["sxy"], np.reshape( latU.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "latV":      (["sxy"], np.reshape( latV.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, nxy) ),
       "dxT":       (["sxy"], np.reshape( dxT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "dyT":       (["sxy"], np.reshape( dyT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "thetaT":    (["sxy"], np.reshape( thetaT.isel(XC=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "thetaU":    (["sxy"], np.reshape( thetaU.isel(XG=slice(imin,imax+1),YC=slice(jmin,jmax+1)).values, nxy) ),
       "thetaV":    (["sxy"], np.reshape( thetaV.isel(XC=slice(imin,imax+1),YG=slice(jmin,jmax+1)).values, nxy) ),
       "depTUV":    (['z'], depTUV.values)
      },
      coords={
      "time": time_val,
      "z": newdepth
      },
      attrs={
      "original_minlat": domain_minlat,
      "original_maxlat": domain_maxlat,
      "original_minlon": domain_minlon,
      "original_maxlon": domain_maxlon
      },
   )

   print('    Load duration: ',datetime.now() - startTime)

   return ds


