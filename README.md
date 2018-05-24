# mitgcm_python
Python tools I built for MITgcm pre- and post-processing. They are designed for my Weddell Sea configuration, but should be general enough that other people will hopefully find them useful.

Disclaimer: I wrote this for an ocean application using a polar spherical (regular lat-lon) grid. It might not work for other sorts of grids, or atmospheric applications.

Second disclaimer: This script assumes the Xp1 and Yp1 axes are the same size as X and Y in all NetCDF files. This is the case if you run with MDS output and convert to NetCDF with xmitgcm, or if you run with MNC output and glue with gluemnc. It is NOT the case if you run with MNC output and glue with gluemncbig.
