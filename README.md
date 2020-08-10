# mitgcm_python
Python tools I built for MITgcm pre- and post-processing. They are designed for my Weddell Sea configuration, but should be general enough that other people will hopefully find them useful.

Please acknowledge the use of these scripts in any publications which make use of them, as "the mitgcm_python package by Kaitlin Naughten".

You must have numpy installed to use any of these functions. Other packages are required for certain functions: netCDF4, matplotlib, scipy, and pynco.

A few functions also use the python tools distributed with MITgcm. Make sure they are in your `PYTHONPATH`. At the bottom of your `~/.bashrc`, add:

```
export PYTHONPATH=$PYTHONPATH:$ROOTDIR/utils/python/MITgcmutils
```

where `$ROOTDIR` is the path to your copy of the MITgcm source code distribution.

Disclaimer: I wrote this for an ocean application using a polar spherical (regular lat-lon) grid. It might not work for other sorts of grids, or atmospheric applications.

Second disclaimer: This script assumes the Xp1 and Yp1 axes are the same size as X and Y in all NetCDF files. This is the case if you run with MDS output and convert to NetCDF with xmitgcm, or if you run with MNC output and glue with gluemnc. It is NOT the case if you run with MNC output and glue with gluemncbig.
