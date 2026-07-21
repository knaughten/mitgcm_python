import xarray as xr
import numpy as np
import os
from nemo_python.utils import add_months

# Loop over all files
for fname in os.listdir('./'):

    if fname.endswith('.nc') and ('OceanA-hind' in fname or 'OceanW-hind' in fname):
        print('Processing '+fname)
    else:
        # Not an output file
        continue

    if fname.startswith('Oce3d_'):
        # Fix a couple of variables that went wrong (since corrected in main code)
        ds = xr.open_dataset(fname)
        if np.count_nonzero(ds['wfoatrli'].notnull()) != 0:
            print('...filling wfoatrli with NaNs')
            # We didn't actually have all the freshwater flux fields that we needed to calculate this
            ds['wfoatrli'] = ds['wfoatrli'].where(False)
            print('...fixing units for wfosicor')
            missval = ds['ficeshelf'].max()
            if 'OceanA-hind' in fname:
                # Swap sign on sea ice fluxes
                factor = -1
            elif 'OceanW-hind' in fname:
                # Multiply by density of freshwater (m/s -> kg/m^2/s)
                factor = 1e3
            ds['wfosicor'] = xr.where(ds['wfosicor']==missval, missval, factor*ds['wfosicor'])
            print('...overwriting file')
            ds.to_netcdf(fname+'.tmp')
            ds.close()
            os.rename(fname+'.tmp', fname)
        else:
            # If wfoatrli is already NaN everywhere, this script was run before (or the file was created using the fixed code)
            print('...wfoatrli and wfosicor corrected previously')
            ds.close()

    # Fix naming convention: MITgcm labels timestamps with the first day after the time period (eg 197902-198001), we want the first day of the time period (197901-197912)    
    # Extract date strings from filename
    i = fname.rfind('_')+1
    date_strings = [fname[i:i+6], fname[i+7:i+13]]
    # Check against first time index in file
    ds = xr.open_dataset(fname)
    date_string_check = str(ds['time'][0].dt.year.item())+str(ds['time'][0].dt.month.item()).zfill(2)
    ds.close()
    if date_strings[0] == date_string_check:
        # Need to fix it
        date_strings_new = []
        for date in date_strings:
            # Subtract one month
            year, month = add_months(int(date[:4]), int(date[4:]), -1)
            date_strings_new.append(str(year)+str(month).zfill(2))
        fname_new = fname[:i]+date_strings_new[0]+'-'+date_strings_new[1]+'.nc'
        print('...renaming to '+fname_new)
        os.rename(fname, fname_new)
    else:
        # This filename was already fixed before
        print('...date strings ok, file starts at '+date_string_check)
    



