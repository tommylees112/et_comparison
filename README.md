<!-- README.md -->
Scripts to allow comparison of MOD16, GLEAM and HOLAPS Evaporation from 2001-2015.
https://www.ntsg.umt.edu/project/modis/mod16.php

# TODO:
- [x] Reproject HOLAPS to geographic lat lon
- [-] Convert GLEAM ET from mm/day -> mm/month
- [x] OR convert MODIS ET -> mm/day
- [x] Copy the HOLAPS mask to other datasets
- [x] CLEAN the exploration code into more useful set of functions (too much copy+paste ...)
- [x] New structure for the code
- [ ] Do P - E calculations (spatial, temporal, subsets)
```
run.py
engineer
    - preprocessing.py [BaseCleaner > HolapsCleaner,ModisCleaner,GleamCleaner]
    - engineer.py
    - plotting.py
```

```python
"""
Problems with the data:
----------------------
[x] Different units:
    holaps = W m-2
    modis = mm/m
    gleam = ???

    Convert: W m-2 to mm of water
    (https://www.researchgate.net/post/How_to_convert_30minute_evapotranspiration_in_watts_to_millimeters)
    1 Watt /m2 = 0.0864 MJ /m2 /day

[x] Different variables:
    holaps: mean monthly latent heat flux
    modis: monthly mean of daily evapotranspiration
    gleam = ???

[x] Timesteps:
     modis starts in February 2001 for some reason

[x] Different masks:
    HOLAPS masks out the water areas
    GLEAM masks out some of the water areas
    MODIS has no mask for the water areas

[x] Different projections
    sincrs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
    llcrs = "+proj=longlat +ellps=WGS84 +datum=WGS84"

[x] Different Resolutions
    GLEAM = lat: 81, lon: 77
    MODIS = longitude: 231, latitude: 242
    HOLAPS = lat: 414, lon: 454

[x] Different Colorbars
"""
```
