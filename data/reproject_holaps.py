"""# reproject_holaps.py

https://medium.com/planet-stories/a-gentle-introduction-to-gdal-part-2-map-projections-gdalwarp-e05173bd710a

Sinusoidal:
----------
http://spatialreference.org/ref/sr-org/modis-sinusoidal-3/

WGS84
-----
EPSG:4326
"""
source_proj = "proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
dest_proj = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

cmd = 'gdalwarp -t_srs "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs" -of netCDF holaps_africa.nc holaps_africa_test.nc'

# TODO: outputs a whole load of bands that need to be reconcatenated
# https://stackoverflow.com/a/48198787/9940782
data_dir="holaps_africa_test.nc"
holaps = xr.open_dataset(data_dir)

h = xr.open_dataset('holaps_africa.nc')
h_times = h.time

band_strings = [key for key in holaps.variables.keys() if 'Band' in key]
bands = [holaps[key] for key in band_strings]
bands = [band.rename('LE_Mean') for band in bands]

assert len(h_times) == len(bands), f"The number of bands should match the number of timesteps. n bands: {len(h_times)} n times: {len(bands)}"

holaps = xr.concat(bands, dim=h_times)
#
# test_cmd1 = "cdoremapycon2,r406x404 holaps_africa.nc holaps_africa_test.nc"
# test_cmd1 = "cdo -remapycon,EA_GLEAM_allvars_2001_2015.nc holaps_africa.nc holaps_africa_tst.nc"
