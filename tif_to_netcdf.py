"""
tif_to_netcdf.py
https://gis.stackexchange.com/questions/199570/how-to-prepare-tiffs-to-create-a-netcdf-file
"""
import xarray as xr
from affine import Affine
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ------------------------------------------------------------------------------
# read in the data
# ------------------------------------------------------------------------------
ds = xr.open_rasterio('MOD16A3_ET_2005.tif')

# ------------------------------------------------------------------------------
# http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html
# ------------------------------------------------------------------------------
transform = Affine.from_gdal(*ds.attrs['transform'])
nx, ny = ds.sizes['x'], ds.sizes['y']
x, y = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5) * transform

# ------------------------------------------------------------------------------
# http://xarray.pydata.org/en/stable/auto_gallery/plot_rasterio.html
# ------------------------------------------------------------------------------
from rasterio.warp import transform

# Compute the lon/lat coordinates with rasterio.warp.transform
ny, nx = len(ds['y']), len(ds['x'])
x, y = np.meshgrid(ds['x'], ds['y'])

# Rasterio works with 1D arrays
lon, lat = transform(ds.crs, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())

# write back to dataset
lon = np.asarray(lon).reshape((ny, nx))
lat = np.asarray(lat).reshape((ny, nx))
ds.coords['lon'] = (('y', 'x'), lon)
ds.coords['lat'] = (('y', 'x'), lat)

# Compute a greyscale out of the rgb image
greyscale = ds.mean(dim='band')

# Plot on a map
# ax = plt.subplot(projection=ccrs.PlateCarree())
ax = plt.subplot()
greyscale.plot(ax=ax, x='lon', y='lat', #transform=ccrs.PlateCarree(),
               cmap='Greys_r', add_colorbar=False)
# ax.coastlines('10km', color='r')
plt.show()
