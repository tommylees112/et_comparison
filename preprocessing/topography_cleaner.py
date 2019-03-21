# topography_cleaner.py

import xarray as xr
import numpy as np
from preprocessing.utils import convert_to_same_grid, select_east_africa, get_holaps_mask



topo = xr.open_rasterio('/soge-home/projects/crop_yield/topography/ETOPO1_Ice_g.tif')
topo = topo.drop('band')
topo = topo.sel(band=0)
topo.name = "elevation"

if not os.path.isfile('/soge-home/projects/crop_yield/EGU_compare/global_topography.nc'):
    topo.to_netcdf('/soge-home/projects/crop_yield/EGU_compare/global_topography.nc')

topo = topo.rename({'y':'lat','x':'lon'})
topo = select_east_africa(topo)

if not os.path.isfile('/soge-home/projects/crop_yield/EGU_compare/EA_topography.nc'):
    topo.to_netcdf('/soge-home/projects/crop_yield/EGU_compare/EA_topography.nc')

# convert to same grid as the other data (ds_valid)
topo = convert_to_same_grid(ds, topo, method="bilinear")

# mask the same areas as other data (ds_valid)
mask = get_holaps_mask(ds.holaps_evapotranspiration)
topo = topo.where(~mask)

if not os.path.isfile('/soge-home/projects/crop_yield/EGU_compare/EA_topo_clean.nc'):
    topo.to_netcdf('/soge-home/projects/crop_yield/EGU_compare/EA_topo_clean.nc')
