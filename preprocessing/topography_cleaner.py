# topography_cleaner.py

import xarray as xr
import numpy as np
from preprocessing.utils import convert_to_same_grid, select_east_africa, get_holaps_mask

def read_tif(data_dir):
    ds = xr.open_rasterio(data_dir)
    return ds

# open the tif file and convert 'band' into 'elevation'
topo = xr.open_rasterio('/soge-home/projects/crop_yield/topography/ETOPO1_Ice_g.tif')
topo = topo.drop('band')
topo = topo.sel(band=0)
topo.name = "elevation"
# rename the x,y dimensions into lon,lat
topo = topo.rename({'y':'lat','x':'lon'})

# create the global file saved as netCDF object
if not os.path.isfile('/soge-home/projects/crop_yield/EGU_compare/global_topography.nc'):
    topo.to_netcdf('/soge-home/projects/crop_yield/EGU_compare/global_topography.nc')

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



if not isinstance(topo, xr.Dataset):
    topo = topo.to_dataset(name='elevation')


# Bin dataset by elevation
topo_bins, intervals = bin_dataset(ds=topo, group_var='elevation', n_bins=10)

# repeat for 60 timesteps (TO BE USED AS `ds` mask)
topo_bins = xr.concat([topo_bins for _ in range(len(ds_valid.time))])
topo_bins = topo_bins.rename({'concat_dims':'time'})
topo_bins['time'] = ds.time
filepaths = [BASE_DATA_DIR / 'intervals_topo1.pickle', BASE_DATA_DIR / 'topo_bins1.pickle']
vars = [intervals, topo_bins]
pickle_files(filepaths, vars)
topo_bins.to_netcdf(BASE_DATA_DIR / 'topo_bins1.nc')
