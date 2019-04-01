# clean_grun.py

import xarray as xr
from pathlib import Path

BASE_DATA_DIR = Path("/soge-home/projects/crop_yield/EGU_compare")
ds = xr.open_dataset(BASE_DATA_DIR/'EA_GRUN.nc')
ds = ds.resample(time='M').first()

ds = ds.sel(time=slice('2001-01-31','2006-01-30'))

ds.to_netcdf(BASE_DATA_DIR/'EA_GRUN_ref.nc')
