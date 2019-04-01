# clean_grun.py

import xarray as xr
from pathlib import Path

BASE_DATA_DIR = Path("/soge-home/projects/crop_yield/EGU_compare")
ds = xr.open_dataset(BASE_DATA_DIR/'EA_GRUN.nc')
ds = ds.resample(time='M').first()

ds = ds.sel(time=slice('2001-01-31','2006-01-30'))

ds.to_netcdf(BASE_DATA_DIR/'EA_GRUN_ref.nc')

#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################

# ==============================================================================
# HISTORY OF IPYTHON WORK TO CLEAN THE DATA
# ==============================================================================
from pathlib import Path
from preprocessing.holaps_cleaner import HolapsCleaner
from preprocessing.modis_cleaner import ModisCleaner
from preprocessing.gleam_cleaner import GleamCleaner
from preprocessing.chirps_cleaner import ChirpsCleaner
from preprocessing.grun_cleaner import GrunCleaner

from preprocessing.esa_cci_lc_cleaner import EsaCciCleaner
from preprocessing.utils import merge_data_arrays, save_netcdf, get_all_valid
BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
gr = GrunCleaner(
    base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
    reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref.nc"),
    reference_ds_variable='Runoff',
    data_filename='EA_GRUN_ref.nc'
)
gr.preprocess2()
# gr.preprocess2()
h = HolapsCleaner(
    base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
    reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/grun_EA_clean2.nc"),
    reference_ds_variable='grun_runoff',
    data_path="holaps_EA_clean.nc"
)
h
h.raw_data
h.regrid_to_reference()
h.clean_data.regrid_to_reference()
h
h.data
h.clean_data
h.clean_data = h.clean_data.holaps_evapotranspiration
h.clean_data.regrid_to_reference()
h.regrid_to_reference()
h.clean_data
h.clean_data = h.raw_data.holaps_evapotranspiration
h.clean_data
h.regrid_to_reference
h.regrid_to_reference(method='bilinear')
h.clean_data
%matplotlib; h.clean_data.plot()
%matplotlib
h.clean_data.plot()
h.clean_data.
h.clean_data.mean(dim='time').plot()
a
from engineering.mask_using_shapefile import add_shape_coord_from_data_array
country_shp_path = BASE_DATA_DIR / "country_shp" / "ne_50m_admin_0_countries.shp"
ds = holaps.clean_data
ds = h.clean_data
ds
country_shp_path = BASE_DATA_DIR / "country_shp" / "ne_50m_admin_0_countries.shp"
add_shape_coord_from_data_array(ds, country_shp_path, coord_name="countries")
ds = add_shape_coord_from_data_array(ds, country_shp_path, coord_name="countries")
ds = ds.where(ds.countries != 2)
ds.mean(dim='time').plot()
ds.mean(dim='time').plot()
d = xr.open_dataset(BASE_DATA_DIR/'EA_GRUN_ref.nc')
import xarray as xr
d = xr.open_dataset(BASE_DATA_DIR/'EA_GRUN_ref.nc')
d
ds.isnull()
ds.isnull().drop(countries)
ds.isnull().drop("countries")
mask = ds.isnull().drop("countries")
d.where(mask).mean(dim='time').plot()
d.where(mask).mean(dim='time')
d.where(mask).mean(dim='time').Runoff
d.where(mask).mean(dim='time').Runoff.plot()
plt.close('all')
import matplotlib.pyplot as plt
plt.close('all')
d.where(mask).mean(dim='time').Runoff.plot()
d.where(~mask).mean(dim='time').Runoff.plot()
d.where(~mask).mean(dim='time').Runoff.plot()
d = d.where(~mask)
d
d = d.rename({'Runoff':'grun_runoff'})
d.to_netcdf(BASE_DATA_DIR/'EA_GRUN_ref_masked.nc')
history
