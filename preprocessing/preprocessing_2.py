# preprocessing_2.py
import sys
sys.path.insert(0, "/soge-home/projects/crop_yield/et_comparison/")

from pathlib import Path
from preprocessing.holaps_cleaner import HolapsCleaner
from preprocessing.modis_cleaner import ModisCleaner
from preprocessing.gleam_cleaner import GleamCleaner
from preprocessing.chirps_cleaner import ChirpsCleaner
from preprocessing.grun_cleaner import GrunCleaner

from preprocessing.esa_cci_lc_cleaner import EsaCciCleaner
from preprocessing.utils import merge_data_arrays, save_netcdf, get_all_valid

# if __name__ == "__main__":
import matplotlib.pyplot as plt
%matplotlib
    BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
    gr = GrunCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff',
        data_filename='EA_GRUN_ref_masked.nc',
        data_variable='grun_runoff'
    )
    gr.preprocess2()
    # gr.preprocess2()

    h = HolapsCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff',
        data_path="holaps_EA_clean.nc"
    )
    h.preprocess2()

    g = GleamCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff'
    )
    g.preprocess2()

    m = ModisCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff'
    )
    m.preprocess2()


    c = ChirpsCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff'
    )
    c.preprocess2()

    ds = merge_data_arrays(h.clean_data, g.clean_data, m.clean_data, c.clean_data, gr.clean_data)
    ds = get_all_valid(ds, ds.holaps_evapotranspiration, ds.modis_evapotranspiration, ds.gleam_evapotranspiration, ds.chirps_precipitation)
    assert (ds.chirps_precipitation.isnull() == ds.holaps_evapotranspiration.isnull()).mean() == 1., "the missing (nan) values should be exactly the same in all products!"

    # drop yemen from the data
    from engineering.mask_using_shapefile import add_shape_coord_from_data_array
    country_shp_path = BASE_DATA_DIR / "country_shp" / "ne_50m_admin_0_countries.shp"
    ds = add_shape_coord_from_data_array(ds, country_shp_path, coord_name="countries")
    ds = ds.where(ds.countries != 2)

    output_ds_path='/soge-home/projects/crop_yield/EGU_compare/processed_ds2.nc'
    save_netcdf(ds, output_ds_path, force=True)
