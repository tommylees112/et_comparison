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

if __name__ == "__main__":
BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
gr = GrunCleaner(
    base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
    reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
    reference_ds_variable='Runoff',
    data_filename='EA_GRUN_ref_masked.nc'
)
gr.preprocess2()
# gr.preprocess2()

h = HolapsCleaner(
    base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
    reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
    reference_ds_variable='grun_runoff',
    data_path="holaps_EA_clean.nc"
)
h.preprocess()
    # h.preprocess2()

    g = GleamCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff'
    )
    g.preprocess()
    # g.preprocess2()

    m = ModisCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff'
    )
    m.preprocess()
    # m.preprocess2()

    c = ChirpsCleaner(
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/EA_GRUN_ref_masked.nc"),
        reference_ds_variable='grun_runoff'
    )
    c.preprocess()
    # c.preprocess2()
