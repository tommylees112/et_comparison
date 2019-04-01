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
    gr = GrunCleaner()
    gr.preprocess()

    
