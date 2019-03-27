"""ipython_test.py"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding
from scipy.stats import pearsonr
from scipy import stats

from pathlib import Path
import itertools
import warnings
import os

import pickle

import cartopy
from shapely import geometry

import sys
sys.path.insert(0, "/soge-home/projects/crop_yield/et_comparison/")

%matplotlib
%load_ext autoreload

from preprocessing.utils import drop_nans_and_flatten
from preprocessing.utils import read_csv_point_data

# from preprocessing.holaps_cleaner import HolapsCleaner
# from preprocessing.gleam_cleaner import GleamCleaner
# from preprocessing.modis_cleaner import ModisCleaner

# import data engineering functions
from engineering.mask_using_shapefile import add_shape_coord_from_data_array
from engineering.regions import regions
from engineering.eng_utils import get_lookup_val, mask_multiple_conditions
from engineering.regions import Region, create_regions, select_bounding_box_xarray
from engineering.eng_utils import get_lookup_val, drop_nans_and_flatten
from engineering.eng_utils import get_unmasked_data
from engineering.eng_utils import bin_dataset, pickle_files
from engineering.eng_utils import load_pickle, create_flattened_dataframe_of_values
from engineering.eng_utils import calculate_monthly_mean, calculate_spatial_mean
from engineering.eng_utils import create_double_year
from engineering.eng_utils import get_non_coord_variables


# import data plotting functions
from plotting.plots import plot_stations_on_region_map
from plotting.plots import plot_seasonal_spatial_means
from plotting.plots import plot_mean_spatial_differences_ET
from plotting.plots import get_variables_for_comparison1, plot_mean_time
from plotting.plots import plot_hexbin_comparisons
from plotting.plots import plot_joint_plot_hex1, hexbin_jointplot_sns
from plotting.plots import plot_xarray_on_map, plot_all_spatial_means
from plotting.plots import plot_seasonal_comparisons_ET_diff
from plotting.plots import plot_marginal_distribution
from plotting.plots import plot_masked_spatial_and_hist
from plotting.plots import plot_geog_location
from plotting.plots import add_points_to_map, add_sub_region_box
from plotting.plots import plot_seasonality
from plotting.plots import plot_normalised_seasonality
from plotting.plots import get_variables_for_comparison1, plot_mean_time, plot_seasonal_comparisons_ET_diff

#
from plotting.plot_utils import get_colors


BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
BASE_DIR = Path('/soge-home/projects/crop_yield/et_comparison')
BASE_FIG_DIR =Path('/soge-home/projects/crop_yield/et_comparison/figs/meeting2')

# clean data
ds = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc")
h = ds.holaps_evapotranspiration.copy()
m = ds.modis_evapotranspiration.copy()
g = ds.gleam_evapotranspiration.copy()

lc = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/esa_lc_EA_clean.nc")

df = ds.to_dataframe()
seasons = ds.groupby('time.season').mean(dim='time')

mthly_mean = ds.groupby('time.month').mean(dim='time')
seasonality = mthly_mean.mean(dim=['lat','lon'])

nonan_h = drop_nans_and_flatten(h)
nonan_g = drop_nans_and_flatten(g)
nonan_m = drop_nans_and_flatten(m)


datasets = ['holaps', 'gleam', 'modis']
evap_das = [f"{ds}_evapotranspiration" for ds in datasets]
colors = [h_col, m_col, g_col, c_col] = get_colors()


lonmin=32.6
lonmax=51.8
latmin=-5.0
latmax=15.2


def create_flattened_dataframe_of_values(h,g,m):
    """ """
    h_ = drop_nans_and_flatten(h)
    g_ = drop_nans_and_flatten(g)
    m_ = drop_nans_and_flatten(m)
    df = pd.DataFrame(dict(
            holaps=h_,
            gleam=g_,
            modis=m_
        ))
    return df


dist_df = create_flattened_dataframe_of_values(h,g,m)

topo = xr.open_dataset('/soge-home/projects/crop_yield/EGU_compare/EA_topo_clean_ds.nc')

all_region = regions[0]
highlands = regions[1]
lake_victoria_region = regions[2]
