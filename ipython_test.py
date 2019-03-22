"""ipython_test.py"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding
from scipy.stats import pearsonr

from pathlib import Path
import itertools
import warnings
import os

import cartopy
from shapely import geometry

import sys
sys.path.insert(0, "/soge-home/projects/crop_yield/et_comparison/")

%matplotlib
%load_ext autoreload

from preprocessing.utils import drop_nans_and_flatten
# from preprocessing.holaps_cleaner import HolapsCleaner
# from preprocessing.gleam_cleaner import GleamCleaner
# from preprocessing.modis_cleaner import ModisCleaner

BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
BASE_DIR = Path('/soge-home/projects/crop_yield/et_comparison')

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

h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]

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
