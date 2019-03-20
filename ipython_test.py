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


import sys
sys.path.insert(0, "/soge-home/projects/crop_yield/et_comparison/")

%matplotlib
%load_ext autoreload

from preprocessing.utils import drop_nans_and_flatten
# from preprocessing.holaps_cleaner import HolapsCleaner
# from preprocessing.gleam_cleaner import GleamCleaner
# from preprocessing.modis_cleaner import ModisCleaner

# clean data
ds = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc")
h = ds.holaps_evapotranspiration.copy()
m = ds.modis_evapotranspiration.copy()
g = ds.gleam_evapotranspiration.copy()

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
