"""ipython_test.py"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding
from scipy.stats import pearsonr
from scipy import stats
import shapely
import geopandas as gpd

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
from engineering.eng_utils import calculate_monthly_mean_std
from engineering.eng_utils import calculate_monthly_std
from engineering.eng_utils import select_pixel, turn_tuple_to_point
from engineering.eng_utils import scalar_xr_to_dict

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
from plotting.plots import add_point_location_to_map
from plotting.plots import plot_pixel_tseries, plot_inset_map

#
from plotting.plot_utils import get_colors


BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
BASE_DIR = Path('/soge-home/projects/crop_yield/et_comparison')
BASE_FIG_DIR =Path('/soge-home/projects/crop_yield/et_comparison/figs/meeting5')

# clean data
ds = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc")
h = ds.holaps_evapotranspiration.copy()
m = ds.modis_evapotranspiration.copy()
g = ds.gleam_evapotranspiration.copy()

# drop yemen from the data
from engineering.mask_using_shapefile import add_shape_coord_from_data_array
country_shp_path = BASE_DATA_DIR / "country_shp" / "ne_50m_admin_0_countries.shp"
ds = add_shape_coord_from_data_array(ds, country_shp_path, coord_name="countries")
ds = ds.where(ds.countries != 2)

# get country lookup
shp_gpd = gpd.read_file(country_shp_path)
country_ids = np.unique(drop_nans_and_flatten(ds.countries))
countries = shp_gpd.loc[country_ids,'SOVEREIGNT']
country_lookup = dict(zip(countries.index, countries.values))


lc = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/esa_lc_EA_clean.nc")

# df = ds.to_dataframe()
seasons = ds.groupby('time.season').mean(dim='time')
mthly_mean = ds.groupby('time.month').mean(dim='time')
seasonality = mthly_mean.mean(dim=['lat','lon'])


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


# dist_df = create_flattened_dataframe_of_values(h,g,m)

topo = xr.open_dataset('/soge-home/projects/crop_yield/EGU_compare/EA_topo_clean_ds.nc')

all_region = regions[0]
highlands = regions[1]
lake_victoria_region = regions[2]


def read_station_metadata():
    """
    Columns of lookup_gdf:
    ---------------------
    ID :            station ID
    StationName :
    RiverName :
    RiverBasin :    basin name
    Country :
    CountryNam :
    Continent :
    Class :
    DrainArLDD :     Drainage Area Local Drain Direction (LDD)
    YCorrected :     latitude
    XCorrected :     longitude
    geometry :       (shapely.Geometry)
    """
    # gpd.read_file(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
    lookup_df = pd.read_csv(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
    lookup_gdf = read_csv_point_data(lookup_df, lat_col='YCorrected', lon_col='XCorrected')
    lookup_gdf['corrected_river_name'] = lookup_gdf.RiverName.apply(str.lower)
    lookup_gdf['NAME'] = lookup_gdf.index

    return lookup_gdf




def read_station_flow_data():
    """ """
    # read raw data
    df = pd.read_csv(BASE_DATA_DIR / 'Qts_Africa_glofas_062016_1971_2005.csv')
    df.index = pd.to_datetime(df.DATE)
    df = df.drop(columns='DATE')
    # select the date range
    df = df['2001-01-01':'2005-12-31']
    df = df.dropna(how='all',axis=1)

    return df


def select_stations_in_river_name(lookup_gdf, river_name="Blue Nile"):
    """ select only the stations in the following river basin"""
    river_name = river_name.lower()
    assert river_name in lookup_gdf.corrected_river_name.values, f"Invalid River name: {river_name}. River name must be one of: \n{np.unique(lookup_gdf.corrected_river_name.values)}"

    # lookup_gdf.loc[lookup_gdf.]
    return lookup_gdf.query(f'corrected_river_name == "{river_name}"').ID


df = read_station_flow_data()
lookup_gdf = read_station_metadata()
col_lookup = dict(zip(evap_das+["chirps_precipitation"], [h_col,g_col,m_col,c_col]))
