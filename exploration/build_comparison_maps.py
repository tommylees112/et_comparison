# build_comparison_maps.py
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding
from scipy.stats import pearsonr
from scipy import stats
import geopandas as gpd
import pickle

import itertools
import warnings
import os
from pathlib import Path

%matplotlib
%load_ext autoreload

from preprocessing.utils import *
from preprocessing.utils import read_csv_point_data


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
from engineering.eng_utils import get_variables_for_comparison1


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

#
from plotting.plot_utils import get_colors

BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
BASE_FIG_DIR =Path('/soge-home/projects/crop_yield/et_comparison/figs/meeting2')

datasets = ['holaps', 'gleam', 'modis']
evap_das = [f"{ds}_evapotranspiration" for ds in datasets]
[h_col, m_col, g_col, c_col] = get_colors()

#%%
# ------------------------------------------------------------------------------
# Working with Precipitation Data
# ------------------------------------------------------------------------------


drop_nans_and_flatten(da)

from engineering.eng_utils import get_variables_for_comparison1
vars_, ds_comparisons = get_variables_for_comparison1()
col_lookup = dict(zip(evap_das, [h_col,g_col,m_col]))

# Plot the comparison between the P-ET products
for ds_comparison in ds_comparisons:
    # get the xlabel and xcolour
    xlabel = ds_comparison[0]
    xcol = col_lookup[xlabel]
    # get the ylabel and ycolour
    ylabel = ds_comparison[1]
    ycol = col_lookup[ylabel]
    # get the dataarrays to compare
    da1 = ds[ds_comparison[0]]
    da2 = ds[ds_comparison[1]]
    hexbin_jointplot_sns(da1, da2, h_col, g_col, bins='log', xlabel=xlabel, ylabel=ylabel)

hexbin_jointplot_sns()




#%%
# ------------------------------------------------------------------------------
# Working with P-E
# ------------------------------------------------------------------------------
# kwargs = {'vmin':,'vmax':}

# plot marginal distribution of chirps
fig,ax = plt.subplots()
plot_marginal_distribution(
    DataArray=ds.chirps_precipitation,
    color=c_col,
    ax=ax,
    title='CHIRPS Precipitation',
    xlabel='Precipitation [mm day-1]',
    **{'kde':False}
)
fig.savefig(BASE_FIG_DIR/'chirps_marginal.png')

#
datasets = ['holaps', 'gleam', 'modis']
evap_das = [f"{ds}_evapotranspiration" for ds in datasets]
[h_col, m_col, g_col, c_col] = get_colors()

for evap_da in evap_das:
    da = ds.chirps_precipitation - ds[evap_da]
    a = drop_nans_and_flatten(da)
    min, max, mean, median = a.min(), a.max(), a.mean(), np.median(a)
    da.name = f"preciptation - {evap_da}"
    fig,ax=plt.subplots()
    plot_marginal_distribution(
        DataArray=da,
        color=h_col,
        ax=ax,
        title=f'CHIRPS Precipitation - {evap_da} Evaporation\nMin: {min:.2f} Max: {max:.2f} Mean: {mean:.2f} Median: {median:.2f}',
        xlabel='P - E [mm day-1]',
        **{'kde':True}
    )
    ax.set_xlim([-10,10])
    ax.set_ylim([-0.05,0.7])
    fig.savefig(BASE_FIG_DIR/f'chirps-{evap_da}_marginal.png')


for ix, evap_da in enumerate(evap_das):
# evap_da_ = ['gleam_evapotranspiration']
# for ix, evap_da in enumerate(evap_das):
    da = ds[evap_da]
    a = drop_nans_and_flatten(da)
    min, max, mean, median = a.min(), a.max(), a.mean(), np.median(a)
    fig,ax=plt.subplots()
    plot_marginal_distribution(
        DataArray=da,
        color=get_colors()[ix],
        ax=ax,
        title=f'{evap_da} \nMin: {min:.2f} Max: {max:.2f} Mean: {mean:.2f} Median: {median:.2f}',
        xlabel='P - E [mm day-1]',
        **{'kde':True}
    )
    ax.set_xlim([-0.1,10])
    ax.set_ylim([-0.15,0.9])
    fig.savefig(BASE_FIG_DIR/f'{evap_da}_marginal.png')


# Calculate the P-E for each evaporation product
all_ds = [ds.chirps_precipitation - ds[evap_da] for evap_da in evap_das]
for i,da in enumerate(all_ds):
    da.name = evap_das[i] + "_minus_P"
P_E_ds = xr.merge(all_ds)


# Plot spatial plots of the comparison between these P-ET
variables, comparisons = get_variables_for_comparison1()
kwargs = {'vmin':-2,'vmax':2}
fig = plot_mean_spatial_differences_ET(P_E_ds, **kwargs)
fig.suptitle('Comparison of Spatial Means between P-ET for different products')
fig.savefig(BASE_FIG_DIR / 'spatial_mean_of_P-E_comparisons.png')

fig = plot_mean_spatial_differences_ET(ds, **kwargs)
fig.suptitle('Comparison of Spatial Means between E for different products')
fig.savefig(BASE_FIG_DIR / 'spatial_mean_of_ET_comparisons.png')

# does the anomaly go away in spatial means? one value per month (60 values)
# how do the products vary by region ?
# how do they time series covary?

xlabel='holaps'; ylabel='gleam'
datasets = ['holaps', 'gleam', 'modis']
evap_das = [f"{ds}_evapotranspiration" for ds in datasets]

from engineering.eng_utils import get_variables_for_comparison1
vars_, ds_comparisons = get_variables_for_comparison1()

# Plot the comparison between the P-ET products
for ds_comparison in ds_comparisons:
    xlabel = ds_comparison[0]
    ylabel = ds_comparison[1]
    da1 = ds.chirps_precipitation - ds[ds_comparison[0]]
    da2 = ds.chirps_precipitation - ds[ds_comparison[1]]
    hexbin_jointplot_sns(da1, da2, h_col, g_col, bins='log', xlabel=xlabel, ylabel=ylabel)
    fig = plt.gcf()
    fig.savefig(BASE_FIG_DIR/f'sns_hexplot_P-E_{xlabel}_vs_{ylabel}.png')


for ds_comparison in ds_comparisons:
    xlabel = ds_comparison[0]
    ylabel = ds_comparison[1]
    da1 = ds[ds_comparison[0]]
    da2 = ds[ds_comparison[1]]
    hexbin_jointplot_sns(da1, da2, h_col, g_col, bins='log', xlabel=xlabel, ylabel=ylabel)
    fig = plt.gcf()
    fig.savefig(BASE_FIG_DIR/f'sns_hexplot_COMPARE_{xlabel}_vs_{ylabel}.png')



#%%
# ------------------------------------------------------------------------------
# Working with FLOW data
# ------------------------------------------------------------------------------
from preprocessing.utils import read_csv_point_data
from plotting.plots import plot_stations_on_region_map
from engineering.regions import regions

all_region = regions[0]
highlands = regions[1]

# gpd.read_file(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
lookup_df = pd.read_csv(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
lookup_gdf = read_csv_point_data(lookup_df, lat_col='YCorrected', lon_col='XCorrected')

# plot locations of all stations
fig, ax = plot_stations_on_region_map(all_region, lookup_gdf)

# read raw data
df = pd.read_csv(BASE_DATA_DIR / 'Qts_Africa_glofas_062016_1971_2005.csv')
df.index = pd.to_datetime(df.DATE)
df = df.drop(columns='DATE')


# do unit conversion (without your previous calcs)
# blue nile metadata (FOR THE STATIONS)
bn_meta = lookup_df.query('RiverName == "Blue Nile"')
plot_stations_on_region_map(all_region, bn_meta)

# what are the unite of the DrainArLDD ??
drainage_area = bn_meta.DrainArLDD
bn_stations = df[bn_meta.ID]

# turn the flow into mm day-1
bn_stations = df[bn_meta.ID]
for ID in bn_meta.ID:
    drainage_area = bn_meta.query(f'ID == "{ID}"').DrainArLDD.values[0]
    bn_stations[ID] = bn_stations[ID] * 86400 / drainage_area

#%%
# ------------------------------------------------------------------------------
# Subset by River Basins (or any other shapefile)
# http://www.fao.org/geonetwork/srv/en/metadata.show?id=30915&currTab=simple
# ------------------------------------------------------------------------------
from engineering.mask_using_shapefile import add_shape_coord_from_data_array
from engineering.eng_utils import get_lookup_val, mask_multiple_conditions
from preprocessing.utils import read_csv_point_data

# READ DATA
# get location of files
base_data_dir = Path("/soge-home/projects/crop_yield/EGU_compare")
river_basins_path = base_data_dir / "hydrosheds" / "h1k_lev6.shp"

# gpd.read_file(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
lookup_df = pd.read_csv(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
lookup_gdf = read_csv_point_data(lookup_df, lat_col='YCorrected', lon_col='XCorrected')


# 1. add new coordinate from shapefile
river_ds = add_shape_coord_from_data_array(ds, river_basins_path, coord_name="river_basins")

# 2. create lookup values for the basins
all_basins = np.unique(river_ds.river_basins)[~np.isnan(np.unique(river_ds.river_basins))]
# first 2 digits are significant! (drop the first one (-0.))
bsns = np.unique(all_basins // 100)

lkup = dict(zip(all_basins, (all_basins // 100)))
lkup[np.nan] = np.nan

# 3. add new variable with lookup values
r = get_lookup_val(xr_obj=river_ds, variable='river_basins',
        new_variable='basin_code', lookup_dict=lkup
)

# basin_codes = array([-0., 22., 23., 24., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 58])
fig,ax = plt.subplots(figsize=(12,10))
bsn_msk = mask_multiple_conditions(r.basin_code, bsns[1:-1])
subset = r.where(bsn_msk)
subset.basin_code.plot.contourf(levels=14)
fig.savefig(BASE_FIG_DIR/"basins_map.png")
# r.sel(basin_code=slice(22,35)).basin_code.plot.contourf(levels=10)

# 4. subset by multiple basins
vals_to_keep = bsns[:3]
da = r.basin_code
bsn_msk = mask_multiple_conditions(da, vals_to_keep)
subset = r.where(bsn_msk)

# 5. plot and assign no. of levels to distinguish all vals
fig,ax= plt.subplots()
subset.basin_code.plot.contourf(levels=10, ax=ax)

# plot the stations ONTOP of the basins
from plotting.plots import plot_stations_on_region_map

# Plot NW most basins (more data)
fig, ax = plot_stations_on_region_map(all_region, lookup_gdf)
# subset.basin_code.plot.contourf(levels=10, ax=ax, zorder=0)
blue_nile = r.where(r.basin_code == 23).basin_code.plot.contourf(levels=10, ax=ax, zorder=0, alpha=0.5,color='y', add_colorbar=False)
fig.suptitle('Location of River Flow Station Measurements and the Blue Nile Basin shown')
fig.savefig(BASE_FIG_DIR/'blue_nile_basin_and_stations_plot.png')


# Plot the mean evaporation over the basin of interest
fig, ax = plot_stations_on_region_map(all_region, lookup_gdf)
blue_nile = r.where(r.basin_code == 23)
blue_nile.holaps_evapotranspiration.mean(dim='time').plot(ax=ax, zorder=0, alpha=0.5)


def label_basins():
    """https://stackoverflow.com/a/38902492/9940782

    pseudo code:
    1) get the shapely geometry objects from the xarray plots
    2) compute a point inside them using
        c['geometry'].apply(lambda x: x.representative_point().coords[:])
        c['coords'] = [coords[0] for coords in c['coords']]
    3) label them by the row 'NAME'
        c.plot()
        for idx, row in c.iterrows():
            plt.annotate(s=row['NAME'], xy=row['coords'],
                         horizontalalignment='center')

    """
    assert False, "Not Implemented!"
    return



#%%
# ------------------------------------------------------------------------------
# Working with RAW DATA
# ------------------------------------------------------------------------------

# from preprocessing.preprocessing import HolapsCleaner, ModisCleaner, GleamCleaner
from preprocessing.holaps_cleaner import HolapsCleaner
from preprocessing.gleam_cleaner import GleamCleaner
from preprocessing.modis_cleaner import ModisCleaner

# h = HolapsCleaner()
# h.preprocess()
# g = GleamCleaner()
# g.preprocess()
# m = ModisCleaner()
# m.preprocess()


ds = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc")

df = ds.to_dataframe()

#%%
# ------------------------------------------------------------------------------
# Seasonal Patterns
# ------------------------------------------------------------------------------

from plotting.plots import plot_seasonal_spatial_means

kwargs = {"vmin":0,"vmax":3.5}

seasons = ds.groupby('time.season').mean(dim='time')
variables = [
 'holaps_evapotranspiration',
 'gleam_evapotranspiration',
 'modis_evapotranspiration',
 'chirps_precipitation',
]

for var in variables:
    seasonal_da = seasons[var]
    plot_seasonal_spatial_means(seasonal_da, **kwargs)
    fig.savefig(f"figs/{var}_season_spatial2.png")

#%%
# ------------------------------------------------------------------------------
# Differences between products
# ------------------------------------------------------------------------------

# mean differences (spatial)
# ----------------
from plotting.plots import plot_mean_spatial_differences_ET

kwargs = {'vmin':-1.5,'vmax':1.5}
fig = plot_mean_spatial_differences_ET(ds, **kwargs)
fig.savefig(f"figs/product_comparison_spatial_means.png")

# differences by season
# ---------------------
from plotting.plots import get_variables_for_comparison1, plot_mean_time, plot_seasonal_comparisons_ET_diff

seasons = ds.groupby('time.season').mean(dim='time')
kwargs = {'vmin':-1.5,'vmax':1.5}
plot_seasonal_comparisons_ET_diff(seasons, **kwargs)
fig.savefig('figs/seasonal_differences_comparison_plot.png')

#%%
# ------------------------------------------------------------------------------
# hexbin of comparisons
# ------------------------------------------------------------------------------

from plotting.plots import plot_hexbin_comparisons


fig, title = plot_hexbin_comparisons(h, g, 'log')
fig.savefig(f'figs/{title}')

fig, title = plot_hexbin_comparisons(h, m, 'log')
fig.savefig(f'figs/{title}')

fig, title = plot_hexbin_comparisons(g, m, 'log')
fig.savefig(f'figs/{title}')


fig, title = plot_hexbin_comparisons(h, g, mincnt=100, bins='log')
fig.savefig(f'figs/{title}')
fig, title = plot_hexbin_comparisons(h, m, mincnt=5)
fig.savefig(f'figs/{title}')
fig, title = plot_hexbin_comparisons(g, m, mincnt=5)
fig.savefig(f'figs/{title}')

# -----------------
# SEABORN HEXPLOTS
# -----------------
from plotting.plots import plot_joint_plot_hex1, hexbin_jointplot_sns

h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]

da1 = drop_nans_and_flatten(h)
da2 = drop_nans_and_flatten(g)

# don't use this one!
# plot_joint_plot_hex1(da1,da2,h_col,g_col)
xlabel='holaps'; ylabel='gleam'
hexbin_jointplot_sns(da1, da2, h_col, g_col, bins='log', xlabel='holaps', ylabel='gleam')
fig = plt.gcf()
fig.savefig(f'figs/sns_hexplot_{xlabel}_vs_{ylabel}.png')


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


# Doing it with a pandas dataframe is potentially easier :(
dist_df = create_flattened_dataframe_of_values(h,g,m)
jp = sns.jointplot('holaps', 'gleam', data=dist_df, kind="hex", annot_kws=dict(stat="r"), joint_kws=dict(bins='log'))
jp.annotate(stats.pearsonr)

# test with random data
d1 = np.random.normal(10,1,100)
# d2 = np.random.randn(10,1,100)
d2 = np.random.gamma(1,2,100)
col1 = sns.color_palette()[0]
col2 = sns.color_palette()[1]
col3 = sns.color_palette()[2]
hexbin_jointplot_sns(d1,d2,col1,col2)

#%%
# ------------------------------------------------------------------------------
# Subset by regions
# ------------------------------------------------------------------------------

from engineering.regions import Region, create_regions, select_bounding_box_xarray

highlands_region, lake_vict_region, lowland_region = create_regions()

ds_high = select_bounding_box_xarray(ds, highlands_region)
ds_vic = select_bounding_box_xarray(ds, lake_vict_region)
ds_low = select_bounding_box_xarray(ds, lowland_region)


def comparison_hexbins(ds, area):
    h = ds.holaps_evapotranspiration
    m = ds.modis_evapotranspiration
    g = ds.gleam_evapotranspiration
    plot_hexbin_comparisons(h, g, bins='log', mincnt=0.5, title_extra=area)
    plot_hexbin_comparisons(h, m, bins='log', mincnt=0.5, title_extra=area)
    plot_hexbin_comparisons(g, m, bins='log', mincnt=0.5, title_extra=area)

    return

comparison_hexbins(ds_high, 'highlands_region')
comparison_hexbins(ds_low, 'lowlands_region')
comparison_hexbins(ds_vic, 'victoria_region')

from plotting.plots import plot_xarray_on_map, plot_all_spatial_means

plot_all_spatial_means(ds_high, 'highlands_region')
plot_all_spatial_means(ds_low, 'lowlands_region')
plot_all_spatial_means(ds_vic, 'victoria_region')

#%%
# ------------------------------------------------------------------------------
# Work with ESA CCI Landcover data
# https://annefou.github.io/metos_python/07-LargeFiles/
# ------------------------------------------------------------------------------
lc_legend = pd.read_csv('/soge-home/projects/crop_yield/EGU_compare/ESACCI-LC-Legend.csv',sep=';')
lc = xr.open_dataset('/soge-home/projects/crop_yield/EGU_compare/esa_lc_EA_clean.nc')
# lc = xr.open_dataset('ESACCI-LC-L4-LCCS-Map-300m-P5Y-2005-v1.6.1.nc', chunks={'lat': 1000,'lon': 1000})

# create lookup dictionary from pd.DataFrame
lookup = lc_legend[['NB_LAB','LCCOwnLabel']]
lookup = dict(zip(lookup.iloc[:,0], lookup.iloc[:,1]))

from engineering.eng_utils import get_lookup_val, drop_nans_and_flatten, create_flattened_dataframe_of_values

lc_2 = get_lookup_val(xr_obj=lc, variable='esa_cci_landcover',
        new_variable='lc_string', lookup_dict=lookup
)

#%%
# ------------------------------------------------------------------------------
# Geographic plotting
# ------------------------------------------------------------------------------
import cartopy.crs as ccrs

from plotting.plots import plot_xarray_on_map
variables = [
 'holaps_evapotranspiration',
 'gleam_evapotranspiration',
 'modis_evapotranspiration',
 'chirps_precipitation',
]

kwargs = {"vmin":0,"vmax":3.5}
for variable in variables:
    fig, ax = plot_xarray_on_map(ds[variable].mean(dim='time'),**kwargs)
    fig.suptitle(f'{variable}')
    fig.savefig(f'figs/{variable}_map.png')
    plt.close()

#%%
# ------------------------------------------------------------------------------
# Plot the mean spatial patterns of the RAW data
# ------------------------------------------------------------------------------
kwargs = {"vmin":0,"vmax":3.5}
from plotting.plots import plot_mean_time

# ASSERT THESE ARE THE CLEANING OBJECTS ADN NOT DATAARRAYS
assert isinstance(h, HolapsCleaner), "should be Cleaner objects not the clean dataarrays"
datasets = [h,m,g,c]
variables = ['holaps_evapotranspiration',
             'gleam_evapotranspiration',
             'modis_evapotranspiration',
             'chirps_precipitation']

# NOTE:
titles = [
f"HOLAPS Mean Latent Heat Flux [{raw_data.units}]",
f"MODIS Monthly Actual Evapotranspiration [{raw_data.units}]",
f"GLEAM Monthly mean daily Actual Evapotranspiration [{raw_data.units}]",
]

for var, data in zip(variables, datasets):
    var = var.split('_')[0]
    fig,ax=plt.subplots(figsize=(12,8))
    plot_mean_time(data.raw_data, ax, **kwargs)
    ax.set_title(f'{var} Raw Data')
    fig.savefig('figs/{var}_map1.png')
    # fig.savefig('figs/{var}_map1.svg')

# Plot the spatial patterns of cleaned data
# ------------------------------------------
kwargs = {"vmin":0,"vmax":3.5}

for var, data in zip(variables, datasets):
    fig,ax=plt.subplots(figsize=(12,8))
    plot_mean_time(data.clean_data, ax, **kwargs)
    ax.set_title(f'{var} Data [{data.clean_data.attrs.units}]')
    fig.savefig(f'figs/{var}_clean.png')


#%%
# ------------------------------------------------------------------------------
# Plot the histograms of values
# ------------------------------------------------------------------------------

from plotting.plots import plot_marginal_distribution

# GET colors for each variable
h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]
c_col = sns.color_palette()[3]

variables = ['holaps_evapotranspiration',
             'gleam_evapotranspiration',
             'modis_evapotranspiration',
             'chirps_precipitation']

colors = [h_col,m_col,g_col,c_col]

for i, var in enumerate(variables):
    DataArray = ds[var]
    color = colors[i]
    plot_marginal_distribution(DataArray, color)
    fig.savefig(f'figs/{var}_histogram.png')

#%%
# ------------------------------------------------------------------------------
# PLOT segmented map AND histograms (TOPO)
# ------------------------------------------------------------------------------

h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]
c_col = sns.color_palette()[3]
colors = [h_col, m_col, g_col]


from engineering.eng_utils import get_unmasked_data
from plotting.plots import plot_marginal_distribution, plot_mean_time
# from plotting.plots import plot_masked_spatial_and_hist

topo = xr.open_dataset('/soge-home/projects/crop_yield/EGU_compare/EA_topo_clean_ds.nc')

# ------------------------------------------------------------------------------
# 1. plot topo histogram
title = "Density Plot of Topography/Elevation in East Africa Region"
fig, ax = plot_marginal_distribution(topo, color=sns.color_palette()[-1], ax=None, title=title, xlabel='elevation')
fig.savefig('figs/topo_histogram.png')

# 2. plot topo_histogram with quintiles
# get the qunitile values
fig, ax = plot_marginal_distribution(topo, color=sns.color_palette()[-1], ax=None, title=title, xlabel='elevation')
# create and plot the qunitiles
qs = [float(topo.quantile(q=q).values) for q in np.arange(0,1.2,0.2)]
[ax.axvline(q, ymin=0,ymax=1,color='r',label=f'Quantile') for q in qs]

fig.savefig('figs/topo_histogram_quintiles.png')

# ----------------------------------------------------
# Conditional on Elevation (TOPO)
# ----------------------------------------------------

from engineering.eng_utils import bin_dataset, pickle_files
from engineering.eng_utils import get_unmasked_data
from plotting.plots import plot_marginal_distribution, plot_mean_time
from plotting.plots import plot_masked_spatial_and_hist
from engineering.eng_utils import load_pickle

# CLEAN CODE:
topo_bins = xr.open_dataset(BASE_DATA_DIR/"topo_bins1.nc")
intervals = load_pickle(BASE_DATA_DIR / 'intervals_topo1.pickle')


interval_ranges = [(interval.left, interval.right) for interval in intervals]
# for each of the 10 topography bins
for i in range(10):
    dataMask = topo_bins.isel(elevation_bins=i).elevation.notnull()
    dataArrays = [
        ds.holaps_evapotranspiration,
        ds.modis_evapotranspiration,
        ds.gleam_evapotranspiration
    ]
    colors = [h_col, m_col, g_col]
    titles = [
        "holaps_evapotranspiration",
        "modis_evapotranspiration",
        "gleam_evapotranspiration"
    ]
    kwargs = {"vmin":0,"vmax":3.5}
    fig = plot_masked_spatial_and_hist(dataMask, dataArrays, colors, titles, scale=1.5, **kwargs)

    # add the figure titles and save figures
    elevation_range = interval_ranges[i]
    fig.suptitle(f"Evapotranspiration in elevation range: {elevation_range} ")
    fig.savefig(f'figs/elevation_bin{i}.png')

#%%
# ------------------------------------------------------------------------------
# Plot the bounding Box (sub-regions)
# ------------------------------------------------------------------------------

from engineering.regions import regions
from plotting.plots import plot_geog_location, plot_stations_on_region_map
from plotting.plots import add_points_to_map, add_sub_region_box

all_region = regions[0]
highlands = regions[1]

# plot the stations
fig, ax = plot_stations_on_region_map(all_region, lookup_gdf)

# plot the regions
fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
color = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
add_sub_region_box(ax, highlands, color=color)


#%%
# ------------------------------------------------------------------------------
# Plot the Time Series of the points (spatial mean)
# ------------------------------------------------------------------------------

from matplotlib import cm
from matplotlib.colors import ListedColormap

my_cmap = ListedColormap(sns.color_palette().as_hex())

# GET colors for each variable
h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]

colors = [h_col,g_col,m_col]
my_cmap = ListedColormap(colors)
# get table of time series

tseries = ds.mean(dim=['lat','lon'],skipna=True).to_dataframe()

fig,ax = plt.subplots(figsize=(12,8))
tseries.plot(ax=ax) # ,colormap=my_cmap)
ax.set_title('Comparing the Spatial Mean Time Series from the ET Products')
ax.set_ylabel('Monthly Mean Daily Evapotranspiration [mm day-1]')
plt.legend()
fig.savefig('figs/spatial_mean_timseries.png')


#%%
# ------------------------------------------------------------------------------
# Plot the Time Series of different locations (spatial mean)
# ------------------------------------------------------------------------------

from engineering.regions import Region, create_regions, select_bounding_box_xarray
from engineering.eng_utils import create_double_year

from plotting.plots import plot_seasonality
from plotting.plots import plot_normalised_seasonality



#%%
# ------------------------------------------------------------------------------
# Plot the Time Series of different locations (points)
# ------------------------------------------------------------------------------


#%%
# ------------------------------------------------------------------------------
# Plot the Seasonality of different products
# ------------------------------------------------------------------------------

from engineering.eng_utils import calculate_monthly_mean, calculate_spatial_mean
from engineering.eng_utils import create_double_year
from plotting.plots import plot_seasonality
from plotting.plots import plot_normalised_seasonality


fig,ax = plot_seasonality(ds, double_year=True)
ax.set_ylabel('Monthly Mean Daily Evapotranspiration [mm day-1]')
fig.savefig(BASE_FIG_DIR / 'spatial_mean_seasonality.png')


fig = plot_normalised_seasonality(ds, double_year=True)
fig.savefig(BASE_FIG_DIR / 'spatial_mean_seasonality_normed.png')
