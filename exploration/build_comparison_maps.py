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

from engineering.mask_using_shapefile import add_shape_coord_from_data_array


BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
#%%
# ------------------------------------------------------------------------------
# Working with FLOW data
# ------------------------------------------------------------------------------
from shapely import geometry
import geopandas as gpd


from preprocessing.utils read_csv_point_data

def read_csv_point_data(df, lat_col='lat', lon_col='lon', crs='epsg:4326'):
    """Read in a csv file with lat,lon values in a column and turn those lat lon
        values into geometry.Point objects.
    Arguments:
    ---------
    : df (pd.DataFrame)
    : lat_col (str)
        the column in the dataframe that has the point latitude information
    : lon_col (str)
        the column in the dataframe that has the point longitude information
    : crs (str)
        coordinate reference system (defaults to 'epsg:4326')
    Returns:
    -------
    : gdf (gpd.GeoDataFrame)
        a geopandas.GeoDataFrame object
    """
    df['geometry'] = [geometry.Point(y, x) \
                      for x, y in zip(df[lat_col],
                                      df[lon_col])
                    ]
    crs = {'init': crs}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry="geometry")
    return gdf


# gpd.read_file(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
lookup_df = pd.read_csv(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
lookup_gdf = read_csv_point_data(lookup_df, lat_col='YCorrected', lon_col='XCorrected')


points = lookup_gdf.geometry.values
ax.scatter([point.x for point in points],
           [point.y for point in points],
           transform=ccrs.Geodetic())



# read raw data
df = pd.read_csv(BASE_DATA_DIR / 'Qts_Africa_glofas_062016_1971_2005.csv')
df['DATE'] = pd.to_datetime(df.DATE)

# pd.read_csv('')


#%%
# ------------------------------------------------------------------------------
# Working with RAW DATA
# ------------------------------------------------------------------------------

# from preprocessing.preprocessing import HolapsCleaner, ModisCleaner, GleamCleaner
from preprocessing.holaps_cleaner import HolapsCleaner
from preprocessing.gleam_cleaner import GleamCleaner
from preprocessing.modis_cleaner import ModisCleaner
#
# h = HolapsCleaner()
# h.preprocess()
# g = GleamCleaner()
# g.preprocess()
# m = ModisCleaner()
# m.preprocess()
#

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

xr_obj = lc
variable = 'esa_cci_landcover'
new_variable = 'new_label'
lookup_dict = lookup

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
assert False, "TEST ME GODDAMIT"
lc_2 = get_lookup_val(xr_obj, variable, new_variable, lookup_dict)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



#%%
# ------------------------------------------------------------------------------
# Subset by River Basins (or any other shapefile)
# http://www.fao.org/geonetwork/srv/en/metadata.show?id=30915&currTab=simple
# ------------------------------------------------------------------------------
from engineering.mask_using_shapefile import add_shape_coord_from_data_array

base_data_dir = Path("/soge-home/projects/crop_yield/EGU_compare")
river_basins_path = base_data_dir / "hydrosheds" / "h1k_lev6.shp"

river_ds = add_shape_coord_from_data_array(ds, river_basins_path, coord_name="river_basins")

# >>>>>>>>>>>>>>>>>>>
assert False, "Need to get a dictionary to lookup the values of the river_basins variable in river_ds"
# <<<<<<<<<<<<<<<<<<<

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
from plotting.plots import plot_masked_spatial_and_hist



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
# TOPO CLEANING
# ----------------------------------------------------

from engineering.eng_utils import bin_dataset, pickle_files
from plotting.plots import plot_masked_spatial_and_hist

# CLEAN CODE:
try:
    # try opening already saved files
    topo_bins = xr.open_dataset(BASE_DATA_DIR/"topo_bins1.nc")
    with open(BASE_DATA_DIR / 'intervals_topo1.pickle', 'rb') as f:
        intervals = pickle.load(f)

except:
    if not isinstance(topo, xr.Dataset):
        topo = topo.to_dataset(name='elevation')

    topo_bins, intervals = bin_dataset(ds=topo, group_var='elevation', n_bins=10)

    # repeat for 60 timesteps (TO BE USED AS `ds` mask)
    topo_bins = xr.concat([topo_bins for _ in range(len(ds_valid.time))])
    topo_bins = topo_bins.rename({'concat_dims':'time'})
    topo_bins['time'] = ds.time

    filepaths = [BASE_DATA_DIR / 'intervals_topo1.pickle', BASE_DATA_DIR / 'topo_bins1.pickle']
    vars = [intervals, topo_bins]
    pickle_files(filepaths, vars)

    topo_bins.to_netcdf(BASE_DATA_DIR / 'topo_bins1.nc')

interval_ranges = [(interval.left, interval.right) for interval in intervals]

# ----------------------------------------------------
# CLEAN CODE
def plot_masked_spatial_and_hist(dataMask, DataArrays, colors, titles, scale=1.5, **kwargs):
    """ SPATIAL and HISTOGRAM plots to show the conditional distributions given
         a particular mask.

    Arguments:
    ---------
    : dataMask (xr.DataArray)
        Mask for a particular area
    : DataArrays (list, tuple, iterable?)
        list of xr.DataArrays to use for the data.
    """
    assert all([isinstance(da, xr.DataArray) for da in DataArrays]), f"Currently only works when every member of DataArrays are xr.DataArray. Currently: {[type(da) for da in DataArrays]}"
    assert len(colors) == len(DataArrays), f"Len of the colors has to be equal to the len of the DataArrays \n Currently len(colors): {len(colors)} \tlen(DataArrays): {len(DataArrays)}"
    assert len(titles) == len(DataArrays), f"Len of the titles has to be equal to the len of the DataArrays \n Currently len(titles): {len(titles)} \tlen(DataArrays): {len(DataArrays)}"

    fig, axs = plt.subplots(2, len(DataArrays), figsize=(12*scale,8*scale))
    for j, DataArray in enumerate(DataArrays):
        if 'time' in DataArray.dims:
            # if time variable e.g. Evapotranspiration
            dataArray = get_unmasked_data(DataArray.mean(dim='time'), dataMask)
        else:
            # if time constant e.g. landcover
            dataArray = get_unmasked_data(DataArray, dataMask)

        # get the axes for the spatial plots and the histograms
        ax_map = axs[0,j]
        ax_hist = axs[1,j]
        color = colors[j]
        title = titles[j]

        ax_map.set_title(f'{dataArray.name}')
        ylim = [0,1.1]; xlim = [0,7]
        ax_hist.set_ylim(ylim)
        ax_hist.set_xlim(xlim)

        # plot the map
        plot_mean_time(dataArray, ax_map, add_colorbar=True, **kwargs)
        # plot the histogram
        plot_marginal_distribution(dataArray, color, ax=ax_hist, title=None, xlabel=dataArray.name)
        # plot_masked_histogram(ax_hist, dataArray, color, dataset)

    return fig


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
    # fig.savefig(f'figs/elevation_bin{i}.png')
# ----------------------------------------------------








for i in range(10):
    scale=1.5
    fig,axs = plt.subplots(2, 3, figsize=(12*scale,8*scale))
    dataMask = topo_bins.isel(elevation_bins=i).elevation.notnull()

    for j, dataset in enumerate(['holaps','modis','gleam']):
        dataArray = ds[f'{dataset}_evapotranspiration']
        dataArray = get_unmasked_data(dataArray.mean(dim='time'), dataMask)
        color = colors[j]
        # get the axes that correspond to the different rows
        ax_map = axs[0,j]
        ax_hist = axs[1,j]

        ax_map.set_title(f'{dataset} Evapotranspiration')

        # plot the maps

        # plot the histograms
        plot_masked_histogram(ax_hist, dataArray, color, dataset, ylim=[0,1.1],xlim=[0,7])

    elevation_range = interval_ranges[i]
    fig.suptitle(f"Evapotranspiration in elevation range: {elevation_range} ")
    fig.savefig(f'figs/elevation_bin{i}.png')

#%%
# ------------------------------------------------------------------------------
# Plot the bounding Box
# ------------------------------------------------------------------------------
# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
# https://stackoverflow.com/questions/14589600/matplotlib-insets-in-subplots
#

# CLEAN CODE
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
fig.savefig('figs/spatial_mean_seasonality.png')


fig = plot_normalised_seasonality(ds, double_year=True)
fig.savefig('figs/spatial_mean_seasonality_normed.png')

#%%
# ------------------------------------------------------------------------------
# Analysis by Elevation: group by elevation
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
# USING BASEMAP (depreceated code)
# ------------------------------------------------------------------------------
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
# OLD CODE (basemap)
from itertools import chain

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

def plot_bounding_box_map(latmin,latmax,lonmin,lonmax):
    fig = plt.figure(figsize=(8, 6), edgecolor='w')
    m = Basemap(projection='cyl', resolution='h',
                llcrnrlat=latmin, urcrnrlat=latmax,
                llcrnrlon=lonmin, urcrnrlon=lonmax, )
    draw_map(m)
    return fig

plot_bounding_box_map(latmin,latmax,lonmin,lonmax)
