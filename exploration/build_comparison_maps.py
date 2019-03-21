# build_comparison_maps.py
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding
from scipy.stats import pearsonr
from scipy import stats

import itertools
import warnings
import os

%matplotlib
%load_ext autoreload

from preprocessing.utils import *

from engineering.mask_using_shapefile import add_shape_coord_from_data_array

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

# GET colors for each variable
h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]
c_col = sns.color_palette()[3]

from plotting.plots import plot_marginal_distribution

# Plot holaps
# -----------
def plot_marginal_distribution(DataArray, color):
    """ """
    fig,ax = plt.subplots(figsize=(12,8))
    # flatten the DataArray
    da_flat = drop_nans_and_flatten(DataArray)
    # plot the histogram
    sns.distplot(da_flat, ax=ax, color=h_col)
    title= f'Density Plot of {DataArray.name} [{DataArray.units}]'
    xlabel = f'Mean Monthly {DataArray.name} [{DataArray.units}]'
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    return fig, ax


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

fig1,ax1=plt.subplots(figsize=(12,8))
h_flat = drop_nans_and_flatten(ds.holaps_evapotranspiration)

sns.set_color_codes()
sns.distplot(h_flat, ax=ax1, color=h_col)

ax1.set_title(f'Density Plot of HOLAPS Mean Monthly Evapotranspiration [{ds.holaps_evapotranspiration.units}]')
ax1.set_xlabel(f'Mean Monthly Evapotranspiration [{ds.holaps_evapotranspiration.units}]')
fig1.savefig('figs/holaps_hist1.png')
# fig1.savefig('figs/holaps_hist1.svg')

# Plot modis
# -----------
fig2,ax2=plt.subplots(figsize=(12,8))
m_flat = drop_nans_and_flatten(ds.modis_evapotranspiration)

sns.set_color_codes()
sns.distplot(m_flat,ax=ax2, color=m_col)

ax2.set_title(f'Density Plot of MODIS Monthly Actual Evapotranspiration [{ds.modis_evapotranspiration.units}]')
ax2.set_xlabel(f'Monthly Actual Evapotranspiration [{ds.modis_evapotranspiration.units}]')
fig2.savefig('figs/modis_hist1.png')
# fig2.savefig('figs/modis_hist1.svg')

# Plot gleam
# -----------
fig3,ax3=plt.subplots(figsize=(12,8))
g_flat = drop_nans_and_flatten(ds.gleam_evapotranspiration)

sns.set_color_codes()
sns.distplot(g_flat,ax=ax3, color=g_col)

ax3.set_title(f'Density Plot of GLEAM Monthly mean daily Actual Evapotranspiration [{g.clean_data.units}] ')
ax3.set_xlabel(f'Monthly mean daily Actual Evapotranspiration [{g.clean_data.units}]')
fig3.savefig('figs/gleam_hist1.png')
# fig3.savefig('figs/gleam_hist1.svg')

#%%
# ------------------------------------------------------------------------------
# PLOT segmented map AND histograms
# ------------------------------------------------------------------------------

def get_unmasked_data(dataArray, dataMask):
    """ """
    return dataArray.where(dataMask)

h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]
colors = [h_col, m_col, g_col]
kwargs = {"vmin":0,"vmax":3.5}

for i in range(10):
    scale=1.5
    fig,axs = plt.subplots(2, 3, figsize=(12*scale,8*scale))
    dataMask = topo_bins.isel(elevation_bins=i).elevation.notnull()

    for j, dataset in enumerate(['holaps','modis','gleam']):
        dataArray = ds_valid[f'{dataset}_evapotranspiration']
        dataArray = get_unmasked_data(dataArray.mean(dim='time'),dataMask)
        color = colors[j]
        # get the axes that correspond to the different rows
        ax_map = axs[0,j]
        ax_map.set_title(f'{dataset} Evapotranspiration')
        ax_hist = axs[1,j]
        ax_hist.set_ylim([0,1.1])
        ax_hist.set_xlim([0,7])
        # plot the maps
        dataArray.mean(dim='time').plot(ax=ax_map,**kwargs)
        # plot the histograms
        d = drop_nans_and_flatten(dataArray)
        sns.distplot(d, ax=ax_hist,color=color)

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

import cartopy
import cartopy.feature as cpf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt

import matplotlib.patches as mpatches
from shapely.geometry.polygon import LinearRing


# whole
region = dict(lonmin=32.6,
    lonmax=51.8,
    latmin=-5.0,
    latmax=15.2,
)

from collections import namedtuple
Region = namedtuple('Region',field_names=['lonmin','lonmax','latmin','latmax'])

region = Region(
    lonmin=32.6,
    lonmax=51.8,
    latmin=-5.0,
    latmax=15.2,
)

highlands_region = Region(
    lonmin=35,
    lonmax=40,
    latmin=5.5,
    latmax=12.5
    )

lake_vict_region = Region(
    lonmin=32.6,
    lonmax=38.0,
    latmin=-5.0,
    latmax=2.5
    )

lowland_region = Region(
    lonmin=32.6,
    lonmax=42.5,
    latmin=0.0,
    latmax=12
)


def plot_geog_location(region):
    """ use cartopy to plot the region (defined as a namedtuple object)"""
    lonmin,lonmax,latmin,latmax = region.lonmin,region.lonmax,region.latmin,region.latmax
    ax = plt.figure().gca(projection=cartopy.crs.PlateCarree())
    ax.add_feature(cpf.COASTLINE)
    ax.add_feature(cpf.BORDERS, linestyle=':')
    ax.add_feature(cpf.LAKES)
    ax.set_extent([lonmin, lonmax, latmin, latmax])

    # plot the lat lon labels
    # https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    xticks = np.linspace(lonmin, lonmax, 5)
    yticks = np.linspace(latmin, latmax, 5)

    ax.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
    ax.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    fig = plt.gcf()

    return fig


def plot_all_regions():
    fig = plot_geog_location(region)
    plt.gca().set_title('ALL_region')
    fig.savefig('figs/ALL_region_location.png')
    fig = plot_geog_location(highlands_region)
    plt.gca().set_title('highlands_region')
    fig.savefig('figs/highlands_region_location.png')
    fig = plot_geog_location(lake_vict_region)
    plt.gca().set_title('lake_vict_region')
    fig.savefig('figs/lake_vict_region_location.png')
    fig = plot_geog_location(lowland_region)
    plt.gca().set_title('lowland_region')
    fig.savefig('figs/lowland_region_region_location.png')

def plot_polygon(fig, lonmin, lonmax, latmin, latmax):
    """
    Note:
    ----
    order is important:
        lower-left, upper-left, upper-right, lower-right
        2 -- 3
        |    |
        1 -- 4
    """
    # ax = fig.axes[0]
    lons = [latmin, latmin, latmax, latmax]
    lats = [lonmin, lonmax, lonmax, lonmin]
    ring = LinearRing(list(zip(lons, lats)))
    ax.add_geometries([ring], cartopy.crs.PlateCarree(), facecolor='none', edgecolor='black')


    return fig
#
# ax = plt.figure().gca(projection=cartopy.crs.PlateCarree())
# ax = plt.subplot(projection=cartopy.crs.PlateCarree())
# ax.add_feature(cpf.COASTLINE)
# ax.add_feature(cpf.BORDERS, linestyle=':')
# ax.add_feature(cpf.LAKES)
# ax.set_extent([lonmin, lonmax, latmin, latmax])
# # plot the lat lon labels
# # https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
# # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
# xticks = np.linspace(lonmin, lonmax, 5)
# yticks = np.linspace(latmin, latmax, 5)
# ax.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
# ax.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
#
# lons = [lake_vict_region['latmin'], lake_vict_region['latmin'], lake_vict_region['latmax'], lake_vict_region['latmax']]
# lats = [lake_vict_region['lonmin'], lake_vict_region['lonmax'], lake_vict_region['lonmax'], lake_vict_region['lonmin']]
# ax.fill(lons, lats, color='coral', transform=cartopy.crs.PlateCarree(), alpha=0.4)
# # ring = LinearRing(list(zip(lons, lats)))
# # ax.add_geometries([ring], cartopy.crs.PlateCarree(), facecolor='none', edgecolor='black')

lake_vict_region = dict(lonmin=32.6,
                        lonmax=38.0,
                        latmin=-5.0,
                        latmax=2.5,
                    )
highlands_region = dict(lonmin=35,
                        lonmax=40,
                        latmin=5.5,
                        latmax=12.5,
                    )

fig = plot_geog_location(region['lonmin'],region['lonmax'],region['latmin'],region['latmax'])

fig = plot_polygon(fig, lake_vict_region['lonmin'],lake_vict_region['lonmax'],lake_vict_region['latmin'],lake_vict_region['latmax'])
fig.suptitle('Whole Region')
fig.savefig('figs/whole_region.png')

fig = plot_geog_location(lake_vict_region['lonmin'],lake_vict_region['lonmax'],lake_vict_region['latmin'],lake_vict_region['latmax'])
fig.suptitle('Lake Victoria Region')
fig.savefig('figs/lake_vict_region.png')

fig = plot_geog_location(highlands_region['lonmin'],highlands_region['lonmax'],highlands_region['latmin'],highlands_region['latmax'])
fig.suptitle('Ethiopian Highlands Region')
fig.savefig('figs/highlands_region.png')


# ax.get_xlim(), ax.get_ylim()
# cb = fig.colorbar(hb, ax=ax)

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

mthly_mean = ds.groupby('time.month').mean(dim='time')
seasonality = mthly_mean.mean(dim=['lat','lon'])

fig, ax = plt.subplots(figsize=(12,8))
seasonality.to_dataframe().plot(ax=ax)
ax.set_title('Spatial Mean Seasonal Time Series from the ET Products')
ax.set_ylabel('Monthly Mean Daily Evapotranspiration [mm day-1]')
plt.legend()
fig.savefig('figs/spatial_mean_seasonality.png')

# ------------------------------------------------------------------------------
# Plot the NORMALISED Seasonality (% of the total)
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12,8))
norm_seasonality = seasonality.apply(lambda x: (x / x.sum(dim='month'))*100)
norm_seasonality.to_dataframe().plot(ax=ax)
ax.set_title('Normalised Seasonality of Data Products')
ax.set_ylabel('Contribution of that month ET to total ET (%)')
plt.legend()
fig.savefig('figs/spatial_mean_seasonality_normed.png')




#%%
# ------------------------------------------------------------------------------
# Analysis by Elevation: group by elevation
# ------------------------------------------------------------------------------

topo = xr.open_rasterio('../topography/ETOPO1_Ice_g.tif')
topo = topo.drop('band')
topo = topo.sel(band=0)
topo.name = "elevation"
# topo = topo.to_dataset()

def select_east_africa(ds):
    """ """
    lonmin=32.6
    lonmax=51.8
    latmin=-5.0
    latmax=15.2

    return ds.sel(y=slice(latmax,latmin),x=slice(lonmin, lonmax))

if not os.path.isfile('global_topography.nc'):
    topo.to_netcdf('global_topography.nc')

topo = select_east_africa(topo)
topo = topo.rename({'y':'lat','x':'lon'})

if not os.path.isfile('EA_topography.nc'):
    topo.to_netcdf('EA_topography.nc')

# convert to same grid as the other data (ds_valid)
topo = convert_to_same_grid(ds_valid, topo, method="bilinear")

# mask the same areas as other data (ds_valid)
mask = get_holaps_mask(ds_valid.holaps_evapotranspiration)
topo = topo.where(~mask)



# ------------------------------------------------------------------------------
# plot topo histogram
t = drop_nans_and_flatten(topo)
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(t,ax=ax, color=sns.color_palette()[-1])
ax.set_title(f'Density Plot of Topography/Elevation in East Africa Region')
fig.savefig('figs/topo_histogram.png')
plt.close()

# plot topo histogram WITH QUINTILES (0,0.2,0.4,0.6,0.8,1.0)
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(t,ax=ax, color=sns.color_palette()[-1])
ax.set_title(f'Density Plot of Topography/Elevation in East Africa Region')
# get the qunitile values
qs = [float(topo.quantile(q=q).values) for q in np.arange(0,1.2,0.2)]
# plot vertical lines at the given quintile value
[ax.axvline(q, ymin=0,ymax=1,color='r',label=f'Quantile') for q in qs]
fig.savefig('figs/topo_histogram_quintiles.png')
plt.close()

# ------------------------------------------------------------------------------
# GROUPBY
topo.name = 'elevation'
topo = topo.to_dataset()
bins = topo.groupby_bins(group='elevation',bins=10)
intervals = bins.mean().elevation_bins.values
left_bins = [interval.left for interval in intervals]

# plot to check WHERE the bins are
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(t,ax=ax, color=sns.color_palette()[-1])
[ax.axvline(bin, ymin=0,ymax=1,color='r',label=f'Bin') for bin in left_bins];

# [bins for bins in
topo_bins = xr.concat([topo.where(
                (topo['elevation'] > interval.left) & (topo['elevation'] < interval.right)
            )
            for interval in intervals ]
)
topo_bins = topo_bins.rename({'concat_dims':'elevation_bins'})

# repeat for 60 timesteps
topo_bins = xr.concat([topo_bins for _ in range(len(ds_valid.time))])
topo_bins = topo_bins.rename({'concat_dims':'time'})
topo_bins['time'] = ds_valid.time

# select and plot the values at different elevations
topo_bins.isel(elevation_bins=0)

#
# ds_valid.where(topo_bins.isel(elevation_bins=0).elevation.notnull())

def get_unmasked_data(dataArray, dataMask):
    """ """
    return dataArray.where(dataMask)


    # data = ds_valid.where(
    #  topo_bins.isel(elevation_bins=i).elevation.notnull()
    # ).holaps_evapotranspiration.mean(dim='time')

h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]
colors = [h_col, m_col, g_col]
kwargs = {"vmin":0,"vmax":3.5}
interval_ranges = [(interval.left, interval.right) for interval in intervals]

for i in range(10):
    scale=1.5
    fig,axs = plt.subplots(2, 3, figsize=(12*scale,8*scale))
    dataMask = topo_bins.isel(elevation_bins=i).elevation.notnull()

    for j, dataset in enumerate(['holaps','modis','gleam']):
        dataArray = ds_valid[f'{dataset}_evapotranspiration']
        dataArray = get_unmasked_data(dataArray.mean(dim='time'),dataMask)
        color = colors[j]
        # get the axes that correspond to the different rows
        ax_map = axs[0,j]
        ax_map.set_title(f'{dataset} Evapotranspiration')
        ax_hist = axs[1,j]
        ax_hist.set_ylim([0,1.1])
        ax_hist.set_xlim([0,7])
        # plot the maps
        dataArray.mean(dim='time').plot(ax=ax_map,**kwargs)
        # plot the histograms
        d = drop_nans_and_flatten(dataArray)
        sns.distplot(d, ax=ax_hist,color=color)

    elevation_range = interval_ranges[i]
    fig.suptitle(f"Evapotranspiration in elevation range: {elevation_range} ")
    fig.savefig(f'figs/elevation_bin{i}.png')


#
# # plot holaps_reprojected
# # -----------------------
# fig4,ax4=plt.subplots(figsize=(12,8))
# h_repr = drop_nans_and_flatten(holaps_repr)
#
# sns.set_color_codes()
# sns.distplot(h_repr,ax=ax4, color=h_col)
#
# ax4.set_title(f'Density Plot of HOLAPS Monthly Actual Evapotranspiration [mm day-1] ')
# ax4.set_xlabel(f'Monthly mean daily Actual Evapotranspiration [mm day-1]')
# fig4.savefig('figs/holaps_repr_hist1.png')
# # fig4.savefig('figs/holaps_repr_hist1.svg')

#%%
# ------------------------------------------------------------------------------
# JointPlots of variables
# ------------------------------------------------------------------------------

#%%
# FIRST have to be mapped onto the same grid

#
# #%%
# fig1,ax1=plt.subplots(figsize=(12,8))
# h = drop_nans_and_flatten(holaps)
# m = drop_nans_and_flatten(modis)
#
# g = sns.JointGrid(x=h, y=m)
# g = g.plot_joint(plt.scatter, color="k", edgecolor="white")
# _ = g.ax_marg_x.hist(h, color=h_col, alpha=.6,)
#                      # bins=np.arange(0, 60, 5))
#
# _ = g.ax_marg_y.hist(m, color=m_col, alpha=.6,
#                      orientation="horizontal",)
#                      # bins=np.arange(0, 12, 1))







#
