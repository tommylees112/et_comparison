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
from engineering.eng_utils import get_non_coord_variables
from engineering.eng_utils import calculate_monthly_mean_std
from engineering.eng_utils import calculate_monthly_std
from engineering.eng_utils import select_pixel, turn_tuple_to_point

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
from plotting.plots import add_point_location_to_map

#
from plotting.plot_utils import get_colors

BASE_DATA_DIR = Path('/soge-home/projects/crop_yield/EGU_compare')
BASE_FIG_DIR =Path('/soge-home/projects/crop_yield/et_comparison/figs/meeting4')

datasets = ['holaps', 'gleam', 'modis']
evap_das = [f"{ds}_evapotranspiration" for ds in datasets]
[h_col, m_col, g_col, c_col] = get_colors()
col_lookup = dict(zip(evap_das+["chirps_precipitation"], [h_col,g_col,m_col,c_col]))


#%%
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------

shp_path = BASE_DATA_DIR / "" / ".shp"
coord_name = "watershed_for_pourpoint"
# 1. add shapefile to xarray object
wsheds_shp = gpd.read_file(shp_path)
wsheds = add_shape_coord_from_data_array(xr_da=ds,
    shp_path=shp_path,
    coord_name=coord_name
)
# 2. subset by the regions of interest
pour_points = lookup_gdf[['ID','StationName', 'DrainArLDD', 'YCorrected', 'XCorrected','geometry','corrected_river_name']]

# 3. normalise the precip / evapotranspiration values by the areas
unique_wsheds = np.unique(drop_nans_and_flatten(wsheds.coord_name))

.where()

#%%
# ------------------------------------------------------------------------------
# GRUN runoff data
# ------------------------------------------------------------------------------
r = xr.open_dataset(BASE_DATA_DIR/"GRUN_v1_GSWP3_WGS84_05_1902_2014.nc").Runoff
r = r.sel(lat=slice(region.latmin,region.latmax),lon=slice(region.lonmin,region.lonmax))
r = r.sel(time=slice(ds.time.min(), ds.time.max()))
#%%
# ------------------------------------------------------------------------------
# ADDING Climate zones
# ------------------------------------------------------------------------------
from engineering.mask_using_shapefile import add_shape_coord_from_data_array
climate_zone_shp_path = BASE_DATA_DIR / "climate_zone_shp" / "WC05_1975H_Koppen.shp"

d = add_shape_coord_from_data_array(ds, climate_zone_shp_path, coord_name="climate_zone")
fig,ax=plt.subplots()
d.climate_zone.plot(ax=ax)

# climate zone lookup
shp_gpd = gpd.read_file(climate_zone_shp_path)
zone_ids = np.unique(drop_nans_and_flatten(d.climate_zone))
zones = shp_gpd.loc[zone_ids,'Koppen']
zone_lookup = dict(zip(zones.index, zones.values))



# add climate zone information to the xarray object
d = get_lookup_val(
        xr_obj=d,
        variable='climate_zone',
        new_variable='koppen',
        lookup_dict=zone_lookup
)
zones_in_region = np.unique(d.koppen)
zone_lookup_r = dict(zip(zones_in_region, [i for i in range(len(zones_in_region))]))

d = get_lookup_val(
        xr_obj=d,
        variable='koppen',
        new_variable='koppen_code',
        lookup_dict=zone_lookup_r
)

mask = d.chirps_precipitation.isel(time=0).isnull()
d['koppen_code'] = d.koppen_code.where(~mask)

fig,ax = plt.subplots()
d.koppen_code.plot.contourf(ax=ax, levels=15)


#%%
# ------------------------------------------------------------------------------
# ADDING COUNTRY INFORMATION
# ------------------------------------------------------------------------------
from engineering.mask_using_shapefile import add_shape_coord_from_data_array

country_shp_path = BASE_DATA_DIR / "country_shp" / "ne_50m_admin_0_countries.shp"


d = add_shape_coord_from_data_array(ds, country_shp_path, coord_name="countries")

# get country lookup
shp_gpd = gpd.read_file(country_shp_path)
country_ids = np.unique(drop_nans_and_flatten(d.countries))
countries = shp_gpd.loc[country_ids,'SOVEREIGNT']
country_lookup = dict(zip(countries.index, countries.values))
# remove yemen from lookup
if 2 in country_lookup.keys():
    country_lookup.pop(2)

#
dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code']
variables = [var for var in ds.variables.keys() if var not in dims]

# PLOTTING SEASONAL PATTERNS BY COUNTRY
for country_id in country_lookup.keys():
    country = country_lookup[country_id]
    print(f"working on country {country}")

    # select that countries mean monthly pattern (temporal)
    d = ds.where(ds.countries == country_id)
    d_mth = calculate_monthly_mean(d)
    norm_mth = d_mth.apply(lambda x: (x / x.sum(dim='month'))*100)
    norm_mth = norm_mth.mean(dim=['lat','lon'])

    fig,axs = plt.subplots(2,2,figsize=(12,8))
    ylim = [-0.1, 35]
    for ix, var in enumerate(variables):
        ax_ix = np.unravel_index(ix, (2,2))
        ax = axs[ax_ix]
        color = col_lookup[var]
        norm_mth[var].plot.line(ax=ax, marker='o', color=color)
        ax.set_title(var)
        ax.set_ylim(ylim)

    plt.tight_layout()
    fig.suptitle(f"Seasonal Patterns for Country: {country}")
    fig.savefig(BASE_FIG_DIR / f"seasonal_patterns_{country}.png")
# look at each country rainfall pattern


#%%
# ------------------------------------------------------------------------------
# Adding Dunning Seasonality
# ------------------------------------------------------------------------------

from preprocessing.utils import convert_to_same_grid

dunning_dir = BASE_DATA_DIR / "dunning_seasonality_mask" / "clean_dunning_mask.nc"
mask = ds.holaps_evapotranspiration.isel(time=0).isnull()

if not dunning_dir.is_file():
    dunning = xr.open_dataset(BASE_DATA_DIR/"dunning_seasonality_mask/"/"chirps_seasonality_mask.nc")
    dunning = convert_to_same_grid(ds, dunning.seasonality_mask, method="nearest_s2d")
    dunning = dunning.where(~mask)
    dunning = dunning.drop('time')
    dunning.to_netcdf(BASE_DATA_DIR / "dunning_seasonality_mask" / "clean_dunning_mask.nc")
else:
    dunning = xr.open_dataset(dunning_dir)


def chop_to_region_extent(ds, region):
    """

    # dunning = chop_to_region_extent(dunning, all_region)

    """

    return ds.sel(
        lat=slice(region.latmax,region.latmin),
        lon=slice(region.lonmin,region.lonmax)
    )



# chirps rainfall for harmonic analysis
kwargs = {'vmin':0, 'vmax':1.8, 'cmap':'inferno'}
fig,ax = plt.subplots()
dunning.plot.contourf(ax=ax, **kwargs)


kwargs = {'vmin':0, 'vmax':4.0, 'cmap':'inferno'}
# fig,ax = plt.subplots()
# condition = (dunning.lat > 10) | (((dunning.lon < 44.0)) & (dunning < 1.0) & (dunning.lat > 1.0))
condition = (dunning > 1.0) | (dunning.lat < 5)

# condition.plot(ax=ax)
fig,ax = plot_geog_location(all_region, borders=True)
dunning.where(condition).plot(ax=ax, **kwargs)

fig,ax = plot_geog_location(all_region, borders=True)
dunning.where(dunning > 1.0).plot(ax=ax, **kwargs)

# 1 = BIMODAL, 0 = UNIMODAL
seasonality_mask = ((dunning > 1.0) ) | ((dunning.lat < 0) )
# seasonality_mask = ((dunning < 1.0) & (dunning.lon < 40)) | (dunning.lat > 4)
mask = ds.holaps_evapotranspiration.isel(time=0).isnull()
seasonality_mask = seasonality_mask.where(~mask)

fig,ax = plot_geog_location(all_region, borders=True)
seasonality_mask.plot(ax=ax)
fig.suptitle('Bimodal/Unimodal Mask\n(dunning > 1.0) ) | ((dunning.lat < 0)')
fig.savefig(BASE_FIG_DIR/'dunning_bimodal_mask.png')

fig,ax = plot_geog_location(all_region, borders=True)
ds.chirps_precipitation.mean(dim='time').where(seasonality_mask).plot(ax=ax)

bimodal = ds.where(seasonality_mask)
unimodal = ds.where((~seasonality_mask.astype(bool)) & (~mask))

fig,ax = plot_geog_location(all_region, borders=True)
bimodal.chirps_precipitation.mean(dim='time').plot(ax=ax)
fig.suptitle('Bimodal Rainfall')
fig.savefig(BASE_FIG_DIR / 'bimodal_rainfall_mean_dim_time.png')

fig,ax = plot_geog_location(all_region, borders=True)
unimodal.chirps_precipitation.mean(dim='time').plot(ax=ax)
fig.suptitle('Unimodal Rainfall')
fig.savefig(BASE_FIG_DIR / 'unimodal_rainfall_mean_dim_time.png')


dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code']
variables = [var for var in ds.variables.keys() if var not in dims]


def normalised_monthly_mean_std():
    """ """
    d_mean = calculate_monthly_mean(d)
    d_std = calculate_monthly_std(d)

    norm_mth = d_mth.apply(lambda x: (x / x.sum(dim='month'))*100)
    norm_mth = norm_mth.mean(dim=['lat','lon'])


# TOMMY I WAS HERE
# TODO: turn this into a function call
col_lookup = dict(zip(evap_das+["chirps_precipitation"], [h_col,g_col,m_col,c_col]))
ylim = [0,28]
for dname, d in zip(["bimodal", "unimodal"], [bimodal, unimodal]):
    d_mean = calculate_monthly_mean(d)
    d_std = calculate_monthly_std(d)
    norm_mth = d_mean.apply(lambda x: (x / x.sum(dim='month'))*100)
    norm_mth = norm_mth.mean(dim=['lat','lon'])

    fig,axs = plt.subplots(2,2,figsize=(12,8))
    ylim = [-0.1, 35]
    for ix, var in enumerate(variables):
        ax_ix = np.unravel_index(ix, (2,2))
        ax = axs[ax_ix]
        color = col_lookup[var]
        norm_mth[var].plot.line(ax=ax, marker='o', color=color)
        ax.set_title(var)
        ax.set_ylim(ylim)

    # plt.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"Seasonal Patterns for {dname} Rainfall Region")
    fig.savefig(BASE_FIG_DIR / f"seasonal_patterns_{dname}_region.png")




#%%
# ------------------------------------------------------------------------------
# plotting inset maps
# ------------------------------------------------------------------------------
import shapely

mean_std = calculate_monthly_mean_std(ds)
ds_mth = calculate_monthly_mean(ds)
norm_mth = ds_mth.apply(lambda x: (x / x.sum(dim='month'))*100)
normed_pcp = norm_mth.chirps_precipitation
norm_seas = ds.groupby('time.season').mean(dim='time').apply(lambda x: (x / x.sum(dim='season'))*100)
seas_pcp = norm_seas.chirps_precipitation

# -----------------------------------
# which is more indicative? MIN or MAX
fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
seas_pcp.max(dim='season').plot(ax=ax)
fig.suptitle("MAX Normalised Monthly Precip (% of total)")
# add_point_location_to_map(point1, ax)

fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
seas_pcp.min(dim='season').plot(ax=ax)
fig.suptitle("MIN Normalised Monthly Precip (% of total)")

fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
(seas_pcp.max(dim='season') - seas_pcp.min(dim='season')).plot(ax=ax)
fig.suptitle("MAX-MIN Normalised Monthly Precip (% of total)")

fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
seas_pcp.std(dim='season').plot(ax=ax)
fig.suptitle("STD Normalised Monthly Precip (% of total)")

# Attempt to mask out bimodal/unimodal regimes
seas_std = seas_pcp.std(dim='season')
fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
seas_std.where((ds.lat < 3.7) | (seas_std > 13) | ).plot(ax=ax)

# -----------------------------------

# lon,lat
loc1 = (38.1,2.407)
loc2 = (37.3,10.29)
loc3 = (12.7,39.4)

point1 = turn_tuple_to_point(loc1)
point2 = turn_tuple_to_point(loc2)


# TODO extract these functions
from engineering.eng_utils import select_pixel, turn_tuple_to_point
from plotting.plots import plot_pixel_tseries, plot_inset_map



loc3 = (40,5)
# pixel_normed = select_pixel(normed_pcp, loc3)

# PLOT THE SEASONAL CYCLE FOR 100 POINTS IN THE ROI
scale=5
lons=np.linspace(round(ds.lon.min().values.item()), int(ds.lon.max().values.item()), 10)
lats=np.linspace(round(ds.lat.min().values.item()), int(ds.lat.max().values.item()), 10)
lats=lats[::-1]

fig,axs = plt.subplots(10,10,figsize=(12*scale,8*scale),sharey=True)
for lon_ix, lon in enumerate(lons):
    for lat_ix, lat in enumerate(lats):
        ax = axs[lat_ix,lon_ix]
        loc = (lon,lat)
        plot_pixel_tseries(normed_pcp, loc, ax, map_plot=True)
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels('')
        ax.set_yticklabels('')

fig = plt.gcf()
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.suptitle('Seasonality of Rains Across East Africa')
fig.savefig(BASE_FIG_DIR / 'SEASONALITY_of_RAINS_across_EA.png')

fig,ax = plt.subplots()
plot_pixel_tseries(normed_pcp, loc3, ax, map_plot=True)


fig, ax = plt.subplots()
ax.plot([4,5,3,1,2])
axins = inset_axes(ax, width="40%", height="40%", loc="upper right",
                   axes_class=cartopy.mpl.geoaxes.GeoAxes,
                   axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()))
axins.add_feature(cartopy.feature.COASTLINE)
axins.add_feature(cartopy.feature.BORDERS)
axins.scatter(point.x,
       point.y,
       transform=cartopy.crs.PlateCarree(),
       c='black')
plt.show()


# TEST
loc3 = (40,5)
point = turn_tuple_to_point(loc3)
fig,ax = plt.subplots()
ax2 = plot_inset_map(ax, all_region, borders=True , lakes=True)
add_point_location_to_map(point, ax2, **{'color':'black'})

pixel_normed = select_pixel(normed_pcp, loc2)
fig,ax = plt.subplots()
plot_pixel_tseries(normed_pcp, loc2, ax, map_plot=False)

pixel_normed = select_pixel(normed_pcp, loc3)
fig,ax = plt.subplots()
plot_pixel_tseries(normed_pcp, loc3, ax, map_plot=False)







fig, ax = plot_geog_location(all_region, borders=True, lakes=True, rivers=False)
point1 = turn_tuple_to_point(loc1)
point2 = turn_tuple_to_point(loc2)
points = [point]
kwargs={'label':'POINT1','color':(1,0,0)}
add_point_location_to_map(point1, ax)
kwargs={'label':'POINT2','color':(0,1,0)}
add_point_location_to_map(point2, ax)
plt.legend()

from engineering.eng_utils import calculate_monthly_mean_std



#%%
# ------------------------------------------------------------------------------
# Working with Precipitation Data
# ------------------------------------------------------------------------------


from engineering.eng_utils import get_variables_for_comparison1
vars_, ds_comparisons = get_variables_for_comparison1()
col_lookup = dict(zip(evap_das+["chirps_precipitation"], [h_col,g_col,m_col,c_col]))

# Plot the comparison between the P-ET products
for var_ in vars_:
    # get the xlabel and xcolour
    xlabel = 'chirps_precipitation'
    xcol = col_lookup[xlabel]
    # get the ylabel and ycolour
    ylabel = var_
    ycol = col_lookup[ylabel]
    # get the dataarrays to compare
    da1 = drop_nans_and_flatten(ds.chirps_precipitation)
    da2 = drop_nans_and_flatten(ds[var_])
    hexbin_jointplot_sns(da1, da2, h_col, g_col, bins='log', xlabel=xlabel, ylabel=ylabel)
    fig = plt.gcf()
    fig.savefig(BASE_FIG_DIR/f'sns_hexplot_{xlabel}_vs_{ylabel}.png')


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
[h_col, m_col, g_col, c_col] = colors = get_colors()

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


# Calculate the P-E for each evaporation product (FOR ANNUAL TIMESCALES)
ds_annual = ds.resample(time = 'Y').mean()
all_ds = [ds_annual.chirps_precipitation - ds_annual[evap_da] for evap_da in evap_das]
for i,da in enumerate(all_ds):
    da.name = "P_minus_" + evap_das[i]
P_E_ds = xr.merge(all_ds)

# compute 5 year P-E
ds_5yr = ds.mean(dim='time')
all_ds = [ds_5yr.chirps_precipitation - ds_5yr[evap_da] for evap_da in evap_das]
for i,da in enumerate(all_ds):
    da.name = "P_minus_" + evap_das[i]
pe_5ds = xr.merge(all_ds)

# Plot spatial plots of the comparison between these P-ET
variables, comparisons = get_variables_for_comparison1()
kwargs = {'vmin':-2,'vmax':2}

# dims_ = [dims for dims in P_E_ds.dims.keys()]
# vars_ = [var for var in P_E_ds.variables.keys() if var not in dims_]
# for var_ in vars_:
#     with xr.set_options(cmap_sequential='RdBu'):
#         fig,ax = plt.subplots()
#         plot_mean_time(P_E_ds[var_], ax=ax, add_colorbar=True, **kwargs)
#         fig.suptitle(f'Spatial Means of Annual P-ET for {var_}')
#         fig.savefig(BASE_FIG_DIR / f'annual_spatial_mean_of_P-E_{var_}.png')

# PLOT MARGINALS
vars_ = [var for var in P_E_ds.variables.keys() if var not in dims_]
col_lookup = dict(zip(vars_,colors[:-1]))
for var_ in vars_:
    da = P_E_ds[var_]
    color = col_lookup[var_]
    title = f"Annual P-E Hisogram plots {var_}"

    plot_marginal_distribution(da, color, ax=None, title=title, xlabel=var_)

# PLOT 5 year marginals
dims_ = [dim for dim in pe_5ds.dims.keys()]
vars_ = [var for var in pe_5ds.variables.keys() if var not in dims_]
col_lookup = dict(zip(vars_,colors[:-1]))
for var_ in vars_:
    da = pe_5ds[var_]
    color = col_lookup[var_]
    title = f"5 Yearly P-E Hisogram plots {var_}"

    fig,ax = plt.subplots(figsize=(12,8))
    plot_marginal_distribution(da, color, ax=ax, title='', xlabel=var_, summary=True)
    fig.savefig(BASE_FIG_DIR / f"5year_P-E_distribution_{var_}.png")


# plot spatial mean of differences
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
    return lookup_gdf




def read_station_flow_data():
    """ """
    # read raw data
    df = pd.read_csv(BASE_DATA_DIR / 'Qts_Africa_glofas_062016_1971_2005.csv')
    df.index = pd.to_datetime(df.DATE)
    df = df.drop(columns='DATE')
    # select the date range
    df = df['2001-01-01':'2005-12-31']

    return df


def select_stations_in_river_name(lookup_gdf, river_name="Blue Nile"):
    """ select only the stations in the following river basin"""
    river_name = river_name.lower()
    assert river_name in lookup_gdf.corrected_river_name.values, f"Invalid River name: {river_name}. River name must be one of: \n{np.unique(lookup_gdf.corrected_river_name.values)}"

    # lookup_gdf.loc[lookup_gdf.]
    return lookup_gdf.query(f'corrected_river_name == "{river_name}"').ID


df = read_station_flow_data()
lookup_gdf = read_station_metadata()

# plot locations of all stations
fig, ax = plot_stations_on_region_map(all_region, lookup_gdf)
# fig,ax = plot_geog_location(all_region, lakes=True, borders=True, rivers=True, scale=0.8)
# lookup_gdf.plot(ax=ax)
fig.suptitle('All Stations')


def plot_blue_nile_stations():
    # blue nile metadata (FOR THE STATIONS)
    bn_ids = select_stations_in_river_name(lookup_gdf, river_name="Blue Nile")
    bn_meta = lookup_gdf.loc[lookup_gdf.ID.isin(bn_ids)]
    plot_stations_on_region_map(all_region, bn_meta)
    fig = plt.gcf()
    fig.suptitle('Runoff Station Measurements in the Blue Nile basin')
    fig.savefig(BASE_FIG_DIR / 'runoff_blue_nile_stations.png')

    # what are the units of the DrainArLDD ??
    drainage_area = bn_meta.DrainArLDD
    bn_stations = df[bn_meta.ID]

    return bn_meta
# do unit conversion (without your previous calcs)



def calculate_flow_per_day(df, lookup_gdf):
    """ convert flow in m3/s => mm/day in new columns, `colnames` = ID + '_perday'

    Steps:
    1) normalise per unit area
        runoff (m3) / m2
    2) Convert m => mm
        * 1000
    3) convert s => days
        * 86,400
    4)
    """
    for ID in lookup_gdf.ID:
        drainage_area = lookup_gdf.query(f'ID == "{ID}"').DrainArLDD.values[0]
        # TODO: what units is DrainArLDD in?
        # df[ID+'_norm'] = df[ID].apply(lambda runoff: ((runoff*1e9) / 86_400) / drainage_area )
        df[ID + '_perday'] = df[ID].apply(lambda runoff: ((runoff/(drainage_area)) * 86_400 * 1000)  )

    return df


df = calculate_flow_per_day(df, lookup_gdf)
df_normed = df[[col for col in df.columns if 'perday' in col]]
fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(drop_nans_and_flatten(df_normed), ax=ax, bins=1000)

# turn the flow into mm day-1
drainage_area_lookup = {}

# # val = flow in m3 sec-1
# # divide per unit area (in m2, m3 => m)
# val / drainage_area
# # convert from m to mm
# val * 1000
# # convert from s to days
# val / 86_400
# # convert per km^2
# val / drainage_area

# normalised flows mm day-1
df_normed = df[[col for col in df.columns if 'perday' in col]]
fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(drop_nans_and_flatten(df_normed), ax=ax, bins=1000)
fig.suptitle('Distribution of Flows\nrunoff / drainage_area * 86,400 * 1000')
fig.savefig(BASE_FIG_DIR/'runoff_calculation_histogram3.png')


#%%
# ------------------------------------------------------------------------------
# working with pysheds
# ------------------------------------------------------------------------------

river_network = gpd.read_file(BASE_DATA_DIR / "hydrosheds_rivers" / "af_riv_15s.shp")
ymin, ymax, xmin, xmax = all_region.latmin, all_region.latmax, all_region.lonmin, all_region.lonmax,
e_africa_rivers = river_network.cx[xmin:xmax, ymin:ymax]

def return_geoms_within_region(geo_df, region):
    """ Select the rows from a geopandas dataframe (read in from shapefile) that within bounding box.
    Arguments:
    ---------
    : geo_df (gpd.GeoDataFrame)

    : region (namedtuple Region object)
        object with following attributes:
         region.latmin, region.latmax, region.lonmin, region.lonmax


    Uses gpd.cx:
        > In addition to the standard pandas methods, GeoPandas also provides coordinate based indexing with the cx indexer, which slices using a bounding box. Geometries in the GeoSeries or GeoDataFrame that intersect the bounding box will be returned.

    https://gis.stackexchange.com/a/266833/123489
    """
    ymin, ymax, xmin, xmax = region.latmin, region.latmax, region.lonmin, region.lonmax
    return geo_df.cx[xmin:xmax, ymin:ymax]

fig,ax = plot_geog_location(all_region)
river_network.plot(ax=ax)
fig.suptitle('RIVER NETWORK (from HydroSHEDS)')
fig.savefig(BASE_FIG_DIR / "ALL_RIVERS.png")






# ==============================================================================

from pysheds.grid import Grid

grid = Grid.from_raster(BASE_DATA_DIR / "hydrosheds_direction" / "af_dir_15s" /"af_dir_15s", data_name='dir')


x=lookup_gdf.iloc[0].XCorrected
y=lookup_gdf.iloc[0].YCorrected
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label')
grid.clip_to('catch')
grid.view('catch')

# grid = Grid.from_raster(BASE_DATA_DIR/'hydrosheds'/'n30w100_con', data_name='dem')
# grid.read_raster('n30w100_dir', data_name='dir')
# grid.view('dem')

dem_dir = BASE_DATA_DIR / "hydrosheds_dem"
dem_files = [p.stem for p in dem_dir.glob('*') if (p.is_dir()) & ('WHOLE' not in p.stem)]
direction_dir = BASE_DATA_DIR / "hydrosheds_direction" / "af_dir_15s" / "af_dir_15s"
flow_accum_dir = BASE_DATA_DIR / "hydrosheds_flow" / "af_acc_15s/" / "af_acc_15s"

from rasterio.plot import show

grid = Grid.from_raster(dem_dir/dem_files[0]/dem_files[0], data_name='dem')
grid.fill_depressions(data='dem', out_name='flooded_dem')
grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
grid.flowdir(data='inflated_dem', out_name='dir')

fig, ax = plt.subplots()
show(grid.dem)

fig, ax = plt.subplots()
show(grid.flooded_dem)

fig, ax = plt.subplots()
show(grid.inflated_dem)

fig, ax = plt.subplots()
show(grid.dir)


directions_lookup = {
    64: 'North',
    128: 'Northeast',
    1: 'East',
    2: 'Southeast',
    4: 'South',
    8: 'Southwest',
    16: 'West',
    32: 'Northwest'
}

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
kwargs = {'vmin':2200,'vmax':3500}
# fig,ax = plot_geog_location(all_region,lakes=True, rivers=True, borders=True)
fig, ax = plot_stations_on_region_map(all_region, lookup_gdf, scale=1)
river_ds.river_basins.plot.contourf(levels=1000, ax=ax, zorder=0, **kwargs)

# 2. create lookup values for the basins
all_basins = np.unique(drop_nans_and_flatten(river_ds.river_basins))
shp_gpd = gpd.read_file(river_basins_path)
basins_subset = shp_gpd.loc[all_basins]


# PLOT ALL
fig,ax = plot_geog_location(all_region, lakes=True, borders=True, rivers=True)
basins_subset.plot(ax=ax, zorder=0, alpha=0.8, color=c_col)
lookup_gdf.plot(ax=ax)
fig.suptitle('Three Layers on top of one another: Basins, Stations, Boundaries')


for polygon in basins_subset.geometry:
    for point in lookup_gdf.geometry:
        polygon.contains(point)

# which CATCHMENT does each polygon belong to?
catchments = []
for point in lookup_gdf.geometry:
    ix = [polygon.contains(point) for polygon in basins_subset.geometry]
    watershed_ID = basins_subset[ix].LEVEL6.values
    try:
        watershed_ID = watershed_ID[0]
    except:
        watershed_ID = np.nan

    catchments.append(watershed_ID)

lookup_gdf['catchment_id'] = catchments

# get lookup dict
lookup_basin = dict(zip(basins_subset.index, basins_subset.LEVEL6))
lookup_basin[np.nan] = np.nan

r = get_lookup_val(xr_obj=river_ds, variable='river_basins',
        new_variable='basin_code', lookup_dict=lookup_basin
)


from engineering.eng_utils import compute_area_of_geom

import pyproj
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial


def compute_area_of_geom(geom):
    """ compute the area of a polygon using pyproj on a shapely object

    https://gis.stackexchange.com/a/166421/123489
    https://gis.stackexchange.com/questions/127607/area-in-km-from-polygon-of-coordinates
    """
    # assert isinstance(geom, shapely.geometry.multipolygon.MultiPolygon), f"geom should be of type: shapely.geometry.multipolygon.MultiPolygon, currently: {type(geom)}"

    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=geom.bounds[1],
                lat2=geom.bounds[3])),
        geom)

    return geom_area.area


#
for row in range(len(basins_subset)):
    geom = basins_subset.iloc[row,-1]
    # geom = basins_subset.iloc[0,-1]
    area = compute_area_of_geom(geom)

# basins_subset.apply(lambda x: compute_area_of_geom(x.geometry), axis=1)
# COMPUTE AREAS IN m-2
areas = [compute_area_of_geom(row['geometry']) for (ix,row) in basins_subset.iterrows()]
basins_subset['areas'] = areas


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


import shapefile as shp
from osgeo import osr


dst_file =BASE_DATA_DIR / 'all_region' / 'all_region.shp'
target_epsg = 4326

latmin = all_region.latmin
latmax = all_region.latmax
lonmin = all_region.lonmin
lonmax = all_region.lonmax

ul = [lonmin,latmax]
ur = [lonmax,latmax]
ll = [lonmin,latmin]
lr = [lonmax,latmin]

print(ul)
print(ur)
print(ll)
print(lr)



w = shp.Writer()
w.shapeType = 5
# add a "name" field of type "Character"
w.field('name', 'C')
w.poly([[
    ul,
    ur,
    lr,
    ll,
    ul
]])
w.record('testpolygon')
w.save(dst_file)
# create ESRI prj-file
sr = osr.SpatialReference()
sr.ImportFromEPSG(target_epsg)
sr.MorphToESRI()
with open(dst_file.as_posix().replace('.shp', '.prj'), 'w') as prj:
    prj.write(sr.ExportToWkt())


region_test = gpd.read_file(dst_file)

# fig,ax = plot_geog_location()
fig,ax=plt.subplots()
region_test.plot(ax=ax)


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


fig,ax = plot_seasonality(ds, double_year=True, variance=True)
ax.set_ylabel('Monthly Mean Daily Evapotranspiration [mm day-1]')
fig.suptitle('Monthly Mean Seasonality With +-1 S.D variability')
fig.savefig(BASE_FIG_DIR / 'spatial_mean_seasonality_VAR.png')


fig = plot_normalised_seasonality(ds, double_year=True, variance=True)
fig.suptitle('Monthly Mean Normalised Seasonality With +-1 S.D variability')
fig.savefig(BASE_FIG_DIR / 'spatial_mean_seasonality_normed_VAR.png')


fig,ax = plot_seasonality(ds.drop('chirps_precipitation'), double_year=True, variance=True)
ax.set_ylabel('Monthly Mean Daily Evapotranspiration [mm day-1]')
fig.suptitle('Monthly Mean Seasonality With +-1 S.D variability')
fig.savefig(BASE_FIG_DIR / 'AA_spatial_mean_seasonality_VAR.png')
