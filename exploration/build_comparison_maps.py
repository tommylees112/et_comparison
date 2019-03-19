# build_comparison_maps.py
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding

import warnings
import os

%matplotlib
%load_ext autoreload

from preprocessing.utils import *

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

seasons = ds.groupby('time.season').mean(dim='time')

variables = [
 'holaps_evapotranspiration',
 'gleam_evapotranspiration',
 'modis_evapotranspiration',
]

kwargs = {"vmin":0,"vmax":3.5}

for var in variables:
    scale=1
    fig,axs = plt.subplots(2,2,figsize=(12*scale,8*scale))
    for i in range(4):
        ax = axs[np.unravel_index(i,(2,2))]
        seasons[var].isel(season=i).plot(ax=ax, **kwargs)
        season_str = str(seasons[var].isel(season=i).season.values)
        ax.set_title(f"{var} {season_str}")

    plt.tight_layout()
    fig.savefig(f"{var}_season_spatial.png")

#%%
# ------------------------------------------------------------------------------
# Differences
# ------------------------------------------------------------------------------
import itertools
variables = [
 'holaps_evapotranspiration',
 'gleam_evapotranspiration',
 'modis_evapotranspiration',
]
comparisons = [i for i in itertools.combinations(variables,2)]

# mean differences (spatial)
# ----------------
kwargs = {'vmin':-1.5,'vmax':1.5}
fig,axs = plt.subplots(1,3, figsize=(15,12))

for i, cmprson in enumerate(comparisons):
    ax = axs[i]
    diff = ds[cmprson[0]] - ds[cmprson[1]]
    if i!=3:
        diff.mean(dim='time').plot(ax=ax, **kwargs, add_colorbar=False)
    else:
        diff.mean(dim='time').plot(ax=ax, **kwargs)
    ax.set_title(f"{cmprson[0].split('_')[0]} - {cmprson[1].split('_')[0]} Temporal Mean")
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.suptitle('Comparison of Spatial Means Between Products')
fig.savefig(f"product_comparison_spatial_means.png")

# differences by season
# ---------------------
seasons = ds.groupby('time.season').mean(dim='time')

kwargs = {'vmin':-1.5,'vmax':1.5}
fig, axs = plt.subplots(2,6, figsize=(15,12))

# get the axes indexes upfront
row_s = [0,1]; col_s = [0,1,2,3,4,5];
axes_ix = [i for i in itertools.product(row_s,col_s)]; axes.sort(key=lambda x: x[1])
axes_ix = [(0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3),(0,4),(0,5),(1,4),(1,5)]

ix_counter = 0
for i, cmprson in enumerate(comparisons):
    # for each comparison calculate the SEASONAL difference
    seas_diff = seasons[cmprson[0]] - seasons[cmprson[1]]
    if i != 0: ix_counter += 1
    # select each season from seasonal difference to plot
    for j in range(4):
        # get the correct axes index
        print("i: ",i)
        print("j: ",j)
        print(ix_counter)
        ax_ix = axes_ix[ix_counter]
        ix_counter += 1
        print(ax_ix)
        ax = axs[ax_ix]

        # plot the given seasonal difference in correct axis
        seas_diff.isel(season=i).plot(ax=ax, add_colorbar=False, **kwargs)
        season_str = str(seas_diff.isel(season=j).season.values)
        # if i!=3:
            # diff.mean(dim='time').plot(ax=ax, **kwargs, add_colorbar=False)
        # else:
            # diff.mean(dim='time').plot(ax=ax, **kwargs)
        ax.set_title(f"{cmprson[0].split('_')[0]} - {cmprson[1].split('_')[0]} {season_str}")
        ax.set_xlabel('')
        ax.set_ylabel('')

plt.tight_layout()

#%%
# ------------------------------------------------------------------------------
# hexbin of comparisons
# ------------------------------------------------------------------------------



# var_dataset_x =
# var_dataset_y =
fig, ax = plt.subplots()

# plot the data
hb = ax.hexbin(var_dataset_x, var_dataset_y, bins='log',gridsize=40, mincnt=0.5)

# draw the 1:1 line (showing datasets exactly the same)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label="1:1")

dataset_name_x = whole_df.columns[0].split("_")[1]
dataset_name_y = whole_df.columns[1].split("_")[1]
if whole_df.columns[0].split("_")[0] == 'albedo':
    variable_name = whole_df.columns[0].split("_")[0].capitalize()
else:
    variable_name = whole_df.columns[0].split("_")[0]
title = variable_name + ": " + dataset_name_x + " vs. " + dataset_name_y
ax.set_xlabel(dataset_name_x)
ax.set_ylabel(dataset_name_y)
ax.set_title(title)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(counts)')




#%%
# ------------------------------------------------------------------------------
# Geographic plotting
# ------------------------------------------------------------------------------
import cartopy.crs as ccrs

def plot_xarray_on_map(da,borders=True,coastlines=True,**kwargs):
    """"""
    # get the center points for the maps
    mid_lat = np.mean(da.lat.values)
    mid_lon = np.mean(da.lon.values)
    # create the base layer
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(mid_lon, mid_lat))
    # ax = plt.axes(projection=ccrs.Orthographic(mid_lon, mid_lat))

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    da.plot(ax=ax, transform=ccrs.PlateCarree(),vmin=vmin, vmax=vmax);

    ax.coastlines();
    ax.add_feature(cartopy.feature.BORDERS,linestyle=':');
    fig = plt.gcf()
    return fig, ax

kwargs = {"vmin":0,"vmax":3.5}
fig1, ax1 = plot_xarray_on_map(ds_valid.holaps_evapotranspiration.mean(dim='time'),**kwargs)
fig1.suptitle('Holaps Evapotranspiration')
plt.gca().outline_patch.set_visible(False)
fig1.savefig('figs/holaps_clean_et_map.png')
plt.close()

fig2, ax2 = plot_xarray_on_map(ds_valid.modis_evapotranspiration.mean(dim='time'),**kwargs)
fig2.suptitle('MODIS Evapotranspiration')
plt.gca().outline_patch.set_visible(False)
fig2.savefig('figs/modis_clean_et_map.png')
plt.close()

fig3, ax3 = plot_xarray_on_map(ds_valid.gleam_evapotranspiration.mean(dim='time'),**kwargs)
fig3.suptitle('GLEAM Evapotranspiration')
plt.gca().outline_patch.set_visible(False)
fig3.savefig('figs/gleam_clean_et_map.png')
plt.close()

#%%
# ------------------------------------------------------------------------------
# Plot the mean spatial patterns of the RAW data
# ------------------------------------------------------------------------------
kwargs = {"vmin":0,"vmax":3.5}

fig1,ax1=plt.subplots(figsize=(12,8))
h.raw_data.mean(dim='time').plot(ax=ax1,**kwargs)
ax1.set_title(f"HOLAPS Mean Latent Heat Flux [{h.raw_data.units}]")
fig1.savefig('figs/holaps_map1.png')
# fig1.savefig('figs/svg/holaps_map1.svg')

fig2,ax2=plt.subplots(figsize=(12,8))
# transpose because longitude/latitude => latitude/longitude for plotting
m.raw_data.mean(dim='time').plot(ax=ax2,**kwargs)
ax2.set_title(f"MODIS Monthly Actual Evapotranspiration [{m.raw_data.units}]")
fig2.savefig('figs/modis_map1.png')
# fig2.savefig('figs/svg/modis_map1.svg')

fig3,ax3=plt.subplots(figsize=(12,8))
g.raw_data.mean(dim='time').plot(ax=ax3,**kwargs)
ax3.set_title(f"GLEAM Monthly mean daily Actual Evapotranspiration [{g.raw_data.units}]")
fig3.savefig('figs/gleam_map1.png')
# fig3.savefig('figs/svg/gleam_map1.svg')

# ------------------------------------------------------------------------------
# Plot the spatial patterns of cleaned data
# ------------------------------------------------------------------------------
kwargs = {"vmin":0,"vmax":3.5}

fig4,ax4=plt.subplots(figsize=(12,8))
h.clean_data.mean(dim='time').plot(ax=ax4,**kwargs)
ax4.set_title(f"HOLAPS Mean Evapotranspiration [{h.clean_data.units}]")
fig4.savefig('figs/holaps_mm_map1.png')
# fig4.savefig('figs/svg/holaps_mm_map1.svg')

fig6,ax6=plt.subplots(figsize=(12,8))
g.clean_data.mean(dim='time').plot(ax=ax6,**kwargs)
ax6.set_title(f"GLEAM Monthly mean daily Actual Evapotranspiration [{g.clean_data.units}]")
fig6.savefig('figs/gleam_msk_map1.png')
# fig5.savefig('figs/svg/holaps_repr_map1.svg')

fig7,ax7=plt.subplots(figsize=(12,8))
m.clean_data.mean(dim='time').plot(ax=ax7,**kwargs)
ax7.set_title(f"MODIS Monthly Actual Evapotranspiration [{modis_msk.units}]")
fig7.savefig('figs/modis_msk_map1.png')
# fig5.savefig('figs/svg/holaps_repr_map1.svg')

#%%
# ------------------------------------------------------------------------------
# Plot the histograms of values
# ------------------------------------------------------------------------------

# GET colors for each variable
h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]

# Plot holaps
# -----------
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

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

import cartopy
import cartopy.feature as cpf


lonmin=32.6
lonmax=51.8
latmin=-5.0
latmax=15.2

ax = plt.figure().gca(projection=cartopy.crs.PlateCarree())
ax.add_feature(cpf.COASTLINE)
ax.add_feature(cpf.BORDERS, linestyle=':')
ax.set_extent([lonmin, lonmax, latmin, latmax])


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
