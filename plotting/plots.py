import numpy as np
import pandas as pd
import xarray as xr

# statistical tests
from scipy import stats
from scipy.stats import pearsonr

# General Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Geographic Plotting
import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cpf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import warnings
import itertools
import os

# Custom functions
from engineering.eng_utils import drop_nans_and_flatten
from engineering.eng_utils import calculate_monthly_mean, calculate_spatial_mean, create_double_year


from engineering.eng_utils import get_unmasked_data


# ------------------------------------------------------------------------------
# Histograms (Marginal Distributions)
# ------------------------------------------------------------------------------

def plot_marginal_distribution(DataArray, color, ax=None, title='', xlabel='DEFAULT', **kwargs):
    """ """
    # if no ax create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))

    # flatten the DataArray
    da_flat = drop_nans_and_flatten(DataArray)
    # plot the histogram
    sns.distplot(da_flat, ax=ax, color=color, **kwargs)
    warnings.warn('Hardcoding the values of the units becuase they should have already been converted to mm day-1')

    if title is None:
        ax.set_title('')
    else:
        title= f'Density Plot of {DataArray.name} [mm day-1]'
        ax.set_title(title)

    if xlabel == 'DEFAULT':
        xlabel = f'Mean Monthly {DataArray.name} [mm day-1]'

    ax.set_xlabel(xlabel)

    if ax is None:
        return fig, ax
    else:
        return ax


# ------------------------------------------------------------------------------
# Joint Distributions
# ------------------------------------------------------------------------------

def plot_hexbin_comparisons(da1, da2, bins=None, mincnt=0.5, title_extra=None):
    """
    Arguments:
    ---------
    : bins (str, int, list, None)
        The binning of the colors for the histogram.
        Can be 'log', None, an integer for dividing into number of bins.
        If a list then used to define the lower bound of the bins to be used
    : mincnt (int, float)
        The minimum count for a color to be shown
    """
    data_array1 = drop_nans_and_flatten(da1)
    data_array2 = drop_nans_and_flatten(da2)

    var_dataset_x = data_array1
    var_dataset_y = data_array2
    r_value = pearsonr(data_array1,data_array2)

    fig, ax = plt.subplots(figsize=(12,8))

    # plot the data
    hb = ax.hexbin(var_dataset_x, var_dataset_y, bins=bins, gridsize=40, mincnt=mincnt)

    # draw the 1:1 line (showing datasets exactly the same)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label="1:1")

    # axes options
    dataset_name_x = da1.name.split('_')[0]
    dataset_name_y = da2.name.split('_')[0]
    title = f"Evapotranspiration: {dataset_name_x} vs. {dataset_name_y} \n Pearsons R: {r_value[0]:.2f} \n {title_extra}"

    ax.set_xlabel(dataset_name_x)
    ax.set_ylabel(dataset_name_y)
    ax.set_title(title)

    # colorbar
    cb = fig.colorbar(hb, ax=ax)
    if bins == 'log':
        cb.set_label('log10(counts)')
    else:
        cb.set_label('counts')

    title = f"{title_extra}{dataset_name_x}_v_{dataset_name_y}{bins}_{mincnt}_hexbin.png"
    return fig, title


def hexbin_jointplot_sns(d1, d2, col1, col2, bins='log', mincnt=0.5, xlabel='', ylabel=''):
    """
    Arguments:
    ---------
    : da1 (np.ndarray)
        numpy array of data (should be same lengths!)
    : da2 (np.ndarray)
        numpy array of data (should be same lengths!)
    : col1 (tuple)
        seaborn color code as tuple (rgba) e.g. `sns.color_palette()[0]`
    : col2 (tuple)
        seaborn color code as tuple (rgba) e.g. `sns.color_palette()[0]`
    : bins (str,list,None)
        how to bin your variables. If str then should be 'log'
    : mincnt (int, float)
        the minimum count for a value to be shown on the plot
    """
    assert False, "Need to implement a colorbar and fix the colorbar values for all products (shouldn't matter too much because all products now have teh exact same number of pixels)"
    jp = sns.jointplot(d1, d2, kind="hex", joint_kws=dict(bins=bins, mincnt=mincnt))
    jp.annotate(stats.pearsonr)

    # plot the 1:1 line
    ax = jp.ax_joint
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label="1:1")

    # color the marginal distributions separately
    for patch in jp.ax_marg_x.patches:
        patch.set_facecolor(col1)

    for patch in jp.ax_marg_y.patches:
        patch.set_facecolor(col2)

    # label the axes appropriately
    jp.ax_joint.set_xlabel(xlabel)
    jp.ax_joint.set_ylabel(ylabel)

    plt.tight_layout()
    return jp


def plot_joint_plot_hex1(da1, da2, col1, col2, bins='log', xlabel='da1', ylabel='da2', mincnt=0.5):
    """
    Arguments:
    ---------
    : da1 (np.ndarray)
        numpy array of data (should be same lengths!)
    : da2 (np.ndarray)
        numpy array of data (should be same lengths!)
    : col1 (tuple)
        seaborn color code as tuple (rgba)
    : col2 (tuple)
        seaborn color code as tuple (rgba)
    """
    assert False, "This method uses the more basic functions of sns.JointGrid to construct the hexbin joint plot. The issue is that it has a very odd way of selecting the number of bins for the marginal histograms. You should not use this function but use the other `hexbin_jointplot_sns` function which will 'intelligently' select the number of bins for you. Keeping here for longevity. Future me might want to edit this more"
    g = sns.JointGrid(x=da1, y=da2)
    g.plot_joint(plt.hexbin, bins='log', mincnt=mincnt) #plt.scatter) # color="m", edgecolor="white")
    g.annotate(stats.pearsonr)
    ax = g.ax_joint
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label="1:1")

    # assert False, "number of bins!"
    # n_bins = int(len(da1) / 1000)
    _ = g.ax_marg_x.hist(da1, color=col1, alpha=.6) # ,bins=n_bins)
    _ = g.ax_marg_y.hist(da2, color=col2, alpha=.6, orientation="horizontal") # ,bins=n_bins)

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)

    g = g.annotate(pearsonr)

    fig = plt.gcf()
    return fig, g

# ------------------------------------------------------------------------------
# Spatial Plots
# ------------------------------------------------------------------------------

def plot_mean_time(DataArray, ax, add_colorbar=True, **kwargs):
    """ plot the SPATIAL variability by collapsing the 'time' dimension

    NOTE: must have 'time' in the coordinate dimensions of the xr.DataArray
    """

    DataArray.mean(dim='time').plot(ax=ax, **kwargs, add_colorbar=add_colorbar)
    return


def plot_all_spatial_means(ds, area):
    """ For a given subsetted xr.Dataset `ds`, plot the temporal mean spatial
    plot.

    Arguments:
    ---------
    : ds (xr.Dataset)
        needs to have the following data variables:
            holaps_evapotranspiration
            modis_evapotranspiration
            gleam_evapotranspiration
            chirps_precipitation
    : area (str)
        Name of the region of interest (for title and plot names)
    """
    h = ds.holaps_evapotranspiration.mean(dim='time')
    fig,ax = plot_xarray_on_map(h)
    ax.set_title(f'{area}')
    fig.savefig(f'figs/holaps_temporal_mean_{area}.png')
    m = ds.modis_evapotranspiration.mean(dim='time')
    fig,ax = plot_xarray_on_map(m)
    ax.set_title(f'{area}')
    fig.savefig(f'figs/modis_temporal_mean_{area}.png')
    g = ds.gleam_evapotranspiration.mean(dim='time')
    fig,ax = plot_xarray_on_map(g)
    ax.set_title(f'{area}')
    fig.savefig(f'figs/gleam_temporal_mean_{area}.png')
    c = ds.chirps_precipitation.mean(dim='time')
    fig,ax = plot_xarray_on_map(d)
    ax.set_title(f'{area}')
    fig.savefig(f'figs/chirps_temporal_mean_{area}.png')
    return






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

# ------------------------------------------------------------------------------
# Temporal Plots
# ------------------------------------------------------------------------------


def plot_seasonality(ds, ylabel=None, double_year=False):
    """ """
    mthly_ds = calculate_monthly_mean(ds)
    seasonality = calculate_spatial_mean(mthly_ds)

    if double_year:
        seasonality = create_double_year(seasonality)

    fig, ax = plt.subplots(figsize=(12,8))
    seasonality.to_dataframe().plot(ax=ax)
    ax.set_title('Spatial Mean Seasonal Time Series')
    plt.legend()

    if ylabel != None:
        ax.set_ylabel(ylabel)

    return fig, ax


def plot_normalised_seasonality(ds, double_year=False):
    """ Normalise the seasonality by each months contribution to the annual mean total.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the dataset to calculate the seasonality from
    : double_year (bool)
        if True then show two annual cycles to get a better picture of the
         seasonality.
    """
    fig, ax = plt.subplots(figsize=(12,8))
    mthly_ds = calculate_monthly_mean(ds)
    norm_seasonality = monthly_ds.apply(lambda x: (x / x.sum(dim='month'))*100)

    if double_year:
        norm_seasonality = create_double_year(norm_seasonality)
    # convert to dataframe (useful for plotting values)
    norm_seasonality.to_dataframe().plot(ax=ax)
    ax.set_title('Normalised Seasonality')
    ax.set_ylabel('Contribution of month to annual total (%)')
    plt.legend()

    return fig



# ------------------------------------------------------------------------------
# Spatio-temporal plots
# ------------------------------------------------------------------------------

def plot_seasonal_spatial_means(seasonal_da, **kwargs):
    """ for a given seasonal xarday object plot the 4 seasons spatial means"""
    assert 'season' in [key for key in seasonal_da.coords.keys()], f"'season' should be a coordinate in the seasonal_da object for using this plotting functionality. \n Currently: {[key for key in seasonal_da.coords.keys()]}"
    assert isinstance(seasonal_da, xr.DataArray), f"seasonal_da should be of type: xr.DataArray. Currently: {type(seasonal_da)}"
    scale=1
    fig,axs = plt.subplots(2,2,figsize=(12*scale,8*scale))
    try:
        var = seasonal_da.name
    except:
        assert False, "sesaonal_da needs to be named!"
    for i in range(4):
        ax = axs[np.unravel_index(i,(2,2))]
        seasonal_da.isel(season=i).plot(ax=ax, **kwargs)
        season_str = str(seasonal_da.isel(season=i).season.values)
        ax.set_title(f"{var} {season_str}")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# Differences
# - plot_mean_spatial_differences_ET
# - plot_seasonal_comparisons_ET_diff
# ------------------------------------------------------------------------------

def get_variables_for_comparison1():
    """ Return the variables for intercomparison ()
    REturns:
    -------
    : variables (list)
        list of strings of the variable names to create all combinations of
    : comparisons (list)
        list of strings. combinations of all the variables listed in variables
    """
    import itertools
    variables = [
     'holaps_evapotranspiration',
     'gleam_evapotranspiration',
     'modis_evapotranspiration',
    ]
    comparisons = [i for i in itertools.combinations(variables,2)]
    return variables, comparisons




def plot_mean_spatial_differences_ET(ds, **kwargs):
    """ """
    # TODO: make this more dynamic and less hard-coded
    variables, comparsions = get_variables_for_comparison1()
    fig,axs = plt.subplots(1,3, figsize=(15,12))

    for i, cmprson in enumerate(comparisons):
        # calculate the difference between the variables
        diff = ds[cmprson[0]] - ds[cmprson[1]]
        ax = axs[i]
        # plot the temporal mean (MAP)
        if i!=3:
            plot_mean_time(diff, ax, add_colorbar=False, **kwargs)
        else:
            plot_mean_time(diff, ax, add_colorbar=True, **kwargs)

        # set the axes options
        ax.set_title(f"{cmprson[0].split('_')[0]} - {cmprson[1].split('_')[0]} Temporal Mean")
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.suptitle('Comparison of Spatial Means Between Products')
    return fig


def compute_difference(ds, cmprson1, cmprson2):
    ds_diff = ds[cmprson1] - ds[cmprson2]
    return ds_diff


def plot_seasonal_comparisons_ET_diff(seasonal_ds, **kwargs):
    """ """
    assert 'season' in [key for key in seasonal_ds.coords.keys()], f"'season' should be a coordinate in the seasonal_ds object for using this plotting functionality. \n Currently: {[key for key in seasonal_ds.coords.keys()]}"
    assert isinstance(seasonal_ds, xr.Dataset), f"seasonal_ds should be of type: xr.Dataset. Currently: {type(seasonal_ds)}"

    fig, axs = plt.subplots(2,6, figsize=(15,12))
    # NOTE: hardcoding the axes indexes because haven't figured out the logic
    axes_ix = [(0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3),(0,4),(0,5),(1,4),(1,5)]

    ix_counter = 0

    _, comparisons = get_variables_for_comparison1()
    for i, cmprson in enumerate(comparisons):
        # for each comparison calculate the SEASONAL difference
        seas_diff = compute_difference(seasons, cmprson[0], cmprson[1])
        # plot each season
        for j in range(4):
            # get the correct axis to plot on
            ax_ix = axes_ix[ix_counter]
            ax = axs[ax_ix]
            ix_counter += 1

            # extract just ONE seasons data to plot
            da = seas_diff.isel(season=j)
            # plot the seasonal mean
            da.plot(ax=ax, add_colorbar=False, **kwargs)
            # plot_mean_time(da, ax, add_colorbar=False, **kwargs)
            # get the name of that season
            season_str = str(seas_diff.isel(season=j).season.values)

            # set axes options
            ax.set_title(f"{cmprson[0].split('_')[0]} - {cmprson[1].split('_')[0]} {season_str}")
            ax.set_xlabel('')
            ax.set_ylabel('')

    plt.tight_layout()

    return fig

# ------------------------------------------------------------------------------
# Geographical Plotting (cartopy helpers)
# ------------------------------------------------------------------------------

def plot_xarray_on_map(da,borders=True,coastlines=True,**kwargs):
    """ Plot the LOCATION of an xarray object """
    # get the center points for the maps
    mid_lat = np.mean(da.lat.values)
    mid_lon = np.mean(da.lon.values)
    # create the base layer
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(mid_lon, mid_lat))
    # ax = plt.axes(projection=cartopy.crs.Orthographic(mid_lon, mid_lat))

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    da.plot(ax=ax, transform=cartopy.crs.PlateCarree(),vmin=vmin, vmax=vmax);

    ax.coastlines();
    ax.add_feature(cartopy.feature.BORDERS,linestyle=':');
    ax.add_feature(cartopy.feature.LAKES,facecolor=None);
    fig = plt.gcf()
    ax.outline_patch.set_visible(False)
    return fig, ax



def get_river_features():
    """ Get the 10m river features from NaturalEarth and turn into shapely.geom
    Note: https://github.com/SciTools/cartopy/issues/945

    """
    shp_path = cartopy.io.shapereader.natural_earth(
        resolution='10m',
        category='physical',
        name='rivers_lake_centerlines')

    water_color = '#3690f7'
    shp_contents = cartopy.io.shapereader.Reader(shp_path)
    river_generator = shp_contents.geometries()
    river_feature = cartopy.feature.ShapelyFeature(
        river_generator,
        cartopy.crs.PlateCarree(),
        edgecolor=water_color,
        facecolor='none')

    return river_feature



def plot_geog_location(region, lakes=False, borders=False, rivers=False):
    """ use cartopy to plot the region (defined as a namedtuple object)

    Arguments:
    ---------
    : region (Region namedtuple)
        region of interest bounding box defined in engineering/regions.py
    : lakes (bool)
        show lake features
    : borders (bool)
        show lake features
    : rivers (bool)
        show river features (@10m scale from NaturalEarth)
    """
    lonmin,lonmax,latmin,latmax = region.lonmin,region.lonmax,region.latmin,region.latmax
    ax = plt.figure().gca(projection=cartopy.crs.PlateCarree())
    ax.add_feature(cartopy.feature.COASTLINE)
    if borders:
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    if lakes:
        ax.add_feature(cartopy.feature.LAKES)
    if rivers:
        # assert False, "Rivers are not yet working in this function"
        river_feature = get_river_features()
        ax.add_feature(river_feature)

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

    return fig, ax


def add_points_to_map(ax, geodf, point_colors="#0037ff"):
    """ Add the point data stored in `geodf.geometry` as points to ax
    Arguments:
    ---------
    : geodf (geopandas.GeoDataFrame)
        gpd.GeoDataFrame with a `geometry` column containing shapely.Point geoms
    : ax (cartopy.mpl.geoaxes.GeoAxesSubplot)
    """
    assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot), f"Axes need to be cartopy.mpl.geoaxes.GeoAxesSubplot. Currently: {type(ax)}"
    points = geodf.geometry.values
    ax.scatter([point.x for point in points],
               [point.y for point in points],
               transform=cartopy.crs.PlateCarree(),
               color=point_colors)

    return ax


def plot_stations_on_region_map(region, station_location_df, point_colors="#0037ff"):
    """ Plot the station locations in `station_location_df` on a map of the region

    Arguments:
    ---------
    : region (Region, namedtuple)
    : station_location_df (geopandas.GeoDataFrame)
        gpd.GeoDataFrame with a `geometry` column containing shapely.Point geoms

    Returns:
    -------
    : fig (matplotlib.figure.Figure)
    : ax (cartopy.mpl.geoaxes.GeoAxesSubplot)
    """
    fig, ax = plot_geog_location(region, lakes=True, borders=True, rivers=True)
    ax =  add_points_to_map(ax, station_location_df, point_colors=point_colors)

    return fig, ax


def add_sub_region_box(ax, subregion, color):
    """ Plot a box for the subregion on the cartopy axes.
    TODO: implement a check where the subregion HAS TO BE inside the axes limits

    Arguments:
    ---------
    : ax (cartopy.mpl.geoaxes.GeoAxesSubplot)
        axes that you are plotting on
    : subregion (Region namedtuple)
        region of interest bounding box defined in engineering/regions.py
    """
    geom = geometry.box(minx=subregion.lonmin,maxx=subregion.lonmax,miny=subregion.latmin,maxy=subregion.latmax)
    ax.add_geometries([geom], crs=cartopy.crs.PlateCarree(), color=color, alpha=0.3)
    return ax


def plot_all_regions(regions):
    """
    : regions (list, tuple)
        list of Region objects to plot geographic locations
    """
    for region in regions:
        fig = plot_geog_location(region, rivers=True, borders=True)
        plt.gca().set_title(region.name)
        fig.savefig(f'figs/{region.name}.png')

    return

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
