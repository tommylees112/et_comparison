import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy

import itertools

import os

# ------------------------------------------------------------------------------
# Histograms (Marginal Distributions)
# ------------------------------------------------------------------------------



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


# ------------------------------------------------------------------------------
# Temporal Plots
# ------------------------------------------------------------------------------



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


def plot_mean_time(DataArray, ax, add_colorbar=True, **kwargs):
    DataArray.mean(dim='time').plot(ax=ax, **kwargs, add_colorbar=add_colorbar)


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
# Geographical Plotting
# ------------------------------------------------------------------------------

def plot_xarray_on_map(da,borders=True,coastlines=True,**kwargs):
    """ Plot the LOCATION of an xarray object """
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
    ax.add_feature(cartopy.feature.LAKES,facecolor=None);
    fig = plt.gcf()
    ax.outline_patch.set_visible(False)
    return fig, ax


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
