import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import itertools

import os

# ------------------------------------------------------------------------------
# Histograms (Marginal Distributions)
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Joint Distributions
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Spatial Plots
# ------------------------------------------------------------------------------



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
# ------------------------------------------------------------------------------

def get_variables_for_comparison1():
    """ Return the variables for intercomparison () """
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


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
