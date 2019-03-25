"""make_seasonal_averages.py
http://xarray.pydata.org/en/stable/examples/monthly-means.html
"""
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import num2date
import matplotlib.pyplot as plt


dpm = {
    "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
}


def leap_year(year, calendar="standard"):
    """Determine if year is a leap year"""
    leap = False
    if (calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"]) and (
        year % 4 == 0
    ):
        leap = True
        if (
            (calendar == "proleptic_gregorian")
            and (year % 100 == 0)
            and (year % 400 != 0)
        ):
            leap = False
        elif (
            (calendar in ["standard", "gregorian"])
            and (year % 100 == 0)
            and (year % 400 != 0)
            and (year < 1583)
        ):
            leap = False
    return leap


def get_dpm(time, calendar="standard"):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


assert False, "Need to make these into functions"

def compute_weighted_seasonal_anomaly(ds):
    """ Calculate seasonal anomalies taking into account that each month has
         a different number of days.
        NB. Currently implemented to work with daily data
    """
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = xr.DataArray(get_dpm(ds.time.to_index(), calendar='noleap'),
                                coords=[ds.time], name='month_length')

    # Calculate the weights by grouping by 'time.season'.
    # Conversion to float type ('astype(float)') only necessary for Python 2.x
    weights = month_length.groupby('time.season') / month_length.astype(float).groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')

    return ds_weighted


def compare_to_unweighted(ds):
    """ Compare the weighted and unweighted seasonal anomalies """
    ds_weighted = compute_weighted_seasonal_anomaly(ds)

    # only used for comparisons
    ds_unweighted = ds.groupby('time.season').mean('time')
    ds_diff = ds_weighted - ds_unweighted
    return ds_diff


def plot_comparison():
    assert False, "need to make this into a general plotting routine"
    # Quick plot to show the results
    notnull = pd.notnull(ds_unweighted['Tair'][0])

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))
    for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        ds_weighted['Tair'].sel(season=season).where(notnull).plot.pcolormesh(
            ax=axes[i, 0], vmin=-30, vmax=30, cmap='Spectral_r',
            add_colorbar=True, extend='both')

        ds_unweighted['Tair'].sel(season=season).where(notnull).plot.pcolormesh(
            ax=axes[i, 1], vmin=-30, vmax=30, cmap='Spectral_r',
            add_colorbar=True, extend='both')

        ds_diff['Tair'].sel(season=season).where(notnull).plot.pcolormesh(
            ax=axes[i, 2], vmin=-0.1, vmax=.1, cmap='RdBu_r',
            add_colorbar=True, extend='both')

        axes[i, 0].set_ylabel(season)
        axes[i, 1].set_ylabel('')
        axes[i, 2].set_ylabel('')

    for ax in axes.flat:
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.axis('tight')
        ax.set_xlabel('')

    axes[0, 0].set_title('Weighted by DPM')
    axes[0, 1].set_title('Equal Weighting')
    axes[0, 2].set_title('Difference')

    plt.tight_layout()

    fig.suptitle('Seasonal Surface Air Temperature', fontsize=16, y=1.02)

    return fig, ax

# Wrap it into a simple function
def season_mean(ds, calendar='standard'):
    # Make a DataArray of season/year groups
    year_season = xr.DataArray(ds.time.to_index().to_period(freq='Q-NOV').to_timestamp(how='E'),
                               coords=[ds.time], name='year_season')

    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = xr.DataArray(get_dpm(ds.time.to_index(), calendar=calendar),
                                coords=[ds.time], name='month_length')
    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby('time.season').sum(dim='time')
