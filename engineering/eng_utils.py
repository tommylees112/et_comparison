# eng_utils.py
import pandas as pd
import numpy as np
import xarray as xr
import os
import pickle
from pathlib import Path

# ------------------------------------------------------------------------------
# Working with Masks (subsets of your data)
# ------------------------------------------------------------------------------

def get_unmasked_data(dataArray, dataMask):
    """ get the data INSIDE the dataMask
    Keep values if True, remove values if False
     (doing the opposite of a 'mask' - perhaps should rename)
    """
    return dataArray.where(dataMask)


def create_new_binned_dimensions(ds, group_var, intervals):
    """ Get the values in `ds` for `group_var` WITHIN the `interval` ranges.
         Return a new xr.Dataset with a new set of variables called `{group_var}_bins`.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the dataset in which we are finding the values that lie within an interval
         range.
    : group_var (str)
        the variable that we are trying to bin
    : intervals (list, np.ndarray)
        list of `pandas._libs.interval.Interval` with methods `interval.left`
         and `interval.right` for extracting the values that fall within that
         range.

    Returns:
    -------
    : ds_bins (xr.Dataset)
        dataset with new `Variables` one for each bin. Pixels outside of the
         interval range are masked with np.nan
    """
    ds_bins = xr.concat([ds.where(
                             (ds[group_var] > interval.left) & (ds[group_var] < interval.right)
                           )
                    for interval in intervals
                   ]
    )
    ds_bins = ds_bins.rename({'concat_dims':f'{group_var}_bins'})
    return ds_bins



def bin_dataset(ds, group_var, n_bins):
    """
    Arguments:
    ---------
    : ds (xr.Dataset)
        the dataset that you want to group / bin
    : group_var (str)
        the data variable that you want to group into bins

    Returns:
    -------
    : topo_bins (xr.Dataset)
        dataset object with number of variables equal to the number of bins
    : intervals (tuple)
        tuple of tuples with the bin left and right edges
         (intervals[0][0] = left edge;
          intervals[0][0] = right edge
         )
    """
    # groupby and collaps to the MID ELEVATION for the values (allows us to extract )
    bins = ds.groupby_bins(group=group_var,bins=n_bins).mean()
    # assert False, "hardcoding the elevation_bins here need to do this dynamically"
    binned_var = [key for key in bins.coords.keys()]
    assert len(binned_var) == 1, "The binned Var should only be one variable!"
    binned_var = binned_var[0]

    # extract the bin locations
    intervals = bins[binned_var].values
    left_bins = [interval.left for interval in intervals]
    # use bin locations to create mask variables of those values inside that
    ds_bins = create_new_binned_dimensions(ds, group_var, intervals)
    # ds_bins = xr.concat([ds.where(
    #                              (ds[group_var] > interval.left) & (ds[group_var] < interval.right)
    #                            )
    #                     for interval in intervals
    #                    ]
    # )
    # ds_bins = ds_bins.rename({'concat_dims':f'{group_var}_bins'})

    return ds_bins, intervals



# ------------------------------------------------------------------------------
# Collapsing Time Dimensions
# ------------------------------------------------------------------------------


def calculate_monthly_mean(ds):
    assert 'time' in [dim for dim in ds.dims.keys()], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby('time.month').mean(dim='time')


def calculate_spatial_mean(ds):
    assert ('lat' in [dim for dim in ds.dims.keys()]) & ('lon' in [dim for dim in ds.dims.keys()]), f"Must have 'lat' 'lon' in the dataset dimensisons"
    return ds.mean(dim=['lat','lon'])


def create_double_year(seasonality):
    """for seasonality data (values for each month) return a copy for a second
        year to account for the cross-over between DJF

    Returns:
    -------
    : (xr.Dataset)
        a Dataset object with 24 months (2 annual cycles)
    """
    assert 'month' in [coord for coord in seasonality.coords.keys()], f"`month` must be a present coordinate in the seasonality data passed to the `create_double_year` function! Currently: {[coord for coord in seasonality.coords.keys()]}"

    seas2 = seasonality.copy()
    seas2['month'] = np.arange(13,25)

    # merge the 2 datasets
    return xr.merge([seasonality, seas2])


# ------------------------------------------------------------------------------
# Lookup values from xarray in a dict
# ------------------------------------------------------------------------------

def replace_with_dict(ar, dic):
    """ Replace the values in an np.ndarray with a dictionary

    https://stackoverflow.com/a/47171600/9940782

    """
    assert isinstance(ar, np.ndarray), f"`ar` shoule be a numpy array! (np.ndarray). To work with xarray objects, first select the values and pass THESE to the `replace_with_dict` function (ar = da.values) \n Type of `ar` currently: {type(ar)}"
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    # NOTE: something going wrong with the number for the indices (0 based vs. 1 based)
    warnings.warn('We are taking one from the index. need to check this is true!!!')
    return v[sidx[ np.searchsorted(k, ar, sorter=sidx) -1 ] ]



def replace_with_dict2(ar, dic):
    """Replace the values in an np.ndarray with a dictionary

    https://stackoverflow.com/a/47171600/9940782
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    warnings.warn('We are taking one from the index. need to check this is true!!!')
    return vs[np.searchsorted(ks,ar) -1 ]



def get_lookup_val(xr_obj, variable, new_variable, lookup_dict):
    """
    Arguments:
    ---------
    : xr_obj (xr.Dataset, xr.DataArray)
        the xarray object we want to look values up from
    : variable (str)
        the INPUT variable we are hoping to look the values up from (the dictionary keys)
    : new_variable (str)
        the name of the OUTPUT variable we want to put the dictionary values in
    : lookup_dict (dict)
        the dictionary we want to lookup the values of 'variable' in to return values to 'new_variable'
    """
    # get the values as a numpy array
    if isinstance(xr_obj, xr.Dataset):
        ar = xr_obj[variable].values
    elif isinstance(xr_obj, xr.DataArray):
        ar = xr_obj.values
    else:
        assert False, f"This function only works with xarray objects. Currently xr_obj is type: {type(xr_obj)}"

    assert isinstance(ar, np.ndarray), f"ar should be a numpy array!"
    assert isinstance(lookup_dict, dict), f"lookup_dict should be a dictionary object!"

    # replace values in a numpy array with the values from the lookup_dict
    new_ar = replace_with_dict2(ar, lookup_dict)

    # assign the values looked up from the dictionary to a new variable in the xr_obj
    ipdb.set_trace()
    xr_obj[new_variable] = new_ar

    return xr_obj

# ------------------------------------------------------------------------------
# creating histograms
# ------------------------------------------------------------------------------

def drop_nans_and_flatten(dataArray):
    """flatten the array and drop nans from that array. Useful for plotting histograms.

    Arguments:
    ---------
    : dataArray (xr.DataArray)
        the DataArray of your value you want to flatten
    """
    # drop NaNs and flatten
    return dataArray.values[~np.isnan(dataArray.values)]




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


# ------------------------------------------------------------------------------
# General utils
# ------------------------------------------------------------------------------

def pickle_files(filepaths, vars):
    """ """
    assert len(filepaths) == len(vars), f"filepaths should be same size as vars because each variable needs a filepath! currently: len(filepaths): {len(filepaths)} len(vars): {len(vars)}"

    for filepath in filepaths:
        with open(filepath, 'wb') as f:
            pickle.dump(intervals, f)
