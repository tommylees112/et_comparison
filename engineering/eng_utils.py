# eng_utils.py
import pandas as pd
import numpy as np
import xarray as xr
import os


# ------------------------------------------------------------------------------
# Collapsing Time Dimensions 
# ------------------------------------------------------------------------------


def calculate_monthly_mean(ds):
    assert 'time' in [dim for dim in ds.dims.keys()], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby('time.month').mean(dim='time')


def calculate_spatial_mean(ds):
    assert ('lat' in [dim for dim in ds.dims.keys()]) & ('lon' in [dim for dim in ds.dims.keys()]), f"Must have 'lat' 'lon' in the dataset dimensisons"
    return ds.mean(dim=['lat','lon'])


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
