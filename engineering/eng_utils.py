# eng_utils.py
import pandas as pd
import numpy as np
import xarray as xr
import os
import pickle
from pathlib import Path
import warnings
import itertools

# package for comparing soil moisture datasets:
# https://pytesmo.readthedocs.io/en/latest/introduction.html

# ------------------------------------------------------------------------------
# Working with spatial analysis
# ------------------------------------------------------------------------------
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



# ------------------------------------------------------------------------------
# Working with Time Variables
# ------------------------------------------------------------------------------


def compute_anomaly(da, time_group='time.month'):
    """ Return a dataarray where values are an anomaly from the MEAN for that
         location at a given timestep. Defaults to finding monthly anomalies.

    Arguments:
    ---------
    : da (xr.DataArray)
    : time_group (str)
        time string to group.
    """
    mthly_vals = da.groupby(time_group).mean('time')
    da = da.groupby(time_group) - mthly_vals

    return da




# ------------------------------------------------------------------------------
# Working with Comparisons
# ------------------------------------------------------------------------------

def get_variables_for_comparison1():
    """ Return the variables for intercomparison (HARDCODED)
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


def get_variables_for_comparison2(ds):
    """ Return the variables for intercomparison for variables from ds
    REturns:
    -------
    : variables (list)
        list of strings of the variable names to create all combinations of
    : comparisons (list)
        list of strings. combinations of all the variables listed in variables
    """
    import itertools
    datasets = ['holaps', 'gleam', 'modis']
    coords = [coord for coord in ds.coords.keys()]
    variables = np.array([var_ for var_ in ds.variables.keys() if var_ not in coords])

    # TODO: this needs some fixing/love to be general
    # currently only working with 3 evaporation products so mask out precip vals
    valid_vars = [any(dst in var for dst in datasets) for var in variables]
    variables = variables[valid_vars]

    comparisons = [i for i in itertools.combinations(variables,2)]
    return variables, comparisons


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

    return ds_bins, intervals


def mask_multiple_conditions(da, vals_to_keep):
    """
    Arguments:
    ---------
    : da (xr.DataArray)
        data that you want to mask
    : variable (str)
        variable to search for the values in vals_to_keep
    : vals_to_keep (list)
        list of values to keep from variable

    Returns:
    -------
    : msk (xr.DataArray)
        a mask showing True for matches and False for non-matches

    Note: https://stackoverflow.com/a/40556458/9940782
    """
    msk = xr.DataArray(np.in1d(da, vals_to_keep).reshape(da.shape),
                       dims=da.dims, coords=da.coords)

    return msk


#
# Extracting individual pixels
#

def select_pixel(ds, loc):
    """ (lat,lon) """
    return ds.sel(lat=loc[1],lon=loc[0],method='nearest')


def turn_tuple_to_point(loc):
    """ (lat,lon) """
    from shapely.geometry.point import Point
    point = Point(loc[1], loc[0])
    return point



# ------------------------------------------------------------------------------
# Collapsing Time Dimensions
# ------------------------------------------------------------------------------


def caclulate_std_of_mthly_seasonality(ds,double_year=False):
    """Calculate standard deviataion of monthly variability """
    std_ds = calculate_monthly_std(ds)
    seasonality_std = calculate_spatial_mean(std_ds)

    # rename vars
    var_names = get_non_coord_variables(seasonality_std)
    new_var_names = [var + "_std" for var in var_names]
    seasonality_std = seasonality_std.rename(dict(zip(var_names, new_var_names)))

    #
    if double_year:
        seasonality_std = create_double_year(seasonality_std)

    return seasonality_std


def calculate_monthly_mean(ds):
    assert 'time' in [dim for dim in ds.dims.keys()], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby('time.month').mean(dim='time')


def calculate_monthly_std(ds):
    assert 'time' in [dim for dim in ds.dims.keys()], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby('time.month').std(dim='time')


def calculate_monthly_mean_std(ds):
    """ """
    # calculate mean and std
    mean = calculate_monthly_mean(ds)
    std = calculate_monthly_std(ds)

    # get var names
    dims = [dim for dim in mean.dims.keys()]
    vars = [var for var in mean.variables.keys() if var not in dims]

    # rename vars so can return ONE ds
    mean_vars = [var+'_monmean' for var in vars]
    std_vars = [var+'_monstd' for var in vars]
    mean = mean.rename(dict(zip(vars, mean_vars)))
    std = std.rename(dict(zip(vars, std_vars)))

    return xr.merge([mean, std])


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


# TODO: rename this function
def get_lookup_val(xr_obj, variable, new_variable, lookup_dict):
    """ Assign a new Variable to xr_obj with values from lookup_dict.
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
    new_da = xr.DataArray(new_ar, coords=[xr_obj.lat, xr_obj.lon], dims=['lat', 'lon'])
    new_da.name = new_variable
    xr_obj = xr.merge([xr_obj, new_da])

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

    for i, filepath in enumerate(filepaths):
        save_pickle(filepath, variable)


def load_pickle(filepath):
    """ """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(filepath, variable):
    """ """
    with open(filepath, 'wb') as f:
        pickle.dump(variable, f)
    return


def get_non_coord_variables(ds):
    """ Return a list of the variable names EXCLUDING the coordinates (lat,lon,time) """
    var_names = [var for var in ds.variables.keys() if var not in ds.coords.keys()]
    return var_names
