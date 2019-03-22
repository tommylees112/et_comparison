import xarray as xr
import numpy as np

from collections import namedtuple

Region = namedtuple('Region',field_names=['region_name','lonmin','lonmax','latmin','latmax'])

def create_regions():
    """ hardcoded region creator object and """

    all_region = Region(
        region_name="all_region",
        lonmin = 32.6,
        lonmax = 51.8,
        latmin = -5.0,
        latmax = 15.2,
    )

    highlands_region = Region(
        region_name="highlands_region",
        lonmin=35,
        lonmax=40,
        latmin=5.5,
        latmax=12.5
        )

    lake_vict_region = Region(
        region_name="lake_vict_region",
        lonmin=32.6,
        lonmax=38.0,
        latmin=-5.0,
        latmax=2.5
        )

    lowland_region = Region(
        region_name="lowland_region",
        lonmin=32.6,
        lonmax=42.5,
        latmin=0.0,
        latmax=12
    )

    return [all_region, highlands_region, lake_vict_region, lowland_region]


def select_bounding_box_xarray(ds, region):
    """ using the Region namedtuple defined in engineering.regions.py select
    the subset of the dataset that you have defined that region for.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the data (usually from netcdf file) that you want to subset a bounding
         box from
    : region (Region)
        namedtuple object defined in engineering/regions.py

    Returns:
    -------
    : ds (xr.DataSet)
        Dataset with a subset of the whol region defined by the Region object
    """
    print(f"selecting region: {region.name} from ds")
    assert isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray), f"ds Must be an xarray object! currently: {type(ds)}"
    lonmin = region.lonmin
    lonmax = region.lonmax
    latmin = region.latmin
    latmax = region.latmax
    return ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))



# RUN THE create_regions() function to get access to regions
regions = create_regions()
