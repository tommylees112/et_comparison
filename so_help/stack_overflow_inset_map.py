# stack_overflow_inset_map.py
from collections import namedtuple
import numpy as np
import xarray as xr
import shapely
import cartopy

# The data that I have looks like this.
I have a region of interest (defined here as `all_region`). I have an `xr.DataArray` which contains my variable.
What I want to do it to select one PIXEL (lat,lon pair) and to plot a small map in the corner of the lineplot showing here that pixel is located.

```python
Region = namedtuple('Region',field_names=['region_name','lonmin','lonmax','latmin','latmax'])
all_region = Region(
    region_name="all_region",
    lonmin = 32.6,
    lonmax = 51.8,
    latmin = -5.0,
    latmax = 15.2,
)

data = np.random.normal(0,1,(12, 414, 395))
lats = np.linspace(-4.909738, 15.155708, 414)
lons = np.linspace(32.605801, 51.794488, 395)
months = np.arange(1,13)
da = xr.DataArray(data, coords=[months, lats, lons], dims=['month','lat','lon'])
```

# These are the functions that I need to fix to work with inset axes.

I have these functions which plot my timeseries from the xarray object, and also the location of the lat,lon point.
```python
def plot_location(region):
    """ use cartopy to plot the region (defined as a namedtuple object)
    """
    lonmin,lonmax,latmin,latmax = region.lonmin,region.lonmax,region.latmin,region.latmax
    fig = plt.figure()
    ax = fig.gca(projection=cartopy.crs.PlateCarree())
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.set_extent([lonmin, lonmax, latmin, latmax])

    return fig, ax


def select_pixel(ds, loc):
    """ (lat,lon) """
    return ds.sel(lat=loc[1],lon=loc[0],method='nearest')


def turn_tuple_to_point(loc):
    """ (lat,lon) """
    from shapely.geometry.point import Point
    point = Point(loc[1], loc[0])
    return point


def add_point_location_to_map(point, ax, color=(0,0,0,1), **kwargs):
    """ """
    ax.scatter(point.x,
           point.y,
           transform=cartopy.crs.PlateCarree(),
           c=[color],
           **kwargs)
    return
```

# Here I do the plotting

```
# choose a lat lon location that want to plot
loc = (2.407,38.1)

# 1. plot the TIME SERIES FOR THE POINT
fig,ax = plt.subplots()
pixel_da = select_pixel(da, loc)
pixel_da.plot.line(ax=ax, marker='o')

# 2. plot the LOCATION for the point
fig,ax = plot_location(all_region)
point = turn_tuple_to_point(loc)
add_point_location_to_map(point, ax)

```

# I have my function for plotting a region, but I want to put this on an axis in the corner of my figure! Like this:

How would I go about doing this? I have had a look at the [`inset_locator` method](https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html) but as far as I can tell the `mpl_toolkits.axes_grid1.parasite_axes.AxesHostAxes` has no means of assigning a projection, which is required for cartopy.

```python
proj=cartopy.crs.PlateCarree
axins = inset_axes(ax, width="20%", height="20%", loc=2, projection=proj)

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-162-9b5fd4f34c3e> in <module>
----> 1 axins = inset_axes(ax, width="20%", height="20%", loc=2, projection=proj)

TypeError: inset_axes() got an unexpected keyword argument 'projection'
```
