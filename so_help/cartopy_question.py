# cartopy_question.py
import cartopy
import cartopy.feature as cpf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely import geometry
from collections import namedtuple
from shapely.geometry.polygon import LinearRing

Region = namedtuple('Region',field_names=['region_name','lonmin','lonmax','latmin','latmax'])

region = Region(
    region_name="all_region",
    lonmin = 32.6,
    lonmax = 51.8,
    latmin = -5.0,
    latmax = 15.2,
)
sub_region =  Region(
        region_name="highlands_region",
        lonmin=35,
        lonmax=40,
        latmin=5.5,
        latmax=12.5
)


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
    ax.add_feature(cpf.COASTLINE)
    if borders:
        ax.add_feature(cpf.BORDERS, linestyle=':')
    if lakes:
        ax.add_feature(cpf.LAKES)
    if rivers:
        # assert False, "Rivers are not yet working in this function"
        water_color = '#3690f7'
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


fig, ax = plot_geog_location(region,borders=True, lakes=True, rivers=False)



def plot_polygon(ax, sub_region):
    """
    https://groups.google.com/forum/#!topic/scitools-iris/LxR0EbQolyE
    Note:
    ----
    order is important:
        lower-left, upper-left, upper-right, lower-right
        2 -- 3
        |    |
        1 -- 4
    """
    # ax = fig.axes[0]
    lons = [sub_region.latmin, sub_region.latmin, sub_region.latmax, sub_region.latmax]
    lats = [sub_region.lonmin, sub_region.lonmax, sub_region.lonmax, sub_region.lonmin]
    ring = LinearRing(list(zip(lons, lats)))
    ax.add_geometries([ring], cartopy.crs.PlateCarree(), facecolor='b', edgecolor='black', alpha=0.5)
    return ax


def add_sub_region_box(ax, subregion):
    """ """
    geom = geometry.box(minx=subregion.lonmin,maxx=subregion.lonmax,miny=subregion.latmin,maxy=subregion.latmax)
    ax.add_geometries(geom, crs=cartopy.crs.PlateCarree(), alpha=0.3)
    return ax


fig, ax = plot_geog_location(region,borders=True, lakes=True, rivers=False)
plot_polygon(ax, sub_region)


fig, ax = plot_geog_location(region,borders=True, lakes=True, rivers=False)
add_sub_region_box(ax, sub_region)
