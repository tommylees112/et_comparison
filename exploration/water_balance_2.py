"""
# water_balance_calcs.py

: wsheds (xr.Dataset)
    evaporation & precipitation values with a watershed variable determining the areas of the watersheds for different locations

: lgt_stations (pd.DataFrame)
    the raw daily runoff (monthly) in [mm day-1] for the stations we have data for

: station_lkup (gpd.GeoDataFrame)
    the metadata associated with the stations that we have data for (LEGIT STATIONS)


:  wsheds_shp (gpd.GeoDataFrame)


: wshed_masks (xr.Dataset)
    netcdf/xr.Dataset of the station watersheds as boolean fields
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import geopandas as gpd
import shapely
from pathlib import Path

BASE_FIG_DIR = Path('/soge-home/projects/crop_yield/et_comparison/figs/meeting6')

# pour point id to geoid
pp_to_geoid_map = {
    # blue_nile
    17 : 'G1686',
    16 : 'G1685',
    # aw,
    3 : 'G1067',
    2 : 'G1074',
    1 : 'G1053',
    0 : 'G1045',
    # ??,
    4 : "G1603",
}

pp_to_polyid_map = {
    'G1686': [ 1,  0,  3, 10],
    'G1685': [ 3, 10],
    'G1067': [15],
    'G1074': [15,  7],
    'G1053': [15,  7,  5],
    'G1045': [15,  7,  5,  2],
    'G1603': [11]
}


my_stations = [
    'G1045',
    'G1053',
    'G1067',
    'G1074',
    'G1603',
    'G1685',
    'G1686'
]



# #
# pp_to_poly_map
#
# # construct other dictionaries
# geoid_to_pp_map = dict(zip(pp_to_geoid_map.values(), pp_to_geoid_map.keys()))
# geoid_to_poly_map = dict(zip(geoid_to_pp_map.keys(), [pp_to_poly_map[val] for val in geoid_to_pp_map.values()]
# ))

# READ in watersheds

def read_station_flow_data():
    """ """
    # read raw data
    df = pd.read_csv(BASE_DATA_DIR / 'Qts_Africa_glofas_062016_1971_2005.csv')
    df.index = pd.to_datetime(df.DATE)
    df = df.drop(columns='DATE')
    # select the date range
    df = df['2001-01-01':'2005-12-31']
    df = df.dropna(how='all',axis=1)

    return df


def read_station_metadata():
    """
    Columns of lookup_gdf:
    ---------------------
    ID :            station ID
    StationName :
    RiverName :
    RiverBasin :    basin name
    Country :
    CountryNam :
    Continent :
    Class :
    DrainArLDD :     Drainage Area Local Drain Direction (LDD)
    YCorrected :     latitude
    XCorrected :     longitude
    geometry :       (shapely.Geometry)
    """
    # gpd.read_file(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
    lookup_df = pd.read_csv(BASE_DATA_DIR / 'Qgis_GHA_glofas_062016_forTommy.csv')
    lookup_gdf = read_csv_point_data(lookup_df, lat_col='YCorrected', lon_col='XCorrected')
    lookup_gdf['corrected_river_name'] = lookup_gdf.RiverName.apply(str.lower)
    lookup_gdf['NAME'] = lookup_gdf.index

    return lookup_gdf


def read_watershed_shp():
    """ """
    shp_path = BASE_DATA_DIR / "marcus_help" / "watershed_areas_shp" / "Watershed_Areas.shp"
    coord_name = "watershed_for_pourpoint"
    wsheds_shp = gpd.read_file(shp_path)
    wsheds_shp = wsheds_shp.rename(columns={'OBJECTID':'polygon_id', 'gridcode':'pour_point_id'})
    return wsheds_shp


# change the units to equivalent for region
def calculate_flow_per_day(df, lookup_gdf):
    """ convert flow in m3/s => mm/day in new columns, `colnames` = ID + '_perday'

    value = runoff / (size(km2) * 1e6 (=>m2)) * 1000 (# mm in m) * 86,400 (# s in day)
    Steps:
    1) normalise per unit area (DrainArLDD = km^2)
        runoff (m3) / (km2 * 1e6)
    2) Convert m => mm
        * 1000
    3) convert s => days
        * 86,400
    """
    for ID in lookup_gdf.ID:
        drainage_area = lookup_gdf.query(f'ID == "{ID}"').DrainArLDD.values[0]
        # TODO: what units is DrainArLDD in?
        # df[ID+'_norm'] = df[ID].apply(lambda runoff: ((runoff*1e9) / 86_400) / drainage_area )
        print('Converting to [mm day-1] using:\n value = runoff / (size(km2) * 1e6 (=>m2)) * 1000 (# mm in m) * 86,400 (# s in day)')
        df[ID] = df[ID].apply(lambda runoff: ((runoff/(drainage_area * 1e6)) * 86400 * 1000)  )

    return df


def create_basins_merged_map(wsheds, pp_to_polyid_map):
    """
    Arguments:
    ---------
    : wsheds (xr.Dataset)
        data for the precipitation and evaporation estimates for different
         products.

    : pp_to_polyid_map (dict)
        a dictionary mapping the station id (keys) to the basin_ids for the
         `watershed_for_pourpoint` variable in the wsheds data (values).
    """
    # CREATE MERGED basin_products
    basins_mask_map= {
        'G1045': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1045']),
        'G1053': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1053']),
        'G1067': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1067']),
        'G1074': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1074']),
        'G1603': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1603']),
        'G1685': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1685']),
        'G1686': np.isin(wsheds.watershed_for_pourpoint,pp_to_polyid_map['G1686'])
    }

    return basins_mask_map


def create_wshed_mask_da(
    wsheds,
    basins_mask_map,
    BASE_DATA_DIR=Path('/soge-home/projects/crop_yield/EGU_compare'),
):
    """ create a xr.DataArray / netcdf with each station watershed as a mask
    """

    # create one xarray MASK netcdf file
    all_masks = []
    # for each watershed defined by the list of polygon IDs in basin_mask_map
    for station_id in my_stations:
        mask = basins_mask_map[station_id]
        da = xr.DataArray(
            mask,
            name=station_id,
            coords=[wsheds.lat, wsheds.lon],
            dims=['lat','lon']
        )
        # mask out the oceans
        da = da.where(~wsheds.chirps_precipitation.isel(time=0).isnull())
        da.name = station_id
        all_masks.append(da)


    wshed_masks = xr.merge(all_masks)
    wshed_masks.to_netcdf(BASE_DATA_DIR / 'PP_wshed_masks.nc')

    return wshed_masks


def scalar_xr_to_dict(xr_ds):
    """ """
    raw_dict = xr_ds.to_dict()['data_vars']
    keys = [key for key in raw_dict.keys()]
    new_dict = {}
    for key in keys:
        new_dict[key] = raw_dict[key]['data']

    return new_dict



def get_mask_for_all_watersheds(wshed_masks):
    valid_mask = (
        wshed_masks["G1045"].where(wshed_masks["G1045"] == 1).isnull().astype(bool) &
        wshed_masks["G1053"].where(wshed_masks["G1053"] == 1).isnull().astype(bool) &
        wshed_masks["G1067"].where(wshed_masks["G1067"] == 1).isnull().astype(bool) &
        wshed_masks["G1074"].where(wshed_masks["G1074"] == 1).isnull().astype(bool) &
        wshed_masks["G1603"].where(wshed_masks["G1603"] == 1).isnull().astype(bool) &
        wshed_masks["G1685"].where(wshed_masks["G1685"] == 1).isnull().astype(bool) &
        wshed_masks["G1686"].where(wshed_masks["G1686"] == 1).isnull().astype(bool)
    )
    ALL_SHEDS = ~valid_mask
    return ALL_SHEDS.drop('time')


coord_name = "watershed_for_pourpoint"
shp_path = BASE_DATA_DIR / "marcus_help" / "watershed_areas_shp" / "Watershed_Areas.shp"

df = read_station_flow_data()
wsheds_shp = read_watershed_shp()
lookup_gdf = read_station_metadata()


# JOIN into the associated basin shapes
pp_to_polyid_map = {
    'G1686': [ 1,  0,  3, 10],
    'G1685': [ 3, 10],
    'G1067': [15],
    'G1074': [15,  7],
    'G1053': [15,  7,  5],
    'G1045': [15,  7,  5,  2],
    'G1603': [11]
}


def merge_shapefiles(wsheds_shp, pp_to_polyid_map):
    # https://stackoverflow.com/a/40386377/9940782
    # unary_union or #geo_df.loc[pp_to_polyid_map[geoid]].dissolve('geometry')
    # https://stackoverflow.com/a/40386377/9940782
    out_shp_geoms=[]
    for geoid in pp_to_polyid_map.keys():
        geoms = wsheds_shp.loc[pp_to_polyid_map[geoid]].geometry

        out_shp_geoms.append(shapely.ops.unary_union(geoms))

    # OUTPUT into one dataframe
    gdf = gpd.GeoDataFrame(
        {
            "geoid":[geoid for geoid in pp_to_polyid_map.keys()],
            "number":np.arange(0,7),
            "geometry":out_shp_geoms
        },
        geometry='geometry'
    )

    return gdf


# COMPUTE AREA OF GEOM
from engineering.eng_utils import compute_area_of_geom

# FROM THE MERGED DATA
gdf = merge_shapefiles(wsheds_shp, pp_to_polyid_map)
areas = {}
for geom in gdf.geometry.values:
    print(compute_area_of_geom(geom))

# FROM THE UNMERGED DATA
areas = {}
for geoid in pp_to_polyid_map.keys():
    geoms = wsheds_shp.loc[pp_to_polyid_map[geoid]].geometry
    area = 0
    for geom in geoms:
        area += compute_area_of_geom(geom)

    areas[geoid] = area
    print(area)

def compute_areas_for_pour_points():
    """ """
    return

# compare the areas
lookup_gdf.loc[np.isin(lookup_gdf.ID,[key for key in areas.keys()])]
[(geoid,area//1e6) for (geoid,area) in zip(areas.keys(), areas.values())]

def plot_polygon(ax, polygon, color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)):
    from descartes import PolygonPatch
    x,y = polygon.exterior.xy
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='#6699cc', alpha=0.7,
        linewidth=3, solid_capstyle='round', zorder=2)
    ax.set_title('Polygon')
    # pp = PolygonPatch(polygon,color=color)
    # ax.add_patch(pp)

    return ax

# TODO: how to merge these shapefiles?
gdf= merge_shapefiles(wsheds_shp, pp_to_polyid_map)


# 1. CLEAN THE STATION RUNOFF DATA
# get only the relevant stations
df2 = df[my_stations]
station_lkup = lookup_gdf.loc[np.isin(lookup_gdf.ID, my_stations)]
df_norm = calculate_flow_per_day(df2, station_lkup)

# PUT INTO MEAN MONTHLY
lgt_stations = df_norm.resample('M').mean()

# 2. ADD THE WATERSHED SHAPEFILE TO THE XARRAY DATA
wsheds = add_shape_coord_from_data_array(
    xr_da=ds,
    shp_path=shp_path,
    coord_name=coord_name
)

basins_mask_map = create_basins_merged_map(wsheds, pp_to_polyid_map)
wshed_masks = create_wshed_mask_da(wsheds,basins_mask_map,BASE_DATA_DIR)
dims = [dim for dim in wshed_masks.dims.keys()] + ['time']
wshed_keys = [var for var in wshed_masks.variables.keys() if var not in dims]

ALL_SHEDS = get_mask_for_all_watersheds(wshed_masks)

# vars for plotting help
dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint', 'grun_runoff']
variables = [var for var in ds.variables.keys() if var not in dims]
evap_variables = [var for var in variables if "precip" not in var]

colors  = [h_col, g_col, m_col, c_col] = get_colors()

def read_dunning_mask(ds):
    dunning_dir = BASE_DATA_DIR / "dunning_seasonality_mask" / "clean_dunning_mask.nc"
    mask = ds.holaps_evapotranspiration.isel(time=0).isnull()

    # if not dunning_dir.is_file():
    dunning = xr.open_dataset(BASE_DATA_DIR/"dunning_seasonality_mask/"/"chirps_seasonality_mask.nc")
    dunning = convert_to_same_grid(ds, dunning.seasonality_mask, method="nearest_s2d")
    dunning = dunning.where(~mask)
    dunning = dunning.drop('time')
    # dunning.to_netcdf(BASE_DATA_DIR / "dunning_seasonality_mask" / "clean_dunning_mask.nc")
    # else:
    #     dunning = xr.open_dataset(dunning_dir)
    #     dunning = dunning.rename({"__xarray_dataarray_variable__":"dunning_mask"})

    seasonality_mask = ((dunning > 1.0) ) | ((dunning.lat < 0) )
    # seasonality_mask = ((dunning < 1.0) & (dunning.lon < 40)) | (dunning.lat > 4)
    seasonality_mask = seasonality_mask.where(~mask)

    return seasonality_mask.drop('time')


seasonality_mask = read_dunning_mask(ds)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
"""
lookup_gdf =    station metadata (location, watershed area etc.)
df =            mm day-1 runoff for stations in lookup_gdf
grun =          raw GRUN data from 1971-2005
"""

# INTERCOMPARISON of GRUN STATIONS
grun = xr.open_dataset(BASE_DATA_DIR/'GRUN_v1_GSWP3_WGS84_05_1902_2014.nc')
grun = grun.sel(time=slice('1971-01-01','2005-12-31'))

# read raw station data
df = pd.read_csv(BASE_DATA_DIR / 'Qts_Africa_glofas_062016_1971_2005.csv')
df.index = pd.to_datetime(df.DATE)
df = df.drop(columns='DATE')
df = df.dropna(how='all',axis=1)
df = df.resample('M').mean()

# get the flows per day for the stations
df = calculate_flow_per_day(df, lookup_gdf)
df = df.loc[:, np.isin(df.columns,lookup_gdf.ID)]



def scalar_xr_ob_to_df(xr_ds):
    """ convert a scalar xarray object (no dims) to df"""
    df = (
        pd.DataFrame(
            xr_ds.to_dict()['data_vars']
        )
        .loc['data']
        .to_frame()
    )

    return df.T

# from engineering.eng_utils import scalar_xr_to_dict
def scalar_xr_to_dict(xr_ds):
    """ """
    raw_dict = xr_ds.to_dict()['data_vars']
    keys = [key for key in raw_dict.keys()]
    new_dict = {}
    for key in keys:
        new_dict[key] = raw_dict[key]['data']

    return new_dict



plt.close('all')

LOW = False

flws = []
hlps = []
glm = []
mdis = []
run = []
stns = []
for geoid in wshed_keys:
    print(geoid)
    d = grun.where(wshed_masks[geoid])

    # get the flows for the NON NULL timesteps
    flows = df[geoid]
    nonnull_times = flows.index[~flows.isnull()]
    print(f"There are {len(nonnull_times)} timesteps for the station {geoid} with data")
    flows = flows[nonnull_times]

    # get the point location of the basin
    point = lookup_gdf.loc[lookup_gdf.ID == geoid].geometry.values[0]
    river = lookup_gdf.loc[lookup_gdf.ID == geoid].RiverName.values[0]

    # plot the time series of the STATION vs. the GRIDDED RUNOFF
    fig,ax=plt.subplots(figsize=(12,8))
    flows.plot.line(ax=ax, marker='o')
    d.to_dataframe().plot(ax=ax,marker='o')
    ax.set_title(f"{geoid} Station in the {river} River")
    fig.savefig(BASE_FIG_DIR / f"{geoid}.png")


# ----------------
    # plot all times ()
    # calculate mean all time for each basin
    d_mean = d.mean()
    d_p_min_e_mean = d_p_min_e.mean()
    p_min_e_dict = scalar_xr_to_dict(d_p_min_e_mean)
    mean_dict = scalar_xr_to_dict(d_mean)

    mean_hlps = p_min_e_dict['holaps_evapotranspiration']
    mean_modis = p_min_e_dict['modis_evapotranspiration']
    mean_gleam = p_min_e_dict['gleam_evapotranspiration']
    mean_runoff = mean_dict['grun_runoff']
    mean_flow = flows.mean()












#
