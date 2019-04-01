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

BASE_FIG_DIR =Path('/soge-home/projects/crop_yield/et_comparison/figs/meeting4')
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


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# --------- PLOTS ------------
fig,ax = plt.subplots()
sns.distplot(drop_nans_and_flatten(df_norm), bins=100, kde=False, ax=ax)
ax.set_title('Runoff Values (DAILY) [mm day-1]')
fig.savefig(BASE_FIG_DIR / "hist_daily_runoff_values_all_stations_daily.png")

fig,ax = plt.subplots()
sns.distplot(drop_nans_and_flatten(lgt_stations), bins=100, kde=False, ax=ax)
ax.set_title('Runoff Values (MONTHLY MEAN) [mm day-1]')
fig.savefig(BASE_FIG_DIR / "hist_monthly_runoff_values_all_stations_daily.png")

# kwargs = {'xlim': [-0.1,10]}
for ix, var in enumerate(evap_variables):
    fig,ax = plt.subplots()
    da = wsheds[var]
    color = colors[ix]
    plot_marginal_distribution(DataArray=da, color=color, ax=ax, title=f'{var} over the whole region', xlabel='DEFAULT', summary=True)#, **kwargs)
    ax.set_xlim([-0.1,10])
    fig.savefig(BASE_FIG_DIR / f"hist_{var}_over_whole_domain_distribution_of_values.png")

fig,ax = plt.subplots()
var = 'chirps_precipitation'
da = wsheds[var]
color = colors[-1]
plot_marginal_distribution(DataArray=da, color=color, ax=ax, title=f'{var} over the whole region', xlabel='DEFAULT', summary=True)#, **kwargs)
ax.set_xlim([-0.1,10])
fig.savefig(BASE_FIG_DIR / f"hist_{var}_over_whole_domain_distribution_of_values.png")

# plot watershed masks
for station in my_stations:
    fig,ax=plot_geog_location(all_region,borders=True,rivers=True)
    lookup_gdf.loc[lookup_gdf.ID == station].plot(ax=ax,color='black')
    wshed_masks[station].plot(ax=ax,cmap='Wistia',zorder=0,alpha=0.7)
    ax.set_title(f'{station} Watershed Delineation')
    fig.savefig(BASE_FIG_DIR/f'{station}_watershed_delineation.png')

# ------------------------------

# --------- PLOTS ------------
# plot the watersheds
# plot the watersheds and their labels
fig,ax = plot_geog_location(all_region, rivers=True, borders=True)
wsheds_shp.plot(ax=ax)
wsheds_shp.apply(lambda x: ax.annotate(s=x.polygon_id, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);

# assign values to coods (latlon) and the index as NAME for plotting
lookup_gdf['coords'] = lookup_gdf['geometry'].apply(lambda x: x.representative_point().coords[:][0])
lookup_gdf['NAME'] = lookup_gdf.index

# PLOT THE STATIONS (POUR POINTS (PP)) WITH THEIR CODES
fig,ax = plot_geog_location(all_region, rivers=True, borders=True)
lookup_gdf.plot(ax=ax)
lookup_gdf.apply(lambda x: ax.annotate(s=x.ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
# pour_points = lookup_gdf[['ID','StationName', 'DrainArLDD', 'YCorrected', 'XCorrected','geometry','corrected_river_name']]

# sense checking are the basins further down the river have more flow?
df.index = pd.to_datetime(df.index)
fig,ax = plt.subplots()
stations = ["G1686","G1684"]
stations = ["G1607","G6088"]
df[stations].mean()
# ------------------------------

# 2. subset by the regions of interest
# drop the stations where there are NO values
df = df.dropna(how='all',axis=1)
stations = df.columns.values
station_lkup = lookup_gdf.loc[np.isin(lookup_gdf.ID, stations)]
station_lkup['NAME'] = station_lkup.index

# --------- PLOTS ------------
fig,ax = plot_geog_location(all_region, rivers=True, borders=True)
station_lkup.plot(ax=ax)
station_lkup.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);

# ALSO drop hte stations outside of ROI
fig,ax = plot_geog_location(all_region, borders=True);
station_lkup.plot(ax=ax)
station_lkup.apply(lambda x: ax.annotate(s=x.ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
# ------------------------------

# drop these codes
station_lkup = station_lkup.drop(station_lkup.loc[np.isin(station_lkup.ID, ["G1687","G1688"])].index)
unique_wsheds = np.unique(drop_nans_and_flatten(wsheds.coord_name))


# 3. normalise the precip / evapotranspiration values by the areas
from engineering.eng_utils import mask_multiple_conditions


# --------- PLOTS ------------
for ix, station in station_lkup.iterrows():
    geoid = station.ID
    pp_id = station.NAME
    polyids = pp_to_polyid_map[geoid]
    print(geoid, polyids)

    # get the watersheds of interest
    fig,ax = plot_geog_location(all_region, borders=True, rivers=True);
    shed = mask_multiple_conditions(wsheds.watershed_for_pourpoint, polyids)
    shed.plot(ax=ax,zorder=0, cmap='Wistia')
    station_lkup.plot(ax=ax)
    station_lkup.apply(lambda x: ax.annotate(s=x.ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    wsheds_shp.plot(ax=ax,alpha=0.5)
    wsheds_shp.apply(lambda x: ax.annotate(s=x.polygon_id, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    fig.suptitle(f'The Watershed for {geoid}')
    fig.savefig(BASE_FIG_DIR / f"watershed_mask_for_{geoid}.png")

# ------------------------------

# ------------------------------------------------------------------------------
# mask this shit
# ------------------------------------------------------------------------------
# legit stations MONTHLY values
lgt_stations = df[station_lkup.ID.values]
lgt_stations = calculate_flow_per_day(lgt_stations, station_lkup)
lgt_stations = lgt_stations[[col for col in lgt_stations.columns if "_perday" in col]]
lgt_stations = lgt_stations.resample('M').mean()


fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(drop_nans_and_flatten(lgt_stations),ax=ax,bins=100)
fig.suptitle('Monthly Mean Runoff Values [mm day-1]\nvalue = runoff / size(m2) * 1000 (mm in m) * 86,400 (s in day)')
fig.savefig(BASE_FIG_DIR / "AAstation_histogram_of_values.png")


fig,ax = plt.subplots(figsize=(12,8))
lgt_stations.plot.line(ax=ax, marker='o')
fig.suptitle('Monthly Mean Runoff Values [mm day-1]\nvalue = runoff / size(m2) * 1000 (mm in m) * 86,400 (s in day)')
fig.savefig(BASE_FIG_DIR / "AAstation_timeseries_for_analysis.png")

# plot stations
fig,ax = plot_geog_location(all_region, borders=True, rivers=True);
station_lkup.plot(ax=ax)
station_lkup.apply(lambda x: ax.annotate(s=x.ID, xy=x.geometry.centroid.coords[0], ha='left'),axis=1);
fig.suptitle('Location of Stations [mm day-1]')
fig.savefig(BASE_FIG_DIR / "AAstation_locations_for_analysis.png")

dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint', 'grun_runoff']
variables = [var for var in ds.variables.keys() if var not in dims]
evap_variables = [var for var in variables if "precip" not in var]

df = df.dropna(how='all',axis=1)
stations = df.columns.values
station_lkup = lookup_gdf.loc[np.isin(lookup_gdf.ID, stations)]
station_lkup['NAME'] = station_lkup.index

colors  = [h_col, g_col, m_col, c_col] = get_colors()


# ------------------------------------------------------------------------------
# Plot the monthly mean values for different watersheds
# ------------------------------------------------------------------------------
for ix, station in station_lkup.iterrows():
    print(geoid, polyids)
    geoid = station.ID
    pp_id = station.NAME
    geoid_col = geoid# + '_perday'
    polyids = pp_to_polyid_map[geoid]
    station_meta = station_lkup.loc[station_lkup.ID == geoid]

    # get the MONTHLY flows for the stations
    flows = lgt_stations[geoid_col]
    #
    legit_timesteps
    monthly_vals = flows.groupby(by=[flows.index.month]).mean()
    shed = mask_multiple_conditions(wsheds.watershed_for_pourpoint, polyids)

    # get watershed area
    d = ds[variables].where(shed)

    # create monthly mean
    d = d.groupby('time.month').mean()

    # PLOT THE CLIMATOLOGY FOR THAT BASIN AREA
    fig,axs = plt.subplots(2,2,figsize=(12,8))

    for ix, var in enumerate(variables):
        ax_ix = np.unravel_index(ix, (2,2))
        color = colors[ix]
        ax = axs[ax_ix]
        d[var].plot.line(ax=ax,marker='o',color=color)
        ax.set_title(f"{var}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'{geoid} Station Seasonality [mm day-1]')
    fig.savefig(BASE_FIG_DIR / f"_{geoid}_station_variable_seasonality.png")

    # PLOT THE watershed delineation
    fig,ax = plot_geog_location(all_region, borders=True, rivers=True)
    shed.plot.contourf(ax=ax,cmap='Wistia', zorder=0)
    station_meta.plot(ax=ax)
    station_meta.apply(lambda x: ax.annotate(s=x.ID, xy=x.geometry.centroid.coords[0], ha='left'),axis=1);
    fig.suptitle(f'_{geoid} Station Watershed Area')
    fig.savefig(BASE_FIG_DIR / f"_{geoid}_station_watershed_area_plot.png")

    # PLOT the seasonality of streamflow
    fig,ax = plt.subplots();
    monthly_vals.plot.line(ax=ax, marker="o")
    fig.suptitle(f'{geoid} Station Seasonality [mm day-1]')
    fig.savefig(BASE_FIG_DIR / f"_{geoid}_station_watershed_area_plot.png")

# ------------------------------------------------------------------------------
# PLOT FOR EACH STATION
# ------------------------------------------------------------------------------
dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint', 'grun_runoff']
variables = [var for var in ds.variables.keys() if var not in dims]
evap_variables = [var for var in variables if "precip" not in var]

colors  = [h_col, g_col, m_col, c_col] = get_colors()

mult_10 = False
for ix, station in station_lkup.iterrows():
    geoid = station.ID
    pp_id = station.NAME
    geoid_col = geoid
    polyids = pp_to_polyid_map[geoid]
    print(geoid, polyids)
    station_meta = station_lkup.loc[station_lkup.ID == geoid]

    # get the flows for the NON NULL timesteps
    flows = lgt_stations[geoid_col]
    nonnull_times = flows.index[~flows.isnull()]
    print(f"There are {len(nonnull_times)} timesteps for the station {geoid} with data")
    flows = flows[nonnull_times]

    monthly_flows = flows.groupby(by=[flows.index.month]).mean()
    # get a (lat,lon) mask for the watershed
    shed = mask_multiple_conditions(wsheds.watershed_for_pourpoint, polyids)

    # get watershed area
    d = ds[variables + ['grun_runoff']].where(shed)
    # get the times where there is data for that station
    d = d.sel(time=nonnull_times)
    # P-E calculation
    d_p_min_e = d.chirps_precipitation - d.drop('grun_runoff')

    # mean spatial patterns?
    d_time = d.mean(['lat','lon'])
    # mean over the year
    d_annual = d.resample(time='Y').mean()

    # PLOT ALL timeseries
    fig,ax = plt.subplots(figsize=(12,8))
    ax = d.mean(['lat','lon']).to_dataframe().plot.line(ax=ax)
    fig.suptitle(f'{geoid} Station RAW Rainfall and Evapotranspiration')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_watershed_legit_time_series.png")

    # plot timeseries of the flows (MONTHLY)
    fig,ax = plt.subplots(figsize=(12,8))
    if mult_10:
        (flows*10)[nonnull_times].plot(ax=ax, color=h_col, alpha=0.7, kind='bar')
        fig.suptitle(f'{geoid} Station RAW Flows (* 10 ????)')
        fig.savefig(BASE_FIG_DIR/f"{geoid}_station_flow_legit_*10.png")
    else:
        (flows)[nonnull_times].plot(ax=ax, color=h_col, alpha=0.7, kind='bar')
        fig.suptitle(f'{geoid} Station RAW Flows')
        fig.savefig(BASE_FIG_DIR/f"{geoid}_station_flow_legit_NORMAL.png")

    # PLOT P-E timeseries
    fig,ax = plt.subplots(figsize=(12,8))
    d_p_min_e.mean(['lat','lon']).drop('chirps_precipitation').to_dataframe().plot.line(ax=ax)
    ax.axhline(y=0, linestyle=":", alpha=0.7,color='black')
    fig.suptitle(f'{geoid} Station Chirps Rainfall minus Evapotranspiration (P-E)')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_watershed_p-e_legit_timeseries.png")

    # PLOT P-E timeseries ANNUALLY
    annual_p_e = d_p_min_e.groupby('time.year').mean(['lat','lon','time']).drop('chirps_precipitation')

    if mult_10:
        # TODO: wtf are you multiplying by 10?
        warnings.warn('MULTIPLYING FLOWS BY 10 because more realistic numbers but makes no senese')
        annual_flow = flows.resample('Y').mean() * 10
        label = 'Annual Runoff (mean) *10'
    else:
        annual_flow = flows.resample('Y').mean()
        label='Annual Runoff (mean)'

    grun_annual = d_annual.grun_runoff.mean(dim=['lat','lon'])
    fig,ax = plt.subplots(figsize=(12,8))
    annual_p_e.to_dataframe().plot.bar(ax=ax,zorder=0)
    annual_flow.plot.bar(ax=ax,alpha=0.4, color='black',label=label)
    ax.axhline(y=0, linestyle=":", alpha=0.7,color='black')
    plt.legend()
    fig.suptitle(f'{geoid} Annual (P-E) vs. Runoff values')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_watershed_p-e_ANNUAL.png")

    # Plot the mean spatial patterns in the basin
    fig,axs = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True);
    spatial = ds[variables].where(shed).mean(dim='time')
    for jx, var in enumerate(variables):
        ax_jx = np.unravel_index(jx, (2,2))
        ax = axs[ax_jx]
        fig,ax = plot_geog_location(all_region, borders=True, rivers=True)
        spatial[var].plot(ax=ax, zorder=0)
        ax.set_title(f"{var}")
    fig.savefig(BASE_FIG_DIR/f"{geoid}_spatial_dist_of_vals_for_watershed.png")

    # distributions for that basin
    fig,axs = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True);
    for jx, var in enumerate(variables):
        ax_jx = np.unravel_index(jx, (2,2))
        color = colors[jx]
        ax = axs[ax_jx]
        distn = drop_nans_and_flatten(d[var])

        sns.distplot(distn, bins=30, kde=False, color=color, ax=ax)
        ax.set_title(f"{var} for Watershed of Station {geoid}")
        ax.axvline(x=0, linestyle=":")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(BASE_FIG_DIR/f"{geoid}_histogram_of_vals_for_watershed.png")

    # WATER BALANCE HISTOGRAMS for every time point
    fig,axs = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True);
    for jx, var in enumerate(evap_variables):
        color = colors[jx]
        wbalance = d_time.chirps_precipitation - d_time[var] - flows
        ax = axs[np.unravel_index(jx, (2,2))]

        sns.distplot(drop_nans_and_flatten(wbalance), ax=ax, bins=10, kde=False, color=color)
        ax.set_title(f' CHIRPS - {var} - {geoid} flow')
        ax.axvline(x=0, linestyle=":")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"{geoid} Station Histograms of WaterBalance calculations")
    fig.savefig(BASE_FIG_DIR / f"{geoid}_station_watershed_water_balance_histograms.png")

    # WATER BALANCE 2 SEASONALITY
    fig,axs = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True);
    for jx, var in enumerate(evap_variables):
        color = colors[jx]
        wbalance = d.chirps_precipitation - d[var]
        # ======== GROUP BY MONTH ===============
        wbalance_seasonality = wbalance.groupby('time.month').mean()
        flow_seasonality = flows.groupby(by=[flows.index.month]).mean()
        # =======================================
        ax = axs[np.unravel_index(jx, (2,2))]

        wbalance_seasonality.plot.line(marker='o', color=color, ax=ax)
        if mult_10:
            warnings.warn('L496: MULTIPLYING FLOWS BY 10 because more realistic numbers but makes no sense')
            (flow_seasonality*10).plot.bar(color=color, ax=ax)
        else:
            flow_seasonality.plot.bar(color=color, ax=ax)
        ax.set_title(f' CHIRPS - {var} (LINES), and {geoid} (BARS) flow')
        ax.axhline(y=0, linestyle=":")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"{geoid} Station Seasonality of WaterBalance calculations")
    fig.savefig(BASE_FIG_DIR / f"{geoid}_station_watershed_water_balance_seasonality.png")

    # WATER BALANCE 3 ANNUAL
    fig,axs = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True);
    for jx, var in enumerate(evap_variables):
        color = colors[jx]
        p_minus_e = d_annual.chirps_precipitation - d_annual[var]
        # ======== GROUP BY YEAR ===============
        p_minus_e_annual = p_minus_e.mean(dim=['lat','lon'])
        if mult_10:
            warnings.warn('L506: MULTIPLYING FLOWS BY 10 because more realistic numbers but makes no sense')
        flow_annual = flows.groupby(by=[flows.index.year]).mean()
        time = pd.to_datetime(p_minus_e_annual.time.values)
        # =======================================
        ax = axs[np.unravel_index(jx, (2,2))]
        sns.lineplot(y=p_minus_e_annual.values, x=np.arange(len(time)), color=color, marker='o', ax=ax)
        sns.barplot(x=np.arange(len(time)), y=flow_annual.values, color=color, ax=ax)
        # p_minus_e_annual.plot.line(marker='o', color=color, ax=ax)
        # flow_annual.plot.bar(color=color, ax=ax)
        ax.set_title(f' CHIRPS - {var} (LINES), and {geoid} (BARS) flow for each YEAR (2001-2005)')
        ax.axhline(y=0, linestyle=":")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"{geoid} Station Yearly (P-ET) WaterBalance calculations")
    fig.savefig(BASE_FIG_DIR / f"{geoid}_station_watershed_water_balance_YEARLY.png")

    plt.close('all')

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
fig,ax = plt.subplots()
p_minus_e_annual = p_minus_e.mean(dim=['lat','lon'])
flow_annual = flows.groupby(by=[flows.index.year]).mean()
time = pd.to_datetime(p_minus_e_annual.time.values)
color = h_col
# ax.plot(y=p_minus_e_annual.values, x=time, color=color, marker='o')
# ax.bar(x=time, height=flow_annual.values, color=color)
sns.lineplot(y=p_minus_e_annual.values, x=np.arange(len(time)), color=color, marker='o', ax=ax)
sns.barplot(x=np.arange(len(time)), y=flow_annual.values, color=color, ax=ax)
# p_minus_e_annual.plot.line(marker='o', color=color, ax=ax)
# flow_annual.plot.bar(color=color, ax=ax)
ax.set_title(f' CHIRPS - {var} (LINES), and {geoid} (BARS) flow for each YEAR (2001-2005)')
ax.axhline(y=0, linestyle=":")
plt.show()

# ==============================================================================
# GRUN runoff data
# ==============================================================================
mod = ds.chirps_precipitation - ds.modis_evapotranspiration - ds.grun_runoff
hlp = ds.chirps_precipitation - ds.holaps_evapotranspiration - ds.grun_runoff
glm = ds.chirps_precipitation - ds.gleam_evapotranspiration - ds.grun_runoff
p_minus_r = ds.chirps_precipitation - ds.grun_runoff

# PLOT MARGINALS
plot_marginal_distribution(mod,m_col)
fig=plt.gcf()
fig.savefig(BASE_FIG_DIR / 'modis_WATERBALANCE_with_grun_data.png')

plot_marginal_distribution(hlp,h_col)
fig=plt.gcf()
fig.savefig(BASE_FIG_DIR / 'holaps_WATERBALANCE_with_grun_data.png')

plot_marginal_distribution(hlp,g_col)
fig=plt.gcf()
fig.savefig(BASE_FIG_DIR / 'gleam_WATERBALANCE_with_grun_data.png')

# PLOT OVER TIME STEP (YEARS)
mod_an = mod.groupby('time.year').mean(dim='time')
hlp_an = hlp.groupby('time.year').mean(dim='time')
glm_an = glm.groupby('time.year').mean(dim='time')
p_minus_r_an = p_minus_r.groupby('time.year').mean(dim='time')


# Spatial Patterns
mod_spatial = mod_an.mean(dim='year')
hlp_spatial = hlp_an.mean(dim='year')
glm_spatial = glm_an.mean(dim='year')
p_minus_r_spatial = p_minus_r_an.mean(dim='year')

kwargs = {'vmin':-2.0,'vmax':2}
fig,ax=plot_geog_location(all_region,rivers=True,borders=True)
mod_spatial.plot(ax=ax,cmap='RdBu',**kwargs)
fig.suptitle('Mean Annual Waterbalance: P - MODIS E - GRUN runoff')
fig.savefig(BASE_FIG_DIR / 'modis_SPTIAL_WATERBALANCE_with_grun_data.png')

fig,ax=plot_geog_location(all_region,rivers=True,borders=True)
fig.suptitle('Mean Annual Waterbalance: P - HOLAPS E - GRUN runoff')
hlp_spatial.plot(ax=ax,cmap='RdBu',**kwargs)
fig.savefig(BASE_FIG_DIR / 'holaps_SPTIAL_WATERBALANCE_with_grun_data.png')

fig,ax=plot_geog_location(all_region,rivers=True,borders=True)
fig.suptitle('Mean Annual Waterbalance: P - GLEAM E - GRUN runoff')
glm_spatial.plot(ax=ax,cmap='RdBu',**kwargs)
fig.savefig(BASE_FIG_DIR / 'gleam_SPTIAL_WATERBALANCE_with_grun_data.png')

fig,ax=plot_geog_location(all_region,rivers=True,borders=True)
fig.suptitle('Mean Annual: P - GRUN runoff')
p_minus_r_spatial.plot(ax=ax)
fig.savefig(BASE_FIG_DIR / 'P-Runoff_SPTIAL_WATERBALANCE_with_grun_data.png')




# Plot
plot_marginal_distribution(mod_spatial,m_col)
ax=plt.gca()
ax.axvline(x=0,color='black',alpha=0.5,linestyle=':')
fig=plt.gcf()
fig.suptitle('Annual Waterbalance: P - MODIS E - GRUN runoff')
fig.savefig(BASE_FIG_DIR / 'modis_ANNUAL_WATERBALANCE_with_grun_data.png')

plot_marginal_distribution(hlp_spatial,h_col)
ax=plt.gca()
ax.axvline(x=0,color='black',alpha=0.5,linestyle=':')
fig=plt.gcf()
fig.suptitle('Annual Waterbalance: P - HOLAPS E - GRUN runoff')
fig.savefig(BASE_FIG_DIR / 'holaps_ANNUAL_WATERBALANCE_with_grun_data.png')

plot_marginal_distribution(glm_spatial,g_col)
ax=plt.gca()
ax.axvline(x=0,color='black',alpha=0.5,linestyle=':')
fig=plt.gcf()
fig.suptitle('Annual Waterbalance: P - GLEAM E - GRUN runoff')
fig.savefig(BASE_FIG_DIR / 'gleam_ANNUAL_WATERBALANCE_with_grun_data.png')



def get_bounding_box(xr_mask, name='mask'):
    """ from an xarray boolean field, get the bounding box of the 1 values (TRUE)

    Returns:
    -------
    : region (Region namedtuple)
    """
    from engineering.regions import Region
    region = Region(
        region_name=name,
        latmin=,
        latmax=,
        lonmin=,
        lonmax=
    )

    return region


def netcdf_mask_to_shapefile():
    """ convert an xr.DataArray boolean mask (.nc) to a .shp file """
    raise NotImplementedError
    return


#
for ix, station in station_lkup.iterrows():
    geoid = station.ID
    pp_id = station.NAME
    geoid_col = geoid
    polyids = pp_to_polyid_map[geoid]
    print(geoid, polyids)
    station_meta = station_lkup.loc[station_lkup.ID == geoid]

    # get the flows for the NON NULL timesteps
    flows = lgt_stations[geoid_col]
    nonnull_times = flows.index[~flows.isnull()]
    print(f"There are {len(nonnull_times)} timesteps for the station {geoid} with data")
    flows = flows[nonnull_times]

    # get the watershed mask
    shed = wshed_masks[geoid]

    # get the runoff for the wshed mask
    ds_shed = ds.where(shed)
    ds_shed = ds_shed.sel(time=nonnull_times)
    # get pcp and
    grun_shed = ds_shed.grun_runoff
    pcp_shed = ds_shed.chirps_precipitation
    # wb calcs
    wb_shed = ds_shed.chirps_precipitation - ds_shed
    wb_shed = wb_shed[[var for var in variables if 'evap' in var]]


    # Plot comparison of GRUN and Stations
    fig,ax=plt.subplots(figsize=(12,8))
    grun_shed.mean(dim=['lat','lon']).plot.line(ax=ax, label='GRUN monthly mean [mm day-1]')
    flows.plot.line(ax=ax, label='Station Runoff Monthly Mean [mm day-1]')
    plt.legend()
    fig.suptitle(f"{geoid} Station Comparison of GRUN and Station Data")
    fig.savefig(BASE_FIG_DIR / f"{geoid}_comparison_of_GRUN_w_stations.png")

    # plot comparison of the RAW SEASONALITY
    fig,ax=plt.subplots(figsize=(12,8))
    grun_shed.mean(dim=['lat','lon']).groupby('time.month').mean().plot.line(ax=ax, label='GRUN monthly mean [mm day-1]')
    flows.groupby(by=[flows.index.month]).mean().plot.line(ax=ax, label='Station Runoff Monthly Mean [mm day-1]')
    plt.legend()
    fig.suptitle(f"{geoid} Station Comparison of GRUN and Station Data Seasonality")
    fig.savefig(BASE_FIG_DIR / f"{geoid}_comparison_of_GRUN_w_stations_SEASONALITY.png")


    # PLOT ALL VARIABLES
    fig,axs = plt.subplots(3,2,figsize=(12,8))
    for ix, var in enumerate(variables + ['grun_runoff']):
        ax_ix = np.unravel_index(ix,(3,2))
        ax = axs[ax_ix]
        if 'evapo' in var:
            kwargs={'vmin':0,'vmax':3}
            ds_shed[var].mean(dim='time').plot(ax=ax,**kwargs)
        else:
            ds_shed[var].mean(dim='time').plot(ax=ax)

        ax.set_title(f'{var}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels('')
        ax.set_yticklabels('')

    # the final axes plot a timeseries
    ax = axs[2,1]
    # wb_shed.mean(dim=['lat','lon']).to_dataframe().plot.line(ax=ax)
    flows.plot.bar(ax=ax,color='gray',label='Station Runoff',zorder=0)
    ax.set_title('Station Ru noff (Monthly Mean [mm day-1])')
    ax.set_xticklabels('')

    sns.barplot(x=flows.index, y=flows.values, ax=ax)
    fig.suptitle(f"{geoid} Comparison of Mean Spatial Patterns")
    fig.savefig(BASE_FIG_DIR / f"{geoid}_comparison_of_mean_spatial_patterns.png")


# ==============================================================================
# Plot over all timesteps
# ==============================================================================

dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint', 'grun_runoff']
variables = [var for var in ds.variables.keys() if var not in dims]
evap_variables = [var for var in variables if "precip" not in var]

colors  = [h_col, g_col, m_col, c_col] = get_colors()


mult_10 = False
for ix, station in station_lkup.iterrows():
    geoid = station.ID
    pp_id = station.NAME
    geoid_col = geoid
    polyids = pp_to_polyid_map[geoid]
    print(geoid, polyids)
    station_meta = station_lkup.loc[station_lkup.ID == geoid]

    # get the flows for the NON NULL timesteps
    flows = lgt_stations[geoid_col]
    nonnull_times = flows.index[~flows.isnull()]
    print(f"There are {len(nonnull_times)} timesteps for the station {geoid} with data")
    flows = flows[nonnull_times]

    monthly_flows = flows.groupby(by=[flows.index.month]).mean()
    # get a (lat,lon) mask for the watershed
    shed = mask_multiple_conditions(wsheds.watershed_for_pourpoint, polyids)

    # get watershed area
    d = ds[variables + ['grun_runoff']].where(shed)
    # get the times where there is data for that station
    d = d.sel(time=nonnull_times)
    # P-E calculation
    d_p_min_e = (d.chirps_precipitation - d.drop('grun_runoff')).drop('chirps_precipitation')

    # mean spatial patterns?
    d_time = d.mean(['lat','lon'])
    # mean over the year
    d_annual = d.resample(time='Y').mean()

    grun_annual = d_annual.grun_runoff.mean(dim=['lat','lon'])

    # -------------------------------
    # PLOT over all times -----------
    # -------------------------------

    d_all_time = d.mean(dim=['time','lat','lon'])
    d_all_p_min_e = (d_all_time.chirps_precipitation - d_all_time.drop('grun_runoff')).drop('chirps_precipitation')
    df_ = pd.DataFrame({
        "holaps_evapotranspiration":d_all_p_min_e.holaps_evapotranspiration.values,
        "gleam_evapotranspiration":d_all_p_min_e.gleam_evapotranspiration.values,
        "modis_evapotranspiration":d_all_p_min_e.modis_evapotranspiration.values
    }, index=[0])
    all_flow = flows.mean()


    fig,ax = plt.subplots(figsize=(12,8))

    # plot the P-E products
    df_.plot.bar(ax=ax,zorder=0)
    # plot the station flows
    ax.bar(x=0,height=all_flow, alpha=0.4, color='black',label='Station Runoff')
    # plot the gridded runoff vals
    ax.bar(x=0,height=d_all_time.grun_runoff.values, alpha=0.4, color='m',label='Gridded Runoff')

    ax.axhline(y=0, linestyle=":", alpha=0.7,color='black')
    plt.legend()
    fig.suptitle(f'{geoid} All Time (P-E) vs. Runoff values')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_watershed_p-e_ALL_TIMES.png")

# ==============================================================================
# Plot over all basins over all time
# ==============================================================================
# wshed_masks
for ix, station in station_lkup.iterrows():
    geoid = station.ID
    pp_id = station.NAME
    geoid_col = geoid
    polyids = pp_to_polyid_map[geoid]
    print(geoid, polyids)
    station_meta = station_lkup.loc[station_lkup.ID == geoid]

    # get the flows for the NON NULL timesteps
    flows = lgt_stations[geoid_col]
    nonnull_times = flows.index[~flows.isnull()]
    print(f"There are {len(nonnull_times)} timesteps for the station {geoid} with data")
    flows = flows[nonnull_times]

    # get the watershed mask
    shed = wshed_masks[geoid]

    # get the runoff for the wshed mask
    ds_shed = ds.where(shed)
    ds_shed.mean()

dims = [dim for dim in wshed_masks.dims.keys()] + ['time']
wshed_keys = [var for var in wshed_masks.variables.keys() if var not in dims]



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
flws = []
hlps = []
glm = []
mdis = []
run = []
stns = []
for geoid in wshed_keys:
    print(geoid)
    d = ds.where(wshed_masks[geoid]).drop(["countries", "watershed_for_pourpoint"])
    d_p_min_e = (d.chirps_precipitation - d.drop('grun_runoff')).drop('chirps_precipitation')

    # get the flows for the NON NULL timesteps
    flows = lgt_stations[geoid]
    nonnull_times = flows.index[~flows.isnull()]
    print(f"There are {len(nonnull_times)} timesteps for the station {geoid} with data")
    flows = flows[nonnull_times]

    # get the point location of the basin
    point = station_lkup.loc[station_lkup.ID == geoid].geometry.values[0]

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

    # ONE for each basin mean
    stns.append(geoid)
    hlps.append(mean_hlps)
    mdis.append(mean_modis)
    glm.append(mean_gleam)
    run.append(mean_runoff)
    flws.append(mean_flow)

    # plot for the basin
    stn_df = pd.DataFrame({
        "holaps_evapotranspiration":mean_hlps,
        "modis_evapotranspiration":mean_modis,
        "gleam_evapotranspiration":mean_gleam,
        "grun_runoff":mean_runoff,
        "station_flows":mean_flow,
    }, index=[0])


    fig,ax = plt.subplots(figsize=(12,8))
    # all_mean[["holaps_evapotranspiration","modis_evapotranspiration","gleam_evapotranspiration"]].plot(kind='bar',ax=ax)
    ax.bar(x=0,height=mean_hlps, label='P - HOLAPS')
    ax.bar(x=1,height=mean_modis, label='P - GLEAM')
    ax.bar(x=2,height=mean_gleam, label='P - MODIS')
    ax.bar(x=1, width=3, height=mean_runoff, color='m', alpha=0.5, label='GRUN Runoff')
    ax.bar(x=1, width=3, height=mean_flow, color='b', alpha=0.5, label='Station Runoff')
    ax.set_title(f'Average P-E and Runoff for Catchment of Station {geoid}')
    plt.xticks(rotation=10)
    ax.set_xticklabels('')
    ax.set_ylim([-0.5,2.2])
    plt.legend(loc='lower left')

    # add the inset map location for the location of the station
    ax2 = plot_inset_map(ax, all_region, borders=True , rivers=True, height="25%",width="25%",loc='lower right')
    add_point_location_to_map(point, ax2, **{'color':'black'})
    # plot mask for the location
    (~d.isel(time=0).holaps_evapotranspiration.isnull()).drop('time').plot(ax=ax2,zorder=0,alpha=0.3,cmap='Wistia',add_colorbar=False)
    ax2.set_title('Watershed')
    fig.savefig(BASE_FIG_DIR / f"{geoid}_average_basin_water_balance_comparison_ALL_timesteps.png")


out_df = pd.DataFrame({
    "stations":stns,
    "holaps_evapotranspiration":hlps,
    "modis_evapotranspiration":mdis,
    "gleam_evapotranspiration":glm,
    "grun_runoff":run,
    "station_flows":flws,
})
all_mean = out_df.mean()

# ALL BASINS
fig,ax = plt.subplots(figsize=(12,8))
# all_mean[["holaps_evapotranspiration","modis_evapotranspiration","gleam_evapotranspiration"]].plot(kind='bar',ax=ax)
ax.bar(x=0,height=all_mean.holaps_evapotranspiration, label='P - HOLAPS')
ax.bar(x=1,height=all_mean.gleam_evapotranspiration, label='P - GLEAM')
ax.bar(x=2,height=all_mean.modis_evapotranspiration, label='P - MODIS')
ax.bar(x=1, width=3, height=all_mean.grun_runoff, color='m', alpha=0.5, label='GRUN Runoff')
ax.bar(x=1, width=3, height=all_mean.station_flows, color='b', alpha=0.5, label='Station Runoff')
ax.set_xticklabels('')
ax.set_title('Average P-E and Runoff (Gridded (GRUN) and Station) for all Basins')
plt.xticks(rotation=10)
ax.set_ylim([-0.5,2.2])
plt.legend(loc='lower left')

# add the inset map location for the location of ALL BASINS
ax2 = plot_inset_map(ax, all_region, borders=True , rivers=True, height="25%",width="25%",loc='upper right')
# plot mask for the location
ALL_SHEDS.plot(ax=ax2,zorder=0,alpha=0.3,cmap='Wistia',add_colorbar=False)
ax2.set_title('')
fig.savefig(BASE_FIG_DIR / f"ALL_STATIONS_average_basin_water_balance_comparison_ALL_timesteps.png")


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

fig,ax=plt.subplots()
ALL_SHEDS.plot(ax=ax)
    # plot only valid station data times


# DO FOR ENTIRE REGION
p_min_e = (ds.chirps_precipitation - ds)[evap_variables]
wb_all = p_min_e - ds.grun_runoff
results = scalar_xr_to_dict(wb_all.mean())
std = scalar_xr_to_dict(wb_all.std())
variation = {station : ((results[station]-std[station]), (results[station]+std[station])) for station in results.keys()}
y_errs = {station : ((results[station]+std[station]) - (results[station]-std[station])) for station in results.keys()}
ALL_AREA = (~ds.chirps_precipitation.isel(time=0).isnull()).drop('time')


#
fig,ax=plt.subplots(figsize=(12,8))
ax.bar(color=h_col,x=0,height=results['holaps_evapotranspiration'], yerr=y_errs['holaps_evapotranspiration'], label='P - HOLAPS - Runoff')
ax.bar(color=g_col,x=1,height=results['gleam_evapotranspiration'], yerr=y_errs['gleam_evapotranspiration'], label='P - GLEAM - Runoff')
ax.bar(color=m_col,x=2,height=results['modis_evapotranspiration'], yerr=y_errs['modis_evapotranspiration'], label='P - MODIS - Runoff')
# ax.set_ylim([-0.5,0.5])
ax.set_xticklabels('')
ax.axhline(y=0, linestyle=':', color='black', alpha=0.5)
ax.set_ylabel('Water Balance [mm day-1]')
plt.legend(loc='upper left')
ax2 = plot_inset_map(ax, all_region, borders=True , rivers=True, height="25%",width="25%",loc='upper right')
# plot mask for the location
ALL_AREA.plot(ax=ax2,zorder=0,alpha=0.3,cmap='Wistia',add_colorbar=False)
ax2.set_title('')
ax.set_title('Average P-E - Runoff (Gridded [GRUN]) for Whole Area')
fig.savefig(BASE_FIG_DIR/"ALL_AREA_BAR_CHART_OF_WATER_BALANCE.png")


# PLOT HISTOGRAMS
col_lookup = dict(zip(evap_das+["chirps_precipitation"], [h_col,g_col,m_col,c_col]))
DATA=wb_all.mean(dim='time')
for var in evap_variables:
    fig,ax=plt.subplots(figsize=(12,8))
    color = col_lookup[var]
    var_name = var.split('_')[0].upper()
    plot_marginal_distribution(DATA[var], ax=ax,color=color, title=f'Water Balance Calculation Histogram with {var_name}', summary=True, **{'kde':False,'hist':True, 'bins':300})
    ax.axvline(x=0,linestyle=':',color='black',alpha=0.5)
    ax.set_xlim([-4,3])
    ax.set_ylim([0,4100])
    fig.savefig(BASE_FIG_DIR / f"{var_name}_distribution_of_water_balance_calcs_ALLTIME.png")

DATA=wb_all.mean(dim='time')
fig,ax=plt.subplots(figsize=(12,8))
for var in evap_variables:
    color = col_lookup[var]
    var_name = var.split('_')[0].upper()
    plot_marginal_distribution(DATA[var],ax=ax,color=color, title=f'Water Balance Calculation Histograms', summary=True, **{'kde':False,'hist':True, 'bins':300, 'label':var_name})
    ax.axvline(x=0,linestyle=':',color='black',alpha=0.5)
    ax.set_xlim([-4,3])
    ax.set_ylim([0,4100])
    plt.legend()

fig.savefig(BASE_FIG_DIR / f"ALLVARS_distribution_of_water_balance_calcs_ALLTIME.png")


# drop_nans_and_flatten(wb_all.holaps_evapotranspiration)
# drop_nans_and_flatten(wb_all.modis_evapotranspiration)
# all_gleam = drop_nans_and_flatten(wb_all.gleam_evapotranspiration)







#
