"""
# water_balance_calcs.py

: wsheds (xr.Dataset)
    evaporation & precipitation values with a watershed variable determining the areas of the watersheds for different locations

: lgt_stations (pd.DataFrame)
    the raw daily runoff (monthly) in [mm day-1] for the stations we have data for

: station_lkup (gpd.GeoDataFrame)
    the metadata associated with the stations that we have data for


:  wsheds_shp (gpd.GeoDataFrame)

"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import geopandas as gpd
import shapely


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


coord_name = "watershed_for_pourpoint"
shp_path = BASE_DATA_DIR / "marcus_help" / "watershed_areas_shp" / "Watershed_Areas.shp"

df = read_station_flow_data()
wsheds_shp = read_watershed_shp()
lookup_gdf = read_station_metadata()


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

# vars for plotting help
dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint']
variables = [var for var in ds.variables.keys() if var not in dims]
evap_variables = [var for var in variables if "precip" not in var]

colors  = [h_col, g_col, m_col, c_col] = get_colors()


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

dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint']
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

#
dims = [dim for dim in ds.dims.keys()] + ['countries', 'climate_zone', 'koppen', 'koppen_code', 'watershed_for_pourpoint']
variables = [var for var in ds.variables.keys() if var not in dims]
evap_variables = [var for var in variables if "precip" not in var]

colors  = [h_col, g_col, m_col, c_col] = get_colors()

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
    d = ds[variables].where(shed)
    # get the times where there is data for that station
    d = d.sel(time=nonnull_times)
    # P-E calculation
    d_p_min_e = d.chirps_precipitation - d

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
    (flows*10)[nonnull_times].plot(ax=ax, color=h_col, alpha=0.7, kind='bar')
    fig.suptitle(f'{geoid} Station RAW Flows (* 10 ????)')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_station_flow_legit_*10.png")

    # PLOT P-E timeseries
    fig,ax = plt.subplots(figsize=(12,8))
    d_p_min_e.mean(['lat','lon']).drop('chirps_precipitation').to_dataframe().plot.line(ax=ax)
    ax.axhline(y=0, linestyle=":", alpha=0.7,color='black')
    fig.suptitle(f'{geoid} Station Chirps Rainfall minus Evapotranspiration (P-E)')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_watershed_p-e_legit_timeseries.png")

    # PLOT P-E timeseries ANNUALLY
    annual_p_e = d_p_min_e.groupby('time.year').mean(['lat','lon','time']).drop('chirps_precipitation')
    # TODO: wtf are you multiplying by 10?
    warnings.warn('MULTIPLYING FLOWS BY 10 because more realistic numbers but makes no senese')
    annual_flow = flows.resample('Y').mean() * 10
    fig,ax = plt.subplots(figsize=(12,8))
    annual_p_e.to_dataframe().plot.bar(ax=ax)
    annual_flow.plot.bar(ax=ax,zorder=0,alpha=0.4, color='black',label='Annual Runoff (mean) *10')
    ax.axhline(y=0, linestyle=":", alpha=0.7,color='black', )
    plt.legend()
    fig.suptitle(f'{geoid} Annual (P-E) vs. Runoff values')
    fig.savefig(BASE_FIG_DIR/f"{geoid}_watershed_p-e_ANNUAL.png")



# fig,ax=plt.subplots(); d_p_min_e.groupby('time.year').mean(dim='time').isel(year=0).holaps_evapotranspiration.plot(cmap="RdBu",ax=ax)
# fig,ax=plt.subplots(); d_p_min_e.groupby('time.year').mean(dim='time').isel(year=1).holaps_evapotranspiration.plot(cmap="RdBu",ax=ax)

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
        warnings.warn('L496: MULTIPLYING FLOWS BY 10 because more realistic numbers but makes no sense')
        (flow_seasonality*10).plot.bar(color=color, ax=ax)
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
        warnings.warn('L4506: MULTIPLYING FLOWS BY 10 because more realistic numbers but makes no sense')
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


# # GRUN runoff data
# # TODO: put onto same grid as other products
# grun = xr.open_dataset(BASE_DATA_DIR/"GRUN_v1_GSWP3_WGS84_05_1902_2014.nc").Runoff
# grun = grun.sel(lat=slice(region.latmin,region.latmax),lon=slice(region.lonmin,region.lonmax))
# grun = grun.sel(time=slice(ds.time.min(), ds.time.max()))
