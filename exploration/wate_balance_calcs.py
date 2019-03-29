# water_balance_calcs.py

# map the pour point ids to the polygons
mapping_dict = {
    # blue_nile
    18 : [2,1,4,11],
    17 : [4,11],
    16 : [11],
    # juba-shabelle
    14 : [21],
    12 : [18,17],
    11 : [17],
    # awash
    4 : [16],
    3 : [16,8],
    2 : [16,8,6],
    1 : [16,8,6,3],
    # ??
    5 : [12],
    # ???
    8 : [7],
    7 : [15],
    9 : [14],
}


shp_path = BASE_DATA_DIR / "marcus_help" / "watershed_areas_shp" / "Watershed_Areas.shp"
coord_name = "watershed_for_pourpoint"
wsheds_shp = gpd.read_file(shp_path)
wsheds_shp = wsheds_shp.rename(columns={'OBJECTID':'polygon_id', 'gridcode':'pour_point_id'})


# 1. add shapefile to xarray object
wsheds = add_shape_coord_from_data_array(xr_da=ds,
    shp_path=shp_path,
    coord_name=coord_name
)
# 2. subset by the regions of interest
pour_points = lookup_gdf[['ID','StationName', 'DrainArLDD', 'YCorrected', 'XCorrected','geometry','corrected_river_name']]

# 3. normalise the precip / evapotranspiration values by the areas
unique_wsheds = np.unique(drop_nans_and_flatten(wsheds.coord_name))
