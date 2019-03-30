# original_lookups.py

# map the pour point ids to the polygons
pp_to_poly_map = {
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

# map the pour point ids to the original rows in the GeoDataFrame
pp_to_orig_map = {
    # blue_nile
    18 : 17,
    17 : 26,
    16 : 25,
    # ju,
    14 : 13,
    12 : 11,
    11 : 10,
    # aw,
    4 : 3,
    3 : 2,
    2 : 1,
    1 : 0,
    # ??,
    5 : 4,
    # ???,
    8 : 7,
    7 : 6,
    9 : 8,
    # EXTRA
    10 :9,
    6 :5
}

#
pp_to_geoid_map = {
    # blue_nile
    18 : 'G1686',
    17 : 'G1685',
    # aw,
    4 : 'G1067',
    3 : 'G1074',
    2 : 'G1053',
    1 : 'G1045',
    # ??,
    5 : "G1603",
}
