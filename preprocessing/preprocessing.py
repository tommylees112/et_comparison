import sys
sys.path.insert(0, "/soge-home/projects/crop_yield/et_comparison/")

from pathlib import Path
from preprocessing.holaps_cleaner import HolapsCleaner
from preprocessing.modis_cleaner import ModisCleaner
from preprocessing.gleam_cleaner import GleamCleaner
from preprocessing.chirps_cleaner import ChirpsCleaner
from preprocessing.grun_cleaner import GrunCleaner

from preprocessing.esa_cci_lc_cleaner import EsaCciCleaner
from preprocessing.utils import merge_data_arrays, save_netcdf, get_all_valid

if __name__ == "__main__":

    h = HolapsCleaner()
    h.preprocess()
    g = GleamCleaner()
    g.preprocess()
    m = ModisCleaner()
    m.preprocess()

    gr = GrunCleaner()
    gr.preprocess()

    c = ChirpsCleaner()
    c.preprocess()

    e = EsaCciCleaner()
    e.preprocess()

    # merge the preprocessed data and save to netcdf
    ds = merge_data_arrays(h.clean_data, g.clean_data, m.clean_data, c.clean_data, gr.clean_data)
    ds = get_all_valid(ds, ds.holaps_evapotranspiration, ds.modis_evapotranspiration, ds.gleam_evapotranspiration, ds.chirps_precipitation)
    assert (ds.chirps_precipitation.isnull() == ds.holaps_evapotranspiration.isnull()).mean() == 1., "the missing (nan) values should be exactly the same in all products!"

    # drop yemen from the data
    from engineering.mask_using_shapefile import add_shape_coord_from_data_array

    country_shp_path = BASE_DATA_DIR / "country_shp" / "ne_50m_admin_0_countries.shp"
    ds = add_shape_coord_from_data_array(ds, country_shp_path, coord_name="countries")
    ds = ds.where(ds.countries != 2)

    output_ds_path='/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc'
    save_netcdf(ds, output_ds_path, force=True)



# def convert_to_same_grid(self, method="nearest_s2d"):
#     """ Use xEMSF package to regrid ds to the same grid as self.reference_ds """
#     assert ("lat" in self.reference_ds.dims)&("lon" in self.reference_ds.dims), f"Need (lat,lon) in self.reference_ds dims Currently: {self.reference_ds.dims}"
#     assert ("lat" in self.clean_data.dims)&("lon" in self.clean_data.dims), f"Need (lat,lon) in self.clean_data dims Currently: {self.clean_data.dims}"
#
#     # create the grid you want to convert TO (from self.reference_ds)
#     ds_out = xr.Dataset({
#         'lat': (['lat'], self.reference_ds.lat),
#         'lon': (['lon'], self.reference_ds.lon),
#     })
#
#     # create the regridder object
#     # xe.Regridder(grid_in, grid_out, method='bilinear')
#     regridder = xe.Regridder(self.clean_data, ds_out, method, reuse_weights=True)
#
#     # IF it's a dataarray just do the original transformations
#     if isinstance(self.clean_data, xr.core.dataarray.DataArray):
#         repr_data = regridder(self.clean_data)
#     # OTHERWISE loop through each of the variables, regrid the datarray then recombine into dataset
#     elif isinstance(self.clean_data, xr.core.dataset.Dataset):
#         vars = [i for i in self.clean_data.var().variables]
#         if len(vars) ==1 :
#             repr_data = regridder(self.clean_data)
#         else:
#             output_dict = {}
#             # LOOP over each variable and append to dict
#             for var in vars:
#                 print(f"- regridding var {var} -")
#                 da = self.clean_data[var]
#                 da = regridder(da)
#                 output_dict[var] = da
#             # REBUILD
#             repr_data = xr.Dataset(output_dict)
#     else:
#         assert False, "This function only works with xarray dataset / dataarray objects"
#
#     print(f"Regridded from {(regridder.Ny_in, regridder.Nx_in)} to {(regridder.Ny_out, regridder.Nx_out)}")
#
#     # UPDATE THE SELF.CLEAN_DATA
#     self.update_clean_data(repr_data, msg="Data Reprojected")
#
#     return repr_data
