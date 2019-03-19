from .holaps_cleaner import HolapsCleaner
from .modis_cleaner import ModisCleaner
from .gleam_cleaner import GleamCleaner
from utils import merge_data_arrays, save_netcdf, get_all_valid


if __name__ == "__main__":
    h = HolapsCleaner()
    h.preprocess()
    g = GleamCleaner()
    g.preprocess()
    m = ModisCleaner()
    m.preprocess()

    # merge the preprocessed data and save to netcdf
    ds = merge_data_arrays(h.clean_data, g.clean_data, m.clean_data)
    ds = get_all_valid(ds, ds.holaps_evapotranspiration, ds.modis_evapotranspiration, ds.gleam_evapotranspiration)

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
