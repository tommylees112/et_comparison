from pathlib import Path
import xarray as xr
import numpy as np
import xesmf as xe # for regridding
import ipdb


from utils import gdal_reproject, bands_to_time, convert_to_same_grid

class Cleaner:
    """Base class for preprocessing the input data.

    Tasks include:
    - Reprojecting
    - Putting datasets onto a consistent spatial grid (spatial resolution)
    - Converting to equivalent units
    - Converting to the same temporal resolution
    - Selecting the same time slice

    Design Considerations:
    - Have an attribute, clean_data', that is constantly updated
    - Keep a copy of the raw data for reference
    - Update the 'clean_data' each time a transformation is applied
    """

    def __init__(self, data_path,
                 reproject_path='/soge-home/projects/crop_yield/EGU_compare/holaps_africa_reproject.nc'
                 ):
        self.data_path = Path(data_path)
        self.reproject_path = Path(reproject_path)

        if ('modis' in self.data_path.as_posix()) or ('gleam' in self.data_path.as_posix()):
            assert self.reference_data_path.exists(), f"The HOLAPS data has to be preprocessed and saved BEFORE you can preprocess the MODIS or GLEAM data. This is because the preprocessed HOLAPS data is needed for converting to the same spatial and temporal resolutions."

        # open the datasets
        self.raw_data = xr.open_dataset(self.data_path)
        self.reference_ds = xr.open_dataset(self.reference_data_path)

        # start with clean data as a copy of the raw data
        self.clean_data = self.raw_data.copy()


    def update_clean_data(self, clean_data, msg=""):
        """ """
        self.clean_data = clean_data
        print("***** self.clean_data Updated: ", msg," *****")

        return


    def resample_time(self, resample_str="M"):
        """ should resample to the given timestep """
        clean_data = self.clean_data.resample(time=resample_str).first()
        self.update_clean_data(clean_data, msg="Resampled time ")

        return


    def convert_units(self):
        """ convert to the equivalent units """

        raise NotImplementedError





class HolapsCleaner(Cleaner):
    """Preprocess the HOLAPS dataset"""


    def __init__(self, data_path='/soge-home/projects/crop_yield/EGU_compare/holaps_africa.nc'):
        ipdb.set_trace()
        super().__init__(data_path=data_path)



    def reproject(self, times):
        """ reproject to WGS84 / geographic latlon """
        if not self.reproject_path.is_file():
            gdal_reproject(infile=self.clean_data, outfile=self.reproject_path)
        repr_data = xr.open_dataset(self.reproject_path)
        repr_data = bands_to_time(repr_data, times, var_name="LE_Mean")

        self.clean_data = repr_data

    def preprocess():
        # reproject the file from MODIS to
        self.reproject()

class ModisCleaner(Cleaner):
    """Preprocess the MODIS dataset"""


    def __init__(self,
        reference_data_path='/soge-home/projects/crop_yield/EGU_compare/holaps_africa_test.nc'
        ):
        self.reference_data_path = Path(reference_data_path)

    def modis_to_holaps_grid(self):
        regrid_data = convert_to_same_grid(self.reference_ds, self.clean_data, method="nearest_s2d")
        # UPDATE THE SELF.CLEAN_DATA
        self.update_clean_data(regrid_data, msg="MODIS Data Regridded to same as HOLAPS")
        return repr_data


class GleamCleaner(Cleaner):
    """Preprocess the GLEAM dataset"""


    def __init__(self,
        reference_data_path='/soge-home/projects/crop_yield/EGU_compare/holaps_africa_test.nc'
        ):
        self.reference_data_path = Path(reference_data_path)

    def gleam_to_holaps_grid(self):
        regrid_data = convert_to_same_grid(self.reference_ds, self.clean_data, method="nearest_s2d")
        # UPDATE THE SELF.CLEAN_DATA
        self.update_clean_data(regrid_data, msg="GLEAM Data Regridded to same as HOLAPS")
        return repr_data


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

h = HolapsCleaner()
