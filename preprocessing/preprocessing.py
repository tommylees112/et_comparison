from pathlib import Path
import xarray as xr
import numpy as np
import xesmf as xe # for regridding
import ipdb


from utils import gdal_reproject, bands_to_time, convert_to_same_grid, select_same_time_slice, save_netcdf, get_holaps_mask, merge_data_arrays


# ------------------------------------------------------------------------------
# Base cleaner
# ------------------------------------------------------------------------------

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

    def __init__(self, data_path):
        self.data_path = Path(data_path)

# -------------------- THIS IS NOT THE BEST WAY TO CODE THIS -------------------
        # if ('modis' in self.data_path.as_posix()) or ('gleam' in self.data_path.as_posix()) or ('GLEAM' in self.data_path.as_posix()):
        #     assert self.reference_data_path.exists(), f"The HOLAPS data has to be preprocessed and saved BEFORE you can preprocess the MODIS or GLEAM data. This is because the preprocessed HOLAPS data is needed for converting to the same spatial and temporal resolutions."
        #
        #     # modis and gleam need a reference_ds
        #     self.reference_ds = xr.open_dataset(self.reference_data_path)
        #     # modis and gleam need a mask from the reference_ds
# -------------------- THIS IS NOT THE BEST WAY TO CODE THIS -------------------

        # open the datasets
        self.raw_data = xr.open_dataset(self.data_path)

        # start with clean data as a copy of the raw data
        self.clean_data = self.raw_data.copy()



    def update_clean_data(self, clean_data, msg=""):
        """ """
        self.clean_data = clean_data
        print("***** self.clean_data Updated: ", msg," *****")

        return



    def correct_time_slice(self):
        """select the same time slice as the reference data"""
        assert self.reference_ds, "self.reference_ds does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"
        correct_time_slice = select_same_time_slice(self.reference_ds, self.clean_data)

        self.update_clean_data(correct_time_slice, msg='Selected the same time slice as reference data')
        return



    def resample_time(self, resample_str="M"):
        """ should resample to the given timestep """
        resampled_time_data = self.clean_data.resample(time=resample_str).first()
        self.update_clean_data(resampled_time_data, msg="Resampled time ")

        return



    def regrid_to_reference(self):
        """ regrid data (spatially) onto the same grid as referebce data """
        assert self.reference_ds, "self.reference_ds does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"

        regrid_data = convert_to_same_grid(self.reference_ds, self.clean_data, method="nearest_s2d")
        # UPDATE THE SELF.CLEAN_DATA
        self.update_clean_data(regrid_data, msg="Data Regridded to same as HOLAPS")
        return



    def use_reference_mask():
        assert self.reference_ds, "self.reference_ds does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"
        assert self.mask, "self.mask does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"

        masked_d = self.clean_data.where(~self.mask)
        self.update_clean_data(masked_d, msg='Copied the mask from HOLAPS to GLEAM')
        return



    def mask_illegitimate_values(self):
        # mask out the missing values (coded as something else)
        return NotImplementedError



    def convert_units(self):
        """ convert to the equivalent units """
        raise NotImplementedError



    def regrid_to_reference(self):
        raise NotImplementedError



    def rename_xr_object(self,name):
        renamed_data = self.clean_data.rename(name)
        self.update_clean_data(renamed_data, msg=f'Data renamed {name}')
        return


    def preprocess(self):
        """ The preprocessing steps (relatively unique for each dtype) """
        raise NotImplementedError

# ------------------------------------------------------------------------------
# HOLAPS cleaner
# ------------------------------------------------------------------------------


class HolapsCleaner(Cleaner):
    """Preprocess the HOLAPS dataset"""


    def __init__(self):
        # init data paths (should be arguments)
        self.base_data_path = Path('/soge-home/projects/crop_yield/EGU_compare/')
        data_path = self.base_data_path / "holaps_africa.nc"
        reproject_path= self.base_data_path / 'holaps_africa_reproject.nc'

        super(HolapsCleaner, self).__init__(data_path=data_path)
        self.reproject_path = Path(reproject_path)


    def chop_EA_region(self):
        """ cheeky little bit of bash scripting with string interpolation (kids don't try this at home) """
        infile = self.base_data_path / 'holaps_reprojected.nc'
        outfile = self.base_data_path / 'holaps_EA.nc'
        lonmin=32.6
        lonmax=51.8
        latmin=-5.0
        latmax=15.2

        cmd = f"cdo sellonlatbox,{lonmin},{lonmax},{latmin},{latmax} {in_file} {out_file}"
        print(f"Running command: {cmd}")
        os.system(cmd)
        print("Chopped East Africa from the Reprojected data")
        re_chopped_data = xr.open_dataset(outfile)
        self.update_clean_data(re_chopped_data,msg='Opened the reprojected & chopped data')
        return



    def reproject(self):
        """ reproject to WGS84 / geographic latlon """
        if not self.reproject_path.is_file():
            gdal_reproject(infile=self.data_path, outfile=self.reproject_path)

        repr_data = xr.open_dataset(self.reproject_path)

        # get the timestamps from the original holaps data
        h_times = self.clean_data.time
        # each BAND is a time (multiple raster images 1 per time)
        repr_data = bands_to_time(repr_data, h_times, var_name="LE_Mean")

        # TODO: ASSUMPTION / PROBLEM
        warnings.warn('TODO: No idea why but the values appear to be 10* bigger than the pre-reprojected holaps data')
        holaps_repr /= 10 # WHY ARE THE VALUES 10* bigger?

        self.update_clean_data(repr_data, "Data Reprojected to WGS84")

        save_netcdf(self.clean_data, filepath=self.base_data_path / 'holaps_reprojected.nc')
        return



    def convert_units(self):
        # Convert from latent heat (w m-2) to evaporation (mm day-1)
        holaps_mm = self.clean_data / 28
        holaps_mm.name = 'Evapotranspiration'
        holaps_mm['units'] = "mm day-1 [w m-2 / 28]"
        self.update_clean_data(holaps_mm, msg="Transform Latent Heat (w m-2) to Evaporation (mm day-1)")

        return


    def preprocess(self):
        # reproject the file from sinusoidal to WGS84
        self.reproject()
        # chop out the correct lat/lon (changes when reprojected)
        self.chop_EA_region()
        # convert the units
        self.convert_units()
        # rename data
        self.rename_xr_object('holaps_evapotranspiration')
        # save the netcdf file (used as reference data for MODIS and GLEAM)
        save_netcdf(self.clean_data, filepath=self.base_data_path/'holaps_EA_clean.nc')
        # ipdb.set_trace()
        return

# ------------------------------------------------------------------------------
# MODIS cleaner
# ------------------------------------------------------------------------------

class ModisCleaner(Cleaner):
    """Preprocess the MODIS dataset"""


    def __init__(self):
        self.base_data_path = Path('/soge-home/projects/crop_yield/EGU_compare/')
        reference_data_path = self.base_data_path / 'holaps_EA_clean.nc'
        data_path = self.base_data_path / "EA_evaporation_modis.nc"

        self.reference_data_path = Path(reference_data_path)
        self.reference_ds = xr.open_dataset(self.reference_data_path)
        super(ModisCleaner, self).__init__(data_path=data_path)

        self.update_clean_data(self.raw_data.monthly_ET, msg="Extract monthly_ET from MODIS xr.Dataset")
        self.get_mask()



    def get_mask():
        self.mask = get_holaps_mask(self.reference_ds)



    def modis_to_holaps_grid(self):
        regrid_data = convert_to_same_grid(self.reference_ds, self.clean_data, method="nearest_s2d")
        # UPDATE THE SELF.CLEAN_DATA
        self.update_clean_data(regrid_data, msg="MODIS Data Regridded to same as HOLAPS")
        return repr_data



    def mask_illegitimate_values(self):
        # mask out the negative values (missing values)
        masked_vals = self.clean_data.where(modis >=0)
        self.update_clean_data(masked_vals, msg='Masked out the ET values LESS THAN 0 mm day-1')
        return


    def swap_modis_axes(self):
        """ longitude/latitude => latitude/longitude """
        m = xr.DataArray(np.swapaxes(self.clean_data.data, -2,-1),
            dims=('time','latitude','longitude')
            )
        m['time'] = modis.time
        m['latitude'] = modis.latitude
        m['longitude'] = modis.longitude
        self.update_clean_data(m, "Swapped the dimensions: longitude/latitude => latitude/longitude")

        return



    def convert_units(self):
        # convert from monthly (mm month-1) to daily (mm day-1)
        warnings.warn('Monthly -> Daily should be unique to each month (month length). Currently dividing by an average of all month lengths (30.417)')
        daily_et = self.clean_data / 30.417
        daily_et.attrs['units'] ='mm day-1 [mm/month / 30.417]'
        self.update_clean_data(daily_et)
        return



    def rename_lat_lon(self):
        rename_latlon = self.clean_data.rename({'longitude':'lon','latitude':'lat'})
        update_clean_data(rename_latlon, msg='Renamed latitude,longitude => lat,lon')
        return


    def preprocessing(self):
        # Resample the timesteps to END OF MONTH
        self.resample_time(resample_str="M")
        # select the correct time slice
        self.correct_time_slice()
        # mask the bad values
        self.mask_illegitimate_values()
        # swap the axes around!
        self.swap_modis_axes()
        # convert the units
        self.convert_units()
        # regrid to the same grid as holaps
        self.rename_lat_lon()
        self.regrid_to_reference()
        # use same mask as holaps
        self.use_reference_mask()
        # rename data
        self.rename_xr_object('modis_evapotranspiration')
        return


# ------------------------------------------------------------------------------
# GLEAM cleaner
# ------------------------------------------------------------------------------

class GleamCleaner(Cleaner):
    """Preprocess the GLEAM dataset"""


    def __init__(self):
        self.base_data_path = Path('/soge-home/projects/crop_yield/EGU_compare/')
        reference_data_path = self.base_data_path / 'holaps_EA_clean.nc'
        data_path = self.base_data_path / "EA_GLEAM_evap_transp_2001_2015.nc"

        self.reference_data_path = Path(reference_data_path)
        self.reference_ds = xr.open_dataset(self.reference_data_path)
        super(GleamCleaner, self).__init__(data_path=data_path)

        # extract the variable of interest (TO xr.DataArray)
        self.update_clean_data(self.raw_data.evaporation, msg="Extract evaporation from GLEAM xr.Dataset")

        # make the mask (FROM REFERENCE_DS) to copy to this dataset too
        self.get_mask()



    def get_mask():
        self.mask = get_holaps_mask(self.reference_ds)



    def convert_units(self):
        # convert unit label to 'mm day-1'
        self.clean_data.attrs['units'] = "mm day-1"



    def preprocessing(self):
        # Resample the timesteps to END OF MONTH
        self.resample_time(resample_str="M")
        # select the correct time slice
        self.correct_time_slice()
        # update the units
        self.convert_units()
        # regrid to same as reference data (holaps)
        self.regrid_to_reference()
        # use the same mask as HOLAPS
        self.use_reference_mask()
        # rename data
        self.rename_xr_object('gleam_evapotranspiration')
        return


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
h.preprocess()
g = GleamCleaner()
m = ModisCleaner()

# ds = merge_data_arrays(h.clean_data, g.clean_data, m.clean_data)
# save_netcdf(ds, "/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc", force=False)
