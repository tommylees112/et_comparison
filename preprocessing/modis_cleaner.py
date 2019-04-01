from pathlib import Path
import xarray as xr
import numpy as np

import ipdb
import warnings
import os

from preprocessing.utils import (
    gdal_reproject,
    bands_to_time,
    convert_to_same_grid,
    select_same_time_slice,
    save_netcdf,
    get_holaps_mask,
    merge_data_arrays,
)

from preprocessing.cleaner import Cleaner

# ------------------------------------------------------------------------------
# MODIS cleaner
# ------------------------------------------------------------------------------


class ModisCleaner(Cleaner):
    """Preprocess the MODIS dataset"""

    def __init__(
        self,
        base_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/"),
        reference_data_path=Path("/soge-home/projects/crop_yield/EGU_compare/") / "holaps_EA_clean.nc",
        reference_ds_variable='holaps_evapotranspiration'
    ):
        self.base_data_path = Path(base_data_path)
        # reference_data_path = self.base_data_path / "holaps_EA_clean.nc"
        data_path = self.base_data_path / "EA_evaporation_modis.nc"

        self.reference_data_path = Path(reference_data_path)
        self.reference_ds = xr.open_dataset(self.reference_data_path)[reference_ds_variable]
        super(ModisCleaner, self).__init__(data_path=data_path)

        self.update_clean_data(
            self.raw_data.monthly_ET, msg="Extract monthly_ET from MODIS xr.Dataset"
        )
        self.get_mask()
        # self.mask = self.mask.drop('units')

    def get_mask(self):
        self.mask = get_holaps_mask(self.reference_ds)

    def modis_to_holaps_grid(self):
        regrid_data = convert_to_same_grid(
            self.reference_ds, self.clean_data, method="nearest_s2d"
        )
        # UPDATE THE SELF.CLEAN_DATA
        self.update_clean_data(
            regrid_data, msg="MODIS Data Regridded to same as HOLAPS"
        )
        return repr_data

    def mask_illegitimate_values(self):
        # mask out the negative values (missing values)
        masked_vals = self.clean_data.where(self.clean_data >= 0)
        self.update_clean_data(
            masked_vals, msg="Masked out the ET values LESS THAN 0 mm day-1"
        )
        return

    def swap_modis_axes(self):
        """ longitude/latitude => latitude/longitude """
        m = xr.DataArray(
            np.swapaxes(self.clean_data.data, -2, -1),
            dims=("time", "latitude", "longitude"),
        )
        m["time"] = self.clean_data.time
        m["latitude"] = self.clean_data.latitude
        m["longitude"] = self.clean_data.longitude
        self.update_clean_data(
            m, "Swapped the dimensions: longitude/latitude => latitude/longitude"
        )

        return

    def convert_units(self):
        # convert from monthly (mm month-1) to daily (mm day-1)
        warnings.warn(
            "Monthly -> Daily should be unique to each month (month length). Currently dividing by an average of all month lengths (30.417)"
        )
        daily_et = self.clean_data / 30.417
        daily_et.attrs["units"] = "mm day-1 [mm/month / 30.417]"
        self.update_clean_data(daily_et)
        return

    def rename_lat_lon(self):
        rename_latlon = self.clean_data.rename({"longitude": "lon", "latitude": "lat"})
        self.update_clean_data(rename_latlon, msg="Renamed latitude,longitude => lat,lon")
        return

    def preprocess(self):
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
        self.rename_xr_object("modis_evapotranspiration")
        # save data
        save_netcdf(
            self.clean_data, filepath=self.base_data_path / "modis_EA_clean.nc"
        )
        print("\n\n MODIS Preprocessed \n\n")
        return
