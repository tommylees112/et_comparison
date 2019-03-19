from pathlib import Path
import xarray as xr
import numpy as np

import ipdb
import warnings
import os

from utils import (
    gdal_reproject,
    bands_to_time,
    convert_to_same_grid,
    select_same_time_slice,
    save_netcdf,
    get_holaps_mask,
    merge_data_arrays,
)


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
        print("***** self.clean_data Updated: ", msg, " *****")

        return

    def correct_time_slice(self):
        """select the same time slice as the reference data"""
        assert (
            self.reference_ds is not None
        ), "self.reference_ds does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"
        correct_time_slice = select_same_time_slice(self.reference_ds, self.clean_data)

        self.update_clean_data(
            correct_time_slice, msg="Selected the same time slice as reference data"
        )
        return

    def resample_time(self, resample_str="M"):
        """ should resample to the given timestep """
        resampled_time_data = self.clean_data.resample(time=resample_str).first()
        self.update_clean_data(resampled_time_data, msg="Resampled time ")

        return

    def regrid_to_reference(self):
        """ regrid data (spatially) onto the same grid as referebce data """
        assert (
            self.reference_ds is not None
        ), "self.reference_ds does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"

        regrid_data = convert_to_same_grid(
            self.reference_ds, self.clean_data, method="nearest_s2d"
        )
        # UPDATE THE SELF.CLEAN_DATA
        self.update_clean_data(regrid_data, msg="Data Regridded to same as HOLAPS")
        return

    def use_reference_mask(self):
        assert (
            self.reference_ds is not None
        ), "self.reference_ds does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"
        assert (
            self.mask is not None
        ), "self.mask does not exist! Likely because you're not using the MODIS or GLEAM cleaners / correct data paths"

        masked_d = self.clean_data.where(~self.mask)
        self.update_clean_data(masked_d, msg="Copied the mask from HOLAPS to GLEAM")
        return

    def mask_illegitimate_values(self):
        # mask out the missing values (coded as something else)
        return NotImplementedError

    def convert_units(self):
        """ convert to the equivalent units """
        raise NotImplementedError

    def rename_xr_object(self, name):
        renamed_data = self.clean_data.rename(name)
        self.update_clean_data(renamed_data, msg=f"Data renamed {name}")
        return

    def preprocess(self):
        """ The preprocessing steps (relatively unique for each dtype) """
        raise NotImplementedError
