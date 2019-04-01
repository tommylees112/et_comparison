from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

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
# GLEAM cleaner
# ------------------------------------------------------------------------------


class EsaCciCleaner(Cleaner):
    """Preprocess the ESA CCI landcover dataset"""

    def __init__(self):
        self.base_data_path = Path("/soge-home/projects/crop_yield/EGU_compare/")
        reference_data_path = self.base_data_path / "holaps_EA_clean.nc"

        # CHANGE ME >>>>>>>>>>>>>>>
        data_path = self.base_data_path / "ESACCI_LC_L4-Map_300m.nc"
        # <<<<<<<<<<<<<<<<<<<<<<<<<

        self.reference_data_path = Path(reference_data_path)
        self.reference_ds = xr.open_dataset(self.reference_data_path).holaps_evapotranspiration
        super(EsaCciCleaner, self).__init__(data_path=data_path)

        # extract the variable of interest (TO xr.DataArray)
        self.update_clean_data(
            self.raw_data.lccs_class, msg="Extract land cover class from ESA CCI xr.Dataset"
        )

        # make the mask (FROM REFERENCE_DS) to copy to this dataset too
        self.get_mask()
        # self.mask = self.mask.drop('units')

    def preprocess(self):
        # regrid to same as reference data (holaps)
        self.regrid_to_reference() # method='bilinear'
        # ipdb.set_trace()
        # use the same mask as HOLAPS
        self.use_reference_mask(one_time=True) # THIS GOING WRONG (NEEDS only one time dim)
        # rename data
        self.rename_xr_object("esa_cci_landcover")
        # save data
        save_netcdf(
            self.clean_data, filepath=self.base_data_path / "esa_lc_EA_clean.nc"
        )
        print("\n\n ESA CCI LandCover Preprocessed \n\n")
        return

        def preprocess2(self):
            # regrid to same as reference data (holaps)
            self.regrid_to_reference(method='bilinear')
            # ipdb.set_trace()
            # use the same mask as HOLAPS
            self.use_reference_mask(one_time=True) # THIS GOING WRONG (NEEDS only one time dim)
            # rename data
            self.rename_xr_object("esa_cci_landcover")
            # save data
            save_netcdf(
                self.clean_data, filepath=self.base_data_path / "esa_lc_EA_clean.nc"
            )
            print("\n\n ESA CCI LandCover Preprocessed \n\n")
            return
