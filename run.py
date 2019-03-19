import fire
from pathlib import Path

from preprocessing.utils import merge_data_arrays, save_netcdf, get_all_valid
from preprocessing.preprocessing import HolapsCleaner, ModisCleaner, GleamCleaner


class RunTask:

    @staticmethod
    def preprocess(holaps_path='/soge-home/projects/crop_yield/EGU_compare/holaps_africa.nc',
                   modis_path='/soge-home/projects/crop_yield/EGU_compare/EA_evaporation_modis.nc',
                   gleam_path='/soge-home/projects/crop_yield/EGU_compare/EA_GLEAM_evap_transp_2001_2015.nc',
                   output_ds_path='/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc'):
        holaps_path, modis_path, gleam_path = Path(holaps_path), Path(modis_path), Path(gleam_path)

        # TODO: PATHS hardcoded (needs to be changed)
        h = HolapsCleaner()
        h.preprocess()
        g = GleamCleaner()
        g.preprocess()
        m = ModisCleaner()
        m.preprocess()

        # merge the preprocessed data and save to netcdf
        ds = merge_data_arrays(h.clean_data, g.clean_data, m.clean_data)
        ds = get_all_valid(ds.holaps_evapotranspiration, ds.modis_evapotranspiration, ds.gleam_evapotranspiration)
        save_netcdf(ds, output_ds_path, force=True)


    @staticmethod
    def plot_raw():
        return


    @staticmethod
    def plot_reprojected(clean_ds_path=''):
        return


if __name__=='__main__':
    fire.Fire(RunTask)
