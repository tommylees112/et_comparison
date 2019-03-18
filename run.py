import fire
from pathlib import Path

class RunTask:

    @staticmethod
    def preprocess(holaps_path='',
                   modis_path='',
                   gleam_path='',
                   output_ds_path=''):
        holaps_path, modis_path, gleam_path = Path(holaps_path), Path(modis_path), Path(gleam_path)
        cleaner = 
        return

    @staticmethod
    def plot_raw():
        return

    @staticmethod
    def plot_reprojected(clean_ds_path=''):
        return

if __name__=='__main__':
    fire.Fire(RunTask)
