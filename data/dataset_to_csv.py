"""
script for converting
"""
import xarray as xr
from pathlib import Path
# import click
import pandas as pd
import fire


def convert_to_dataframe(ds):
    return ds.to_dataframe()


def save_as_csv(nc_filepath, out_csv_filepath):
    """ """
    ds = xr.open_dataset(nc_filepath)
    df = convert_to_dataframe(ds)
    df.to_csv(out_csv_filepath)

    return


def chunk_csv_file(in_file, chunkrows=100_000, numrows=1_250_000_000):
    """
    Note:
    ----
    https://stackoverflow.com/a/40023798/9940782
    """
    assert False, "This needs some love"
    # open multiple pandas dataframes
    for i, chunk in enumerate(pd.read_csv(in_file,chunksize=chunkrows)):
        # save to csv files
        # chunk.drop('Unnamed: 0', axis=1, inplace=True)
        chunk.to_csv('chunk{}.csv'.format(i), index=False)

        for chunk in df: #for each 100k rows
            if count <= numrows/chunkrows: #if 1GB threshold has not been reached
                GB_files += 1
                outname = "csv_big_file2_1stHalf.csv"
            else:
                GB_files += 1
                outname = "csv_big_file2_2ndHalf.csv"
            #append each output to same csv, using no header
            chunk.to_csv(outname, mode='a', header=None, index=None)
            count+=1




if __name__ == "__main__":
    # save_as_csv()
    # python save_as_csv nc_filepath out_csv_filepath
    fire.Fire()
