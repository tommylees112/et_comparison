# build_comparison_maps.py
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xesmf as xe # for regridding

import warnings
import os

%matplotlib


def convert_to_same_grid(reference_ds, ds, method="nearest_s2d"):
    """ Use xEMSF package to regrid ds to the same grid as reference_ds """
    assert ("lat" in reference_ds.dims)&("lon" in reference_ds.dims), f"Need (lat,lon) in reference_ds dims Currently: {reference_ds.dims}"
    assert ("lat" in ds.dims)&("lon" in ds.dims), f"Need (lat,lon) in ds dims Currently: {ds.dims}"

    # create the grid you want to convert TO (from reference_ds)
    ds_out = xr.Dataset({
        'lat': (['lat'], reference_ds.lat),
        'lon': (['lon'], reference_ds.lon),
    })

    # create the regridder object
    # xe.Regridder(grid_in, grid_out, method='bilinear')
    regridder = xe.Regridder(ds, ds_out, method, reuse_weights=True)

    # IF it's a dataarray just do the original transformations
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = regridder(ds)
    # OTHERWISE loop through each of the variables, regrid the datarray then recombine into dataset
    elif isinstance(ds, xr.core.dataset.Dataset):
        vars = [i for i in ds.var().variables]
        if len(vars) ==1 :
            ds = regridder(ds)
        else:
            output_dict = {}
            # LOOP over each variable and append to dict
            for var in vars:
                print(f"- regridding var {var} -")
                da = ds[var]
                da = regridder(da)
                output_dict[var] = da
            # REBUILD
            ds = xr.Dataset(output_dict)
    else:
        assert False, "This function only works with xarray dataset / dataarray objects"

    print(f"Regridded from {(regridder.Ny_in, regridder.Nx_in)} to {(regridder.Ny_out, regridder.Nx_out)}")

    return ds



def select_same_time_slice(reference_ds, ds):
    """ Select the values for the same timestep as the reference ds"""
    # CHECK THEY ARE THE SAME FREQUENCY
    # get the frequency of the time series from reference_ds
    freq = pd.infer_freq(reference_ds.time.values)
    old_freq = pd.infer_freq(ds.time.values)
    warnings.warn('Disabled the assert statement. ENSURE FREQUENCIES THE SAME (e.g. monthly)')
    # assert freq == old_freq, f"The frequencies should be the same! currenlty ref: {freq} vs. old: {old_freq}"

    # get the STARTING time point from the reference_ds
    min_time = reference_ds.time.min().values
    max_time = reference_ds.time.max().values
    orig_time_range = pd.date_range(min_time, max_time, freq=freq)
    # EXTEND the original time_range by 1 (so selecting the whole slice)
    # because python doesn't select the final in a range
    periods = len(orig_time_range) # + 1
    # create new time series going ONE EXTRA PERIOD
    new_time_range = pd.date_range(min_time, freq=freq, periods=periods)
    new_max = new_time_range.max()

    # select using the NEW MAX as upper limit
    ds = ds.sel(time=slice(min_time, new_max))
    # assert reference_ds.time.shape[0] == ds.time.shape[0],"The time dimensions should match, currently reference_ds.time dims {reference_ds.time.shape[0]} != ds.time dims {ds.time.shape[0]}"

    print_time_min = pd.to_datetime(ds.time.min().values)
    print_time_max = pd.to_datetime(ds.time.max().values)
    try:
        vars = [i for i in ds.var().variables]
    except:
        vars = ds.name
    # ref_vars = [i for i in reference_ds.var().variables]
    print(f"Select same timeslice for ds with vars: {vars}. Min {print_time_min} Max {print_time_max}")

    return ds



def drop_nans_and_flatten(dataArray):
    """flatten the array and drop nans from that array. Useful for plotting histograms.

    Arguments:
    ---------
    : dataArray (xr.DataArray)
        the DataArray of your value you want to flatten
    """
    # drop NaNs and flatten
    return dataArray.values[~np.isnan(dataArray.values)]



def bands_to_time(da, times, var_name="LE_Mean"):
    """ For a dataArray with each timestep saved as a different band, create
         a time Coordinate
    """
    # get a list of all the bands as dataarray objects (for concatenating later)
    band_strings = [key for key in da.variables.keys() if 'Band' in key]
    bands = [da[key] for key in band_strings]
    bands = [band.rename(var_name) for band in bands]

    # check the number of bands matches n timesteps
    assert len(times) == len(bands), f"The number of bands should match the number of timesteps. n bands: {len(times)} n times: {len(bands)}"
    # concatenate into one array
    timestamped_da = xr.concat(bands, dim=times)

    return timestamped_da


def gdal_reproject(infile, outfile, **kwargs):
    """Use gdalwarp to reproject one file to another

    Help:
    ----
    https://www.gdal.org/gdalwarp.html
    """
    to_proj4_string = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    resample_method = 'near'

    # check options
    valid_resample_methods = ['average','near','bilinear','cubic','cubicspline','lanczos','mode','max','min','med','q1','q3']
    assert resample_method in valid_resample_methods, f"Resample method not Valid. Must be one of: {valid_resample_methods} Currently: {resample_method}"

    cmd = f'gdalwarp -t_srs "{to_proj4_string}" -of netCDF -r average -dstnodata -9999 -ot Float32 {infile} {outfile}'

    # run command
    print(f"#### Running command: {cmd} ####")
    os.system(cmd)
    print(f"#### Run command {cmd} \n FILE REPROJECTED ####")

    return


def get_holaps_mask(ds):
    """
    NOTE:
    - assumes that all of the null values from the HOLAPS file are valid null values (e.g. water bodies). Could also be invalid nulls due to poor data processing / lack of satellite input data for a pixel!
    """
    warnings.warn('assumes that all of the null values from the HOLAPS file are valid null values (e.g. water bodies). Could also be invalid nulls due to poor data processing / lack of satellite input data for a pixel!')
    warnings.warn('How to collapse the time dimension in the holaps mask? Here we just select the first time because all of the valid pixels are constant for first, last second last. Need to check this is true for all timesteps')
    mask = ds.isnull().isel(time=0).drop('time')
    mask.name = 'holaps_mask'

    return mask


#%%
# ------------------------------------------------------------------------------
# Working with RAW DATA
# ------------------------------------------------------------------------------

# HOLAPS
# ------
# assert False, "\n\n #### You need to ask Jian to run your reprojection code BEFORE he subsets the data #### \n\n"
data_dir="holaps_africa.nc"
holaps = xr.open_dataset(data_dir).LE_Mean
# RESAMPLE to the end of the month
holaps = holaps.resample(time='M').first()

# Convert from latent heat (w m-2) to evaporation (mm day-1)
holaps_mm = holaps / 28
holaps_mm.name = 'Evapotranspiration'
holaps_mm['units'] = "mm day-1 [w m-2 / 28]"

# READ HOLAPS REPROJECTED
# ----------------------
# reproject if necessary
outfile="holaps_africa_test.nc"
if not os.path.isfile(outfile):
    # cmd = 'gdalwarp -t_srs "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs" -of netCDF -r average -dstnodata -9999 -ot Float32 holaps_africa.nc holaps_africa_test.nc'
    # print(f"#### Running command: {cmd} ####")
    # os.system(cmd)
    # print(f"#### Run command {cmd} \n FILE REPROJECTED ####")

    # functional form
    gdal_reproject(infile="holaps_africa.nc", outfile=outfile)

# read the data
# data_dir="holaps_africa_test.nc"
# holaps_repr = xr.open_dataset(data_dir)

data_dir="holaps_east_africa.nc"
holaps_repr = xr.open_dataset(data_dir)

# turn the multiple bands into timesteps
# get the ORIGINAL timestamps from the non-reprojected data
h_times = holaps.time
holaps_repr = bands_to_time(holaps_repr, h_times, var_name="LE_Mean")
# ------------------------------------------------------------------------------
warnings.warn('TODO: No idea why but the values appear to be 10* bigger than the pre-reprojected holaps data')
holaps_repr /= 10 # WHY ARE THE VALUES 10* bigger?
# ------------------------------------------------------------------------------

# convert to mm day-1
holaps_repr /= 28
holaps_repr.name = 'Evapotranspiration'
holaps_repr.attrs['units'] = "mm day-1 [w m-2 / 28]"

# holaps_repr = holaps_repr.resample(time='M').first()
# # Convert from latent heat (w m-2) to evaporation (mm day-1)
# holaps_repr = holaps_repr / 28
# holaps_repr['units'] = "mm day-1 [w m-2 / 28]"

# GLEAM
# -----
data_dir="EA_GLEAM_evap_transp_2001_2015.nc"
gleam = xr.open_dataset(data_dir).evaporation
# resample to monthly & select same time range as
gleam = gleam.resample(time='M').mean(dim='time')
gleam = select_same_time_slice(holaps, gleam)
gleam.attrs['units'] = "mm day-1"
# REGRID onto same grid as HOLAPS
gleam = convert_to_same_grid(holaps_repr, gleam, method="nearest_s2d")

# MODIS
# -----
data_dir="EA_evaporation_modis.nc"
modis = xr.open_dataset(data_dir).monthly_ET
modis = modis.resample(time='M').first()
modis = select_same_time_slice(holaps,modis)
# mask out the negative values (missing values)
modis = modis.where(modis >=0)

# transpose because longitude/latitude => latitude/longitude for plotting
# modis = modis.T

# SWAP the order of the dimensions
#   longitude/latitude => latitude/longitude
m = xr.DataArray(np.swapaxes(modis.data, -2,-1),
    dims=('time','latitude','longitude')
    )
m['time'] = modis.time
m['latitude'] = modis.latitude
m['longitude'] = modis.longitude
modis = m

# convert from monthly (mm month-1) to daily (mm day-1)
modis = modis / 30.417

# REGRID onto same grid as HOLAPS
modis = modis.rename({'longitude':'lon','latitude':'lat'})
modis = convert_to_same_grid(holaps_repr, modis, method="nearest_s2d")

# reassign the units to the modis DataArray object
modis.attrs['units'] ='mm day-1 [mm/month / 30.417]'

#%%
# ------------------------------------------------------------------------------
# Get the HOLAPS mask and apply it to all other datasets
# ------------------------------------------------------------------------------
mask  = get_holaps_mask(holaps_repr)
mask = xr.concat([mask for _ in range(len(holaps.time))])
mask = mask.rename({'concat_dims':'time'})
mask['time'] = holaps.time

# mask the other datasets
gleam_msk = gleam.where(~mask)
gleam_msk.attrs['units'] = "mm day-1"
modis_msk = modis.where(~mask)

#%%
# ------------------------------------------------------------------------------
# Merge into one big xr.Dataset
# ------------------------------------------------------------------------------
gleam_msk = gleam_msk.rename('gleam_evapotranspiration')
modis_msk = modis_msk.rename('modis_evapotranspiration')
holaps_repr = holaps_repr.rename('holaps_evapotranspiration')
ds = xr.merge([holaps_repr,gleam_msk,modis_msk])

# if the .nc file doesnt exist then save it
if not os.path.isfile('all_vars_ds.nc'):
    ds.to_netcdf('all_vars_ds.nc')
else:
    ds = xr.open_dataset('all_vars_ds.nc')

# Get the values where ALL DATASETS are not null
valid_mask = (
    ds.holaps_evapotranspiration.notnull()
    & ds.modis_evapotranspiration.notnull()
    & ds.gleam_evapotranspiration.notnull()
)
ds_valid = ds.where(valid_mask)
