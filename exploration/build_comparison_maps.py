# build_comparison_maps.py

import xarray as xr
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import os
import xesmf as xe # for regridding


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
    regridder = xe.Regridder(ds, ds_out, 'nearest_s2d', reuse_weights=True)

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
    mask = ds.isnull().isel(time=0)
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
gleam = select_same_time_slice(holaps,gleam)

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

"""
Problems with the data:
----------------------
[x] Different units:
    holaps = W m-2
    modis = mm/m
    gleam = ???

    Convert: W m-2 to mm of water
    (https://www.researchgate.net/post/How_to_convert_30minute_evapotranspiration_in_watts_to_millimeters)
    1 Watt /m2 = 0.0864 MJ /m2 /day

[x] Different variables:
    holaps: mean monthly latent heat flux
    modis: monthly mean of daily evapotranspiration
    gleam = ???

[x] Timesteps:
     modis starts in February 2001 for some reason

[] Different masks:
    HOLAPS masks out the water areas
    GLEAM masks out some of the water areas
    MODIS has no mask for the water areas

[x] Different projections
    sincrs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
    llcrs = "+proj=longlat +ellps=WGS84 +datum=WGS84"

[x] Different Resolutions
    GLEAM = lat: 81, lon: 77
    MODIS = longitude: 231, latitude: 242
    HOLAPS = lat: 414, lon: 454

[x] Different Colorbars

"""

#%%
# ------------------------------------------------------------------------------
# Plot the mean spatial patterns of the RAW data
# ------------------------------------------------------------------------------

kwargs = {"vmin":0,"vmax":3.5}

fig1,ax1=plt.subplots(figsize=(12,8))
holaps.mean(dim='time').plot(ax=ax1,**kwargs)
ax1.set_title(f"HOLAPS Mean Latent Heat Flux [{holaps.units}]")
fig1.savefig('figs/holaps_map1.png')
# fig1.savefig('figs/holaps_map1.svg')

fig2,ax2=plt.subplots(figsize=(12,8))
# transpose because longitude/latitude => latitude/longitude for plotting
modis.mean(dim='time').plot(ax=ax2,**kwargs)
ax2.set_title(f"MODIS Monthly Actual Evapotranspiration [{modis.units}]")
fig2.savefig('figs/modis_map1.png')
# fig2.savefig('figs/modis_map1.svg')

fig3,ax3=plt.subplots(figsize=(12,8))
gleam.mean(dim='time').plot(ax=ax3,**kwargs)
ax3.set_title(f"GLEAM Monthly mean daily Actual Evapotranspiration [mm day-1]")
fig3.savefig('figs/gleam_map1.png')
# fig3.savefig('figs/gleam_map1.svg')

fig4,ax4=plt.subplots(figsize=(12,8))
holaps_mm.mean(dim='time').plot(ax=ax4,**kwargs)
ax4.set_title(f"HOLAPS Mean Evapotranspiration [{holaps_mm.units}]")
fig4.savefig('figs/holaps_mm_map1.png')
# fig4.savefig('figs/holaps_mm_map1.svg')

fig5,ax5=plt.subplots(figsize=(12,8))
holaps_repr.mean(dim='time').plot(ax=ax5,**kwargs)
ax5.set_title(f"HOLAPS Reprojected Mean Evapotranspiration [{holaps_repr.units}]")
fig5.savefig('figs/holaps_repr_map1.png')
# fig5.savefig('figs/holaps_repr_map1.svg')

#%%
# ------------------------------------------------------------------------------
# Plot the histograms of values
# ------------------------------------------------------------------------------

# GET colors for each variable
h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]

# Plot holaps
# -----------
fig1,ax1=plt.subplots(figsize=(12,8))
h = drop_nans_and_flatten(holaps)

sns.set_color_codes()
sns.distplot(h,ax=ax1, color=h_col)

ax1.set_title(f'Density Plot of HOLAPS Mean Latent Heat Flux [{holaps.units}]')
ax1.set_xlabel(f'Mean Latent Heat Flux [{holaps.units}]')
fig1.savefig('figs/holaps_hist1.png')
# fig1.savefig('figs/holaps_hist1.svg')

# Plot modis
# -----------
fig2,ax2=plt.subplots(figsize=(12,8))
m = drop_nans_and_flatten(modis)

sns.set_color_codes()
sns.distplot(m,ax=ax2, color=m_col)

ax2.set_title(f'Density Plot of MODIS Monthly Actual Evapotranspiration [{modis.units}]')
ax2.set_xlabel(f'Monthly Actual Evapotranspiration [{modis.units}]')
fig2.savefig('figs/modis_hist1.png')
# fig2.savefig('figs/modis_hist1.svg')

# Plot gleam
# -----------
fig3,ax3=plt.subplots(figsize=(12,8))
g = drop_nans_and_flatten(gleam)

sns.set_color_codes()
sns.distplot(g,ax=ax3, color=g_col)

ax3.set_title(f'Density Plot of GLEAM Monthly mean daily Actual Evapotranspiration [mm / day] ')
ax3.set_xlabel(f'Monthly mean daily Actual Evapotranspiration [mm / day]')
fig3.savefig('figs/gleam_hist1.png')
# fig3.savefig('figs/gleam_hist1.svg')

# plot holaps_reprojected
# -----------------------
fig4,ax4=plt.subplots(figsize=(12,8))
h_repr = drop_nans_and_flatten(holaps_repr)

sns.set_color_codes()
sns.distplot(h_repr,ax=ax4, color=h_col)

ax4.set_title(f'Density Plot of HOLAPS Monthly Actual Evapotranspiration [mm day-1] ')
ax4.set_xlabel(f'Monthly mean daily Actual Evapotranspiration [mm day-1]')
fig4.savefig('figs/holaps_repr_hist1.png')
# fig4.savefig('figs/holaps_repr_hist1.svg')

#%%
# ------------------------------------------------------------------------------
# JointPlots of variables
# ------------------------------------------------------------------------------

#%%
# FIRST have to be mapped onto the same grid

#
# #%%
# fig1,ax1=plt.subplots(figsize=(12,8))
# h = drop_nans_and_flatten(holaps)
# m = drop_nans_and_flatten(modis)
#
# g = sns.JointGrid(x=h, y=m)
# g = g.plot_joint(plt.scatter, color="k", edgecolor="white")
# _ = g.ax_marg_x.hist(h, color=h_col, alpha=.6,)
#                      # bins=np.arange(0, 60, 5))
#
# _ = g.ax_marg_y.hist(m, color=m_col, alpha=.6,
#                      orientation="horizontal",)
#                      # bins=np.arange(0, 12, 1))







#
