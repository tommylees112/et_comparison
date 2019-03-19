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

from preprocessing.preprocessing import HolapsCleaner, ModisCleaner, GleamCleaner

h = HolapsCleaner()
h.preprocess()
g = GleamCleaner()
g.preprocess()
m = ModisCleaner()
m.preprocess()

ds = xr.open_dataset("/soge-home/projects/crop_yield/EGU_compare/processed_ds.nc")

df = ds.to_dataframe()

#%%
# ------------------------------------------------------------------------------
# Analysis by Elevation: group by elevation
# ------------------------------------------------------------------------------

topo = xr.open_rasterio('../topography/ETOPO1_Ice_g.tif')
topo = topo.drop('band')
topo = topo.sel(band=0)
topo.name = "elevation"
# topo = topo.to_dataset()

def select_east_africa(ds):
    """ """
    lonmin=32.6
    lonmax=51.8
    latmin=-5.0
    latmax=15.2

    return ds.sel(y=slice(latmax,latmin),x=slice(lonmin, lonmax))

if not os.path.isfile('global_topography.nc'):
    topo.to_netcdf('global_topography.nc')

topo = select_east_africa(topo)
topo = topo.rename({'y':'lat','x':'lon'})

if not os.path.isfile('EA_topography.nc'):
    topo.to_netcdf('EA_topography.nc')

# convert to same grid as the other data (ds_valid)
topo = convert_to_same_grid(ds_valid, topo, method="bilinear")

# mask the same areas as other data (ds_valid)
mask = get_holaps_mask(ds_valid.holaps_evapotranspiration)
topo = topo.where(~mask)



# ------------------------------------------------------------------------------
# plot topo histogram
t = drop_nans_and_flatten(topo)
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(t,ax=ax, color=sns.color_palette()[-1])
ax.set_title(f'Density Plot of Topography/Elevation in East Africa Region')
fig.savefig('figs/topo_histogram.png')
plt.close()

# plot topo histogram WITH QUINTILES (0,0.2,0.4,0.6,0.8,1.0)
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(t,ax=ax, color=sns.color_palette()[-1])
ax.set_title(f'Density Plot of Topography/Elevation in East Africa Region')
# get the qunitile values
qs = [float(topo.quantile(q=q).values) for q in np.arange(0,1.2,0.2)]
# plot vertical lines at the given quintile value
[ax.axvline(q, ymin=0,ymax=1,color='r',label=f'Quantile') for q in qs]
fig.savefig('figs/topo_histogram_quintiles.png')
plt.close()

# ------------------------------------------------------------------------------
# GROUPBY
topo.name = 'elevation'
topo = topo.to_dataset()
bins = topo.groupby_bins(group='elevation',bins=10)
intervals = bins.mean().elevation_bins.values
left_bins = [interval.left for interval in intervals]

# plot to check WHERE the bins are
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(t,ax=ax, color=sns.color_palette()[-1])
[ax.axvline(bin, ymin=0,ymax=1,color='r',label=f'Bin') for bin in left_bins];

# [bins for bins in
topo_bins = xr.concat([topo.where(
                (topo['elevation'] > interval.left) & (topo['elevation'] < interval.right)
            )
            for interval in intervals ]
)
topo_bins = topo_bins.rename({'concat_dims':'elevation_bins'})

# repeat for 60 timesteps
topo_bins = xr.concat([topo_bins for _ in range(len(ds_valid.time))])
topo_bins = topo_bins.rename({'concat_dims':'time'})
topo_bins['time'] = ds_valid.time

# select and plot the values at different elevations
topo_bins.isel(elevation_bins=0)

#
# ds_valid.where(topo_bins.isel(elevation_bins=0).elevation.notnull())

def get_unmasked_data(dataArray, dataMask):
    """ """
    return dataArray.where(dataMask)


    # data = ds_valid.where(
    #  topo_bins.isel(elevation_bins=i).elevation.notnull()
    # ).holaps_evapotranspiration.mean(dim='time')

h_col = sns.color_palette()[0]
m_col = sns.color_palette()[1]
g_col = sns.color_palette()[2]
colors = [h_col, m_col, g_col]
kwargs = {"vmin":0,"vmax":3.5}
interval_ranges = [(interval.left, interval.right) for interval in intervals]

for i in range(10):
    scale=1.5
    fig,axs = plt.subplots(2, 3, figsize=(12*scale,8*scale))
    dataMask = topo_bins.isel(elevation_bins=i).elevation.notnull()

    for j, dataset in enumerate(['holaps','modis','gleam']):
        dataArray = ds_valid[f'{dataset}_evapotranspiration']
        dataArray = get_unmasked_data(dataArray.mean(dim='time'),dataMask)
        color = colors[j]
        # get the axes that correspond to the different rows
        ax_map = axs[0,j]
        ax_map.set_title(f'{dataset} Evapotranspiration')
        ax_hist = axs[1,j]
        ax_hist.set_ylim([0,1.1])
        ax_hist.set_xlim([0,7])
        # plot the maps
        dataArray.mean(dim='time').plot(ax=ax_map,**kwargs)
        # plot the histograms
        d = drop_nans_and_flatten(dataArray)
        sns.distplot(d, ax=ax_hist,color=color)

    elevation_range = interval_ranges[i]
    fig.suptitle(f"Evapotranspiration in elevation range: {elevation_range} ")
    fig.savefig(f'figs/elevation_bin{i}.png')

#%%
# ------------------------------------------------------------------------------
# Geographic plotting
# ------------------------------------------------------------------------------
import cartopy.crs as ccrs

def plot_xarray_on_map(da,borders=True,coastlines=True,**kwargs):
    """"""
    # get the center points for the maps
    mid_lat = np.mean(da.lat.values)
    mid_lon = np.mean(da.lon.values)
    # create the base layer
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(mid_lon, mid_lat))
    # ax = plt.axes(projection=ccrs.Orthographic(mid_lon, mid_lat))

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    da.plot(ax=ax, transform=ccrs.PlateCarree(),vmin=vmin, vmax=vmax);

    ax.coastlines();
    ax.add_feature(cartopy.feature.BORDERS,linestyle=':');
    fig = plt.gcf()
    return fig, ax

kwargs = {"vmin":0,"vmax":3.5}
fig1, ax1 = plot_xarray_on_map(ds_valid.holaps_evapotranspiration.mean(dim='time'),**kwargs)
fig1.suptitle('Holaps Evapotranspiration')
plt.gca().outline_patch.set_visible(False)
fig1.savefig('figs/holaps_clean_et_map.png')
plt.close()

fig2, ax2 = plot_xarray_on_map(ds_valid.modis_evapotranspiration.mean(dim='time'),**kwargs)
fig2.suptitle('MODIS Evapotranspiration')
plt.gca().outline_patch.set_visible(False)
fig2.savefig('figs/modis_clean_et_map.png')
plt.close()

fig3, ax3 = plot_xarray_on_map(ds_valid.gleam_evapotranspiration.mean(dim='time'),**kwargs)
fig3.suptitle('GLEAM Evapotranspiration')
plt.gca().outline_patch.set_visible(False)
fig3.savefig('figs/gleam_clean_et_map.png')
plt.close()

#%%
# ------------------------------------------------------------------------------
# Plot the mean spatial patterns of the RAW data
# ------------------------------------------------------------------------------
kwargs = {"vmin":0,"vmax":3.5}

fig1,ax1=plt.subplots(figsize=(12,8))
h.raw_data.mean(dim='time').plot(ax=ax1,**kwargs)
ax1.set_title(f"HOLAPS Mean Latent Heat Flux [{h.raw_data.units}]")
fig1.savefig('figs/holaps_map1.png')
# fig1.savefig('figs/svg/holaps_map1.svg')

fig2,ax2=plt.subplots(figsize=(12,8))
# transpose because longitude/latitude => latitude/longitude for plotting
m.raw_data.mean(dim='time').plot(ax=ax2,**kwargs)
ax2.set_title(f"MODIS Monthly Actual Evapotranspiration [{m.raw_data.units}]")
fig2.savefig('figs/modis_map1.png')
# fig2.savefig('figs/svg/modis_map1.svg')

fig3,ax3=plt.subplots(figsize=(12,8))
g.raw_data.mean(dim='time').plot(ax=ax3,**kwargs)
ax3.set_title(f"GLEAM Monthly mean daily Actual Evapotranspiration [{g.raw_data.units}]")
fig3.savefig('figs/gleam_map1.png')
# fig3.savefig('figs/svg/gleam_map1.svg')

# ------------------------------------------------------------------------------
# Plot the spatial patterns of cleaned data
# ------------------------------------------------------------------------------
kwargs = {"vmin":0,"vmax":3.5}

fig4,ax4=plt.subplots(figsize=(12,8))
h.clean_data.mean(dim='time').plot(ax=ax4,**kwargs)
ax4.set_title(f"HOLAPS Mean Evapotranspiration [{h.clean_data.units}]")
fig4.savefig('figs/holaps_mm_map1.png')
# fig4.savefig('figs/svg/holaps_mm_map1.svg')

fig6,ax6=plt.subplots(figsize=(12,8))
g.clean_data.mean(dim='time').plot(ax=ax6,**kwargs)
ax6.set_title(f"GLEAM Monthly mean daily Actual Evapotranspiration [{g.clean_data.units}]")
fig6.savefig('figs/gleam_msk_map1.png')
# fig5.savefig('figs/svg/holaps_repr_map1.svg')

fig7,ax7=plt.subplots(figsize=(12,8))
m.clean_data.mean(dim='time').plot(ax=ax7,**kwargs)
ax7.set_title(f"MODIS Monthly Actual Evapotranspiration [{modis_msk.units}]")
fig7.savefig('figs/modis_msk_map1.png')
# fig5.savefig('figs/svg/holaps_repr_map1.svg')

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
h_flat = drop_nans_and_flatten(ds.holaps_evapotranspiration)

sns.set_color_codes()
sns.distplot(h_flat, ax=ax1, color=h_col)

ax1.set_title(f'Density Plot of HOLAPS Mean Monthly Evapotranspiration [{ds.holaps_evapotranspiration.units}]')
ax1.set_xlabel(f'Mean Monthly Evapotranspiration [{ds.holaps_evapotranspiration.units}]')
fig1.savefig('figs/holaps_hist1.png')
# fig1.savefig('figs/holaps_hist1.svg')

# Plot modis
# -----------
fig2,ax2=plt.subplots(figsize=(12,8))
m_flat = drop_nans_and_flatten(ds.modis_evapotranspiration)

sns.set_color_codes()
sns.distplot(m_flat,ax=ax2, color=m_col)

ax2.set_title(f'Density Plot of MODIS Monthly Actual Evapotranspiration [{ds.modis_evapotranspiration.units}]')
ax2.set_xlabel(f'Monthly Actual Evapotranspiration [{ds.modis_evapotranspiration.units}]')
fig2.savefig('figs/modis_hist1.png')
# fig2.savefig('figs/modis_hist1.svg')

# Plot gleam
# -----------
fig3,ax3=plt.subplots(figsize=(12,8))
g_flat = drop_nans_and_flatten(ds.gleam_evapotranspiration)

sns.set_color_codes()
sns.distplot(g_flat,ax=ax3, color=g_col)

ax3.set_title(f'Density Plot of GLEAM Monthly mean daily Actual Evapotranspiration [{g.clean_data.units}] ')
ax3.set_xlabel(f'Monthly mean daily Actual Evapotranspiration [{g.clean_data.units}]')
fig3.savefig('figs/gleam_hist1.png')
# fig3.savefig('figs/gleam_hist1.svg')

#%%
# ------------------------------------------------------------------------------
# Plot the Time Series of the points (spatial mean)
# ------------------------------------------------------------------------------

# get table of time series
tseries = ds.mean(dim=['lat','lon']).to_dataframe().drop(columns='units')

fig,ax = plt.subplots(figsize=(12,8))
ds.mean(dim=['lat','lon']).holaps_evapotranspiration.plot(ax=ax)

#%%
# ------------------------------------------------------------------------------
# Plot the Time Series of different locations (spatial mean)
# ------------------------------------------------------------------------------



#%%
# ------------------------------------------------------------------------------
# Plot the Time Series of different locations (points)
# ------------------------------------------------------------------------------


#%%
# ------------------------------------------------------------------------------
# Plot the Climatology of different products
# ------------------------------------------------------------------------------


#
# # plot holaps_reprojected
# # -----------------------
# fig4,ax4=plt.subplots(figsize=(12,8))
# h_repr = drop_nans_and_flatten(holaps_repr)
#
# sns.set_color_codes()
# sns.distplot(h_repr,ax=ax4, color=h_col)
#
# ax4.set_title(f'Density Plot of HOLAPS Monthly Actual Evapotranspiration [mm day-1] ')
# ax4.set_xlabel(f'Monthly mean daily Actual Evapotranspiration [mm day-1]')
# fig4.savefig('figs/holaps_repr_hist1.png')
# # fig4.savefig('figs/holaps_repr_hist1.svg')

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
