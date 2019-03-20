# subset_esa_cci.py
import xarray as xr
from pathlib import Path

lonmin=32.6
lonmax=51.8
latmin=-5.0
latmax=15.2

base_data_path = Path("/soge-home/projects/crop_yield/EGU_compare")
in_file = base_data_path / "ESACCI-LC-L4-LCCS-Map-300m-P5Y-2005-v1.6.1.nc"
mid_file = base_data_path / "ESACCI_LC_L4-Map_300m.nc"
# out_file = base_data_path / "EA_ESACCI_LC.nc"

# chunk the data using DASK (BETTER THAN NCO!!!!!)
lc = xr.open_dataset(in_file, chunks={'lat': 1000,'lon': 1000})

lc = lc.sel(lat=slice(latmax,latmin), lon=slice(lonmin,lonmax))
lc.to_netcdf(mid_file)

print(f"WRITTEN TO NETCDF {mid_file}")

#
cmd = """
base_data_path=/soge-home/projects/crop_yield/EGU_compare

lonmin=32.6
lonmax=51.8
latmin=-5.0
latmax=15.2

in_file=$base_data_path/ESACCI_LC_L4-Map_300m.nc
out_file=$base_data_path/EA_ESACCI_LC.nc

cdo sellonlatbox,$lonmin,$lonmax,$latmin,$latmax $in_file $out_file
"""

os.system(cmd)
