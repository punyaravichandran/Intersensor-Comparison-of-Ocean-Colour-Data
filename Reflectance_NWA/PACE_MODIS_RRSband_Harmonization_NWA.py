# -*- coding: utf-8 -*-

"""
PACE–MODIS RRS Band Harmonization + NWA Alignment
Author: Punya P
"""

# Imports necessary python packages
import earthaccess
import numpy as np
import xarray as xr
import dask.array as da
import xesmf as xe

from sklearn.ensemble import RandomForestRegressor

# Earthdata Login
earthaccess.login(persist=True)

# Parameters
bbox = (41, -1, 102, 33)
tspan = ("2024-03-05", "2024-03-31")

target_wavelengths = [412, 443, 469, 488, 531, 547, 555, 645, 667, 678]

# Data Search
PACE_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_RRS",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

MODIS_results = earthaccess.search_data(
    short_name="MODISA_L3m_RRS",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

# Attach time coordinate and subset region
def time_from_attr(ds):
    dt = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["date"] = ((), np.datetime64(dt, "ns"))
    ds = ds.set_coords("date")
    return ds.sel(lat=slice(33, -1), lon=slice(41, 102))

# Load Data
fileset1 = earthaccess.open(PACE_results)
fileset2 = earthaccess.open(MODIS_results)

datasetP = xr.open_mfdataset(
    fileset1,
    preprocess=time_from_attr,
    combine="nested",
    concat_dim="date",
    parallel=True
)

datasetM = xr.open_mfdataset(
    fileset2,
    preprocess=time_from_attr,
    combine="nested",
    concat_dim="date",
    parallel=True
)

# PACE Processing
PACE_rrs = datasetP["Rrs"].sel(wavelength=slice(400, 701))
PACE_rrs_mean = PACE_rrs.mean(dim="date")
PACE_interp = PACE_rrs_mean.interp(wavelength=target_wavelengths)

# MODIS Processing
MOD_mean = datasetM.mean(dim="date")

MOD_bands = MOD_mean[
    [
        'Rrs_412','Rrs_443','Rrs_469','Rrs_488',
        'Rrs_531','Rrs_547','Rrs_555','Rrs_645',
        'Rrs_667','Rrs_678'
    ]
]

# Band Selection (678 nm)
pace_678 = PACE_interp.sel(wavelength=678)
modis_678 = MOD_bands["Rrs_678"]

rrs_443 = MOD_bands["Rrs_443"]
rrs_555 = MOD_bands["Rrs_555"]

# Region Classification
ratio = rrs_555 / rrs_443

def classify_water(r):
    """0=open ocean, 1=coastal"""
    return xr.where(r > 1.0, 1, 0)

region = classify_water(ratio)

# Log Transform
mask = (modis_678 > 0) & (pace_678 > 0)

X_log = np.log10(modis_678.where(mask))
Y_log = np.log10(pace_678.where(mask))

# DataFrame Conversion
stacked = xr.Dataset({
    "X": X_log,
    "Y": Y_log,
    "region": region
}).stack(z=("lat", "lon")).dropna("z")

df = stacked.to_dataframe()

# Random Forest Harmonization
models = {}
df["Y_pred"] = np.nan

for r in [0, 1]:
    sub = df[df["region"] == r]

    if len(sub) < 100:
        print(f"Skipping region {r}")
        continue

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    X_r = sub["X"].values.reshape(-1, 1)
    Y_r = sub["Y"].values

    model.fit(X_r, Y_r)
    models[r] = model

    df.loc[sub.index, "Y_pred"] = model.predict(X_r)

# Reconstruct Field
corrected = xr.DataArray(
    df["Y_pred"],
    dims=["z"],
    coords={"z": stacked["z"]}
).unstack("z")

rrs_corrected = 10 ** corrected

# Regridding
regridder = xe.Regridder(rrs_corrected, pace_678, "bilinear", reuse_weights=True)
rrs_interp = regridder(rrs_corrected)

# Dask Conversion
Pace_da = da.from_array(pace_678.values, chunks=(256, 256))
Modis_da = da.from_array(rrs_interp.values, chunks=(256, 256))

# Fill NaNs
Pace_da = da.where(da.isnan(Pace_da), -999, Pace_da)
Modis_da = da.where(da.isnan(Modis_da), -999, Modis_da)

# Scale
Pace_da *= 1000
Modis_da *= 1000

# Match chunks
chunks = (256, 256)
Pace_da = Pace_da.rechunk(chunks)
Modis_da = Modis_da.rechunk(chunks)

# Needleman-Wunsch (Optimized)
def nw_2d_optimized(img1, img2, gap_penalty=-0.1):

    H, W = img1.shape

    prev = np.arange(W + 1) * gap_penalty
    curr = np.zeros(W + 1)

    for i in range(1, H + 1):
        curr[0] = i * gap_penalty

        for j in range(1, W + 1):
            diff = img1[i-1, j-1] - img2[i-1, j-1]
            match = prev[j-1] + np.exp(-(diff**2) / (2 * 0.05**2))
            delete = prev[j] + gap_penalty
            insert = curr[j-1] + gap_penalty

            curr[j] = max(match, delete, insert)

        prev, curr = curr, prev

    return prev[W]


def block_nw(img1, img2):
    return np.array([[nw_2d_optimized(img1, img2)]])

# Dask Block Processing
alignment_map = da.map_blocks(
    block_nw,
    Modis_da,
    Pace_da,
    dtype=float,
    chunks=(1, 1)
)

result = alignment_map.compute()

# Evaluation Metrics
Modis_412 = rrs_interp.values   # corrected & regridded MODIS 
pace_412  = pace_678.values     # PACE reference

# Bias
bias = Modis_412 - pace_412

# NWA Map
nwa_map = result

# Spectral Angle Mapper (SAM)
sam = np.arccos(
    (Modis_412 * pace_412) /
    (np.sqrt(Modis_412**2) * np.sqrt(pace_412**2) + 1e-8)
)

# Mask invalid values
mask = np.isnan(Modis_412) | np.isnan(pace_412)
bias = np.where(mask, np.nan, bias)
sam  = np.where(mask, np.nan, sam)

# Smart Scaling
# Bias scaling (robust)
bmax = np.nanpercentile(np.abs(bias), 95)
# NWA scaling
nwa_min, nwa_max = np.percentile(nwa_map, [5, 95])
# SAM scaling
sam_min, sam_max = np.percentile(sam, [5, 95])

