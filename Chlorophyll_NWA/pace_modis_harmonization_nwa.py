"""
PACE–MODIS Chlorophyll Harmonization + NWA Alignment
Author: Punya. P
"""

# Imports necessary Python Packages
import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt

import earthaccess
from sklearn.linear_model import HuberRegressor
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Authentication
auth = earthaccess.login(persist=True)

# Search Parameters
tspan = ("2024-03-05", "2024-03-31")

# Data Search
PACE_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_CHL",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

MODIS_results = earthaccess.search_data(
    short_name="MODISA_L3m_CHL",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

# Preprocessing Function
def time_from_attr(ds):
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["date"] = ((), np.datetime64(datetime, "ns"))
    ds = ds.set_coords("date")
    return ds.sel(lat=slice(33, -1), lon=slice(42, 102))

# Load Data
fileset1 = earthaccess.open(PACE_results)
fileset2 = earthaccess.open(MODIS_results)

datasetP = xr.open_mfdataset(
    fileset1, preprocess=time_from_attr,
    combine="nested", concat_dim="date", parallel=True
)

datasetM = xr.open_mfdataset(
    fileset2, preprocess=time_from_attr,
    combine="nested", concat_dim="date", parallel=True
)

PACE_CHL = datasetP.chlor_a.mean(dim=["date"])
MODIS_CHL = datasetM.chlor_a.mean(dim=["date"])

# Alignment and Cleaning for Band Harmonization
pace, modis = xr.align(PACE_CHL, MODIS_CHL, join='inner')

mask = (pace > 0) & (modis > 0)
pace = pace.where(mask)
modis = modis.where(mask)

log_pace = np.log10(pace)
log_modis = np.log10(modis)

# Regime Classification
def classify_regime(chl):
    return xr.where(chl < 0.1, 0,
           xr.where(chl < 1.0, 1, 2))

regime = classify_regime(pace)

# Stack Dataset
stacked = xr.Dataset({
    'log_pace': log_pace,
    'log_modis': log_modis,
    'regime': regime
}).stack(points=('lat', 'lon')).dropna('points')

df = stacked.to_dataframe()

# Regression per Regime
models = {}
df['log_corrected'] = np.nan

for r in [0, 1, 2]:
    sub = df[df['regime'] == r]

    if len(sub) < 5:
        print(f"Skipping regime {r}")
        continue

    X = sub[['log_modis']].values
    y = sub['log_pace'].values

    model = HuberRegressor()
    model.fit(X, y)

    models[r] = model
    df.loc[sub.index, 'log_corrected'] = model.predict(X)

# Back to xarray
corrected = xr.DataArray(
    df['log_corrected'],
    dims=['points'],
    coords={'points': stacked['points']}
).unstack('points')

MODIS_CHL_CORRECTED = 10 ** corrected


# Convert to Dask for NWA analysis
Pace_da = da.from_array(PACE_CHL.values, chunks=(256, 256))
Modis_da = da.from_array(MODIS_CHL_CORRECTED.values, chunks=(256, 256))

Pace_da = da.where(da.isnan(Pace_da), -999, Pace_da)
Modis_da = da.where(da.isnan(Modis_da), -999, Modis_da)

# NWA Functions
def pixel_score(a, b, sigma=0.05):
    return np.exp(-((a - b) ** 2) / (2 * sigma ** 2))

def nw_2d_optimized(img1, img2, gap_penalty=-0.1):
    H, W = img1.shape

    prev = np.arange(W + 1) * gap_penalty
    curr = np.zeros(W + 1)

    for i in range(1, H + 1):
        curr[0] = i * gap_penalty

        for j in range(1, W + 1):
            match = prev[j-1] + pixel_score(img1[i-1, j-1], img2[i-1, j-1])
            delete = prev[j] + gap_penalty
            insert = curr[j-1] + gap_penalty
            curr[j] = max(match, delete, insert)

        prev, curr = curr, prev

    return prev[W]

def block_nw(img1, img2):
    return np.array([[nw_2d_optimized(img1, img2)]])

alignment_map = da.map_blocks(
    block_nw,
    Modis_da,
    Pace_da,
    dtype=float,
    chunks=(1, 1)
)

nwa_map = alignment_map.compute()

# Metrics
bias = MODIS_CHL_CORRECTED.values - PACE_CHL.values

sam = np.arccos(
    (MODIS_CHL_CORRECTED.values * PACE_CHL.values) /
    (np.sqrt(MODIS_CHL_CORRECTED.values**2) *
     np.sqrt(PACE_CHL.values**2) + 1e-8)
)

mask = np.isnan(MODIS_CHL_CORRECTED.values) | np.isnan(PACE_CHL.values)
bias[mask] = np.nan
sam[mask] = np.nan

# Scaling
bmax = np.nanpercentile(np.abs(bias), 95)
sam_min, sam_max = np.nanpercentile(sam, [5, 95])
nwa_min, nwa_max = np.percentile(nwa_map, [5, 95])

# Plotting
def plot_map(data, title, cmap, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label)

    plt.tight_layout()
    plt.show()

# Plots
plot_map(bias, "Bias (MODIS - PACE)", 'RdBu_r', -bmax, bmax, "Chl Difference")
plot_map(nwa_map, "NWA Alignment Score", 'viridis', nwa_min, nwa_max, "Score")
plot_map(sam, "Spectral Angle Mapper", 'plasma', sam_min, sam_max, "Radians")
