"""
Satellite Data Analysis: PACE vs MODIS Chlorophyll & Rrs Comparison
Author: Punya P
"""

# Imports the necessary python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress

from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load Data
chl_path = "data/Chl_March_5_31_wholedataset.csv"

rrs_paths = {
    "412": "data/RRS_March_5_31_wholedataset_412.csv",
    "443": "data/RRS_March_5_31_wholedataset_443.csv",
    "469": "data/RRS_March_5_31_wholedataset_469.csv",
    "488": "data/RRS_March_5_31_wholedataset_488.csv",
    "531": "data/RRS_March_5_31_wholedataset_531.csv",
    "547": "data/RRS_March_5_31_wholedataset_547.csv",
    "555": "data/RRS_March_5_31_wholedataset_555.csv",
    "645": "data/RRS_March_5_31_wholedataset_645.csv",
    "667": "data/RRS_March_5_31_wholedataset_667.csv",
    "678": "data/RRS_March_5_31_wholedataset_678.csv",
}

# Chlorophyll data
chl_df = pd.read_csv(chl_path)
chl_df = chl_df.rename(columns={
    'log_pace': 'logPace',
    'log_corrected': 'logModis'
})

# Load Rrs Data 
rrs_list = []

for band, path in rrs_paths.items():
    df = pd.read_csv(path)

    df = df.rename(columns={
        'X': band,
        'Y_pred': f'Rrs_{band}'
    })

    # Convert log10 to linear
    df[[band, f'Rrs_{band}']] = 10 ** df[[band, f'Rrs_{band}']]

    rrs_list.append(df)

rrs = pd.concat(rrs_list, axis=1)

# Preprocessing
chl_df = chl_df.replace([np.inf, -np.inf], np.nan)
chl_df = chl_df.dropna(subset=['logPace', 'logModis'])

x = chl_df['logPace']
y = chl_df['logModis']

# Remove outliers (1–99%)
q_low, q_high = 0.01, 0.99
mask = (
    (x > x.quantile(q_low)) & (x < x.quantile(q_high)) &
    (y > y.quantile(q_low)) & (y < y.quantile(q_high))
)
x = x[mask]
y = y[mask]

# Chlorophyll Comparison Plot
fig, ax = plt.subplots(figsize=(5, 6))

hb = ax.hexbin(
    x, y,
    gridsize=80,
    cmap='viridis',
    norm=LogNorm(),
    mincnt=1
)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(hb, cax=cax)
cbar.set_label('Counts')

# 1:1 line
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
ax.plot([min_val, max_val], [min_val, max_val],
        '--', color='black', linewidth=1.5)

# Metrics
slope, intercept, r_value, _, _ = linregress(x, y)
r2 = r_value**2
mse = mean_squared_error(x, y)
mae = mean_absolute_error(x, y)

ax.text(0.02, 0.95, f'R² = {r2:.2f}', transform=ax.transAxes)
ax.text(0.02, 0.90, f'MSE = {mse:.2f}', transform=ax.transAxes)
ax.text(0.02, 0.85, f'MAE = {mae:.2f}', transform=ax.transAxes)

ax.set_xlabel('Log Chlorophyll (PACE)')
ax.set_ylabel('Log Chlorophyll (MODIS)')
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal')

ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Multi-band Rrs Comparison Plot
bands = ['412','443','469','488','531','547','555','645','667','678']

fig, axes = plt.subplots(5, 2, figsize=(10, 20), constrained_layout=True)
axes = axes.flatten()

all_hbs = []

for i, band in enumerate(bands):
    ax = axes[i]

    x = rrs[f'Rrs_{band}']
    y = rrs[band]

    # Clean
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Remove outliers
    q_low, q_high = 0.01, 0.99
    mask = (
        (x > np.quantile(x, q_low)) & (x < np.quantile(x, q_high)) &
        (y > np.quantile(y, q_low)) & (y < np.quantile(y, q_high))
    )
    x = x[mask]
    y = y[mask]

    hb = ax.hexbin(
        x, y,
        gridsize=70,
        cmap='viridis',
        norm=LogNorm(),
        mincnt=1
    )
    all_hbs.append(hb)

    # 1:1 line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            '--', color='black', linewidth=1)

    # Regression
    slope, intercept, r_value, _, _ = linregress(x, y)
    r2 = r_value**2
    mse = mean_squared_error(x, y)
    mae = mean_absolute_error(x, y)

    ax.plot(x, slope * x + intercept, linewidth=1)

    # Labels
    ax.set_title(f'{band} nm')
    ax.set_xlabel('MODIS Rrs')
    ax.set_ylabel('PACE Rrs')

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax.grid(alpha=0.3)

    # Metrics
    ax.text(0.05, 0.92, f'R²={r2:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'MSE={mse:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.78, f'MAE={mae:.2f}', transform=ax.transAxes)

# Add colorbars to right column
for row in range(5):
    hb = all_hbs[row * 2 + 1]
    ax = axes[row * 2 + 1]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label('Counts')

plt.show()
