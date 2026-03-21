"""
Satellite Data Analysis: PACE vs MODIS Chlorophyll & Rrs Comparison
Author: Punya P
"""

# Imports necessary Python Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress
from matplotlib.ticker import FormatStrFormatter


# Load Data
chl_file = "Chl_March_5_31_wholedataset.csv"
rrs_file = "RRS_March_5_31_wholedataset.csv"

eez_chl = pd.read_csv(chl_file)
rrs_data = pd.read_csv(rrs_file)


# Preprocess Chlorophyll
df = eez_chl.rename(columns={
    'log_pace': 'logPace',
    'log_corrected': 'logModis'
})

df = df[['logPace', 'logModis']].dropna()

# Create Rrs DataFrames
bands = [
    ('Rrs_412', '412'),
    ('Rrs_443', '443'),
    ('Rrs_469', '469'),
    ('Rrs_488', '488'),
    ('Rrs_531', '531'),
    ('Rrs_547', '547.0'),
    ('Rrs_555', '555.0'),
    ('Rrs_645', '645.0'),
    ('Rrs_667', '667.0'),
    ('Rrs_678', '678.0')
]

rrs_dfs = []

for modis_col, pace_col in bands:
    temp_df = rrs_data[[modis_col, pace_col]].copy()
    temp_df = temp_df.multiply(1000).dropna()
    rrs_dfs.append((temp_df, modis_col, pace_col))


# Plot 1: Chlorophyll Comparison
plt.figure(figsize=(6, 6))

hb = plt.hexbin(df['logPace'], df['logModis'],
                gridsize=60, cmap='Greens', vmax=100)

plt.colorbar(hb, label='Counts')

# 1:1 line
min_val = min(df['logPace'].min(), df['logModis'].min())
max_val = max(df['logPace'].max(), df['logModis'].max())
plt.plot([min_val, max_val], [min_val, max_val], '--', color='black')

# Metrics
slope, intercept, r_value, _, _ = linregress(df['logPace'], df['logModis'])
r2 = r_value**2
mse = mean_squared_error(df['logPace'], df['logModis'])
mae = mean_absolute_error(df['logPace'], df['logModis'])

plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.90, f'MSE = {mse:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'MAE = {mae:.2f}', transform=plt.gca().transAxes)

plt.xlabel('Log Chlorophyll (PACE)')
plt.ylabel('Log Chlorophyll (MODIS)')
plt.grid(True)
plt.tight_layout()

plt.savefig("figures/chl_comparison.png", dpi=300)
plt.show()


# Plot 2: Rrs Comparison
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for i, (df_rrs, modis_col, pace_col) in enumerate(rrs_dfs):
    ax = axes[i]

    x = df_rrs[modis_col]
    y = df_rrs[pace_col]

    hb = ax.hexbin(x, y, gridsize=50, cmap='Greens', mincnt=1, vmax=100)
    fig.colorbar(hb, ax=ax)

    # 1:1 line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='red')

    # Metrics
    slope, intercept, r_value, _, _ = linregress(x, y)
    r2 = r_value**2
    mse = mean_squared_error(x, y)
    mae = mean_absolute_error(x, y)

    ax.text(0.05, 0.95, f'R² = {r2:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.90, f'MSE = {mse:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'MAE = {mae:.2f}', transform=ax.transAxes)

    # Labels
    band = modis_col.split('_')[-1]
    ax.set_xlabel(f'MODIS {band}')
    ax.set_ylabel(f'PACE {band}')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True)

plt.tight_layout()

plt.savefig("figures/rrs_comparison.png", dpi=300)
plt.show()
