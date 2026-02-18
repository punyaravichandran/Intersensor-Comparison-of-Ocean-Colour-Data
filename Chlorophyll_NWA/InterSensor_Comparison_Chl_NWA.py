# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 09:47:09 2026

@author: Punya  P
"""

"""
PACE vs MODIS Chlorophyll Comparison
Author: Your Name
Description:
    - Loads multi-file NetCDF chlorophyll datasets
    - Crops to bounding box
    - Clips to shapefile
    - Applies percentile filtering
    - Resamples to same grid
    - Computes pixel match %, similarity matrix, RMSE
"""


from numba import jit
import xarray as xr
import h5netcdf
import numpy as np
import rioxarray
import cv2
import geopandas

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colormaps
import matplotlib.cbook
from mpl_toolkits.basemap import Basemap

from shapely.geometry import mapping
import warnings


# Set the folder location

PACE_PATH = r"D:\PACE\March24_PACE_CHLdata\PACE_1KM_CHL_8DAY\SingleFile\*.nc"
MODIS_PATH = r"D:\PACE\MODIS\March24_MODISCHLdata\Single_Files\*.nc"
SHAPEFILE_PATH = r"E:\SC21D005\SC21D005\SC21D005\google earth engine codes\SHAPEFILES\ion\ion\Final_nio1.shp"

# DATA LOADING

#For PACE
PACEdataset = xr.open_mfdataset(
    PACE_PATH,
    combine="nested",
    concat_dim="date",
)

PACEdss = PACEdataset["chlor_a"].mean(dim='date')

#Crop the region 
min_lon = 42.2115910
min_lat = -0.8387938
max_lon = 100.12590
max_lat = 31.18586
mask_lon = (PACEdss.lon >= min_lon) & (PACEdss.lon <= max_lon)
mask_lat = (PACEdss.lat >= min_lat) & (PACEdss.lat <= max_lat)
cropped_PACE = PACEdss.where(mask_lon & mask_lat, drop=True)

#Rename the coordinates to x and y    
cropped_PACE_1= cropped_PACE.rename({'lon': 'x','lat': 'y'})
#Set the Projection
cropped_PACE_1.rio.write_crs("EPSG:4326", inplace=True)

#Clip the data with northern Indian Ocean shapefile
ion = geopandas.read_file(SHAPEFILE_PATH)
clipped_PACE_1= cropped_PACE_1.rio.clip(ion.geometry.values, ion.crs)
clipped_PACE_1.plot()
#Rename cordinates to  lat and lon
clipped_PACE_2 = clipped_PACE_1.rename({'x': 'lon','y': 'lat'})

###Percentile Filtering ###
clipped_PACE_3 = clipped_PACE_2.chunk(dict(lat=-1, lon=-1))
# Example: values between 2 and 98th percentile
percentile_25 = clipped_PACE_3.quantile(0.02, dim=('lat', 'lon'))
percentile_75 = clipped_PACE_3.quantile(0.98, dim=('lat', 'lon'))
# Fix: Call .compute() before .item() for printing
filtered_in_range = clipped_PACE_3.where(
    (clipped_PACE_3 > percentile_25) & (clipped_PACE_3 < percentile_75)
)
print(filtered_in_range)

#Plotting with Basemap
lat_min = clipped_PACE_3.lat.min()
lat_max = clipped_PACE_3.lat.max()
lon_min = clipped_PACE_3.lon.min()
lon_max = clipped_PACE_3.lon.max()

fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,resolution='l', epsg=4326)
m.drawcoastlines(linewidth=0.8)
m.fillcontinents(color='#E8E8E8')
im = m.imshow(np.flip(clipped_PACE_3.fillna(0), axis=0),cmap='turbo',interpolation='nearest')
cb = m.colorbar(im,location='right',pad="10%")
cb.set_label('PACE Chlorophyll Concentration (mg/m3)', size=15,weight='bold')
plt.show()



#For MODIS    
MODISdataset = xr.open_mfdataset(
    MODIS_PATH,
    combine="nested",
    concat_dim="date",
)
 
MODISdss = MODISdataset["chlor_a"].mean(dim='date')

#Crop the region 
mask_lon = (MODISdss.lon >= min_lon) & (MODISdss.lon <= max_lon)
mask_lat = (MODISdss.lat >= min_lat) & (MODISdss.lat <= max_lat)
cropped_MODIS = MODISdss.where(mask_lon & mask_lat, drop=True)

#Rename the coordinates to x and y    
cropped_MODIS_1= cropped_MODIS.rename({'lon': 'x','lat': 'y'})
#Set the Projection
cropped_MODIS_1.rio.write_crs("EPSG:4326", inplace=True)

#Clip the data with northern Indian Ocean shapefile
clipped_MODIS_1= cropped_MODIS_1.rio.clip(ion.geometry.values, ion.crs)
#Rename cordinates to  lat and lon
clipped_MODIS_2 = clipped_MODIS_1.rename({'x': 'lon','y': 'lat'})

###Percentile Filtering ###
clipped_MODIS_3 = clipped_MODIS_2.chunk(dict(lat=-1, lon=-1))
# Example: values between 2 and 98th percentile
percentile_25 = clipped_MODIS_3.quantile(0.02, dim=('lat', 'lon'))
percentile_75 = clipped_MODIS_3.quantile(0.98, dim=('lat', 'lon'))
# Fix: Call .compute() before .item() for printing
filtered_in_range = clipped_MODIS_3.where(
    (clipped_MODIS_3 > percentile_25) & (clipped_MODIS_3 < percentile_75)
)
print(filtered_in_range)

#Plotting with Basemap
fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,resolution='l', epsg=4326)
m.drawcoastlines(linewidth=0.8)
m.fillcontinents(color='#E8E8E8')
im = m.imshow(np.flip(clipped_MODIS_3.fillna(0), axis=0),cmap='turbo',interpolation='nearest')
cb = m.colorbar(im,location='right',pad="10%")
cb.set_label('MODIS Chlorophyll Concentration (mg/m3)', size=15,weight='bold')
plt.show()



#convert xarray to numpy for PACE
PACE_CHL_NP = clipped_PACE_1.values
PACE_CHL_RES = cv2.resize(PACE_CHL_NP, dsize=(1390, 762), interpolation=cv2.INTER_CUBIC)
PACE_CHL_RES1 = np.nan_to_num(PACE_CHL_RES, nan=-999)

#convert xarray to numpy for MODIS
MODIS_CHL_NP = clipped_MODIS_1.values
MODIS_CHL_RES = cv2.resize(MODIS_CHL_NP, dsize=(1390, 762), interpolation=cv2.INTER_CUBIC)
MODIS_CHL_RES1 = np.nan_to_num(MODIS_CHL_RES, nan=-999)

# Mask NaNs
mask = ~np.isnan(PACE_CHL_RES) & ~np.isnan(MODIS_CHL_RES)
# Pixel-wise match (exact match)
tol = 0.5 # adjust tolerance
pixel_match = (abs(PACE_CHL_RES - MODIS_CHL_RES) <= tol) & mask
# Count matches and total valid pixels
n_matches = pixel_match.sum().item()
n_total = mask.sum().item()
# Percentage match
percentage_match = (n_matches / n_total) * 100 if n_total > 0 else np.nan
print(f"Pixel match percentage: {percentage_match:.2f}%")

## Needleman Wunsch Algorithm ##
"""
   generating a score matrix for pixel-to-pixel comparison of two 2D arrays.
    arr1, arr2: 2D numpy arrays representing the reflectance datasets.
    match_score: Score for matching reflectance values.
    mismatch_score: Score for mismatched reflectance values.
    gap_penalty: Penalty for gaps in alignment.
    score_matrix: 2D numpy array representing the score of pixel-wise comparison.
"""
# Needleman-Wunsch function for pixel comparison
def needleman_wunsch(modis_data, aviris_data, match_score=1, mismatch_score=-1, gap_penalty=0):
    rows, cols = modis_data.shape
    score_matrix = np.zeros((rows+1, cols+1))

    # Initialize score matrix with gap penalties
    for i in range(1, rows+1):
        score_matrix[i][0] = score_matrix[i-1][0] + gap_penalty
    for j in range(1, cols+1):
        score_matrix[0][j] = score_matrix[0][j-1] + gap_penalty

    # Fill the score matrix
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            match = score_matrix[i-1][j-1] + (match_score if modis_data[i-1, j-1] == aviris_data[i-1, j-1] else mismatch_score)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    return score_matrix

# Similarity matrix function (using squared differences)
def similarity_matrix(modis_data, aviris_data):
    return (modis_data - aviris_data) ** 2  

def rmse_pixelwise(modis_data, aviris_data):
    return np.sqrt((modis_data - aviris_data) ** 2)


# Perform Needleman-Wunsch alignment to get the score matrix
alignment_score_matrix = needleman_wunsch(MODIS_CHL_RES1, PACE_CHL_RES1)
# Generate similarity matrix
sim_matrix = similarity_matrix(MODIS_CHL_RES1, PACE_CHL_RES1)
# Compute RMSE matrix (pixel-wise)
rmse_matrix = rmse_pixelwise(MODIS_CHL_RES1, PACE_CHL_RES1)

# Plotting the score matrix
plt.figure(figsize=(6, 5))
plt.axis('off') 
im1 = plt.imshow(alignment_score_matrix, cmap='viridis', interpolation='nearest')
cbar1 = plt.colorbar(im1, shrink=0.6, aspect=15, pad=0.02)
plt.title('Needleman-Wunsch Score Matrix')

#plot the Squared Difference
fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,resolution='l', epsg=4326)
m.drawcoastlines(linewidth=0.8)
m.fillcontinents(color='#E8E8E8')
im = m.imshow(np.flip(sim_matrix, axis=0),cmap='twilight',interpolation='nearest',vmin=0,vmax=0.8)
cb = m.colorbar(im,location='right',pad="10%")
cb.set_label('Squared Difference', size=15,weight='bold')
plt.show()

#Plot the RMSE
fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,resolution='l', epsg=4326)
m.drawcoastlines(linewidth=0.8)
m.fillcontinents(color='#E8E8E8')
im = m.imshow(np.flip(rmse_matrix, axis=0),cmap='twilight',interpolation='nearest',vmin=0,vmax=10)
cb = m.colorbar(im,location='right',pad="10%")
cb.set_label('RMSE', size=15,weight='bold')
plt.show()
