# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:01:58 2026

@author: Punya  P
"""
from numba import jit
import xarray as xr
import h5netcdf
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.cbook
from mpl_toolkits.basemap import Basemap
from matplotlib import colormaps

import cv2
import warnings


#For PACE Swath Data
PACE_filepath1 = r'D:\PACE\March24_PACE_CHLdata\AS1_PACE_OCI.20240323T085228.L2.OC_AOP.V2_0.NRT.nc'
PACE_filepath2 = r'D:\PACE\March24_PACE_CHLdata\BOB_PACE_OCI.20240323T071408.L2.OC_AOP.V2_0.NRT.nc'

#Merge Arabian Sea & Bay of bengal RRs dataset
ASdata_PACE = xr.open_dataset(PACE_filepath1, group="geophysical_data")
AS_PACE = ASdata_PACE.Rrs
LatAS = xr.open_dataset(PACE_filepath1, group="navigation_data")
LatASS = LatAS.set_coords(("longitude", "latitude"))
FinalASS = xr.merge((AS_PACE, LatASS.coords))

BOBdata_PACE = xr.open_dataset(PACE_filepath2, group="geophysical_data")
BOB_PACE = BOBdata_PACE.Rrs
LatBOB = xr.open_dataset(PACE_filepath2, group="navigation_data")
LatBOB = LatBOB.set_coords(("longitude", "latitude"))
FinalBOB = xr.merge((BOB_PACE, LatBOB.coords))

#Combine Arabian Sea and Bay of Bengal files
min_lines = min(FinalBOB.sizes['number_of_lines'], FinalASS.sizes['number_of_lines'])
ds1_trimmed = FinalBOB.isel(number_of_lines=slice(0, min_lines))
ds2_trimmed = FinalASS.isel(number_of_lines=slice(0, min_lines))
assert ds1_trimmed.sizes["number_of_lines"] == ds2_trimmed.sizes["number_of_lines"]
PACE_mosaic = xr.concat([ds1_trimmed, ds2_trimmed], dim="pixels_per_line")

#PACE_wavelength band number corresponding to wavelength
#30:412, 42:443, 60:488, 84:547, 125:645, 152:678

#For MODIS Swath Data

filepath1 = r'D:\PACE\MODIS\Reflectance-March21_25\MODISA_L2_OC_R2022.0-20240905_061353\AQUA_MODIS.20240320T083500.L2.OC.nc'
filepath2 = r'D:\PACE\MODIS\Reflectance-March21_25\MODISA_L2_OC_R2022.0-20240905_061353\AQUA_MODIS.20240326T092501.L2.OC.nc'
filepath3 = r'D:\PACE\MODIS\Reflectance-March21_25\MODISA_L2_OC_R2022.0-20240905_061353\AQUA_MODIS.20240322T082001.L2.OC.nc'
filepath4 = r'D:\PACE\MODIS\Reflectance-March21_25\MODISA_L2_OC_R2022.0-20240905_061353\AQUA_MODIS.20240322T095501.L2.OC.nc'
filepath5 = r'D:\PACE\MODIS\Reflectance-March21_25\MODISA_L2_OC_R2022.0-20240905_061353\AQUA_MODIS.20240323T103501.L2.OC.nc'
filepath6 = r'D:\PACE\MODIS\Reflectance-March21_25\MODISA_L2_OC_R2022.0-20240905_061353\AQUA_MODIS.20240325T102001.L2.OC.nc'

MODIS1 = xr.open_dataset(filepath1, group="geophysical_data")
MODIS2 = xr.open_dataset(filepath2, group="geophysical_data")
MODIS3 = xr.open_dataset(filepath3, group="geophysical_data")
MODIS4 = xr.open_dataset(filepath4, group="geophysical_data")
MODIS5 = xr.open_dataset(filepath5, group="geophysical_data")
MODIS6 = xr.open_dataset(filepath6, group="geophysical_data")

LatLon1 = xr.open_dataset(filepath1, group="navigation_data")
LatLon1 = LatLon1.set_coords(("longitude", "latitude"))
LatLon2 = xr.open_dataset(filepath2, group="navigation_data")
LatLon2 = LatLon2.set_coords(("longitude", "latitude"))
LatLon3 = xr.open_dataset(filepath3, group="navigation_data")
LatLon3 = LatLon3.set_coords(("longitude", "latitude"))
LatLon4 = xr.open_dataset(filepath4, group="navigation_data")
LatLon4 = LatLon4.set_coords(("longitude", "latitude"))
LatLon5 = xr.open_dataset(filepath5, group="navigation_data")
LatLon5 = LatLon5.set_coords(("longitude", "latitude"))
LatLon6 = xr.open_dataset(filepath6, group="navigation_data")
LatLon6 = LatLon6.set_coords(("longitude", "latitude"))

FinalMODIS1 = xr.merge((MODIS1, LatLon1.coords))
FinalMODIS2 = xr.merge((MODIS2, LatLon2.coords))
FinalMODIS3 = xr.merge((MODIS3, LatLon3.coords))
FinalMODIS4 = xr.merge((MODIS4, LatLon4.coords))
FinalMODIS5 = xr.merge((MODIS5, LatLon5.coords))
FinalMODIS6 = xr.merge((MODIS6, LatLon6.coords))

#Combine all 6 MODIS RRs files
MODIS_42 = FinalMODIS4.combine_first(FinalMODIS2)
MODIS_426 = MODIS_42.combine_first(FinalMODIS6)
MODIS_4261 = MODIS_426.combine_first(FinalMODIS1)
MODIS_42613 = MODIS_4261.combine_first(FinalMODIS3)
MODIS_426135 = MODIS_42613.combine_first(FinalMODIS5)
MODIS_mosaic = MODIS_426135

#Select the Red band 678nm - Wavelength number152
PACErrs1 = PACE_mosaic["Rrs"].sel({"wavelength_3d": 152}) 
MODIS_rrs1 = MODIS_mosaic.Rrs_678

# Convert the xarray DataArray to a NumPy array
PACE_Rrs678 = PACErrs1.values
MODIS_Rrs678 = MODIS_rrs1.to_numpy()

#Resampling using Bicubic Interpolation
PACE_678_res = cv2.resize(PACE_Rrs678, dsize=(2544,1709), interpolation=cv2.INTER_CUBIC)
MODIS_678_res = cv2.resize(MODIS_Rrs678, dsize=(2544,1709), interpolation=cv2.INTER_CUBIC)
#Replace NaN values
PACE_678_res1 = np.nan_to_num(PACE_678_res, nan=-999)
MODIS_678_res1 = np.nan_to_num(MODIS_678_res, nan=-999)
#Scale the Rrs by 1000
PACE_678_res2 = PACE_678_res1 * 1000
MODIS_678_res2 = MODIS_678_res1 * 1000

###Percentage Pixel Match###
# Mask NaNs
mask = ~np.isnan(PACE_678_res2) & ~np.isnan(MODIS_678_res2)
# Pixel-wise match (exact match)
tol = 0.5
pixel_match = (abs(PACE_678_res2 - MODIS_678_res2) <= tol) & mask
# Count matches and total valid pixels
n_matches = pixel_match.sum().item()
n_total = mask.sum().item()
# Percentage match
percentage_match = (n_matches / n_total) * 100 if n_total > 0 else np.nan


### Needleman-Wunsch function for pixel comparison ###

def needleman_wunsch(modis_data, pace_data, match_score=1, mismatch_score=-1, gap_penalty=0):
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
            match = score_matrix[i-1][j-1] + (match_score if modis_data[i-1, j-1] == pace_data[i-1, j-1] else mismatch_score)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    return score_matrix

# Similarity matrix function (using squared differences)
def similarity_matrix(modis_data, pace_data):
    return (modis_data - pace_data) ** 2  

def rmse_pixelwise(modis_data, pace_data):
    return np.sqrt((modis_data - pace_data) ** 2)

# Perform Needleman-Wunsch alignment to get the score matrix
alignment_score_matrix = needleman_wunsch(MODIS_678_res2, PACE_678_res2)
# Generate similarity matrix
sim_matrix = similarity_matrix(MODIS_678_res2, PACE_678_res2)
# Compute RMSE matrix (pixel-wise)
rmse_matrix = rmse_pixelwise(MODIS_678_res2, PACE_678_res2)

