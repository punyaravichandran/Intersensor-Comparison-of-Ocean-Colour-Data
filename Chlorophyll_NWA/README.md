# PACE vs MODIS Chlorophyll Comparison
Spatial comparison between NASA PACE and MODIS-Aqua chlorophyll-a datasets over the Northern Indian Ocean.

## The workflow:

- Loads multi-file NetCDF chlorophyll datasets
- Computes temporal mean chlorophyll
- Crops to a defined geographic bounding box
- Clips data to a shapefile region
- Applies percentile filtering (2nd–98th percentile)
- Resamples datasets to a common grid
- Computes Pixel Match Percentage, Squared Difference Matrix, Pixel-wise RMSE & Needleman–Wunsch alignment score matrix

## Data Sources

- PACE Ocean Color Chlorophyll-a
- MODIS-Aqua Chlorophyll-a
- Northern Indian Ocean shapefile

## Outputs

- PACE chlorophyll map
- MODIS chlorophyll map
- Pixel Match Percentage 
- Needleman–Wunsch score matrix plot
- Squared Difference map
- RMSE map
