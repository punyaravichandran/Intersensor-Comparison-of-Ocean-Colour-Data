# PACE vs MODIS Reflectance (Rrs 678 nm) Comparison
Spatial comparison of Remote Sensing Reflectance (Rrs) between NASA PACE OCI & MODIS-Aqua Ocean Color

## Data Used

1. PACE OCI (Level-2 OC_AOP)
- Variable: Rrs
- Group: geophysical_data
- Navigation: navigation_data
- Selected band index: 152 → 678 nm
- PACE swath regions: Arabian Sea, Bay of Bengal

2. MODIS-Aqua (Level-2 OC)
- Variable: Rrs_678
- Group: geophysical_data
- Navigation: navigation_data
MODIS granules are mosaicked using combine_first().

## Outputs
- Pixel Match Percentage (console output)
- Needleman–Wunsch score matrix
- Squared Difference matrix
- RMSE matrix
