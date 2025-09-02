# GazaEnvironmentalChange
Code Associated with the Publication of "Spectral Markers of Destruction: NDVI and NDBI-Based Assessment of Environmental Warfare in the Gaza Strip from 2023 - 2025"

# Spectral Markers of Destruction: NDVI and NDBI-Based Assessment of Environmental Warfare in the Gaza Strip (2023–2025)

This repository accompanies the manuscript:

**Hoheneder, T. J., Duderstadt, K., Palace, M., & Sowers, J. (2025).  
*Spectral Markers of Destruction: NDVI and NDBI-Based Assessment of Environmental Warfare in the Gaza Strip (2023–2025).***  

The manuscript evaluates the environmental and land use consequences of armed conflict in Gaza using optical satellite remote sensing, land cover classification, and spatiotemporal analysis of NDVI and NDBI. Methods include Random Forest classification trained on ESA WorldCover 2020 data, LOESS-deseasonalized time series, and quadrant-based spectral change analysis.

## Acknowledgements
This research was supported by the National Science Foundation (NSF) Research Trainee Grant #2125868 awarded to the University of New Hampshire. The study uses Sentinel-2 imagery provided by the European Space Agency (ESA) and the Google Earth Engine (GEE) platform.  
We also acknowledge reviewers and colleagues whose insights informed this work including Dr. Erika Weinthal of Duke University.

## GitHub Repository Structure
This repository contains two categories of scripts and one data products folder:

1. **Google Earth Engine (GEE) Scripts**
   - Written in JavaScript, run in the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
   - When deployed, the GEE code will assist with:
     - Land cover classification with Random Forest
     - NDVI/NDBI time series extraction
     - Land cover change mapping
     - Data export to GEE Assets or Google Drive

2. **Python Scripts**
   - Written in Python 3.x, typically run locally or in Jupyter notebooks.
   - Used for:
     - Statistical analysis (ANOVA, regression, LOESS deseasonalization)
     - Plot generation (Matplotlib, Pandas, NumPy)
     - Post-processing of classification outputs
     - Reproduction of figures and tables included in the manuscript

3. **Data Products Folder:**
   -Shapefile of the Gaza Strip Boundary
   -Pre-Conflict Land Cover Classification of the Gaza Strip
   -Active Conflict Period Land Cover Classification of the Gaza Strip
   -Per Land Cover .CSV Time Series Files for NDBI and NDVI  

## Requirements
- **For GEE scripts**:  
  - A Google Earth Engine account.
- **For Python scripts**:  
  - Python 3.9+  
  - Core packages: `pandas`, `numpy`, `matplotlib`, `scipy`, `geopandas`


## How to Use
1. Identify whether the script is for **GEE** (`.js`) or **Python** (`.py` / `.ipynb`).
2. For GEE scripts:
   - Copy the script into the GEE Code Editor.
   - Adjust study area and dates if needed.
3. For Python scripts:
   - Clone this repo.
   - Install required Python packages.
   - Run analysis or plotting scripts to reproduce figures and results.


## Disclaimer
These scripts were developed for academic research in conflict monitoring. They may need adaptation for other study areas, datasets, or computational limits. Users should validate outputs against available reference data.


## Citation
If you use this repository, please cite the manuscript:
> Hoheneder, T. J., Duderstadt, K., Palace, M., & Sowers, J. (2025). *Spectral Markers of Destruction: NDVI and NDBI-Based Assessment of Environmental Warfare in the Gaza Strip (2023–2025).* [Manuscript/Journal info once published].


## Contact: 

If email inquiry is required for assistance or questions regarding this data, products, or other items within, please contact Tim.Hoheneder@unh.edu for help. 
