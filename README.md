# Forest LULC Classification — Anwiaso East, Ghana

A land use/land cover (LULC) classification pipeline for Anwiaso East Forest Reserve, Ghana, combining Sentinel-1 SAR and Sentinel-2 multispectral imagery with both classical and deep learning approaches.

## Overview

This project classifies forest landscapes into 11 land cover types — including dense forest, degraded forest, cocoa farms, illegal mining (galamsey), and water bodies — using 10 m satellite imagery processed via Google Earth Engine.

**Study Area:** Anwiaso East Forest Reserve, Ghana (~6.05–6.36°N, 2.13–2.26°W)  
**Resolution:** 10 m pixels (Sentinel-2 native)  
**CRS:** EPSG:32630 (UTM Zone 30N)  
**Season:** Dry season imagery (November–February) to minimize cloud cover

## Land Cover Classes

| ID | Class | Description |
|----|-------|-------------|
| 1 | Cocoa | Cocoa plantations (~90.7% of study area) |
| 2 | Degraded Forest | Partially cleared/disturbed forest |
| 3 | Dense Forest | Intact closed-canopy forest |
| 4 | Farms | Agricultural land |
| 5 | Galamsey | Illegal artisanal mining sites |
| 6 | Invasives | Invasive plant species |
| 7 | Natural Forest | Secondary/mixed natural forest |
| 8 | Open Areas | Bare land, cleared areas |
| 9 | Road Buffer | Road corridors |
| 10 | Swamp | Wetland/swamp areas |

## Feature Stack (26 bands)

| Category | Features |
|----------|----------|
| Sentinel-2 Multispectral | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| Spectral Indices | NDVI, NDWI, NDRE, EVI, SAVI, NBR, BSI |
| Sentinel-1 SAR Statistics | VV/VH mean, min, max, std + VH/VV ratio |

## Notebooks

Run notebooks in this order:

### 1. [`data_prep.ipynb`](data_prep.ipynb)
Connects to Google Earth Engine, downloads and composites Sentinel-1 and Sentinel-2 imagery, computes spectral indices, samples training polygons, and exports the 26-band feature stack and training samples to Google Drive.

**Outputs:** `s1_s2_features_2026.tif`, `training_samples.csv`, `class_mapping.json`

### 2. [`forest_lulc_rf_full_pipeline.ipynb`](forest_lulc_rf_full_pipeline.ipynb)
Full Random Forest classification pipeline: authenticates GEE, builds the feature stack, trains an sklearn `RandomForestClassifier` with cross-validation, evaluates accuracy (confusion matrix, Kappa coefficient), predicts the full landscape, and post-processes outputs (majority filtering, minimum mapping unit absorption, vectorization).

**Outputs:** `rf_model.joblib`, classified GeoTIFF, uncertainty map, `lulc_classification.gpkg`

### 3. [`forest_classification_dl_and_colab.ipynb`](forest_classification_dl_and_colab.ipynb)
Deep learning pipeline using a ResU-Net (UnetPlusPlus + ResNet50) and MAnet ensemble. Uses tiled patch inference (256×256 px, 128 px stride) to handle full-scene images. Same post-processing outputs as the RF pipeline.

**Outputs:** Model weights (`.pth`), band normalization stats, classified GeoPackage

### 4. [`dl_predict_new_image.ipynb`](dl_predict_new_image.ipynb)
Applies pre-trained DL models to new imagery. Merges tiled GEE exports, runs memory-mapped tiled inference to avoid RAM overflow, and produces classification outputs in a separate `NewImage_Results/` directory.

## Google Drive Output Structure

```
Forest_LULC_Classification/
├── study_area.gpkg
├── phase2.gpkg
├── training_dataset.gpkg
├── s1_s2_features_2026.tif
├── training_samples.csv
├── class_mapping.json
├── RF_Results/
│   ├── rf_model.joblib
│   ├── lulc_classification.gpkg
│   ├── lulc_classification_2024.tif
│   ├── lulc_uncertainty_2024.tif
│   └── [confusion matrix, feature importance plots]
├── DL_Results/
│   ├── model1_best.pth
│   ├── lulc_classification.gpkg
│   └── [TIF rasters, PNG visualizations]
└── NewImage_Results/
    └── [predictions for new imagery]
```

## Local Data Files

| File | Description |
|------|-------------|
| [`study_area.gpkg`](study_area.gpkg) | 709-feature study area boundary (AOI for GEE) |
| [`training_dataset.gpkg`](training_dataset.gpkg) | 709 labeled polygons, 11 land cover classes |
| [`phase2.gpkg`](phase2.gpkg) | Refined AOI boundary for phase 2 predictions |

> Note: `.gpkg` files are excluded from version control (see [.gitignore](.gitignore)).

## Setup

All notebooks are designed to run in **Google Colab**. Required packages are installed at the top of each notebook.

**Key dependencies:**
- `earthengine-api` — Google Earth Engine Python API
- `geemap`, `geopandas`, `rasterio`, `shapely`, `fiona` — Geospatial processing
- `scikit-learn` — Random Forest classifier
- `torch`, `segmentation_models_pytorch` — Deep learning models
- `scikit-image`, `scipy` — Post-processing (morphological filtering, connectivity)
- `google-auth`, `google-api-python-client` — Google Drive integration

**Authentication required:**
1. Google Earth Engine account with project access
2. Google Drive access for reading/writing model files and rasters

## Methods

**Random Forest:** 633K training samples × 26 bands, evaluated with stratified k-fold cross-validation. Post-processing applies a majority filter and minimum mapping unit (MMU) absorption to remove small isolated patches.

**Deep Learning Ensemble:** Two-model softmax average (ResU-Net + MAnet). Tiled inference with overlap-blending handles imagery larger than GPU memory. Training uses the same 633K samples extracted via GEE.

**Post-processing (both approaches):**
1. Majority filter (morphological mode filter)
2. MMU absorption — patches below minimum area threshold absorbed into dominant neighbor
3. Vectorization with geometry simplification
4. Area calculation per patch and per class
