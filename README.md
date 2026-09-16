# Identification of Flood and Drought Events in HUC 0505

Machine-learning workflow for streamflow forecasting flood and drought identification in the Central Appalachian Region (HUC 0505, the Kanawha-New River watershed).

The accompanying manuscript, [`huc0505_manuscript_9_14_2026.docx`](huc0505_manuscript_9_14_2026.docx) (available soon), describes the scientific motivation, methods, results, limitations, and future research directions for this workflow.

## Study Overview

The project combines daily USGS streamflow observations with daily 4 km gridMET climate data for seven gauges spanning HUC 0505:

- New River at Radford, VA (`03171000`)
- New River at Glen Lyn, VA (`03176500`)
- Gauley River at Belva, WV (`03192000`)
- Kanawha River at Kanawha Falls, WV (`03193000`)
- Elk River at Queen Shoals, WV (`03197000`)
- Coal River at Tornado, WV (`03200500`)
- Kanawha River at Charleston, WV (`03198000`)

The analysis covers 2000-2025. It is designed to support identification of extreme floowd and drought events eventually assisting growers in planting delays, irrigation scheduling, livestock protection, and crop and forage risk management.

## Research Questions

The workflow evaluates whether a compact, physically motivated feature set and computationally inexpensive data augmentation can:

1. Forecast daily streamflow across multiple gauges.
2. Detect high-flow flood conditions and low-precipitation drought conditions.
3. Improve learning from rare extreme-event observations through synthetic samples.
4. Provide calibrated prediction intervals.
5. Connect predicted events to eventually assist agriculture.

## Methods

### Inputs

- USGS National Water Information System daily streamflow data.
- gridMET daily climate data at approximately 4 km resolution, including precipitation, reference evapotranspiration, vapor pressure deficit, and temperature variables.
- Watershed boundaries from the USGS Watershed Boundary Dataset (WBD) for grid-cell assignment.

### Feature engineering

The current pipeline creates 20 input features using:

- Streamflow and precipitation lags at 1, 3, 7, 14, and 30 days.
- Rolling means, standard deviations, and cumulative precipitation statistics.
- Seasonal sine and cosine encoding of day of year.
- Daily and rolling water balance (`precipitation - PET`).
- 30- and 90-day SPEI-style water-balance measures.
- Reference evapotranspiration and other gridMET predictors.

Flood and drought labels were threshold-based. Flood days were associated with streamflow above the Q95 threshold, while drought days used the 10th percentile (P10) of monthly 30-day cumulative precipitation.

### Data augmentation

[`14b_gen_augment_v4.py`](14b_gen_augment_v4.py) augmented rare-event records using resampling and correlated Gaussian perturbations. Flood and drought records received different scaling treatments, and compound drought-to-flood cases were also generated. Synthetic rows were marked with a `synthetic` column so they can be distinguished from observations.

### Models and validation

The modeling workflow uses scikit-learn models for four tasks:

- Random forest regression for streamflow.
- Random forest classification for flood detection.
- Random forest classification for drought detection.
- Quantile gradient boosting for 90% streamflow prediction intervals.

Validation was chronological rather than randomly shuffled:

- Training: 2000-2022.
- Testing: 2023-2025.

Reported metrics included NSE, KGE, log-NSE, RMSE, PBIAS, correlation, ROC-AUC, average precision, F1, recall, precision, empirical interval coverage, interval width, and Winkler score.

## Repository Workflow

Run the scripts from the repository root in the following order:

1. [`11_gridmet_download.py`](11_gridmet_download.py) downloads and assigns gridMET data for the gauge watersheds for 2000-2022. It creates `data/gridmet_cells.json` and `data/gridmet_gauges.csv`.
2. [`11b_extend_to_2025.py`](11b_extend_to_2025.py) extends streamflow and gridMET data through 2025.
3. [`18_add_gauge_03193000.py`](18_add_gauge_03193000.py) adds or rebuilds the gridMET series for gauge `03193000` when that gauge is missing from the existing grid assignment.
4. [`13c_build_features_v4.py`](13c_build_features_v4.py) joins streamflow and climate data and writes `data/features_gridmet_v4.csv`.
5. [`14b_gen_augment_v4.py`](14b_gen_augment_v4.py) creates with augmented data through bootstrapping `data/features_augmented_v4.csv`.
6. [`15c_ml_train_v4.py`](15c_ml_train_v4.py) trains the forecasting and classification models and writes model files, `data/predictions_v4.csv`, and `data/metrics_v4.json`.
7. [`15d_fullperiod_preds_v4.py`](15d_fullperiod_preds_v4.py) generates full-period predictions in `data/predictions_full_v4.csv`.
8. [`16c_figures_v4.py`](16c_figures_v4.py) creates the manuscript figures in `figures/`.
9. [`19_bootstrap_improved.py`](19_bootstrap_improved.py) compares uncertainty-interval methods and writes `data/bootstrap_comparison.csv` and `figures/F7_bootstrap_comparison.png`.

The scripts are numbered by workflow stage, but steps 1-3 are data preparation alternatives or repair steps rather than one universally required sequence. Inspect the existing data before running a download or merge step.

## Expected Outputs

Typical generated files include:

```text
data/
  gridmet/                  # Cached annual gridMET files
  gridmet_cells.json        # Gauge-to-grid assignment
  gridmet_gauges.csv        # Daily gauge-level climate data
  streamflow_6gauges.csv    # Daily USGS streamflow data
  features_gridmet_v4.csv   # Observed feature matrix
  features_augmented_v4.csv # Observed plus synthetic training data
  predictions_v4.csv        # Test-period model predictions
  predictions_full_v4.csv   # Full-period predictions
  metrics_v4.json           # Model metrics
  bootstrap_comparison.csv  # Uncertainty-method comparison
models/                     # Serialized trained models
figures/                    # Manuscript figures and diagnostics
```

Generated data, models, and figures were not included in this repository
.. Create the directories before running the pipeline if a script does not create them automatically.

## Environment

The scripts require Python 3 and scientific/data-access packages used by the imports in the workflow, including:

```text
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
requests
geopandas
shapely
xarray
netCDF4
```

Install the packages in an isolated environment appropriate for your system. A typical setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn scipy matplotlib seaborn requests geopandas shapely xarray netCDF4
```

## Configuration Notes

Several scripts currently use the working directory or an absolute path such as `/home/sandbox/huc0505`. Before running them on another machine, update those path settings to the local checkout, or run them in an environment where that path exists.

The data-acquisition scripts require an internet connection and access to the USGS NWIS, USGS WBD, and gridMET services. Downloads can be time-consuming; the gridMET workflow caches annual files under `data/gridmet/`.

## Reproducibility and Interpretation

The test period must remain held out from training and augmentation. Do not augment or resample the 2023-2025 observations when reproducing the reported evaluation. Because the workflow downloads live public data and contains hard-coded paths and script-specific assumptions, rerunning it may produce different intermediate files or metrics unless the source data, package versions, random seeds, and configuration are fixed.

The project reports the following headline test-period results: streamflow NSE of 0.832 and KGE of 0.893, flood ROC-AUC of 0.991 with 94.4% recall, drought ROC-AUC of 0.992 with 94.2% recall, and 92.1% empirical coverage for nominal 90% prediction intervals. 

## Data Sources

- [USGS National Water Information System](https://waterdata.usgs.gov/nwis)
- [gridMET climate data](https://www.climatologylab.org/gridmet.html)
- [USGS Watershed Boundary Dataset](https://nhd.usgs.gov/wbd)

## Limitations

This project identifies several limitations when interpreting this repository:

- gridMET is model-derived and may have interpolation uncertainty in complex Appalachian terrain.
- The current workflow is primarily temporal and does not explicitly model soil, land-cover, or topographic heterogeneity.
- The drought label is a simplified precipitation-based threshold.
- Bootstrapped augmentation cannot guarantee realistic conditions beyond the historical range.
- A three-year held-out period and seven gauges may not represent all hydrological variability in the region.

