# Coral Bleaching Classification — Galapagos Islands

Machine learning classification of coral bleaching alert levels using 40 years of daily sea surface temperature data from a fixed monitoring point in the Galapagos Islands. Three classifiers are compared: Support Vector Machine, Random Forest, and XGBoost.

## Dataset

**Source:** [NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/)  
**Location:** Galapagos Islands (0.15°N, 90.625°W)  
**Period:** January 1, 1985 – February 20, 2025  
**Observations:** 14,660 daily records  

> The raw data file is not included in this repository. Download from NOAA Coral Reef Watch and place as `data/galapagos_islands.csv`.

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `Date` | datetime | Daily timestamp (dropped — non-predictive) |
| `Latitude` | float | Fixed at 0.15°N (dropped — constant) |
| `Longitude` | float | Fixed at 90.625°W (dropped — constant) |
| `Sea_Surface_Temperature` | float | Daily SST in °C |
| `HotSpots` | float | Temperature anomaly above the maximum monthly mean (°C) |
| `Degree_Heating_Weeks` | float | Accumulated thermal stress over a 12-week rolling window |
| `Bleaching_Alert_Area` | int | **Target variable** — NOAA bleaching alert level (0–4) |

### Target Classes

| Class | Alert Level | Condition |
|-------|-------------|-----------|
| 0 | No Stress | HotSpots = 0 |
| 1 | Watch | HotSpots > 0 |
| 2 | Warning | DHW > 0 |
| 3 | Alert Level 1 | DHW ≥ 4 |
| 4 | Alert Level 2 | DHW ≥ 8 |

## Models

Each notebook follows the same structure: data loading and validation, feature preparation, model training, evaluation (accuracy, classification report, confusion matrix), ROC curve, and Precision-Recall curve.

| Notebook | Model | Notes |
|----------|-------|-------|
| [01_svm.ipynb](notebooks/01_svm.ipynb) | Support Vector Machine | Linear kernel; features scaled with StandardScaler |
| [02_random-forest.ipynb](notebooks/02_random-forest.ipynb) | Random Forest | 100 estimators; no scaling required |
| [03_xgboost.ipynb](notebooks/03_xgboost.ipynb) | XGBoost | 100 estimators; no scaling required |

All models use an 80/20 train/test split with `random_state=42`.

## Setup

**Requirements:** Python 3.11, conda recommended.

```bash
conda create --name coral-bleaching python=3.11
conda activate coral-bleaching
pip install -r requirements.txt
python -m ipykernel install --user --name=coral-bleaching --display-name "Python (coral-bleaching)"
```

Open any notebook in VS Code or JupyterLab and select the `coral-bleaching` kernel.
