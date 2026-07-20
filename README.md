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

Each notebook follows the same structure: data loading and validation, feature preparation, model training, evaluation (accuracy, classification report, confusion matrix), cross-validation, feature importance, ROC curve, and Precision-Recall curve.

| Notebook | Model | Notes |
|----------|-------|-------|
| [01_svm.ipynb](notebooks/01_svm.ipynb) | Support Vector Machine | Linear kernel; features scaled with StandardScaler; probability=True for ROC/PR curves |
| [02_random-forest.ipynb](notebooks/02_random-forest.ipynb) | Random Forest | 100 estimators; no scaling required |
| [03_xgboost.ipynb](notebooks/03_xgboost.ipynb) | XGBoost | 100 estimators; no scaling required |

All models use an 80/20 train/test split with `random_state=42`.

## Results & Model Comparison

### Performance

| Metric | SVM | Random Forest | XGBoost |
|--------|-----|----------------|---------|
| Train accuracy | 0.910 | 0.985 | 0.970 |
| Test accuracy | 0.902 | 0.921 | 0.925 |
| Train–test gap | 0.008 | 0.063 | 0.045 |
| CV accuracy (5-fold) | 0.909 ± 0.012 | 0.908 ± 0.008 | 0.903 ± 0.007 |
| Recall — Class 1 (Watch) | 0.44 | 0.59 | 0.61 |
| Recall — Class 3 (Alert 1) | 0.66 | 0.79 | 0.80 |

Overall accuracy is close across all three models (90–93%), which is expected given `Bleaching_Alert_Area` is a rule-based encoding of `HotSpots` and `Degree_Heating_Weeks` thresholds — the classes are largely separable by construction. The more informative differences show up in the minority classes: Class 1 (Watch) is the hardest for every model to recall correctly, likely because it sits closest to the No Stress boundary and captures the noisiest transition region, while both tree-based models recover Alert-level cases (Class 3) noticeably better than SVM.

### Overfitting

Random Forest shows the largest train–test gap (0.063), consistent with tree ensembles' tendency to memorize training splits when left at default depth. XGBoost's gap (0.045) is smaller due to its built-in regularization, and SVM's linear kernel keeps the gap smallest (0.008) by constraining decision boundary complexity — at the cost of underfitting the minority classes. Cross-validation accuracy is consistent with test accuracy for all three, confirming none of these gaps are artifacts of a single unlucky split.

### Feature importance disagreement

Random Forest and SVM disagree on the top feature: RF's mean-decrease-in-impurity metric ranks `Sea_Surface_Temperature` highest, while SVM's mean absolute coefficients rank it lowest, with `HotSpots` and `Degree_Heating_Weeks` dominating instead. This is a known limitation of impurity-based importance — continuous features with a wide range of values (like raw SST) get more candidate split points and are structurally favored, regardless of their true causal role. Since `HotSpots` and `DHW` are the features the alert thresholds are actually defined on, the SVM ranking better reflects the underlying logic of the target variable. It's also worth noting `Sea_Surface_Temperature`, `HotSpots`, and `Degree_Heating_Weeks` are three levels of processing of the same thermal stress signal (`HotSpots` is SST minus a historical baseline, `DHW` is accumulated `HotSpots` over 12 weeks), so all three models are working with a meaningfully multicollinear feature set.

### Which model would I use?

**XGBoost.** It has the best test accuracy, the best recall on the minority alert classes — the cases that matter most for an early-warning system, where missing an Alert-Level event is the costly failure mode — and a smaller train–test gap than Random Forest despite similar model complexity. SVM is the most stable of the three and would be a reasonable choice if interpretability and low variance mattered more than catching rare events, but its weak recall on Classes 1 and 3 is a real liability here.

That said, the honest caveat: because `Bleaching_Alert_Area` is a deterministic threshold function of `HotSpots` and `DHW`, a simple rule-based classifier applying those same thresholds directly would almost certainly outperform all three ML models here while being fully interpretable and auditable. This project is a modeling exercise for comparing classifier behavior, not a claim that ML is the right tool for a target that's already rule-derived — in a real deployment I'd reach for the threshold logic first and reserve ML for cases where the labeling rule isn't already known.

### Limitations

- **Single location.** All data comes from one fixed coordinate (0.15°N, 90.625°W). Models learn thermal stress patterns specific to this point and would not generalize to other reef locations without retraining.
- **No spatial generalization.** 40 years of data at one coordinate means temporal coverage is strong but spatial coverage is absent.

## Setup

**Requirements:** Python 3.11, conda recommended.

```bash
conda create --name coral-bleaching python=3.11
conda activate coral-bleaching
pip install -r requirements.txt
python -m ipykernel install --user --name=coral-bleaching --display-name "Python (coral-bleaching)"
```

Open any notebook in VS Code or JupyterLab and select the `coral-bleaching` kernel.
