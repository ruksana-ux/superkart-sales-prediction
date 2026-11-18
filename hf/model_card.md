---
license: mit
library_name: scikit-learn
tags:
  - tabular-regression
  - xgboost
  - sales-forecasting
  - retail
metrics:
  - r2
  - rmse
  - mae
---

# SuperKart Sales Predictor — XGBoost regression pipeline

Predicts the **total sales** a product generates in a SuperKart store from its
price, weight, shelf allocation, category and the store profile.

This is the model behind the
[SuperKart Sales Predictor demo](https://superkart-prediction.streamlit.app)
(source: [ruksana-ux/superkart-sales-prediction](https://github.com/ruksana-ux/superkart-sales-prediction)).

## Model details

- **Architecture:** scikit-learn `Pipeline` → `StandardScaler` + `OneHotEncoder`
  → `XGBRegressor` (400 trees, `max_depth=3`, `learning_rate=0.06`,
  `tree_method="hist"`)
- **Target:** `Product_Store_Sales_Total`
- **Framework:** scikit-learn 1.9.1, XGBoost 3.4.1, joblib 1.6.0
- **Split:** 80/20 hold-out, `random_state=42` (7,010 train / 1,753 test rows)

## Metrics (hold-out test set)

| Metric | Train | Test |
| --- | --- | --- |
| R² | 0.9333 | **0.9193** |
| Adjusted R² | 0.9332 | 0.9189 |
| RMSE | 272.1 | **300.5** |
| MAE | 114.7 | **128.9** |
| MAPE | 4.12% | **4.50%** |

For reference, always predicting the mean sale (3,464 units) gives RMSE 1,066 and
MAE 840, so the model reduces RMSE by ~72% and MAE by ~85%.

## Usage

```python
import joblib
import pandas as pd

model = joblib.load("SuperKart_model_v1_0.joblib")

row = pd.DataFrame(
    [
        {
            "Product_Weight": 12.66,
            "Product_Sugar_Content": "Low Sugar",
            "Product_Allocated_Area": 0.056,
            "Product_Type": "Frozen Foods",
            "Product_MRP": 147.0,
            "Store_Establishment_Year": 2002,
            "Store_Size": "Medium",
            "Store_Location_City_Type": "Tier 2",
            "Store_Type": "Supermarket Type1",
        }
    ]
)
print(model.predict(row)[0])  # 3487.3
```

The pipeline expects the nine feature columns above (the raw `Store_Id` and
`Product_Id` columns are dropped). Categorical values are one-hot encoded, so
unseen categories are ignored rather than raising.

## Files

| File | Description |
| --- | --- |
| `SuperKart_model_v1_0.joblib` | Fitted pipeline (478 KB) |
| `model_metadata.json` | Training metadata and metrics |

## Training data

[ruksanakhanum/superkart-sales-prediction-data](https://huggingface.co/datasets/ruksanakhanum/superkart-sales-prediction-data)
— 8,763 historical SuperKart sales records.

## Limitations

- Single train/test split; no cross-validation, so metrics carry some split
  variance.
- Trained on a narrow, tidy dataset (4 stores, synthetic-looking distributions).
  Predictions far outside the training ranges are extrapolations.
- Outlier clipping was computed on the full dataset before splitting, a small
  methodological leakage that slightly flatters the scores.
- Not calibrated for business use; the demo's ±RMSE band is a rough guide, not a
  confidence interval.

## Licence

MIT.
