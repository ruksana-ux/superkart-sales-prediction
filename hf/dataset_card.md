---
license: mit
task_categories:
  - tabular-regression
language:
  - en
size_categories:
  - 1K<n<10K
tags:
  - retail
  - sales
  - superkart
configs:
  - config_name: default
    data_files:
      - split: train
        path: SuperKart.csv
---

# SuperKart sales dataset

8,763 historical product/store sales records used to train the
[SuperKart Sales Predictor](https://huggingface.co/ruksanakhanum/superkart-sales-prediction-model)
demo app ([live app](https://superkart-prediction.streamlit.app),
[source](https://github.com/ruksana-ux/superkart-sales-prediction)).

## Columns

| Column | Type | Notes |
| --- | --- | --- |
| `Product_Id` | string | Identifier, e.g. `FDX0001` (not used as a feature) |
| `Product_Weight` | float | Weight in the store's units |
| `Product_Sugar_Content` | category | `Low Sugar`, `Regular`, `No Sugar` (raw file also contains `reg`) |
| `Product_Allocated_Area` | float | Shelf area allocated to the product |
| `Product_Type` | category | 16 product categories, e.g. `Frozen Foods`, `Dairy` |
| `Product_MRP` | float | Maximum retail price |
| `Store_Id` | string | `OUT001`–`OUT004` (not used as a feature) |
| `Store_Establishment_Year` | int | 1987–2009 |
| `Store_Size` | category | `Small`, `Medium`, `High` |
| `Store_Location_City_Type` | category | `Tier 1`, `Tier 2`, `Tier 3` |
| `Store_Type` | category | `Departmental Store`, `Food Mart`, `Supermarket Type1`, `Supermarket Type2` |
| `Product_Store_Sales_Total` | float | **Target** — total sales for that product in that store |

## Notes on the raw data

- `Product_Sugar_Content` contains a dirty value `reg`, normalised to `Regular`
  during preprocessing.
- The modelling pipeline drops `Product_Id` and `Store_Id` and uses the other
  nine columns (eight features + target).
- Target range: 33 – 8,000, mean 3,464, standard deviation 1,066.

## Usage

```python
import pandas as pd

df = pd.read_csv(
    "hf://datasets/ruksanakhanum/superkart-sales-prediction-data/SuperKart.csv"
)
print(df.shape)  # (8763, 12)
```

## Licence

MIT.
