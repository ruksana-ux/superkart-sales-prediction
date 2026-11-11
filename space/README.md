---
title: SuperKart Sales Predictor
emoji: 🛒
colorFrom: purple
colorTo: indigo
sdk: streamlit
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predict total sales for a product in a store
---

# 🛒 SuperKart Sales Predictor

Estimate the **total sales** a product will generate in a SuperKart store from its
price, weight, shelf allocation, category and store profile.

This Space runs the same Streamlit app that is hosted on Streamlit Community
Cloud, with the trained XGBoost pipeline committed as a pre-built artifact — so
the app is ready as soon as the Space starts, with no training at boot.

## Tabs

| Tab | Description |
| --- | --- |
| 🎯 **Predict sales** | Enter product/store details and get an instant sales estimate with an expected error range. |
| 📄 **Batch prediction** | Upload a CSV of many products and download predictions. |
| 📊 **Model performance** | Test-set metrics, actual vs predicted, residuals, feature importance. |
| 🔍 **Data explorer** | Browse and filter the 8,763-row dataset. |

## Model

- **Pipeline:** `StandardScaler` + `OneHotEncoder` → `XGBoostRegressor` (400 trees)
- **Target:** `Product_Store_Sales_Total`
- **Test R²:** 0.919 · **RMSE:** 300.5 · **MAE:** 128.9

## Source code

Full source, tests and CI: <https://github.com/ruksana-ux/superkart-sales-prediction>
