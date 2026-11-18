---
title: SuperKart Sales Predictor
emoji: 🛒
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Predict total sales for a product in a store
---

# 🛒 SuperKart Sales Predictor

Estimate the **total sales** a product will generate in a SuperKart store from
its price, weight, shelf allocation, category and store profile.

Built on a scikit-learn pipeline with an XGBoost regressor trained on 8,763
sales records (test R² 0.919, RMSE 300, MAE 129).

- Trained model: [ruksanakhanum/superkart-sales-prediction-model](https://huggingface.co/ruksanakhanum/superkart-sales-prediction-model)
- Dataset: [ruksanakhanum/superkart-sales-prediction-data](https://huggingface.co/datasets/ruksanakhanum/superkart-sales-prediction-data)
- Source code: [ruksana-ux/superkart-sales-prediction](https://github.com/ruksana-ux/superkart-sales-prediction)
