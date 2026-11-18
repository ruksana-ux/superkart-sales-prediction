---
title: SuperKart Sales Predictor
emoji: 🛒
colorFrom: purple
colorTo: indigo
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predict total sales for a product in a store
---

# 🛒 SuperKart Sales Predictor

Estimate the **total sales** a product will generate in a SuperKart store from
its price, weight, shelf allocation, category and store profile.

Built on a scikit-learn pipeline with an XGBoost regressor trained on 8,763
sales records (test R² 0.919, RMSE 300, MAE 129). Four tabs: single prediction,
batch prediction from a CSV, model performance, and a data explorer.

This is the Hugging Face Spaces twin of the
[Streamlit app](https://superkart-prediction.streamlit.app); both share the same
`superkart_core.py` and joblib artifact.

## Why Gradio?

Hugging Face now charges for `cpu-basic` compute Spaces, but **ZeroGPU** hardware
is still free — and ZeroGPU only runs Gradio SDK Spaces. The XGBoost model is
CPU-only, so inference runs in the main process and no GPU quota is consumed; the
`@spaces.GPU` function that ZeroGPU requires is declared but not exposed in the
UI. The Space requests Python 3.12 because scikit-learn 1.9 needs 3.11+.

## Links

- Trained model: [ruksanakhanum/superkart-sales-prediction-model](https://huggingface.co/ruksanakhanum/superkart-sales-prediction-model)
- Dataset: [ruksanakhanum/superkart-sales-prediction-data](https://huggingface.co/datasets/ruksanakhanum/superkart-sales-prediction-data)
- Source code: [ruksana-ux/superkart-sales-prediction](https://github.com/ruksana-ux/superkart-sales-prediction)
