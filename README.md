# 🛒 SuperKart Sales Predictor

[![smoke-test](https://github.com/ruksana-ux/superkart-sales-prediction/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/ruksana-ux/superkart-sales-prediction/actions/workflows/smoke-test.yml)

A Streamlit web app that predicts the **total sales** a product will generate in a
SuperKart store, based on its price, weight, shelf allocation, category and the
store profile.

The project is the deployment of the original `SuperKart` Colab notebook
(exploratory analysis → preprocessing → Random Forest / XGBoost comparison →
model serialisation), rebuilt as a self-contained Streamlit application.

**🚀 Live app:** <https://superkart-prediction.streamlit.app>

> The first page load after a restart trains the model from the included dataset
> and takes about a minute. Later loads are instant.

---

## What the app does

| Tab | Description |
| --- | --- |
| 🎯 **Predict sales** | Enter product and store details and get an instant sales estimate with an expected error range. |
| 📄 **Batch prediction** | Upload a CSV of many products/stores, download the predictions, and use the ready-made template. |
| 📊 **Model performance** | Train/test metrics, actual-vs-predicted chart, feature importance and residual distribution. |
| 🔍 **Data explorer** | Preview the dataset, numeric summary, category distributions and correlations. |

## The model

- **Target:** `Product_Store_Sales_Total`
- **Features:** `Product_MRP`, `Product_Allocated_Area`, `Product_Weight`,
  `Store_Establishment_Year`, `Product_Sugar_Content`, `Product_Type`,
  `Store_Location_City_Type`, `Store_Type`
- **Preprocessing:** IQR outlier clipping → `StandardScaler` for numeric features →
  `OneHotEncoder(handle_unknown="ignore")` for categorical features
- **Estimator:** `XGBRegressor` (400 trees, depth 3, learning rate 0.06) inside a
  scikit-learn `Pipeline`
- **Split:** 80/20 train–test, `random_state=42`
- **Metrics:** R², adjusted R², RMSE, MAE and MAPE, recomputed on the test set at
  startup

The original notebook saved its tuned pipeline as
`SuperKart_model_v1_0.joblib`, but that file was never downloaded from Colab.
Because the dataset **is** included in this repository, the app trains an
equivalent XGBoost pipeline automatically the first time it starts (about
20 seconds), and caches it for the rest of the session. If a saved artifact is
present at `model/SuperKart_model_v1_0.joblib`, the app loads it instead.

### Measured performance

Results from the automated CI run (`tests/smoke_test.py`, Python 3.12):

| Metric | Test set |
| --- | --- |
| R² | **0.919** |
| RMSE | **300.5** sales units |
| MAE | **128.9** sales units |

## Project structure

```
superkart-streamlit/
├── app.py                     # Streamlit UI
├── superkart_core.py          # Data loading, preprocessing, training, metrics
├── train_model.py             # Optional: train and save a model artifact
├── requirements.txt
├── runtime.txt                # Python version for Streamlit Cloud
├── sample_batch.csv           # Example file for batch prediction
├── data/
│   └── SuperKart.csv          # 8,763 rows of historical sales
├── model/                     # Optional saved model + metadata
└── .streamlit/
    └── config.toml            # Theme
```

## Run locally

> Requires Python 3.10+.

```bash
git clone https://github.com/ruksana-ux/superkart-sales-prediction.git
cd superkart-sales-prediction

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### Optional: create a fixed model artifact

```bash
python train_model.py
```

This writes `model/SuperKart_model_v1_0.joblib` and
`model/model_metadata.json`. Commit them if you want the deployed app to skip
training at startup:

```bash
git add model/
git commit -m "Add trained SuperKart model artifact"
git push
```

## Deploy on GitHub + Streamlit Community Cloud

This repository is already deployed at
<https://superkart-prediction.streamlit.app>. The steps below show how it was set
up and how to redeploy it from scratch.

### 1. Push the code to GitHub

The repository is already initialised and committed locally. To push a fresh
copy yourself:

```bash
git init -b main
git add .
git commit -m "Deploy SuperKart sales prediction app"
git remote add origin https://github.com/ruksana-ux/superkart-sales-prediction.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with **GitHub**.
2. Click **Create app** (or **New app**).
3. Choose **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `ruksana-ux/superkart-sales-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** e.g. `superkart-sales-prediction`
5. Open **Advanced settings** and select **Python 3.12** (matching `runtime.txt`).
6. Click **Deploy**. The first build installs the dependencies, then the app
   trains the model on first load (roughly 20–60 seconds).
7. The public URL is shown at the top of this README; every push to `main`
   automatically redeploys the app.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Dataset not found` | Make sure `data/SuperKart.csv` is committed (it is tracked by Git, not ignored). |
| Build fails on `xgboost` | Check the Streamlit build logs; use Python 3.12 in **Advanced settings**. |
| First page load is slow | Expected: the model trains once per container start. Commit a joblib artifact to skip this. |
| Batch upload shows missing columns | Download the in-app CSV template; all eight feature columns are required. |
| Predictions look off for extreme inputs | The model was trained on the ranges in `SuperKart.csv`; predictions far outside them are extrapolations. |

## Credits

- Dataset: SuperKart retail sales data (8,763 records)
- Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org)
  and [XGBoost](https://xgboost.readthedocs.io)
- Original analysis: `SuperKart (2)` Google Colab notebook
