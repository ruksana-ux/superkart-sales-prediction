# 🛒 SuperKart Sales Predictor

[![smoke-test](https://github.com/ruksana-ux/superkart-sales-prediction/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/ruksana-ux/superkart-sales-prediction/actions/workflows/smoke-test.yml)

A Streamlit web app that predicts the **total sales** a product will generate in a
SuperKart store, based on its price, weight, shelf allocation, category and the
store profile.

The project is the deployment of the original `SuperKart` Colab notebook
(exploratory analysis → preprocessing → Random Forest / XGBoost comparison →
model serialisation), rebuilt as a self-contained Streamlit application.

**🚀 Live app:** <https://superkart-prediction.streamlit.app>

> The trained model ships with the repository, so the app starts immediately.
> No training happens on page load.

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
This repository therefore includes a freshly trained artifact at
`model/SuperKart_model_v1_0.joblib`, built automatically by the
[`build-model`](.github/workflows/build-model.yml) GitHub Actions workflow.
The app loads it at startup, so page loads are fast and use almost no CPU.

If the artifact is missing (for example in a fork), the app falls back to
training a lightweight single-threaded model from `data/SuperKart.csv`. The full
400-tree model is only trained by `train_model.py` and the CI workflow.

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
├── train_model.py             # Trains and saves the model artifact
├── deploy_hf.py               # Stages and uploads the Hugging Face Space
├── requirements.txt           # Pinned dependency versions
├── runtime.txt                # Python version for Streamlit Cloud
├── sample_batch.csv           # Example file for batch prediction
├── data/
│   └── SuperKart.csv          # 8,763 rows of historical sales
├── model/
│   ├── SuperKart_model_v1_0.joblib   # Pre-trained pipeline (committed)
│   └── model_metadata.json           # Metrics and training details
├── space/
│   └── README.md              # Space card + YAML front matter for Hugging Face
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

### Rebuild the model artifact

The committed artifact was produced by the `build-model` GitHub Actions
workflow. To rebuild it manually:

```bash
python train_model.py
git add model/
git commit -m "Rebuild SuperKart model artifact"
git push
```

You can also run the **build-model** workflow from the *Actions* tab on GitHub;
it trains the model and commits the updated artifact for you.

## Staying within free-tier limits

Streamlit Community Cloud throttles apps that use too much CPU. The original
version trained a 400-tree XGBoost model on every container restart, which
caused the throttle warning. The current version:

- ships a pre-trained `model/SuperKart_model_v1_0.joblib` and only loads it;
- pins dependency versions so the artifact always unpickles cleanly;
- uses a lightweight, single-threaded model only if the artifact is missing.

No action is needed when a throttle notice appears — it expires automatically.
To avoid repeat notices, avoid repeated redeploys and keep the artifact committed.

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
6. Click **Deploy**. The first build installs the dependencies; because the model
   artifact is committed, the app is ready as soon as the build finishes.
7. The public URL is shown at the top of this README; every push to `main`
   automatically redeploys the app.

## Deploy on Hugging Face Spaces

Hugging Face Spaces is a second free host; it runs the exact same app. Spaces
use the **Streamlit SDK**, so no Dockerfile is needed and the app listens on
port 7860 automatically.

The Space is a separate repository on Hugging Face (`huggingface.co/spaces/...`)
whose README needs a YAML config block. Instead of duplicating files, this repo
generates that folder with `deploy_hf.py` and uploads it:

1. Create a free account at <https://huggingface.co/join>.
2. Create a token with **write** scope at
   <https://huggingface.co/settings/tokens>.
3. Run one of the two options below.

**Option A — automatic (recommended).** On GitHub open
*Settings → Secrets and variables → Actions* and add:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `HF_TOKEN` | your `hf_...` token |
| Variable | `HF_SPACE_ID` | `<your-hf-username>/superkart-sales-prediction` |

Then run the **deploy-hf-space** workflow (or just push to `main`). It mirrors
`app.py`, `superkart_core.py`, `requirements.txt`, `data/`, `model/` and
`.streamlit/config.toml` into the Space, so the two hosts never drift apart.
If the Space does not exist yet, the workflow creates it as a public Streamlit
Space.

**Option B — manual.** From the project folder:

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxxxxxxx            # PowerShell: $env:HF_TOKEN="hf_xxxxxxxx"
python deploy_hf.py --space-id <your-hf-username>/superkart-sales-prediction
```

To inspect what will be uploaded without contacting Hugging Face:

```bash
python deploy_hf.py --no-upload      # writes build/hf-space/
```

The build page for the Space shows the install progress; the first build takes a
few minutes, later pushes rebuild in under a minute.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Dataset not found` | Make sure `data/SuperKart.csv` is committed (it is tracked by Git, not ignored). |
| Build fails on `xgboost` | Check the Streamlit build logs; use Python 3.12 in **Advanced settings**. |
| "Your app has been throttled" | Temporary free-tier CPU limit. It expires automatically; the pre-built artifact prevents repeat notices. |
| First page load is slow | Check that `model/SuperKart_model_v1_0.joblib` is committed; otherwise the app trains a fallback model. |
| Batch upload shows missing columns | Download the in-app CSV template; all eight feature columns are required. |
| Predictions look off for extreme inputs | The model was trained on the ranges in `SuperKart.csv`; predictions far outside them are extrapolations. |
| Space stuck in *Building* | Open the Space's *Logs* tab; a dependency that cannot be built shows up there first. |
| `401 Unauthorized` when deploying | The token is missing the **write** scope, or belongs to a different account than `HF_SPACE_ID`. |

## Credits

- Dataset: SuperKart retail sales data (8,763 records)
- Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org)
  and [XGBoost](https://xgboost.readthedocs.io)
- Original analysis: `SuperKart (2)` Google Colab notebook
