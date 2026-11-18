# 🛒 SuperKart Sales Predictor

[![smoke-test](https://github.com/ruksana-ux/superkart-sales-prediction/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/ruksana-ux/superkart-sales-prediction/actions/workflows/smoke-test.yml)

A web app that predicts the **total sales** a product will generate in a
SuperKart store, based on its price, weight, shelf allocation, category and the
store profile.

The project is the deployment of the original `SuperKart` Colab notebook
(exploratory analysis → preprocessing → Random Forest / XGBoost comparison →
model serialisation), rebuilt as a self-contained application and shipped twice
from the same code:

| Host | Live app | UI |
| --- | --- | --- |
| Streamlit Community Cloud | <https://superkart-prediction.streamlit.app> | `app.py` |
| Hugging Face Spaces (free ZeroGPU) | <https://ruksanakhanum-superkartproject.hf.space> | `gradio_space/app.py` |

Both hosts import the same `superkart_core.py` and load the same committed joblib
artifact, so they never drift apart.

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
├── app.py                     # Streamlit UI (source of truth for behaviour)
├── superkart_core.py          # Data loading, preprocessing, training, metrics
├── train_model.py             # Trains and saves the model artifact
├── deploy_hf.py               # Stages and uploads the paid Docker Space
├── deploy_hf_gradio.py        # Stages and uploads the free Gradio/ZeroGPU Space
├── publish_hf_assets.py       # Publishes the model + dataset to Hugging Face
├── requirements.txt           # Pinned dependency versions
├── runtime.txt                # Python version for Streamlit Cloud
├── sample_batch.csv           # Example file for batch prediction
├── data/
│   └── SuperKart.csv          # 8,763 rows of historical sales
├── model/
│   ├── SuperKart_model_v1_0.joblib   # Pre-trained pipeline (committed)
│   └── model_metadata.json           # Metrics and training details
├── gradio_space/              # Hugging Face Space (Gradio SDK, free ZeroGPU)
│   ├── app.py                 # Four-tab Gradio UI
│   ├── requirements.txt       # Runtime dependencies for the Space
│   └── README.md              # Space card + YAML config
├── space/
│   ├── README.md              # Space card + YAML config for the Docker Space
│   └── Dockerfile             # Runs the Streamlit app on port 7860
├── hf/
│   ├── model_card.md          # README for the Hugging Face model repo
│   └── dataset_card.md        # README for the Hugging Face dataset repo
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

The Gradio version runs from the same folder and reuses the same model and data:

```bash
pip install gradio joblib scikit-learn xgboost pandas numpy
python gradio_space/app.py     # opens at http://localhost:7860
```

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

## Deploy on Hugging Face Spaces (free)

There are two ways to host on Hugging Face. The **free** one is a Gradio Space on
**ZeroGPU** hardware; the Docker/Streamlit route is kept for anyone with a PRO
plan.

### Free path — Gradio on ZeroGPU

Hugging Face charges for `cpu-basic` compute Spaces (free accounts get
`402 Payment Required`; only *static* Spaces are free), but **ZeroGPU hardware is
still free** for personal accounts in good standing (verified email, older than
30 days, up to two Spaces). ZeroGPU only runs the **Gradio** SDK and refuses to
start a Space that declares no `@spaces.GPU` function, so:

- the app lives in `gradio_space/app.py` and mirrors all four Streamlit tabs;
- it requests Python 3.12 (`python_version` in `gradio_space/README.md`) because
  scikit-learn 1.9 needs 3.11+ — the ZeroGPU base image also offers 3.10.13;
- a `@spaces.GPU` probe is declared so the Space starts, but it is **not wired to
  any UI control** because every call spends the visitor's small daily ZeroGPU
  quota. The XGBoost model is CPU-only, so predictions never touch the GPU.

`deploy_hf_gradio.py` stages `gradio_space/` plus `superkart_core.py`, `data/` and
`model/` and uploads them, so the Space reuses the committed artifact.

**Automatic (recommended).** On GitHub open *Settings → Secrets and variables →
Actions* and add:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `HF_TOKEN` | your `hf_...` token (write scope) |
| Variable | `HF_GRADIO_SPACE` | `<your-hf-username>/Superkartproject` (optional) |

If `HF_GRADIO_SPACE` is unset it is derived from `HF_SPACE_ID` as
`<owner>/Superkartproject`. The **deploy-hf-gradio-space** workflow then uploads
on every push that touches `gradio_space/**` or `deploy_hf_gradio.py`.

**Manual.** From the project folder:

```bash
pip install "huggingface_hub>=0.34"
export HF_TOKEN=hf_xxxxxxxx            # PowerShell: $env:HF_TOKEN="hf_xxxxxxxx"
python deploy_hf_gradio.py --space-id <your-hf-username>/Superkartproject
```

To inspect what will be uploaded without contacting Hugging Face:

```bash
python deploy_hf_gradio.py --no-upload      # writes build/hf-gradio-space/
```

> Create the Space once at <https://huggingface.co/new-space> with **SDK:
> Gradio** and **Hardware: ZeroGPU** if the script cannot create it for you, then
> re-run the command to upload the files. The first build installs scikit-learn
> and XGBoost, which takes a few minutes.

### Paid path — Docker/Streamlit Space

Kept for PRO accounts that prefer the exact Streamlit UI. Spaces are Docker
containers, so `space/Dockerfile` starts Streamlit on the port Spaces expects
(7860).

**Option A — automatic.** Add the same `HF_TOKEN` secret and a `HF_SPACE_ID`
variable (`<your-hf-username>/superkart-sales-prediction`), plus
`HF_DEPLOY_ENABLED=true` once you have PRO. The **deploy-hf-space** workflow is
off until `HF_DEPLOY_ENABLED` is `true`, so pushes to `main` never fail on the
paid-tier limit.

**Option B — manual.**

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxxxxxxx
python deploy_hf.py --space-id <your-hf-username>/superkart-sales-prediction
```

> Why Docker and not the Streamlit SDK? Hugging Face's creation API only accepts
> `gradio`, `docker` or `static` — its own Streamlit template
> (`streamlit/streamlit-template-space`) is a Docker Space too.

## Publish the model and dataset to Hugging Face (free)

Model and dataset repositories are plain file storage, so they stay free even
though Spaces no longer are. `publish_hf_assets.py` uploads:

| Repository | Contents |
| --- | --- |
| `models/<user>/superkart-sales-prediction-model` | `SuperKart_model_v1_0.joblib`, `model_metadata.json`, model card |
| `datasets/<user>/superkart-sales-prediction-data` | `SuperKart.csv` (viewable with the dataset viewer), dataset card |

The **publish-hf-assets** workflow runs this automatically whenever the dataset,
the model artifact or a card changes. To run it yourself:

```bash
HF_TOKEN=hf_xxxx python publish_hf_assets.py
# or: python publish_hf_assets.py --owner <your-username>
```

The owner defaults to whichever account the token belongs to. This gives your
Hugging Face profile a public model and dataset alongside the app, without any
paid plan.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Dataset not found` | Make sure `data/SuperKart.csv` is committed (it is tracked by Git, not ignored). |
| Build fails on `xgboost` | Check the Streamlit build logs; use Python 3.12 in **Advanced settings**. |
| "Your app has been throttled" | Temporary free-tier CPU limit. It expires automatically; the pre-built artifact prevents repeat notices. |
| First page load is slow | Check that `model/SuperKart_model_v1_0.joblib` is committed; otherwise the app trains a fallback model. |
| Batch upload shows missing columns | Download the in-app CSV template; all eight feature columns are required. |
| Predictions look off for extreme inputs | The model was trained on the ranges in `SuperKart.csv`; predictions far outside them are extrapolations. |
| Space stuck in *Building* | Open the Space's *Logs → Build* tab; a dependency that cannot be built shows up there first. |
| `402 Payment Required` when deploying | Hugging Face needs PRO to host `cpu-basic` Spaces; use the free Gradio/ZeroGPU path above, or buy PRO for the Docker Space. |
| `401 Unauthorized` when deploying | The token is missing the **write** scope, or belongs to a different account than `HF_SPACE_ID`. |
| `No @spaces.GPU function detected during startup` | ZeroGPU requires at least one `@spaces.GPU` function; keep the probe defined in `gradio_space/app.py`. |
| `You have exceeded your ZeroGPU runs limit` | The free daily ZeroGPU quota is spent (2 min unauthenticated, 5 min signed in). Predictions are CPU-only, so this only affects the unused probe. |

## Credits

- Dataset: SuperKart retail sales data (8,763 records)
- Built with [Streamlit](https://streamlit.io), [Gradio](https://www.gradio.app),
  [scikit-learn](https://scikit-learn.org) and
  [XGBoost](https://xgboost.readthedocs.io)
- Original analysis: `SuperKart (2)` Google Colab notebook
