"""SuperKart sales prediction dashboard (Gradio).

This is the Hugging Face Spaces version of the app. It mirrors the four tabs of
the Streamlit build (``app.py``) - single prediction, batch prediction, model
performance and data explorer - and reuses ``superkart_core.py`` and the saved
joblib artifact, so the two hosts never drift apart.

It targets Hugging Face's free **ZeroGPU** hardware, which only runs Gradio
Spaces and requires at least one ``@spaces.GPU`` function to exist at startup.
The XGBoost model is CPU-only, so inference runs in the main process and the GPU
function is only an environment probe.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from superkart_core import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    PREDICTION_COLUMN,
    TARGET,
    evaluate,
    get_feature_importance,
    load_data,
    predict_frame,
    split_data,
    train_pipeline,
)

# --------------------------------------------------------------------- ZeroGPU
# ``spaces`` is preinstalled on ZeroGPU hardware. Guard the import so the same
# file also runs on a plain CPU host (for example Streamlit or a local machine).
try:  # pragma: no cover - depends on the host
    import spaces

    HAS_SPACES = True
except Exception:  # noqa: BLE001
    spaces = None
    HAS_SPACES = False


def _gpu_probe() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"ZeroGPU worker ready — CUDA on {torch.cuda.get_device_name(0)}."
        return "GPU worker reachable, but CUDA is not available."
    except Exception as exc:  # noqa: BLE001
        return f"GPU probe failed: {exc}"


if HAS_SPACES:

    @spaces.GPU
    def gpu_probe() -> str:
        """Runs on the ZeroGPU worker; the model itself does not need a GPU."""
        return _gpu_probe()

else:

    def gpu_probe() -> str:  # type: ignore[misc]
        return _gpu_probe() + " (running without ZeroGPU)"


# ------------------------------------------------------------------ load model
def prepare_artifacts() -> dict:
    """Load the saved pipeline (or train it) and compute every metric once."""
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    loaded_from_disk = MODEL_PATH.exists()
    if loaded_from_disk:
        model = joblib.load(MODEL_PATH)
    else:
        # Fallback for forks without the artifact; keep CPU usage low.
        model = train_pipeline(X_train, y_train, lightweight=True)

    test_predictions = model.predict(X_test)

    metadata: dict = {}
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}

    return {
        "data": data,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "metrics": {
            "train": evaluate(model, X_train, y_train),
            "test": evaluate(model, X_test, y_test),
        },
        "importance": get_feature_importance(model),
        "loaded_from_disk": loaded_from_disk,
        "metadata": metadata,
    }


ART = prepare_artifacts()
DATA = ART["data"]
MODEL = ART["model"]
TEST_METRICS = ART["metrics"]["test"]
RMSE = TEST_METRICS["RMSE"]


def options_from(column: str) -> list[str]:
    return sorted(DATA[column].dropna().astype(str).unique().tolist())


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def default_value(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else options[0]


PRODUCT_TYPES = options_from("Product_Type")
SUGAR_OPTIONS = options_from("Product_Sugar_Content")
CITY_OPTIONS = options_from("Store_Location_City_Type")
STORE_OPTIONS = options_from("Store_Type")


def build_template() -> Path:
    """Write a small, valid CSV template for batch predictions."""
    columns = [*FEATURES]
    if "Product_Id" in DATA.columns:
        columns = ["Product_Id", *columns]
    template = DATA[columns].head(10).copy()
    for column in NUMERICAL_FEATURES:
        template[column] = pd.to_numeric(template[column], errors="coerce").round(3)
    path = Path(tempfile.gettempdir()) / "superkart_batch_template.csv"
    template.to_csv(path, index=False)
    return path


TEMPLATE_PATH = build_template()


# ------------------------------------------------------------- single prediction
def predict_single(
    product_type: str,
    sugar_content: str,
    product_mrp: float,
    product_weight: float,
    allocated_area: float,
    city_type: str,
    store_type: str,
    establishment_year: int,
):
    input_row = pd.DataFrame(
        [
            {
                "Product_MRP": product_mrp,
                "Product_Allocated_Area": allocated_area,
                "Product_Weight": product_weight,
                "Store_Establishment_Year": establishment_year,
                "Product_Sugar_Content": sugar_content,
                "Product_Type": product_type,
                "Store_Location_City_Type": city_type,
                "Store_Type": store_type,
            }
        ]
    )

    try:
        prediction = float(predict_frame(MODEL, input_row)[0])
    except Exception as exc:  # noqa: BLE001
        return f"### Prediction failed\n\n`{exc}`", pd.DataFrame()

    lower = max(0.0, prediction - RMSE)
    upper = prediction + RMSE
    card = (
        "### Predicted total sales\n"
        f"# {prediction:,.0f}\n"
        f"Likely range: **{lower:,.0f} – {upper:,.0f}** sales units  \n"
        f"*Range = prediction ± RMSE ({RMSE:,.0f}) on unseen test data.*\n\n"
        "---\n\n"
        f"- **Predicted sales:** {prediction:,.0f}\n"
        f"- **Expected error (MAE):** {TEST_METRICS['MAE']:,.0f}\n"
        f"- **Model R²:** {TEST_METRICS['R-squared']:.3f}\n"
    )

    summary = input_row.T.reset_index()
    summary.columns = ["Feature", "Value"]
    return card, summary


# -------------------------------------------------------------- batch prediction
def predict_batch(file_path: str | None):
    empty = (gr.update(), gr.update(), gr.update())
    if not file_path:
        return ("Upload a CSV file first.", pd.DataFrame(), gr.update())

    try:
        batch_data = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        return (f"Could not read the uploaded CSV: `{exc}`", pd.DataFrame(), gr.update())

    missing = [column for column in FEATURES if column not in batch_data.columns]
    if not batch_data.columns.size:
        return ("The uploaded CSV has no columns.", pd.DataFrame(), gr.update())
    if missing:
        pretty = ", ".join(f"`{column}`" for column in missing)
        return (f"The uploaded CSV is missing required column(s): {pretty}", pd.DataFrame(), gr.update())
    if batch_data.empty:
        return ("The uploaded CSV contains no rows.", pd.DataFrame(), gr.update())

    try:
        predictions = predict_frame(MODEL, batch_data)
    except Exception as exc:  # noqa: BLE001
        return (f"Batch prediction failed: `{exc}`", pd.DataFrame(), gr.update())

    results = batch_data.copy()
    results[PREDICTION_COLUMN] = np.round(predictions, 2)

    out_path = Path(tempfile.gettempdir()) / "superkart_predictions.csv"
    results.to_csv(out_path, index=False)

    status = (
        f"Predicted **{len(results):,}** row(s) successfully.\n\n"
        f"| Rows predicted | Average prediction | Prediction range |\n"
        f"| --- | --- | --- |\n"
        f"| {len(results):,} | {results[PREDICTION_COLUMN].mean():,.0f} | "
        f"{results[PREDICTION_COLUMN].min():,.0f} – {results[PREDICTION_COLUMN].max():,.0f} |"
    )
    return (status, results.head(200), gr.update(value=str(out_path), interactive=True))


# ------------------------------------------------------------ model performance
def metrics_table() -> pd.DataFrame:
    frame = pd.DataFrame(ART["metrics"]).T
    frame = frame[["R-squared", "Adj. R-squared", "RMSE", "MAE", "MAPE"]]
    display = frame.copy()
    display["R-squared"] = display["R-squared"].map("{:.3f}".format)
    display["Adj. R-squared"] = display["Adj. R-squared"].map("{:.3f}".format)
    display["RMSE"] = display["RMSE"].map("{:,.2f}".format)
    display["MAE"] = display["MAE"].map("{:,.2f}".format)
    display["MAPE"] = display["MAPE"].map("{:.2%}".format)
    return display.reset_index().rename(columns={"index": "Split"})


def actual_vs_predicted() -> pd.DataFrame:
    sample_size = min(1_200, len(ART["y_test"]))
    return pd.DataFrame(
        {
            "Actual sales": np.asarray(ART["y_test"])[:sample_size],
            "Predicted sales": ART["test_predictions"][:sample_size],
        }
    )


def importance_frame() -> pd.DataFrame:
    return ART["importance"].head(12).copy()


def residual_frame() -> pd.DataFrame:
    residuals = np.asarray(ART["y_test"]) - ART["test_predictions"]
    counts, edges = np.histogram(residuals, bins=30)
    return pd.DataFrame(
        {
            "Residual bin": [
                f"{edges[i]:,.0f} to {edges[i + 1]:,.0f}" for i in range(len(counts))
            ],
            "Rows": counts,
        }
    )


# ----------------------------------------------------------------- data explorer
def preview_frame() -> pd.DataFrame:
    return DATA.head(50)


def describe_frame() -> pd.DataFrame:
    return DATA.select_dtypes("number").describe().T.round(2).reset_index().rename(
        columns={"index": "Column"}
    )


def correlation_frame() -> pd.DataFrame:
    corr = DATA.select_dtypes("number").corr(numeric_only=True).round(2)
    return corr.reset_index().rename(columns={"index": "Column"})


def category_counts(column: str) -> pd.DataFrame:
    counts = DATA[column].astype(str).value_counts()
    return pd.DataFrame({column: counts.index, "Rows": counts.values})


def data_tiles() -> str:
    parts = [
        f"**Rows:** {len(DATA):,}",
        f"**Columns:** {DATA.shape[1]}",
    ]
    if "Product_Id" in DATA.columns:
        parts.append(f"**Products:** {DATA['Product_Id'].nunique():,}")
    if "Store_Id" in DATA.columns:
        parts.append(f"**Stores:** {DATA['Store_Id'].nunique():,}")
    return " &nbsp;·&nbsp; ".join(parts)


HEADER = f"""
# 🛒 SuperKart Sales Predictor

Estimate the **total sales** a product generates in a SuperKart store from its
price, weight, shelf allocation and store profile. Powered by a scikit-learn
pipeline with an XGBoost regressor trained on {len(DATA):,} sales records.

| Model accuracy (R²) | Typical error (RMSE) | Mean error (MAE) | Training records |
| --- | --- | --- | --- |
| **{TEST_METRICS['R-squared']:.3f}** on unseen test data | **{RMSE:,.0f}** sales units, test set | **{TEST_METRICS['MAE']:,.0f}** sales units, test set | **{len(DATA):,}** cleaned SuperKart rows |
"""

FOOTER = (
    "*SuperKart Sales Predictor · Gradio + scikit-learn + XGBoost · "
    "predictions are estimates, not guarantees.*"
)


# ----------------------------------------------------------------------- layout
with gr.Blocks(title="SuperKart Sales Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)

    with gr.Tabs():
        # ----------------------------------------------------- predict sales
        with gr.Tab("🎯 Predict sales"):
            gr.Markdown(
                "### Single prediction\n"
                "Enter the product and store details, then generate a sales estimate."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Product details**")
                    product_type = gr.Dropdown(
                        PRODUCT_TYPES,
                        value=default_value(PRODUCT_TYPES, "Fruits and Vegetables"),
                        label="Product type",
                    )
                    sugar_content = gr.Dropdown(
                        SUGAR_OPTIONS,
                        value=default_value(SUGAR_OPTIONS, "Low Sugar"),
                        label="Sugar content",
                    )
                    product_mrp = gr.Number(value=147.0, label="Product MRP")
                    product_weight = gr.Number(value=12.66, label="Product weight")
                    allocated_area = gr.Number(value=0.070, label="Allocated shelf area")
                with gr.Column():
                    gr.Markdown("**Store details**")
                    city_type = gr.Dropdown(
                        CITY_OPTIONS,
                        value=default_value(CITY_OPTIONS, "Tier 2"),
                        label="Store location city type",
                    )
                    store_type = gr.Dropdown(
                        STORE_OPTIONS,
                        value=default_value(STORE_OPTIONS, "Supermarket Type2"),
                        label="Store type",
                    )
                    establishment_year = gr.Slider(
                        minimum=1980,
                        maximum=2025,
                        value=2002,
                        step=1,
                        label="Store establishment year",
                    )
            predict_button = gr.Button("Predict sales", variant="primary")
            result_card = gr.Markdown()
            with gr.Accordion("Input summary", open=False):
                summary_table = gr.Dataframe(interactive=False, wrap=True)

            predict_button.click(
                predict_single,
                inputs=[
                    product_type,
                    sugar_content,
                    product_mrp,
                    product_weight,
                    allocated_area,
                    city_type,
                    store_type,
                    establishment_year,
                ],
                outputs=[result_card, summary_table],
            )

        # -------------------------------------------------- batch prediction
        with gr.Tab("📄 Batch prediction"):
            gr.Markdown(
                "### Batch prediction\n"
                "Upload a CSV with the same feature columns as the training data. "
                "Any extra columns (such as `Product_Id`) are preserved in the output."
            )
            gr.DownloadButton(
                "⬇️ Download CSV template", value=str(TEMPLATE_PATH)
            )
            batch_file = gr.File(
                label="Upload a CSV file for batch prediction",
                file_types=[".csv"],
                type="filepath",
            )
            batch_button = gr.Button("Run batch prediction", variant="primary")
            batch_status = gr.Markdown()
            batch_table = gr.Dataframe(interactive=False, wrap=True)
            batch_download = gr.DownloadButton(
                "⬇️ Download predictions as CSV", interactive=False
            )
            batch_button.click(
                predict_batch,
                inputs=[batch_file],
                outputs=[batch_status, batch_table, batch_download],
            )

        # ------------------------------------------------- model performance
        with gr.Tab("📊 Model performance"):
            gr.Markdown(
                "### How well does the model perform?\n"
                "Metrics are calculated on a reproducible 80/20 train–test split "
                f"(random_state=42). The test set contains {len(ART['y_test']):,} rows "
                "the model did not see during training."
            )
            gr.Dataframe(value=metrics_table(), interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Actual vs predicted sales**")
                    gr.ScatterPlot(
                        value=actual_vs_predicted(),
                        x="Actual sales",
                        y="Predicted sales",
                        height=360,
                    )
                    gr.Markdown(
                        "*Points close to a 45° line indicate accurate predictions.*"
                    )
                with gr.Column():
                    gr.Markdown("**What drives the prediction?**")
                    gr.BarPlot(
                        value=importance_frame(),
                        x="Feature",
                        y="Importance",
                        height=360,
                        x_label_angle=-45,
                    )
                    gr.Markdown(
                        "*XGBoost gain-based feature importance, top 12 features.*"
                    )
            gr.Markdown("**Residual distribution (test set)**")
            gr.BarPlot(
                value=residual_frame(),
                x="Residual bin",
                y="Rows",
                height=280,
                x_label_angle=-45,
            )
            gr.Markdown(
                "*Residual = actual − predicted. A distribution centred around zero "
                "suggests the model is not systematically over- or under-predicting.*"
            )

        # ----------------------------------------------------- data explorer
        with gr.Tab("🔍 Data explorer"):
            gr.Markdown("### Explore the training data")
            gr.Markdown(data_tiles())
            gr.Markdown("**Dataset preview**")
            gr.Dataframe(value=preview_frame(), interactive=False, wrap=True)
            gr.Markdown("**Numerical summary**")
            gr.Dataframe(value=describe_frame(), interactive=False, wrap=True)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Category distribution**")
                    category = gr.Dropdown(
                        CATEGORICAL_FEATURES,
                        value=CATEGORICAL_FEATURES[0],
                        label="Choose a column",
                    )
                    category_plot = gr.BarPlot(
                        value=category_counts(CATEGORICAL_FEATURES[0]),
                        x=CATEGORICAL_FEATURES[0],
                        y="Rows",
                        height=320,
                        x_label_angle=-45,
                    )
                    category.change(
                        lambda column: gr.BarPlot(
                            value=category_counts(column),
                            x=column,
                            y="Rows",
                            height=320,
                            x_label_angle=-45,
                        ),
                        inputs=category,
                        outputs=category_plot,
                    )
                with gr.Column():
                    gr.Markdown("**Correlation between numeric columns**")
                    gr.Dataframe(value=correlation_frame(), interactive=False)
            with gr.Accordion("Column dictionary", open=False):
                gr.Markdown(
                    f"""
                    | Column | Role |
                    | --- | --- |
                    | `Product_Id` | Product identifier (not used by the model) |
                    | `Product_MRP` | Maximum retail price |
                    | `Product_Allocated_Area` | Share of store area allocated to the product |
                    | `Product_Weight` | Product weight |
                    | `Product_Sugar_Content` | Sugar category |
                    | `Product_Type` | Product category |
                    | `Store_Id` | Store identifier (not used by the model) |
                    | `Store_Establishment_Year` | Year the store opened |
                    | `Store_Size` | Store size band (not used by the model) |
                    | `Store_Location_City_Type` | City tier |
                    | `Store_Type` | Store format |
                    | `{TARGET}` | Target: total sales |
                    """
                )

    with gr.Accordion("Environment check", open=False):
        gr.Markdown(
            "Hugging Face's free ZeroGPU hardware requires the Space to declare a "
            "`@spaces.GPU` function. The XGBoost model is CPU-only, so this button "
            "just verifies the GPU worker is reachable."
        )
        gpu_button = gr.Button("Probe the GPU worker")
        gpu_output = gr.Textbox(label="Result")
        gpu_button.click(gpu_probe, inputs=[], outputs=[gpu_output])

    gr.Markdown(FOOTER)


if __name__ == "__main__":
    demo.launch()
