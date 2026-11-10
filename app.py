"""SuperKart sales prediction dashboard (Streamlit).

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_artifacts() -> dict:
    """Load the saved pipeline, or train one from the dataset on first run."""
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    loaded_from_disk = MODEL_PATH.exists()
    if loaded_from_disk:
        model = joblib.load(MODEL_PATH)
    else:
        model = train_pipeline(X_train, y_train)

    test_predictions = model.predict(X_test)

    metadata = {}
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


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def options_from(data: pd.DataFrame, column: str) -> list[str]:
    return sorted(data[column].dropna().astype(str).unique().tolist())


def build_template(data: pd.DataFrame) -> pd.DataFrame:
    """Small, valid CSV template for batch predictions."""
    columns = [*FEATURES]
    if "Product_Id" in data.columns:
        columns = ["Product_Id", *columns]
    template = data[columns].head(10).copy()
    for column in NUMERICAL_FEATURES:
        template[column] = pd.to_numeric(template[column], errors="coerce").round(3)
    return template


# ---------------------------------------------------------------- page styling
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
        .hero {
            background: linear-gradient(120deg, #1e1b4b 0%, #4c1d95 55%, #7c3aed 100%);
            border-radius: 18px;
            padding: 34px 38px 30px 38px;
            color: #f8fafc;
            margin-bottom: 26px;
        }
        .hero h1 {font-size: 2.2rem; margin: 0 0 8px 0; font-weight: 750; letter-spacing: -0.02em;}
        .hero p {margin: 0; color: #ddd6fe; font-size: 1.02rem; max-width: 820px; line-height: 1.6;}
        .hero .tag {
            display: inline-block; margin-bottom: 14px; padding: 4px 12px;
            border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
            font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #c4b5fd;
        }
        .stat-card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
            padding: 16px 18px; height: 100%;
        }
        .stat-label {color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;}
        .stat-value {color: #0f172a; font-size: 1.55rem; font-weight: 700; margin-top: 4px;}
        .stat-note {color: #94a3b8; font-size: 0.78rem; margin-top: 2px;}
        .result-card {
            background: linear-gradient(150deg, #f5f3ff 0%, #faf5ff 55%, #ffffff 100%);
            border: 1px solid #ddd6fe; border-radius: 18px;
            padding: 26px 28px; margin-bottom: 16px;
        }
        .result-label {color: #6d28d9; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;}
        .result-value {color: #0f172a; font-size: 2.7rem; font-weight: 800; letter-spacing: -0.03em; margin: 6px 0 2px 0;}
        .result-range {color: #475569; font-size: 0.95rem;}
        .result-note {color: #94a3b8; font-size: 0.78rem; margin-top: 12px;}
        .footer {color: #94a3b8; font-size: 0.8rem; margin-top: 30px; text-align: center;}
        div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------- load model
if not DATA_PATH.exists():
    st.error(
        "The dataset `data/SuperKart.csv` is missing from this deployment. "
        "Add it to the repository and redeploy the app."
    )
    st.stop()

try:
    with st.spinner(
        "Preparing the model — the first run trains XGBoost, which takes about 20 seconds…"
    ):
        art = get_artifacts()
except Exception as exc:  # pragma: no cover - deployment safety net
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

data = art["data"]
model = art["model"]
test_metrics = art["metrics"]["test"]
rmse = test_metrics["RMSE"]

# ------------------------------------------------------------------------ hero
st.markdown(
    f"""
    <div class="hero">
        <div class="tag">Retail analytics · Sales forecasting</div>
        <h1>SuperKart Sales Predictor</h1>
        <p>
            Estimate the total sales generated by a product in a SuperKart store from
            its price, weight, shelf allocation and store profile. Powered by a
            scikit-learn pipeline with an XGBoost regressor trained on
            {len(data):,} sales records.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_cols = st.columns(4)
stats = [
    ("Model accuracy (R²)", f"{test_metrics['R-squared']:.3f}", "on unseen test data"),
    ("Typical error (RMSE)", f"{rmse:,.0f}", "sales units, test set"),
    ("Mean error (MAE)", f"{test_metrics['MAE']:,.0f}", "sales units, test set"),
    ("Training records", f"{len(data):,}", "cleaned SuperKart rows"),
]
for column, (label, value, note) in zip(stat_cols, stats):
    with column:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About this project")
    st.write(
        "SuperKart is a retail chain. This app predicts the sales total of a "
        "product in a given store, using the same feature engineering as the "
        "original notebook."
    )
    st.divider()
    st.subheader("Model")
    st.write(
        "XGBoost regressor wrapped in a scikit-learn `Pipeline` "
        "(standard scaling for numeric features, one-hot encoding for categories)."
    )
    st.caption(
        "Loaded from a saved joblib artifact."
        if art["loaded_from_disk"]
        else "Trained automatically at startup from data/SuperKart.csv."
    )
    st.divider()
    st.subheader("Input features")
    for feature in FEATURES:
        st.markdown(f"- `{feature}`")
    st.divider()
    st.caption(
        "Predictions are estimates based on historical patterns. "
        "Actual sales depend on promotions, seasonality and local demand."
    )

# ------------------------------------------------------------------------- tabs
predict_tab, batch_tab, performance_tab, data_tab = st.tabs(
    ["🎯 Predict sales", "📄 Batch prediction", "📊 Model performance", "🔍 Data explorer"]
)

# ------------------------------------------------------------- single prediction
with predict_tab:
    st.subheader("Single prediction")
    st.caption("Enter the product and store details, then generate a sales estimate.")

    product_types = options_from(data, "Product_Type")
    sugar_options = options_from(data, "Product_Sugar_Content")
    city_options = options_from(data, "Store_Location_City_Type")
    store_options = options_from(data, "Store_Type")

    with st.form("single_prediction_form"):
        product_col, store_col = st.columns(2)

        with product_col:
            st.markdown("**Product details**")
            product_type = st.selectbox(
                "Product type",
                product_types,
                index=default_index(product_types, "Fruits and Vegetables"),
            )
            sugar_content = st.selectbox(
                "Sugar content",
                sugar_options,
                index=default_index(sugar_options, "Low Sugar"),
            )
            product_mrp = st.number_input(
                "Product MRP",
                min_value=1.0,
                max_value=1000.0,
                value=147.0,
                step=1.0,
                help="Maximum retail price of the product.",
            )
            product_weight = st.number_input(
                "Product weight",
                min_value=0.1,
                max_value=100.0,
                value=12.66,
                step=0.1,
                help="Product weight in the dataset's units.",
            )
            allocated_area = st.number_input(
                "Allocated shelf area",
                min_value=0.001,
                max_value=1.0,
                value=0.070,
                step=0.001,
                format="%.3f",
                help="Share of store area allocated to the product.",
            )

        with store_col:
            st.markdown("**Store details**")
            city_type = st.selectbox(
                "Store location city type",
                city_options,
                index=default_index(city_options, "Tier 2"),
            )
            store_type = st.selectbox(
                "Store type",
                store_options,
                index=default_index(store_options, "Supermarket Type2"),
            )
            establishment_year = st.slider(
                "Store establishment year",
                min_value=1980,
                max_value=2025,
                value=2002,
                step=1,
            )

        submitted = st.form_submit_button("Predict sales", type="primary")

    if submitted:
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
            prediction = float(predict_frame(model, input_row)[0])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            lower = max(0.0, prediction - rmse)
            upper = prediction + rmse
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted total sales</div>
                    <div class="result-value">{prediction:,.0f}</div>
                    <div class="result-range">
                        Likely range: {lower:,.0f} – {upper:,.0f} sales units
                    </div>
                    <div class="result-note">
                        Range = prediction ± RMSE ({rmse:,.0f}) on unseen test data.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            detail_col, summary_col = st.columns([3, 2])
            with detail_col:
                st.markdown("**Input summary**")
                summary = input_row.T.reset_index()
                summary.columns = ["Feature", "Value"]
                st.dataframe(summary, hide_index=True)
            with summary_col:
                st.markdown("**At a glance**")
                st.metric("Predicted sales", f"{prediction:,.0f}")
                st.metric("Expected error (MAE)", f"{test_metrics['MAE']:,.0f}")
                st.metric("Model R²", f"{test_metrics['R-squared']:.3f}")
                st.caption(
                    "A higher R² means the model explains more of the variation in sales."
                )

# ---------------------------------------------------------------- batch prediction
with batch_tab:
    st.subheader("Batch prediction")
    st.write(
        "Upload a CSV with the same feature columns as the training data. "
        "Any extra columns (such as `Product_Id`) are preserved in the output."
    )

    template = build_template(data)
    st.download_button(
        "⬇️ Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="superkart_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
        else:
            missing = [column for column in FEATURES if column not in batch_data.columns]
            if not batch_data.columns.size:
                st.error("The uploaded CSV has no columns.")
            elif missing:
                st.error(
                    "The uploaded CSV is missing required column(s): "
                    + ", ".join(f"`{column}`" for column in missing)
                )
            elif batch_data.empty:
                st.warning("The uploaded CSV contains no rows.")
            else:
                try:
                    predictions = predict_frame(model, batch_data)
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")
                else:
                    results = batch_data.copy()
                    results[PREDICTION_COLUMN] = np.round(predictions, 2)

                    st.success(
                        f"Predicted {len(results):,} row(s) successfully."
                    )

                    summary_cols = st.columns(3)
                    summary_cols[0].metric("Rows predicted", f"{len(results):,}")
                    summary_cols[1].metric(
                        "Average prediction", f"{results[PREDICTION_COLUMN].mean():,.0f}"
                    )
                    summary_cols[2].metric(
                        "Prediction range",
                        f"{results[PREDICTION_COLUMN].min():,.0f} – "
                        f"{results[PREDICTION_COLUMN].max():,.0f}",
                    )

                    st.dataframe(results, hide_index=True, height=380)
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                        type="primary",
                    )

# ---------------------------------------------------------------- model performance
with performance_tab:
    st.subheader("How well does the model perform?")
    st.write(
        "Metrics are calculated on a reproducible 80/20 train–test split "
        f"(random_state=42). The test set contains {len(art['y_test']):,} rows "
        "the model did not see during training."
    )

    metrics_frame = pd.DataFrame(art["metrics"]).T
    metrics_frame = metrics_frame[
        ["R-squared", "Adj. R-squared", "RMSE", "MAE", "MAPE"]
    ]
    display_frame = metrics_frame.copy()
    display_frame["R-squared"] = display_frame["R-squared"].map("{:.3f}".format)
    display_frame["Adj. R-squared"] = display_frame["Adj. R-squared"].map("{:.3f}".format)
    display_frame["RMSE"] = display_frame["RMSE"].map("{:,.2f}".format)
    display_frame["MAE"] = display_frame["MAE"].map("{:,.2f}".format)
    display_frame["MAPE"] = display_frame["MAPE"].map("{:.2%}".format)
    st.dataframe(display_frame, hide_index=False)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("**Actual vs predicted sales**")
        sample_size = min(1_200, len(art["y_test"]))
        scatter = pd.DataFrame(
            {
                "Actual sales": art["y_test"].to_numpy()[:sample_size],
                "Predicted sales": art["test_predictions"][:sample_size],
            }
        )
        st.scatter_chart(
            scatter,
            x="Actual sales",
            y="Predicted sales",
            height=360,
            color="#7c3aed",
        )
        st.caption("Points close to a 45° line indicate accurate predictions.")

    with chart_right:
        st.markdown("**What drives the prediction?**")
        top_features = art["importance"].head(12).set_index("Feature")
        st.bar_chart(top_features, height=360, color="#8b5cf6")
        st.caption("XGBoost gain-based feature importance, top 12 features.")

    st.markdown("**Residual distribution (test set)**")
    residuals = art["y_test"].to_numpy() - art["test_predictions"]
    counts, edges = np.histogram(residuals, bins=30)
    residual_frame = pd.DataFrame(
        {
            "Residual bin": [
                f"{edges[i]:,.0f} to {edges[i + 1]:,.0f}" for i in range(len(counts))
            ],
            "Rows": counts,
        }
    ).set_index("Residual bin")
    st.bar_chart(residual_frame, height=280, color="#a78bfa")
    st.caption(
        "Residual = actual − predicted. A distribution centred around zero "
        "suggests the model is not systematically over- or under-predicting."
    )

# ------------------------------------------------------------------- data explorer
with data_tab:
    st.subheader("Explore the training data")

    tile_cols = st.columns(4)
    tile_cols[0].metric("Rows", f"{len(data):,}")
    tile_cols[1].metric("Columns", f"{data.shape[1]}")
    if "Product_Id" in data.columns:
        tile_cols[2].metric("Products", f"{data['Product_Id'].nunique():,}")
    if "Store_Id" in data.columns:
        tile_cols[3].metric("Stores", f"{data['Store_Id'].nunique():,}")

    st.markdown("**Dataset preview**")
    st.dataframe(data.head(50), hide_index=True, height=320)

    st.markdown("**Numerical summary**")
    st.dataframe(
        data.select_dtypes("number").describe().T.round(2),
        hide_index=False,
    )

    explore_left, explore_right = st.columns(2)
    with explore_left:
        st.markdown("**Category distribution**")
        category = st.selectbox(
            "Choose a column",
            CATEGORICAL_FEATURES,
            key="explore_category",
        )
        value_counts = (
            data[category]
            .astype(str)
            .value_counts()
            .rename_axis(category)
            .to_frame("Rows")
        )
        st.bar_chart(value_counts, height=320, color="#7c3aed")

    with explore_right:
        st.markdown("**Correlation between numeric columns**")
        correlation = data.select_dtypes("number").corr(numeric_only=True).round(2)
        st.dataframe(correlation, height=320)

    with st.expander("Column dictionary"):
        st.markdown(
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

st.markdown(
    """
    <div class="footer">
        SuperKart Sales Predictor · Streamlit + scikit-learn + XGBoost ·
        Predictions are estimates, not guarantees.
    </div>
    """,
    unsafe_allow_html=True,
)
