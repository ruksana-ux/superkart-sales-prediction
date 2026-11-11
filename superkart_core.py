"""Shared data and model logic for the SuperKart sales prediction app.

The model predicts ``Product_Store_Sales_Total`` from eight product and store
attributes. Preprocessing (scaling + one-hot encoding) and the regressor live in
a single scikit-learn pipeline so the exact same transformations are applied
during training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SuperKart.csv"
MODEL_PATH = ROOT / "model" / "SuperKart_model_v1_0.joblib"
METADATA_PATH = ROOT / "model" / "model_metadata.json"

TARGET = "Product_Store_Sales_Total"
PREDICTION_COLUMN = "Predicted_Product_Store_Sales_Total"

NUMERICAL_FEATURES = [
    "Product_MRP",
    "Product_Allocated_Area",
    "Product_Weight",
    "Store_Establishment_Year",
]
CATEGORICAL_FEATURES = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Location_City_Type",
    "Store_Type",
]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
OUTLIER_COLUMNS = [*NUMERICAL_FEATURES, TARGET]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the SuperKart dataset and validate its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place SuperKart.csv in the data/ folder."
        )

    data = pd.read_csv(path)
    required = {"Product_Id", *FEATURES, TARGET}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    data = data.drop_duplicates().reset_index(drop=True)
    return data


def treat_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Clip numerical columns to 1.5 x IQR, as done in the notebook."""
    cleaned = data.copy()
    for column in OUTLIER_COLUMNS:
        if column not in cleaned.columns:
            continue
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        cleaned[column] = cleaned[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return cleaned


def split_data(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a reproducible train/test split of the cleaned dataset."""
    cleaned = treat_outliers(data)
    X = cleaned[FEATURES]
    y = cleaned[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def build_pipeline(*, lightweight: bool = False) -> Pipeline:
    """Build the preprocessing + XGBoost pipeline used for training.

    ``lightweight`` is used only when the app has to train at startup on a
    low-resource host (such as Streamlit Community Cloud). It keeps CPU usage
    low by using fewer trees and a single thread.
    """
    numerical_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    regressor = XGBRegressor(
        n_estimators=150 if lightweight else 400,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_lambda=0.6,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1 if lightweight else -1,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", regressor)]
    )


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline | None = None,
    *,
    lightweight: bool = False,
) -> Pipeline:
    """Fit the pipeline on the training data and return it."""
    model = pipeline if pipeline is not None else build_pipeline(lightweight=lightweight)
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Compute the regression metrics used in the notebook."""
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    n_samples, n_features = X.shape
    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))

    return {
        "R-squared": float(r2),
        "Adj. R-squared": float(adjusted_r2),
        "RMSE": float(np.sqrt(mean_squared_error(y, predictions))),
        "MAE": float(mean_absolute_error(y, predictions)),
        "MAPE": float(mean_absolute_percentage_error(y, predictions)),
    }


def _get_step(pipeline: Pipeline, names: Iterable[str]):
    """Fetch a named step, tolerating pipelines built with other step names."""
    for name in names:
        if name in pipeline.named_steps:
            return pipeline.named_steps[name]
    raise KeyError(f"None of the steps {list(names)} exist in this pipeline.")


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return feature importances with readable, de-prefixed feature names."""
    preprocessor = _get_step(pipeline, ("preprocessor", "columntransformer"))
    regressor = _get_step(
        pipeline, ("regressor", "xgbregressor", "randomforestregressor")
    )

    encoded_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    readable = [
        str(name).split("__", 1)[-1].replace("_", " ") for name in encoded_names
    ]
    frame = pd.DataFrame({"Feature": readable, "Importance": importances})
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


def predict_frame(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Validate a feature frame and return non-negative predictions."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
        )

    if frame.empty:
        raise ValueError("The input table is empty.")

    inputs = frame[FEATURES].copy()
    for column in NUMERICAL_FEATURES:
        inputs[column] = pd.to_numeric(inputs[column], errors="coerce")

    invalid_numeric = inputs[NUMERICAL_FEATURES].isna().any()
    if invalid_numeric.any():
        bad = ", ".join(invalid_numeric[invalid_numeric].index)
        raise ValueError(f"Non-numeric or empty values found in: {bad}")

    for column in CATEGORICAL_FEATURES:
        inputs[column] = inputs[column].astype(str).str.strip()

    predictions = model.predict(inputs)
    return np.clip(predictions, 0, None)
