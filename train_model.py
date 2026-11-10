"""Train the SuperKart sales model and save the deployment artifact.

Usage:
    python train_model.py

The Streamlit app trains the same pipeline automatically if
``model/SuperKart_model_v1_0.joblib`` is absent, so running this script is
optional. It is useful when you want a fixed, versioned model artifact that can
be committed to Git or shared with others.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib

from superkart_core import (
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    evaluate,
    load_data,
    split_data,
    train_pipeline,
)


def main() -> None:
    print("Loading dataset…")
    data = load_data()
    print(f"   {len(data):,} rows, {data.shape[1]} columns")

    print("Splitting and cleaning data…")
    X_train, X_test, y_train, y_test = split_data(data)
    print(f"   train: {len(X_train):,} rows | test: {len(X_test):,} rows")

    print("Training the XGBoost pipeline…")
    model = train_pipeline(X_train, y_train)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "test": evaluate(model, X_test, y_test),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    metadata = {
        "project": "SuperKart sales prediction",
        "model": "XGBoostRegressor (scikit-learn Pipeline)",
        "target": "Product_Store_Sales_Total",
        "dataset_rows": int(len(data)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata saved to {METADATA_PATH}")

    print("\nTest-set performance")
    for name, value in metrics["test"].items():
        print(f"   {name:<16} {value:,.4f}")


if __name__ == "__main__":
    main()
