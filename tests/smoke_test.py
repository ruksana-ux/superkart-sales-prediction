"""Headless smoke tests for the SuperKart app.

Runs without a browser:
    python tests/smoke_test.py

1. Trains the real pipeline on the real dataset and checks inference.
2. Boots the Streamlit app with Streamlit's AppTest and checks it renders.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from superkart_core import (  # noqa: E402
    evaluate,
    load_data,
    predict_frame,
    split_data,
    train_pipeline,
)


def check_model() -> None:
    print("Loading data and training the pipeline…")
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_pipeline(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print(
        "Test metrics: "
        f"R2={metrics['R-squared']:.3f} "
        f"RMSE={metrics['RMSE']:.1f} "
        f"MAE={metrics['MAE']:.1f}"
    )
    assert metrics["R-squared"] > 0.5, "Model quality regressed unexpectedly"

    sample = X_test.head(5)
    predictions = predict_frame(model, sample)
    assert len(predictions) == len(sample)
    assert (predictions >= 0).all(), "Predictions must not be negative"
    print(f"Sample predictions: {[round(float(p), 2) for p in predictions]}")


def check_app() -> None:
    from streamlit.testing.v1 import AppTest

    print("Booting the Streamlit app headlessly…")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()

    if at.exception:
        for exception in at.exception:
            print(exception.message)
        raise SystemExit("The Streamlit app raised an exception")

    rendered = " ".join(block.value for block in at.markdown)
    assert "SuperKart Sales Predictor" in rendered, "Hero section did not render"
    assert len(at.selectbox) >= 4, "Expected the four category selectors"
    assert len(at.number_input) >= 3, "Expected the numeric input widgets"
    print(
        f"App rendered: {len(at.selectbox)} selectboxes, "
        f"{len(at.number_input)} number inputs, {len(at.tabs)} tabs"
    )


if __name__ == "__main__":
    check_model()
    check_app()
    print("All smoke tests passed.")
