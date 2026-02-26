import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_data
from src.features import create_features
from src.model import build_model


def train_pipeline(ticker, start_date="2015-01-01", end_date=None, progress_callback=None):
    """
    Full training pipeline. Returns model, metrics dict, and data for visualization.
    progress_callback(fraction, message) is called to report progress.
    """
    def _progress(frac, msg):
        if progress_callback:
            progress_callback(frac, msg)

    _progress(0.05, "Downloading stock data...")
    df = load_data(ticker, start=start_date, end=end_date)

    if df.empty or len(df) < 100:
        raise ValueError(f"Not enough data for {ticker}. Got {len(df)} rows (need >= 100).")

    _progress(0.15, "Engineering features...")
    X, y, df_full = create_features(df)

    _progress(0.25, "Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    _progress(0.35, "Training Random Forest with hyperparameter tuning...")
    model = build_model()
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    _progress(0.80, "Evaluating model...")
    y_pred = best_model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    _progress(0.90, "Saving model...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "best_model.pkl")
    joblib.dump(best_model, model_path)

    _progress(1.0, "Done!")

    return {
        "model": best_model,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "df_full": df_full,
        "X": X,
        "best_params": model.best_params_,
        "feature_importances": dict(zip(X.columns, best_model.feature_importances_)),
    }