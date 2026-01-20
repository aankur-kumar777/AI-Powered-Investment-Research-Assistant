# quant/predictor.py
"""
Predictor pipeline:
- Train an XGBoost regressor to predict next-day returns (or horizon returns).
- Produce a signal column based on predicted return threshold.
- Save/Load model (joblib).
- Small evaluate function (MSE, directional accuracy).

Dependencies: pandas, numpy, scikit-learn, xgboost, joblib
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

try:
    import xgboost as xgb
except Exception as e:
    raise ImportError("xgboost is required for quant/predictor.py. Install with `pip install xgboost`.") from e


def train_xgb_regressor(X: pd.DataFrame, y: pd.Series, save_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None, do_gridsearch: bool = False):
    """
    Train XGBoost regressor and optionally save it.
    Returns the trained model and evaluation metrics on a holdout set.
    """
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if do_gridsearch:
        estimator = xgb.XGBRegressor(**params)
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.05],
        }
        gs = GridSearchCV(estimator, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
    else:
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    directional = np.mean(np.sign(preds) == np.sign(y_test))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(model, save_path)

    return model, {"mse": mse, "directional_accuracy": float(directional)}


def load_model(path: str):
    return joblib.load(path)


def predict_and_signal(model, X: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """
    Predict returns and convert into trading signal:
      +1 => long if predicted return >= threshold
       0 => neutral
      -1 => short if predicted return <= -threshold

    Returns DataFrame with columns: 'pred', 'signal'
    """
    preds = model.predict(X)
    sig = np.where(preds >= threshold, 1, np.where(preds <= -threshold, -1, 0))
    out = pd.DataFrame({"pred": preds, "signal": sig}, index=X.index)
    return out


def evaluate_signals(signals_df: pd.DataFrame, actual_returns: pd.Series) -> Dict[str, Any]:
    """
    Evaluate simple P&L and metrics:
      - cumulative return of following the signal (simple, no transaction costs)
      - hit rate (directional)
    Assumes signals_df.index aligns with actual_returns index and that actual_returns are the forward return.
    """
    # align
    df = signals_df.copy()
    df = df.loc[df.index.intersection(actual_returns.index)]
    rets = actual_returns.loc[df.index]
    # strategy returns: shift signal by 0 (signal applies for the next period if target is forward return)
    strat_rets = df["signal"] * rets
    cumret = (1 + strat_rets).cumprod().iloc[-1] - 1.0
    hit_rate = np.mean(np.sign(df["pred"]) == np.sign(rets))
    return {"cumulative_return": float(cumret), "hit_rate": float(hit_rate), "mean_strat_return": float(strat_rets.mean())}
