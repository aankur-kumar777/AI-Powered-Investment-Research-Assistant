# tests/test_quant.py
"""
Unit tests for quant modules using synthetic GBM price series.
"""
import numpy as np
import pandas as pd
from quant.factor_engineering import compute_basic_factors, add_target_next_return, prepare_features_for_model
from quant.predictor import train_xgb_regressor, predict_and_signal, evaluate_signals
import tempfile
import os

def generate_gbm_prices(n=1000, s0=100, mu=0.0002, sigma=0.01, seed=42):
    np.random.seed(seed)
    dt = 1.0
    eps = np.random.normal(loc=0.0, scale=1.0, size=n)
    returns = mu * dt + sigma * np.sqrt(dt) * eps
    prices = s0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(prices, index=dates, name="close")

def test_factor_and_train_smoke(tmp_path):
    price = generate_gbm_prices(n=500)
    df = pd.DataFrame(price)
    df = compute_basic_factors(df)
    df = add_target_next_return(df, horizon=1)
    # pick a set of features (existing from compute_basic_factors)
    feature_cols = [c for c in df.columns if c.startswith(("ret_lag_", "mom_", "ret_roll_mean_", "ret_roll_std_", "sma_spread_"))]
    X, y = prepare_features_for_model(df, feature_cols, target_col="target_return")
    assert not X.empty
    # train model
    model, metrics = train_xgb_regressor(X, y, save_path=str(tmp_path / "xgb.model"))
    assert "mse" in metrics
    # predict and signal
    preds = predict_and_signal(model, X, threshold=0.0005)
    assert "signal" in preds.columns
    # evaluate
    evals = evaluate_signals(preds, y.loc[preds.index])
    assert "cumulative_return" in evals
    # cleanup
    os.remove(str(tmp_path / "xgb.model"))
