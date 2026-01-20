# quant/factor_engineering.py
"""
Factor engineering utilities.

Functions:
- compute_basic_factors(df, price_col='close'): returns DataFrame with:
    - returns (log and simple)
    - rolling mean returns
    - rolling volatility (std)
    - momentum (price / price_n)
    - zscore of returns
    - lag features
- add_target_next_return(df, horizon=1): adds 'target_return' which is simple return over next `horizon` periods
- prepare_features_for_model(df, feature_cols, dropna=True): returns X, y ready for model training
"""
from typing import List, Tuple
import pandas as pd
import numpy as np


def compute_basic_factors(df: pd.DataFrame, price_col: str = "close", windows: List[int] = None, n_lags: int = 3) -> pd.DataFrame:
    """
    Given a price DataFrame (datetime-index or with a 'date' column) with a 'close' column (or price_col),
    compute several standard factors and return a new DataFrame (copy).
    """
    if windows is None:
        windows = [5, 10, 20]  # short/medium/long

    df = df.copy()
    # Ensure sorting by time
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")
    else:
        df = df.sort_index()

    price = df[price_col].astype(float)

    # Simple returns and log returns
    df["ret_1"] = price.pct_change()               # simple return
    df["logret_1"] = np.log(price).diff()          # log return

    # rolling statistics
    for w in windows:
        df[f"ret_roll_mean_{w}"] = df["ret_1"].rolling(window=w).mean()
        df[f"ret_roll_std_{w}"] = df["ret_1"].rolling(window=w).std()
        # momentum as price ratio vs w periods ago
        df[f"mom_{w}"] = price / price.shift(w) - 1.0

    # volatility (annualized approx) from daily std (if daily)
    # default assume ~252 trading days for annualization if user wants it later
    df["vol_10"] = df["ret_1"].rolling(window=10).std()

    # z-score of recent returns (using 20 window)
    df["zscore_ret_20"] = (df["ret_1"] - df["ret_1"].rolling(20).mean()) / (df["ret_1"].rolling(20).std() + 1e-12)

    # lag features
    for lag in range(1, n_lags + 1):
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)

    # simple moving averages and sma spread
    df["sma_10"] = price.rolling(window=10).mean()
    df["sma_50"] = price.rolling(window=50).mean()
    df["sma_spread_10_50"] = (df["sma_10"] - df["sma_50"]) / (df["sma_50"] + 1e-12)

    return df


def add_target_next_return(df: pd.DataFrame, price_col: str = "close", horizon: int = 1, target_col: str = "target_return") -> pd.DataFrame:
    """
    Add a forward-looking target: simple return from t to t+horizon (non-overlapping).
    target_return at time t = (price_{t+horizon} / price_t) - 1
    """
    df = df.copy()
    price = df[price_col]
    df[target_col] = price.shift(-horizon) / price - 1.0
    return df


def prepare_features_for_model(df: pd.DataFrame, feature_cols: List[str], target_col: str = "target_return", dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return X (DataFrame) and y (Series) for model training.
    """
    df = df.copy()
    X = df[feature_cols]
    y = df[target_col]
    if dropna:
        mask = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[mask]
        y = y.loc[mask]
    return X, y
