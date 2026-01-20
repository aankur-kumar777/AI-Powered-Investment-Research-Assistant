# pipelines/quant_run.py
"""End-to-end quant example pipeline.

Usage:
    python -m pipelines.quant_run --ticker AAPL --period 1y --model-path models/aapl_xgb.joblib

Outputs:
- models/<ticker>_xgb.joblib            # saved model
- reports/<ticker>/report_metrics.csv   # metrics summary
- reports/<ticker>/cumulative_returns.png

Dependencies: yfinance, pandas, numpy, matplotlib, vectorbt
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import vectorbt as vbt

from quant.factor_engineering import compute_basic_factors, add_target_next_return, prepare_features_for_model
from quant.predictor import train_xgb_regressor, predict_and_signal, evaluate_signals, load_model


def fetch_price_series(ticker: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    """Fetch price series (closing prices) using yfinance."""
    df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Failed to fetch price data for {ticker}")
    # yfinance returns DatetimeIndex; use 'Close' or 'close'
    close_col = None
    for c in ["Close", "close", "Adj Close", "Adj_Close"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise RuntimeError("Couldn't find a close column in yfinance data")
    series = df[close_col].rename("close")
    return series


def run_pipeline(ticker: str = "AAPL", period: str = "1y", model_save_path: str = None, report_dir: str = "reports") -> dict:
    os.makedirs(report_dir, exist_ok=True)
    report_dir = Path(report_dir) / ticker
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching price for {ticker} ({period})...")
    price = fetch_price_series(ticker, period=period)
    df = pd.DataFrame(price)

    print("Computing factors...")
    df = compute_basic_factors(df)
    df = add_target_next_return(df, horizon=1)

    # choose feature columns (common prefixes)
    feature_cols = [c for c in df.columns if c.startswith(("ret_lag_", "mom_", "ret_roll_mean_", "ret_roll_std_", "sma_spread_"))]
    if not feature_cols:
        raise RuntimeError("No feature columns found — check factor_engineering outputs")

    X, y = prepare_features_for_model(df, feature_cols, target_col="target_return")
    print(f"Training dataset: X={X.shape}, y={y.shape}")

    # train model
    model_path = model_save_path or f"models/{ticker.lower()}_xgb.joblib"
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    print("Training XGBoost regressor...")
    model, metrics = train_xgb_regressor(X, y, save_path=model_path)
    print("Training complete — metrics:", metrics)

    # predict and signal
    preds_df = predict_and_signal(model, X)

    # Evaluate signals against true forward returns
    evals = evaluate_signals(preds_df, y.loc[preds_df.index])

    # Backtest with vectorbt using boolean entries/exits
    print("Running backtest with vectorbt...")
    # Align price to preds index
    price_bt = price.loc[preds_df.index]
    entries = preds_df['signal'] == 1
    exits = preds_df['signal'] == -1
    # Use from_signals: open on entries, close on exits
    pf = vbt.Portfolio.from_signals(price_bt, entries.values, exits.values, init_cash=10000)
    stats = {
        'total_return': float(pf.total_return()),
        'sharpe': float(pf.sharpe()),
        'max_drawdown': float(pf.max_drawdown()),
        'trades': int(pf.trades_count()),
    }

    # Save report CSV
    metrics_summary = {
        'train_mse': metrics.get('mse'),
        'train_directional_accuracy': metrics.get('directional_accuracy'),
        **evals,
        **stats,
    }
    report_csv = report_dir / 'report_metrics.csv'
    pd.DataFrame([metrics_summary]).to_csv(report_csv, index=False)

    # Plot cumulative returns: buy-and-hold vs strategy
    strat_cumret = (1 + pf.returns).cumprod()
    bh_cumret = (1 + price_bt.pct_change().fillna(0)).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(bh_cumret.index, bh_cumret.values, label=f"{ticker} buy&hold")
    plt.plot(strat_cumret.index, strat_cumret.values, label="strategy")
    plt.title(f"Cumulative returns — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    png_path = report_dir / 'cumulative_returns.png'
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()

    # Save model path and artifacts
    result = {
        'ticker': ticker,
        'model_path': model_path,
        'report_csv': str(report_csv),
        'plot_png': str(png_path),
        'metrics': metrics_summary,
    }
    print("Pipeline complete. Artifacts:", result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run quant example pipeline")
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--period', type=str, default='1y')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--report-dir', type=str, default='reports')
    args = parser.parse_args()

    run_pipeline(ticker=args.ticker, period=args.period, model_save_path=args.model_path, report_dir=args.report_dir)
