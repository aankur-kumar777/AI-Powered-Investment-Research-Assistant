# pipelines/quant_run_with_docs.py
"""
Quant pipeline that *adds document-derived features* to the XGBoost predictor.

What it does:
- Fetch price series (yfinance)
- Compute price-based factors (quant.factor_engineering)
- Extract document-derived features from the Chroma vectorstore:
    - mention_count: number of doc chunks that mention the ticker or metadata.ticker == ticker
    - mean_sentiment: average VADER compound sentiment across mentioned chunks
    - mention_ratio: mention_count / total_chunks_in_collection
  If document chunk metadata contains a parsable 'date' field (ISO string or pandas/np compatible),
  we compute time-aware cumulative counts and rolling counts per day (best-effort).
- Merge doc features into X (as static columns or time-varying when possible)
- Train XGBoost, generate signals, evaluate and backtest with vectorbt
- Save model + report artifacts under reports/<TICKER>/

Usage:
    python -m pipelines.quant_run_with_docs --ticker AAPL --period 1y

Dependencies (in addition to earlier):
    pip install vaderSentiment
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import vectorbt as vbt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# existing quant modules
from quant.factor_engineering import compute_basic_factors, add_target_next_return, prepare_features_for_model
from quant.predictor import train_xgb_regressor, predict_and_signal, evaluate_signals

# embeddings/vector store helpers
from embeddings.embedder import get_embedder, embed_texts
from embeddings.vector_store import init_chroma_client

# For typing convenience
CollectionType = Any


def fetch_price_series(ticker: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Failed to fetch price data for {ticker}")
    # prefer Close / Adj Close / close
    close_col = None
    for c in ["Close", "close", "Adj Close", "Adj_Close"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise RuntimeError("Couldn't find a close column in yfinance data")
    series = df[close_col].rename("close")
    return series


def _get_chroma_collection(collection_name: str = "research_docs") -> CollectionType:
    client = init_chroma_client()
    # If collection doesn't exist, this will raise or return None depending on chroma version
    try:
        return client.get_collection(collection_name)
    except Exception:
        # fallback: try to create and return empty collection
        return client.get_or_create_collection(collection_name)


def _safe_parse_date(v) -> Optional[pd.Timestamp]:
    """Try to parse a date-like value from metadata into a pandas Timestamp; return None if impossible."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v
    if isinstance(v, (np.datetime64, datetime)):
        try:
            return pd.to_datetime(v)
        except Exception:
            return None
    if isinstance(v, str):
        try:
            # Allow many common iso formats
            return pd.to_datetime(v, utc=False, errors="coerce")
        except Exception:
            return None
    return None


def extract_doc_features_for_ticker(
    ticker: str,
    collection_name: str = "research_docs",
    use_timeaware: bool = True,
    rolling_window_days: int = 7,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Extract document-derived features for a given ticker from Chroma collection.

    Returns:
      - static_features: dict with 'mention_count', 'mention_ratio', 'mean_sentiment', 'total_chunks'
      - time_series_df: Optional DataFrame indexed by date with columns:
            'daily_mentions', 'cumulative_mentions', 'daily_mean_sentiment', 'rolling_mentions'
        If the collection's metadata contains parseable dates and use_timeaware=True, a time series is returned.
        Otherwise, time_series_df is None and only static features are provided.
    """
    collection = _get_chroma_collection(collection_name)
    # retrieve all documents + metadata
    try:
        # Many chroma client versions support collection.get()
        all_data = collection.get()
        docs: List[str] = all_data.get("documents", []) or []
        metadatas: List[dict] = all_data.get("metadatas", []) or []
    except Exception:
        # If .get isn't available, try reading via query with empty text (best-effort)
        # Fall back to no docs
        docs = []
        metadatas = []

    total_chunks = len(docs)
    if total_chunks == 0:
        return ({"mention_count": 0, "mention_ratio": 0.0, "mean_sentiment": 0.0, "total_chunks": 0}, None)

    # sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    mention_flags = []
    mention_sentiments = []
    mention_dates = []

    ticker_lower = ticker.lower()
    for doc_text, meta in zip(docs, metadatas):
        text_lower = (doc_text or "").lower()
        meta_ticker = None
        if isinstance(meta, dict):
            meta_ticker = meta.get("ticker") or meta.get("symbol") or meta.get("source_ticker")
        # Check if ticker appears in metadata or in text; simple heuristics:
        text_mentions_ticker = (ticker_lower in text_lower)
        meta_mentions_ticker = False
        if meta_ticker:
            try:
                meta_mentions_ticker = (str(meta_ticker).lower() == ticker_lower)
            except Exception:
                meta_mentions_ticker = False

        is_mention = text_mentions_ticker or meta_mentions_ticker
        mention_flags.append(bool(is_mention))
        if is_mention:
            # sentiment
            try:
                vs = analyzer.polarity_scores(doc_text or "")
                mention_sentiments.append(float(vs.get("compound", 0.0)))
            except Exception:
                mention_sentiments.append(0.0)
            # parse date from metadata if present
            dt = None
            if isinstance(meta, dict):
                # common keys to check
                for dk in ("date", "published", "ingested_at", "pub_date"):
                    if dk in meta:
                        dt = _safe_parse_date(meta.get(dk))
                        if dt is not None:
                            break
            # append parsed date or None
            mention_dates.append(dt)
        else:
            mention_sentiments.append(0.0)
            mention_dates.append(None)

    mention_count = int(sum(1 for f in mention_flags if f))
    mention_ratio = mention_count / float(total_chunks)
    mean_sentiment = float(np.mean([s for s, f in zip(mention_sentiments, mention_flags) if f]) if mention_count > 0 else 0.0)

    static_features = {
        "mention_count": mention_count,
        "mention_ratio": mention_ratio,
        "mean_sentiment": mean_sentiment,
        "total_chunks": total_chunks,
    }

    # If user requested time-aware features and we have at least one parsed date, build a time series
    if use_timeaware:
        # collect only mentions with a valid date
        dated = [(d.date(), s) for d, s, f in zip(mention_dates, mention_sentiments, mention_flags) if (d is not None and f)]
        if len(dated) == 0:
            # cannot build time-aware features
            return static_features, None

        # Build a DataFrame with counts per date and mean sentiment per date
        dates_list, sents = zip(*dated)
        df_time = pd.DataFrame({"date": pd.to_datetime(list(dates_list)), "sentiment": list(sents)})
        df_time = df_time.groupby("date").agg(daily_mentions=("sentiment", "count"), daily_mean_sentiment=("sentiment", "mean"))
        df_time = df_time.sort_index()
        df_time["cumulative_mentions"] = df_time["daily_mentions"].cumsum()
        # rolling mentions
        df_time["rolling_mentions"] = df_time["daily_mentions"].rolling(window=rolling_window_days, min_periods=1).sum()
        # normalize by total_chunks to get ratios if desired
        df_time["daily_mention_ratio"] = df_time["daily_mentions"] / float(total_chunks)
        return static_features, df_time

    return static_features, None


def merge_doc_features_into_X(
    X: pd.DataFrame,
    doc_static: Dict[str, Any],
    doc_time: Optional[pd.DataFrame] = None,
    align_on_index: bool = True,
) -> pd.DataFrame:
    """
    Merge document features into X.

    - If doc_time is None: append static columns to X (constant across rows).
    - If doc_time is present and align_on_index=True: we attempt to align by date index:
        For each X.index date, if doc_time has that date, use doc_time values; otherwise forward-fill the doc_time series
        to create matching index and merge. doc_time must be indexed by date (Date-like).
    """
    X2 = X.copy()
    # add static features
    for k, v in doc_static.items():
        X2[f"doc_{k}"] = float(v)

    if doc_time is not None and align_on_index:
        # reindex doc_time to X index (date alignment)
        # ensure index is datetime
        try:
            doc_ts = doc_time.copy()
            doc_ts.index = pd.to_datetime(doc_ts.index)
            # create full range covering X index
            idx = pd.DatetimeIndex(X2.index)
            # reindex doc_ts to have the same index; forward-fill previous values
            doc_ts_reindexed = doc_ts.reindex(idx, method="ffill").fillna(0.0)
            # merge columns (prefix to avoid conflicts)
            for col in doc_ts_reindexed.columns:
                X2[f"doc_time_{col}"] = doc_ts_reindexed[col].values
        except Exception:
            # if alignment fails, fallback to static-only
            pass
    return X2


def run_pipeline_with_docs(
    ticker: str = "AAPL",
    period: str = "1y",
    collection_name: str = "research_docs",
    model_save_path: Optional[str] = None,
    report_dir: str = "reports",
    use_timeaware_doc_features: bool = True,
) -> Dict[str, Any]:
    os.makedirs(report_dir, exist_ok=True)
    report_dir = Path(report_dir) / ticker
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching price for {ticker} ({period})...")
    price = fetch_price_series(ticker, period=period)
    df = pd.DataFrame(price)

    print("Computing factors...")
    df = compute_basic_factors(df)
    df = add_target_next_return(df, horizon=1)

    # pick feature columns
    feature_cols = [c for c in df.columns if c.startswith(("ret_lag_", "mom_", "ret_roll_mean_", "ret_roll_std_", "sma_spread_"))]
    if not feature_cols:
        raise RuntimeError("No feature columns found — check factor_engineering outputs")
    X, y = prepare_features_for_model(df, feature_cols, target_col="target_return")
    print(f"Base X shape: {X.shape}")

    # Extract document-derived features
    print("Extracting document-derived features from vector store...")
    doc_static, doc_time = extract_doc_features_for_ticker(ticker, collection_name=collection_name, use_timeaware=use_timeaware_doc_features)

    if doc_time is not None:
        print(f"Found time-aware document features spanning {doc_time.index.min()} to {doc_time.index.max()}")
    else:
        print("No time-aware document features available; using static document aggregates.")

    # Merge doc features into X
    X_with_docs = merge_doc_features_into_X(X, doc_static, doc_time, align_on_index=True)
    print(f"X with docs shape: {X_with_docs.shape}")

    # Train model on combined features
    model_path = model_save_path or f"models/{ticker.lower()}_xgb_with_docs.joblib"
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    print("Training XGBoost regressor with document features...")
    model, metrics = train_xgb_regressor(X_with_docs, y, save_path=model_path)
    print("Training complete — metrics:", metrics)

    # predict and signal
    preds_df = predict_and_signal(model, X_with_docs)

    # Evaluate
    evals = evaluate_signals(preds_df, y.loc[preds_df.index])

    # Backtest
    print("Running backtest with vectorbt...")
    price_bt = price.loc[preds_df.index]
    entries = preds_df["signal"] == 1
    exits = preds_df["signal"] == -1
    pf = vbt.Portfolio.from_signals(price_bt, entries.values, exits.values, init_cash=10000)
    stats = {
        "total_return": float(pf.total_return()),
        "sharpe": float(pf.sharpe()),
        "max_drawdown": float(pf.max_drawdown()),
        "trades": int(pf.trades_count()),
    }

    # Save report CSV
    metrics_summary = {
        "train_mse": metrics.get("mse"),
        "train_directional_accuracy": metrics.get("directional_accuracy"),
        **evals,
        **stats,
        # doc static features for visibility
        "doc_mention_count": doc_static.get("mention_count"),
        "doc_mean_sentiment": doc_static.get("mean_sentiment"),
        "doc_mention_ratio": doc_static.get("mention_ratio"),
    }
    report_csv = report_dir / "report_metrics_with_docs.csv"
    pd.DataFrame([metrics_summary]).to_csv(report_csv, index=False)

    # Plot cumulative returns: buy-and-hold vs strategy
    strat_cumret = (1 + pf.returns).cumprod()
    bh_cumret = (1 + price_bt.pct_change().fillna(0)).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(bh_cumret.index, bh_cumret.values, label=f"{ticker} buy&hold")
    plt.plot(strat_cumret.index, strat_cumret.values, label="strategy")
    plt.title(f"Cumulative returns — {ticker} (with doc features)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    png_path = report_dir / "cumulative_returns_with_docs.png"
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    result = {
        "ticker": ticker,
        "model_path": model_path,
        "report_csv": str(report_csv),
        "plot_png": str(png_path),
        "metrics": metrics_summary,
    }
    print("Pipeline complete. Artifacts:", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quant pipeline with document-derived features")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--collection-name", type=str, default="research_docs")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--no-timeaware", dest="timeaware", action="store_false", help="Disable time-aware doc features")
    args = parser.parse_args()

    run_pipeline_with_docs(
        ticker=args.ticker,
        period=args.period,
        collection_name=args.collection_name,
        model_save_path=args.model_path,
        report_dir=args.report_dir,
        use_timeaware_doc_features=args.timeaware,
    )
