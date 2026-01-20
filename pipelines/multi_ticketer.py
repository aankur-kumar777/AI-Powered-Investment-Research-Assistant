# pipelines/prefect_multi_ticker.py
"""Prefect multi-ticker orchestration for the quant-with-docs pipeline.

This flow maps the existing `run_pipeline_task` (from pipelines.prefect_flows) across
a list of tickers and then logs each result to MLflow using `mlflow_log_results`.

Notes:
- Prefect controls concurrency via agents/workers; mapping will schedule each ticker run
  as an independent task which can execute in parallel depending on your Prefect setup.
- If you run locally without a Prefect agent, Prefect will execute tasks sequentially in the local process.

Usage (local run):
    python -m pipelines.prefect_multi_ticker --tickers AAPL,MSFT,GOOG --period 1y

Usage (Prefect deployment):
    prefect deployment build pipelines.prefect_multi_ticker:multi_ticker_flow -n "multi-ticker-quant"
    prefect deployment apply multi-ticker-quant-deployment.yaml
    prefect deployment run "multi-ticker-quant"

"""
from typing import List, Optional
import argparse
from prefect import flow, get_run_logger

# import the tasks/flow we already created in pipelines/prefect_flows.py
from pipelines.prefect_flows import run_pipeline_task, mlflow_log_results


@flow(name="multi-ticker-quant-flow")
def multi_ticker_flow(
    tickers: List[str],
    period: str = "1y",
    collection_name: str = "research_docs",
    model_save_root: Optional[str] = None,
    report_dir: str = "reports",
    use_timeaware_doc_features: bool = True,
):
    """Run the quant-with-docs pipeline in parallel for a list of tickers.

    Args:
        tickers: list of ticker symbols (e.g. ["AAPL","MSFT"]).
        period: yfinance period string (e.g. "1y").
        collection_name: name of Chroma collection to use for doc features.
        model_save_root: optional root path to save models; if provided each model will be saved to
                         <model_save_root>/<ticker>_xgb.joblib
        report_dir: directory root for reports (reports/<TICKER>/...)
        use_timeaware_doc_features: whether to attempt to build time-aware doc features.

    Returns:
        list of mlflow run ids (one per ticker) and summary of artifacts.
    """
    logger = get_run_logger()
    logger.info("Starting multi-ticker flow for %s tickers", len(tickers))

    # prepare per-ticker args and map run_pipeline_task across tickers
    model_paths = []
    for t in tickers:
        if model_save_root:
            model_paths.append(f"{model_save_root}/{t.lower()}_xgb_with_docs.joblib")
        else:
            model_paths.append(None)

    # Map the pipeline task: returns list of Results (dictionaries)
    results = run_pipeline_task.map(
        ticker=tickers,
        period=[period] * len(tickers),
        collection_name=[collection_name] * len(tickers),
        model_save_path=model_paths,
        report_dir=[report_dir] * len(tickers),
        use_timeaware_doc_features=[use_timeaware_doc_features] * len(tickers),
    )

    # Now map mlflow logging across the results
    mlflow_run_ids = mlflow_log_results.map(results, experiment_name=["ai_invest_research"] * len(tickers))

    # Optionally, gather a summary for the caller
    summary = []
    for t, res, run_id in zip(tickers, results, mlflow_run_ids):
        # res and run_id are Prefect results; call .result() to obtain the concrete values if running locally
        try:
            r = res.result()
        except Exception:
            # If the task failed, capture the failure info
            r = {"ticker": t, "error": "task_failed"}
        try:
            rid = run_id.result()
        except Exception:
            rid = None
        summary.append({"ticker": t, "result": r, "mlflow_run_id": rid})

    logger.info("Multi-ticker flow complete")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-ticker Prefect orchestration locally")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g. AAPL,MSFT,GOOG")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--collection-name", type=str, default="research_docs")
    parser.add_argument("--model-save-root", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--no-timeaware", dest="timeaware", action="store_false", help="Disable time-aware doc features")

    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    out = multi_ticker_flow(
        tickers=tickers,
        period=args.period,
        collection_name=args.collection_name,
        model_save_root=args.model_save_root,
        report_dir=args.report_dir,
        use_timeaware_doc_features=args.timeaware,
    )
    print(out)