# pipelines/prefect_flows.py
"""
Prefect flow wrapper for the quant pipeline (with document features).

- Calls run_pipeline_with_docs (from pipelines.quant_run_with_docs).
- Starts an MLflow run and logs:
    - numeric metrics (train mse, directional accuracy, backtest stats)
    - artifacts: report CSV, cumulative returns PNG, saved model
- Can be executed manually, or registered with Prefect for scheduling.

Usage (manual):
    python -m pipelines.prefect_flows --ticker AAPL --period 1y

Usage (run via Prefect):
    prefect deployment build pipelines.prefect_flows:main_flow -n "quant-with-docs" --cron "0 2 * * *"
    prefect deployment apply quant-with-docs-deployment.yaml
    prefect deployment run "quant-with-docs"

Requirements:
    pip install prefect==2.16.0 mlflow
    (adjust Prefect version if you use a newer Prefect)
"""
from prefect import flow, task, get_run_logger
import mlflow
from typing import Dict, Any, Optional
import argparse
import os

# import the existing pipeline runner
from pipelines.quant_run_with_docs import run_pipeline_with_docs

# Prefect task wrapping the pipeline call
@task(retries=1, retry_delay_seconds=30)
def run_pipeline_task(
    ticker: str,
    period: str,
    collection_name: str,
    model_save_path: Optional[str],
    report_dir: str,
    use_timeaware_doc_features: bool,
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting run_pipeline_with_docs for %s", ticker)
    result = run_pipeline_with_docs(
        ticker=ticker,
        period=period,
        collection_name=collection_name,
        model_save_path=model_save_path,
        report_dir=report_dir,
        use_timeaware_doc_features=use_timeaware_doc_features,
    )
    logger.info("Pipeline finished for %s", ticker)
    return result

@task
def mlflow_log_results(result: Dict[str, Any], experiment_name: str = "ai_invest_research"):
    """
    Start an MLflow run and log metrics/artifacts from the pipeline result.
    Expects result to contain model_path, report_csv, plot_png, and metrics dict.
    """
    logger = get_run_logger()
    mlflow.set_experiment(experiment_name)
    metrics = result.get("metrics", {})
    model_path = result.get("model_path")
    report_csv = result.get("report_csv")
    plot_png = result.get("plot_png")
    ticker = result.get("ticker", "unknown")

    # Start MLflow run
    with mlflow.start_run(run_name=f"quant_with_docs_{ticker}") as run:
        logger.info("MLflow run started: %s", run.info.run_id)

        # Log basic params
        mlflow.log_param("ticker", ticker)

        # Log metrics (only numeric ones)
        for k, v in metrics.items():
            try:
                val = float(v)
                mlflow.log_metric(k, val)
            except Exception:
                # skip non-numeric or nested entries
                logger.debug("Skipping non-numeric metric %s => %s", k, v)

        # Log artifacts if they exist
        if report_csv and os.path.exists(report_csv):
            mlflow.log_artifact(report_csv, artifact_path="reports")
            logger.info("Logged report CSV: %s", report_csv)
        else:
            logger.warning("Report CSV missing or not found: %s", report_csv)

        if plot_png and os.path.exists(plot_png):
            mlflow.log_artifact(plot_png, artifact_path="plots")
            logger.info("Logged plot PNG: %s", plot_png)
        else:
            logger.warning("Plot PNG missing or not found: %s", plot_png)

        if model_path and os.path.exists(model_path):
            # record the model file as an artifact
            mlflow.log_artifact(model_path, artifact_path="models")
            logger.info("Logged model artifact: %s", model_path)
            # Optionally, register the model (uncomment if you run a tracking server with registry)
            # mlflow.register_model(f"runs:/{run.info.run_id}/{os.path.basename(model_path)}", f"{ticker}_model_registry")
        else:
            logger.warning("Model path missing or not found: %s", model_path)

        logger.info("MLflow run %s complete", run.info.run_id)
        return run.info.run_id

@flow(name="quant-with-docs-flow")
def main_flow(
    ticker: str = "AAPL",
    period: str = "1y",
    collection_name: str = "research_docs",
    model_save_path: Optional[str] = None,
    report_dir: str = "reports",
    use_timeaware_doc_features: bool = True,
    mlflow_experiment: str = "ai_invest_research",
):
    """
    Orchestrates the full quant pipeline for a single ticker and logs outputs to MLflow.
    """
    logger = get_run_logger()
    logger.info("Starting orchestration for %s", ticker)

    result = run_pipeline_task.submit(
        ticker=ticker,
        period=period,
        collection_name=collection_name,
        model_save_path=model_save_path,
        report_dir=report_dir,
        use_timeaware_doc_features=use_timeaware_doc_features,
    ).result()

    run_id = mlflow_log_results.submit(result, experiment_name=mlflow_experiment).result()
    logger.info("Completed orchestration for %s; mlflow_run_id=%s", ticker, run_id)
    return {"ticker": ticker, "mlflow_run_id": run_id, "artifacts": {"model_path": result.get("model_path"), "report_csv": result.get("report_csv"), "plot_png": result.get("plot_png")}}

# Simple CLI to run the flow locally without installing Prefect UI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prefect orchestration for quant pipeline (with docs).")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--collection-name", type=str, default="research_docs")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--no-timeaware", dest="timeaware", action="store_false", help="Disable time-aware doc features")
    parser.add_argument("--mlflow-experiment", type=str, default="ai_invest_research")
    args = parser.parse_args()

    main_flow(
        ticker=args.ticker,
        period=args.period,
        collection_name=args.collection_name,
        model_save_path=args.model_path,
        report_dir=args.report_dir,
        use_timeaware_doc_features=args.timeaware,
        mlflow_experiment=args.mlflow_experiment,
    )
