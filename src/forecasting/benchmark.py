"""
Forecasting Benchmark Engine
-----------------------------
Trains and evaluates Prophet, NeuralProphet, and Amazon Chronos
on a holdout test set. Returns a unified metrics DataFrame and
the best-model forecast for downstream LLM agent consumption.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of actuals falling within the prediction interval."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)) * 100)


# ─── Prophet ──────────────────────────────────────────────────────────────────

def run_prophet(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )
    model.fit(train[["ds", "y"]])

    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future).tail(len(test))

    yhat = forecast["yhat"].values
    lower = forecast["yhat_lower"].values
    upper = forecast["yhat_upper"].values
    return yhat, lower, upper


# ─── NeuralProphet ────────────────────────────────────────────────────────────

def run_neuralprophet(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")

    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        quantiles=[0.10, 0.90],
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
    )

    df_all = pd.concat([train, test])[["ds", "y"]]
    model.fit(df_all, freq="D", valid_p=len(test) / len(df_all))

    future = model.make_future_dataframe(df_all, periods=len(test))
    forecast = model.predict(future).tail(len(test))

    yhat = forecast["yhat1"].values
    lower = forecast.get("yhat1 10.0%", pd.Series(yhat * 0.9)).values
    upper = forecast.get("yhat1 90.0%", pd.Series(yhat * 1.1)).values
    return yhat, lower, upper


# ─── Amazon Chronos ───────────────────────────────────────────────────────────

def run_chronos(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zero-shot inference with Amazon Chronos (distil variant for CPU efficiency).
    No training required — uses pretrained foundation model.
    """
    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    context = torch.tensor(train["y"].values, dtype=torch.float32).unsqueeze(0)

    forecast = pipeline.predict(
        context=context,
        prediction_length=len(test),
        num_samples=100,
    )

    samples = forecast[0].numpy()  # shape: (num_samples, horizon)
    yhat = samples.mean(axis=0)
    lower = np.percentile(samples, 10, axis=0)
    upper = np.percentile(samples, 90, axis=0)
    return yhat, lower, upper


# ─── Benchmark Orchestrator ───────────────────────────────────────────────────

def run_benchmark(
    df: pd.DataFrame,
    test_size: int = 30,
    models: list = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Split data, run all models, compute metrics, return results table + best forecast.

    Returns:
        metrics_df : DataFrame with MAE, MAPE, RMSE, Coverage per model
        best_result: Dict with best model name, forecast array, and metadata
    """
    if models is None:
        models = ["prophet", "neuralprophet", "chronos"]

    train = df.iloc[:-test_size].copy()
    test  = df.iloc[-test_size:].copy()
    y_true = test["y"].values

    logger.info(f"Train size: {len(train)} | Test size: {len(test)}")

    runners = {
        "prophet":       run_prophet,
        "neuralprophet": run_neuralprophet,
        "chronos":       run_chronos,
    }

    results = []
    forecasts = {}

    for name in models:
        if name not in runners:
            logger.warning(f"Unknown model '{name}', skipping.")
            continue
        logger.info(f"Running {name}...")
        try:
            yhat, lower, upper = runners[name](train, test)
            metrics = {
                "Model":    name.capitalize(),
                "MAE":      round(mae(y_true, yhat), 2),
                "MAPE (%)": round(mape(y_true, yhat), 2),
                "RMSE":     round(rmse(y_true, yhat), 2),
                "Coverage": round(interval_coverage(y_true, lower, upper), 1),
            }
            results.append(metrics)
            forecasts[name] = {
                "yhat": yhat, "lower": lower, "upper": upper,
                "dates": test["ds"].values,
                "y_true": y_true,
            }
            logger.info(f"  {name}: MAE={metrics['MAE']} | MAPE={metrics['MAPE (%)']}% | RMSE={metrics['RMSE']}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    metrics_df = pd.DataFrame(results).sort_values("MAPE (%)")

    # Pick best model by lowest MAPE
    best_name_raw = metrics_df.iloc[0]["Model"].lower()
    best_name = next((k for k in forecasts if k.startswith(best_name_raw[:6])), list(forecasts.keys())[0])
    best_result = {"model_name": best_name, **forecasts[best_name], "metrics": metrics_df}

    logger.info(f"Best model: {best_name} (MAPE {metrics_df.iloc[0]['MAPE (%)']}%)")
    return metrics_df, best_result


if __name__ == "__main__":
    import argparse
    from src.ingestion.pipeline import run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/raw/sample.csv")
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=["prophet", "neuralprophet", "chronos"])
    args = parser.parse_args()

    df = run_pipeline(args.dataset)
    metrics_df, best = run_benchmark(df, test_size=args.test_size, models=args.models)

    print("\n── Benchmark Results ──")
    print(metrics_df.to_string(index=False))
    print(f"\n✓ Best model: {best['model_name']}")
