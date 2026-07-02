"""
Gradio Dashboard — LLM Pricing Intelligence Framework
------------------------------------------------------
Entry point for Hugging Face Spaces deployment.
Upload a CSV with 'ds' and 'y' columns to get:
  1. Interactive forecast chart (all 3 models)
  2. Benchmark metrics table
  3. LLM-powered pricing recommendation
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, tempfile, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.pipeline import run_pipeline
from src.forecasting.benchmark import run_benchmark
from src.agent.pricing_agent import get_recommendation, format_recommendation

DEMO_CSV = """ds,y
2023-01-01,1200
2023-01-02,1350
2023-01-03,980
2023-01-04,1100
2023-01-05,1420
2023-01-06,1600
2023-01-07,1550
2023-01-08,1200
2023-01-09,1300
2023-01-10,950
2023-01-11,1050
2023-01-12,1400
2023-01-13,1700
2023-01-14,1650
2023-01-15,1250
2023-01-16,1380
2023-01-17,1020
2023-01-18,1150
2023-01-19,1480
2023-01-20,1750
2023-01-21,1700
2023-01-22,1320
2023-01-23,1450
2023-01-24,1080
2023-01-25,1200
2023-01-26,1520
2023-01-27,1800
2023-01-28,1750
2023-01-29,1400
2023-01-30,1500
"""


# ─── Core processing ──────────────────────────────────────────────────────────

def process_upload(csv_file, test_size, current_price, margin_floor, competitor_flag, llm_provider, api_key):

    if csv_file is None:
        # Use built-in demo data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(DEMO_CSV)
            tmp_path = f.name
    else:
        tmp_path = csv_file.name

    # Set API key from UI input
    if api_key:
        if llm_provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key

    # ── 1. Ingestion ──
    df = run_pipeline(tmp_path, output_path=tempfile.mktemp(suffix=".parquet"))

    if len(df) < test_size + 10:
        return None, "❌ Not enough data. Need at least test_size + 10 rows.", "", ""

    # ── 2. Benchmark (Prophet only in demo if no extras installed) ──
    available_models = ["prophet"]
    try:
        import neuralprophet
        available_models.append("neuralprophet")
    except ImportError:
        pass
    try:
        import chronos
        available_models.append("chronos")
    except ImportError:
        pass

    metrics_df, best = run_benchmark(df, test_size=test_size, models=available_models)

    # ── 3. Build forecast chart ──
    fig = build_chart(df, best, test_size)

    # ── 4. Metrics table ──
    metrics_md = metrics_df.to_markdown(index=False)

    # ── 5. LLM recommendation ──
    rec = get_recommendation(
        forecast_result=best,
        current_price=float(current_price) if current_price else None,
        margin_floor_pct=float(margin_floor) / 100,
        competitor_flag=competitor_flag,
        provider=llm_provider,
    )
    rec_text = format_recommendation(rec)
    payload_json = json.dumps(rec.get("_payload", {}), indent=2)

    return fig, metrics_md, rec_text, payload_json


def build_chart(df: pd.DataFrame, best: dict, test_size: int) -> go.Figure:
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]

    fig = go.Figure()

    # Actuals
    fig.add_trace(go.Scatter(
        x=train["ds"], y=train["y"],
        mode="lines", name="Historical (train)",
        line=dict(color="#1f3864", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=test["ds"], y=test["y"],
        mode="lines", name="Actual (test)",
        line=dict(color="#2e86ab", width=2, dash="dot"),
    ))

    # Best model forecast
    fig.add_trace(go.Scatter(
        x=best["dates"], y=best["yhat"],
        mode="lines", name=f"Forecast ({best['model_name']})",
        line=dict(color="#e84855", width=2),
    ))

    # Confidence interval
    dates_rev = list(best["dates"]) + list(best["dates"])[::-1]
    bounds     = list(best["upper"]) + list(best["lower"])[::-1]
    fig.add_trace(go.Scatter(
        x=dates_rev, y=bounds,
        fill="toself",
        fillcolor="rgba(232, 72, 85, 0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Prediction Interval",
    ))

    fig.update_layout(
        title=f"Demand Forecast — Best Model: {best['model_name'].capitalize()}",
        xaxis_title="Date",
        yaxis_title="Demand / Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
    )
    return fig


# ─── Gradio UI ────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🔬 LLM-Powered Pricing Intelligence Framework
**Benchmarks Prophet · NeuralProphet · Amazon Chronos → LLM Pricing Recommendations**

Upload a CSV with two columns: `ds` (date, YYYY-MM-DD) and `y` (numeric demand/sales).  
If no file is uploaded, a built-in demo dataset is used.

> **GitHub:** [cdmontoya82/LLM-Powered-Pricing-Intelligence-Framework](https://github.com/cdmontoya82/LLM-Powered-Pricing-Intelligence-Framework)
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Pricing Intelligence") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            csv_input      = gr.File(label="Upload CSV (ds, y)", file_types=[".csv"])
            test_size      = gr.Slider(7, 90, value=30, step=1, label="Test horizon (days)")
            current_price  = gr.Number(label="Current unit price (optional)", precision=2)
            margin_floor   = gr.Slider(5, 50, value=20, step=1, label="Margin floor (%)")
            competitor_flag= gr.Checkbox(label="Competitor pricing pressure detected")
            llm_provider   = gr.Radio(["anthropic", "openai"], value="anthropic", label="LLM Provider")
            api_key_input  = gr.Textbox(label="API Key", type="password", placeholder="sk-... or sk-ant-...")
            run_btn        = gr.Button("🚀 Run Analysis", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Forecast Chart")
            forecast_chart = gr.Plot()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Model Benchmark")
            metrics_output = gr.Markdown()

        with gr.Column():
            gr.Markdown("### 🤖 LLM Pricing Recommendation")
            rec_output = gr.Markdown()

    with gr.Accordion("🔍 LLM Prompt Payload (debug)", open=False):
        payload_output = gr.Code(language="json")

    run_btn.click(
        fn=process_upload,
        inputs=[csv_input, test_size, current_price, margin_floor, competitor_flag, llm_provider, api_key_input],
        outputs=[forecast_chart, metrics_output, rec_output, payload_output],
    )

    gr.Examples(
        examples=[[None, 30, None, 20, False, "anthropic", ""]],
        inputs=[csv_input, test_size, current_price, margin_floor, competitor_flag, llm_provider, api_key_input],
        label="Run with built-in demo data",
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
