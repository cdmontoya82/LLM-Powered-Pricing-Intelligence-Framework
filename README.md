# LLM-Powered Pricing Intelligence Framework

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/cdmontoya82/pricing-intelligence)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)]()

> **An end-to-end applied research framework that integrates probabilistic forecasting models with Large Language Model (LLM) reasoning to generate actionable pricing recommendations from time-series demand data.**

---

## Abstract

Pricing optimization in high-velocity environments (logistics, retail, e-commerce) requires not only accurate demand forecasting but also rapid, interpretable decision support. This framework proposes a modular pipeline that (1) ingests and preprocesses demand time-series data, (2) benchmarks three state-of-the-art forecasting models — Facebook Prophet, NeuralProphet, and Amazon Chronos — using standardized error metrics, and (3) feeds forecast outputs and deviation analyses into an LLM agent (Claude / GPT-4o) that generates natural-language pricing recommendations aligned with business constraints.

Empirical results on public retail datasets demonstrate that hybrid LLM-augmented workflows reduce average pricing decision latency while maintaining forecast accuracy within ±3% MAPE of standalone model baselines.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Live Demo](#live-demo)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)

---

## Motivation

Traditional forecasting pipelines produce numerical outputs that require analyst interpretation before informing pricing decisions. This gap between *model output* and *business action* introduces latency and inconsistency, especially in SMEs without dedicated data teams.

This project addresses three research questions:

1. **RQ1 — Accuracy:** Which forecasting model (Prophet, NeuralProphet, Chronos) achieves the lowest error on short-horizon demand series across multiple product categories?
2. **RQ2 — Interpretability:** Can an LLM agent reliably translate forecast deviations into pricing recommendations without hallucinating business context?
3. **RQ3 — Deployment:** Can the full pipeline be containerized and served as a real-time interactive tool with sub-5s response latency?

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                          │
│   CSV / API (Kaggle / Open Retail) → Parquet (compressed)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 FORECASTING BENCHMARK ENGINE                     │
│                                                                  │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│   │   Prophet    │  │ NeuralProphet │  │   Amazon Chronos    │  │
│   │  (additive   │  │  (neural +    │  │  (zero-shot LLM     │  │
│   │   seasonal)  │  │   seasonal)   │  │   time-series)      │  │
│   └──────┬───────┘  └──────┬────────┘  └──────────┬──────────┘  │
│          └─────────────────┴──────────────────────┘             │
│                    Metrics: MAE · MAPE · RMSE · Coverage        │
└────────────────────────┬────────────────────────────────────────┘
                         │  Best model + deviation report
┌────────────────────────▼────────────────────────────────────────┐
│                    LLM AGENT LAYER                               │
│   Structured prompt → Claude / GPT-4o API                       │
│   Output: Pricing recommendation + confidence + reasoning       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  GRADIO DASHBOARD (HF Spaces)                    │
│   Upload CSV → Select models → View forecast → Get LLM insight  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Methodology

### 1. Data Preprocessing
- IQR-based outlier detection (consistent with production pipeline at Edinsa–Postobón)
- Missing value imputation using forward-fill + seasonal decomposition fallback
- Parquet serialization for efficient downstream loading

### 2. Forecasting Models

| Model | Type | Key Strengths |
|---|---|---|
| **Prophet** | Additive decomposition | Robust to missing data, holidays, strong seasonality |
| **NeuralProphet** | Neural + additive | Auto-regression, multi-step ahead, configurable lags |
| **Amazon Chronos** | Zero-shot LLM for TS | No training required, strong on cold-start products |

### 3. Evaluation Metrics

```
MAE  = mean(|y - ŷ|)
MAPE = mean(|y - ŷ| / y) × 100
RMSE = sqrt(mean((y - ŷ)²))
Coverage = % of actuals within 80% prediction interval
```

### 4. LLM Agent Prompt Design
The agent receives a structured JSON payload containing:
- Forecast horizon and confidence intervals
- Deviation vs. previous period (% change)
- Current pricing context (if provided)
- Business constraints (margin floor, competitor flag)

Output is a structured recommendation: `action`, `magnitude`, `rationale`, `risk_level`.

---

## Results

> Results on **Rossmann Store Sales** dataset (Kaggle, 1,017 stores, 2.5 years).

| Model | MAE | MAPE | RMSE | Interval Coverage |
|---|---|---|---|---|
| Prophet | 512.3 | 14.2% | 731.5 | 81.3% |
| NeuralProphet | **448.7** | **11.8%** | **668.2** | 83.1% |
| Amazon Chronos | 489.1 | 13.4% | 712.3 | **85.7%** |

*NeuralProphet achieves the lowest point-forecast error; Chronos provides the best uncertainty calibration (zero-shot, no fine-tuning).*

**LLM Agent Evaluation (n=50 manual reviews):**
- Factual consistency with forecast data: **96%**
- Actionability score (1–5 Likert, human raters): **4.2 / 5**
- Avg. response latency: **2.8s**

---

## Repository Structure

```
LLM-Powered-Pricing-Intelligence-Framework/
│
├── data/
│   ├── raw/                    # Source datasets (gitignored)
│   └── processed/              # Parquet files after preprocessing
│
├── src/
│   ├── ingestion/
│   │   └── pipeline.py         # Data loading, cleaning, Parquet export
│   ├── forecasting/
│   │   ├── prophet_model.py    # Prophet wrapper + tuning
│   │   ├── neural_prophet.py   # NeuralProphet wrapper
│   │   ├── chronos_model.py    # Chronos zero-shot wrapper
│   │   └── benchmark.py        # Unified evaluation + metrics table
│   ├── agent/
│   │   └── pricing_agent.py    # LLM prompt builder + API call + parser
│   └── dashboard/
│       └── app.py              # Gradio interface (HF Spaces entry point)
│
├── experiments/
│   └── notebooks/
│       ├── 01_EDA.ipynb        # Exploratory Data Analysis
│       ├── 02_Model_Benchmark.ipynb
│       └── 03_LLM_Agent_Eval.ipynb
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_forecasting.py
│   └── test_agent.py
│
├── docs/
│   └── assets/                 # Architecture diagrams, result charts
│
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/cdmontoya82/LLM-Powered-Pricing-Intelligence-Framework
cd LLM-Powered-Pricing-Intelligence-Framework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 4. Run benchmark (uses sample data)
python src/forecasting/benchmark.py --dataset data/raw/sample.csv

# 5. Launch dashboard locally
python src/dashboard/app.py
```

---

## Live Demo

🚀 **[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/cdmontoya82/pricing-intelligence)**

Upload any time-series CSV with `ds` (date) and `y` (value) columns. The system will:
1. Clean and preprocess your data
2. Run all three forecasting models
3. Show a benchmark comparison table
4. Generate an LLM-powered pricing recommendation

---

## Limitations & Future Work

**Current limitations:**
- Chronos inference requires ~4GB RAM; not suitable for free-tier HF Spaces CPU (workaround: use distil-chronos)
- LLM agent does not yet incorporate competitor pricing signals
- Evaluation dataset limited to retail (grocery); generalization to logistics/fintech untested

**Planned extensions:**
- [ ] Fine-tune Chronos on logistics distance data (from `Logistics-Distance-Forecasting-Prophet`)
- [ ] Add reinforcement learning layer for multi-period pricing optimization
- [ ] Integrate real-time data via public APIs (e.g., commodity prices, fuel index)
- [ ] Multi-language LLM output (ES / EN)

---

## Citation

If you use this framework in your research or projects, please cite:

```bibtex
@misc{montoya2025pricingllm,
  author       = {Montoya Henao, Cristian David},
  title        = {LLM-Powered Pricing Intelligence Framework},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/cdmontoya82/LLM-Powered-Pricing-Intelligence-Framework}
}
```

---

## Author

**Cristian David Montoya Henao**
Senior Data Scientist | Quantitative Pricing & Forecasting Specialist
[LinkedIn](https://linkedin.com/in/cristian-david-montoya-420741107) · [GitHub](https://github.com/cdmontoya82)
