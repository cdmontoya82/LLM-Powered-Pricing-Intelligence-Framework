"""
LLM Pricing Agent
-----------------
Takes the best-model forecast output and generates structured
pricing recommendations via Claude (Anthropic) or GPT-4o (OpenAI).

Output schema:
{
    "action":     "increase" | "decrease" | "hold",
    "magnitude":  float (% change suggested),
    "rationale":  str  (business reasoning),
    "risk_level": "low" | "medium" | "high",
    "confidence": float (0–1)
}
"""

import json
import os
import numpy as np
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior pricing analyst assistant specialized in data-driven revenue optimization.
You will receive a JSON payload containing demand forecast results and business context.
Your task is to analyze the data and return a structured pricing recommendation.

RULES:
- Base your recommendation ONLY on the data provided. Do not invent market data.
- Be conservative: only suggest price changes above 3% magnitude.
- Always explain your reasoning in plain business language (1–2 sentences max).
- Return ONLY valid JSON matching the output schema. No markdown, no preamble.

OUTPUT SCHEMA:
{
    "action":     "increase" | "decrease" | "hold",
    "magnitude":  <float, % change, e.g. 5.0 for +5%>,
    "rationale":  "<1-2 sentence business reasoning>",
    "risk_level": "low" | "medium" | "high",
    "confidence": <float 0.0 to 1.0>
}"""


def build_payload(
    forecast_result: Dict,
    current_price: Optional[float] = None,
    margin_floor_pct: float = 0.20,
    competitor_flag: bool = False,
) -> Dict:
    """
    Build structured JSON payload for the LLM prompt.

    Args:
        forecast_result : Output dict from benchmark.run_benchmark()
        current_price   : Current unit price (optional context)
        margin_floor_pct: Minimum acceptable margin (e.g. 0.20 = 20%)
        competitor_flag : Whether competitor pricing pressure is detected
    """
    yhat   = forecast_result["yhat"]
    y_true = forecast_result["y_true"]
    lower  = forecast_result["lower"]
    upper  = forecast_result["upper"]

    avg_forecast     = float(np.mean(yhat))
    avg_actual       = float(np.mean(y_true))
    deviation_pct    = ((avg_forecast - avg_actual) / avg_actual) * 100
    trend            = "upward" if yhat[-1] > yhat[0] else "downward"
    volatility       = float(np.std(yhat) / avg_forecast * 100)
    interval_width   = float(np.mean(upper - lower))

    payload = {
        "model_used":           forecast_result["model_name"],
        "forecast_horizon_days": len(yhat),
        "avg_demand_actual":    round(avg_actual, 2),
        "avg_demand_forecast":  round(avg_forecast, 2),
        "demand_deviation_pct": round(deviation_pct, 2),
        "demand_trend":         trend,
        "forecast_volatility_pct": round(volatility, 2),
        "prediction_interval_width": round(interval_width, 2),
        "business_context": {
            "current_price":    current_price,
            "margin_floor_pct": margin_floor_pct,
            "competitor_pressure": competitor_flag,
        },
    }
    return payload


def call_anthropic(payload: Dict) -> Dict:
    """Call Claude via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_message = f"Analyze this forecast and provide a pricing recommendation:\n\n{json.dumps(payload, indent=2)}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def call_openai(payload: Dict) -> Dict:
    """Call GPT-4o via OpenAI API (fallback)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_message = f"Analyze this forecast and provide a pricing recommendation:\n\n{json.dumps(payload, indent=2)}"

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=512,
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


def get_recommendation(
    forecast_result: Dict,
    current_price: Optional[float] = None,
    margin_floor_pct: float = 0.20,
    competitor_flag: bool = False,
    provider: str = "anthropic",
) -> Dict:
    """
    Main entry point. Builds payload, calls LLM, returns structured recommendation.

    Args:
        provider: "anthropic" | "openai"
    """
    payload = build_payload(forecast_result, current_price, margin_floor_pct, competitor_flag)
    logger.info(f"Calling {provider} API...")

    try:
        if provider == "anthropic":
            result = call_anthropic(payload)
        elif provider == "openai":
            result = call_openai(payload)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"Recommendation: {result['action'].upper()} {result.get('magnitude', 0):.1f}% | Risk: {result['risk_level']}")
        result["_payload"] = payload  # attach for transparency
        return result

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "action":     "hold",
            "magnitude":  0.0,
            "rationale":  f"LLM unavailable ({e}). Manual review recommended.",
            "risk_level": "high",
            "confidence": 0.0,
            "_payload":   payload,
        }


def format_recommendation(rec: Dict) -> str:
    """Human-readable string for display in dashboard."""
    action_emoji = {"increase": "📈", "decrease": "📉", "hold": "⏸️"}.get(rec["action"], "")
    risk_color   = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(rec["risk_level"], "⚪")

    return f"""
## {action_emoji} Pricing Recommendation

**Action:** {rec['action'].capitalize()} by **{rec.get('magnitude', 0):.1f}%**

**Rationale:** {rec['rationale']}

**Risk Level:** {risk_color} {rec['risk_level'].capitalize()}
**Confidence:** {rec.get('confidence', 0) * 100:.0f}%
    """.strip()
