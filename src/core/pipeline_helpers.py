# -*- coding: utf-8 -*-
"""
Pure helper functions extracted from StockAnalysisPipeline.

These functions are stateless transformations that don't require pipeline
instance state. Moving them here:
1. Shrinks pipeline.py (focuses on orchestration)
2. Makes the logic independently testable
3. Avoids accidental coupling to pipeline instance attributes

All functions here MUST be pure (no side effects, no self/cls dependency).
Pipeline methods delegate to these via thin wrappers for backward compatibility.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional

import pandas as pd

from data_provider.base import normalize_stock_code
from src.report_language import (
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.services.daily_market_context import DailyMarketContext
from src.core.trading_calendar import get_effective_trading_date, get_market_for_stock


# ---------------------------------------------------------------------------
# Date / market helpers
# ---------------------------------------------------------------------------
def coerce_daily_market_context_date(value: Any) -> Optional[date]:
    """Coerce various date-like values to a plain ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def resolve_resume_target_date(
    code: str, current_time: Optional[datetime] = None
) -> date:
    """Resolve the trading date used by checkpoint/resume checks."""
    market = get_market_for_stock(normalize_stock_code(code))
    return get_effective_trading_date(market, current_time=current_time)


# ---------------------------------------------------------------------------
# Volume / MA description helpers
# ---------------------------------------------------------------------------
def describe_volume_ratio(volume_ratio: float) -> str:
    """Return a human-readable description of the volume ratio."""
    if volume_ratio < 0.5:
        return "极度萎缩"
    elif volume_ratio < 0.8:
        return "明显萎缩"
    elif volume_ratio < 1.2:
        return "正常"
    elif volume_ratio < 2.0:
        return "温和放量"
    elif volume_ratio < 3.0:
        return "明显放量"
    else:
        return "巨量"


def compute_ma_status(close: float, ma5: float, ma10: float, ma20: float) -> str:
    """Compute MA alignment status from price and MA values."""
    close = close or 0
    ma5 = ma5 or 0
    ma10 = ma10 or 0
    ma20 = ma20 or 0
    if close > ma5 > ma10 > ma20 > 0:
        return "多头排列 📈"
    elif close < ma5 < ma10 < ma20 and ma20 > 0:
        return "空头排列 📉"
    elif close > ma5 and ma5 > ma10:
        return "短期向好 🔼"
    elif close < ma5 and ma5 < ma10:
        return "短期走弱 🔽"
    else:
        return "震荡整理 ↔️"


# ---------------------------------------------------------------------------
# Daily market context attachment
# ---------------------------------------------------------------------------
def attach_daily_market_context(
    target_context: Dict[str, Any],
    daily_market_context: Optional[DailyMarketContext],
    *,
    report_language: str,
) -> None:
    """Embed daily market context into the analysis context dict (in-place).

    Uses ``to_safe_dict()`` to serialise the context before attaching and
    formatting the prompt section, ensuring only safe data enters the snapshot.
    """
    if daily_market_context is None:
        return

    from src.services.daily_market_context import format_daily_market_context_prompt_section

    safe_context = daily_market_context.to_safe_dict()
    prompt_section = format_daily_market_context_prompt_section(
        safe_context,
        report_language=report_language,
    )
    if not prompt_section:
        return
    target_context["daily_market_context"] = safe_context
    target_context["daily_market_context_summary"] = prompt_section


# ---------------------------------------------------------------------------
# Real-time OHLC augmentation
# ---------------------------------------------------------------------------
def augment_historical_with_realtime(
    df: pd.DataFrame,
    realtime_quote: Any,
    code: str,
    *,
    enable_realtime_technical_indicators: bool = True,
) -> pd.DataFrame:
    """Augment historical OHLCV with intraday real-time quote for MA calculation.

    Creates or updates the last row with the latest price/volume data,
    enabling intraday technical indicator calculation.
    """
    from src.core.trading_calendar import get_market_now, is_market_open

    if df is None or df.empty or 'close' not in df.columns:
        return df
    if realtime_quote is None:
        return df
    price = getattr(realtime_quote, 'price', None)
    if price is None or not (isinstance(price, (int, float)) and price > 0):
        return df

    if not enable_realtime_technical_indicators:
        return df
    market = get_market_for_stock(code)
    market_today = get_market_now(market).date()
    if market and not is_market_open(market, market_today):
        return df

    last_val = df['date'].max()
    last_date = (
        last_val.date() if hasattr(last_val, 'date') else
        (last_val if isinstance(last_val, date) else pd.Timestamp(last_val).date())
    )
    yesterday_close = float(df.iloc[-1]['close']) if len(df) > 0 else price
    open_p = getattr(realtime_quote, 'open_price', None) or getattr(
        realtime_quote, 'pre_close', None
    ) or yesterday_close
    high_p = getattr(realtime_quote, 'high', None) or price
    low_p = getattr(realtime_quote, 'low', None) or price
    vol = getattr(realtime_quote, 'volume', None) or 0
    amt = getattr(realtime_quote, 'amount', None)
    pct = getattr(realtime_quote, 'change_pct', None)

    if last_date >= market_today:
        # Update the last row in a copy.
        df = df.copy()
        idx = df.index[-1]
        df.loc[idx, 'close'] = price
        if open_p is not None:
            df.loc[idx, 'open'] = open_p
        if high_p is not None:
            df.loc[idx, 'high'] = high_p
        if low_p is not None:
            df.loc[idx, 'low'] = low_p
        if vol:
            df.loc[idx, 'volume'] = vol
        if amt is not None:
            df.loc[idx, 'amount'] = amt
        if pct is not None:
            df.loc[idx, 'pct_chg'] = pct
    else:
        # Append a virtual intraday bar.
        new_row = {
            'code': code,
            'date': market_today,
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': price,
            'volume': vol,
            'amount': amt if amt is not None else 0,
            'pct_chg': pct if pct is not None else 0,
        }
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Trend analysis fallback helpers
# ---------------------------------------------------------------------------
def trend_score_fallback(trend_result: Any) -> Optional[int]:
    """Extract signal score from trend result, or None."""
    if trend_result is None:
        return None
    try:
        score = int(getattr(trend_result, "signal_score", 0))
    except (TypeError, ValueError):
        return None
    return score if score > 0 else None


def trend_label_fallback(trend_result: Any, report_language: str = "zh") -> str:
    """Produce a trend label from trend result."""
    if trend_result is None:
        return ""
    trend_status = getattr(trend_result, "trend_status", None)
    value = getattr(trend_status, "value", None) or str(trend_status or "").strip()
    if report_language != "en":
        return value
    return localize_trend_prediction(value, report_language)


def trend_signal_fallback(trend_result: Any, report_language: str = "zh") -> str:
    """Produce an operation signal label from trend result."""
    if trend_result is None:
        return ""
    buy_signal = getattr(trend_result, "buy_signal", None)
    value = getattr(buy_signal, "value", None) or str(buy_signal or "").strip()
    return localize_operation_advice(value, report_language)


def trend_decision_fallback(trend_result: Any) -> Optional[str]:
    """Map trend buy_signal to a simple decision category."""
    if trend_result is None:
        return None
    signal_name = getattr(getattr(trend_result, "buy_signal", None), "name", "").lower()
    return {
        "strong_buy": "buy",
        "buy": "buy",
        "hold": "hold",
        "wait": "hold",
        "sell": "sell",
        "strong_sell": "sell",
    }.get(signal_name)


def mark_trend_fallback_source(result: Any) -> None:
    """Tag result.data_sources with 'trend:fallback' (idempotent, in-place)."""
    if "trend:fallback" in (result.data_sources or ""):
        return
    result.data_sources = (
        f"{result.data_sources},trend:fallback"
        if result.data_sources
        else "trend:fallback"
    )


def summary_fallback_from_result(result: Any, report_language: str) -> str:
    """Build a one-sentence summary from result fields."""
    trend = (result.trend_prediction or "").strip()
    advice = (result.operation_advice or "").strip()
    if trend and advice:
        if report_language == "en":
            return f"Trend view: {trend}; action advice: {advice}."
        if report_language == "ko":
            return f"추세 결론: {trend}; 대응 전략: {advice}."
        return f"趋势结论：{trend}；操作建议：{advice}。"
    return ""


# ---------------------------------------------------------------------------
# Agent dashboard helpers
# ---------------------------------------------------------------------------
def is_agent_placeholder_text(text: str) -> bool:
    """Check if agent output is a placeholder/missing value."""
    if not text:
        return True
    return text.lower() in {"n/a", "na", "none", "null", "unknown", "tbd"} or text in {
        "未知", "待补充", "数据缺失", "无",
    }


def _extract_advice_text_from_dict(raw_advice: dict) -> str:
    """Extract meaningful advice text from a structured advice dict (internal)."""
    for field in ("has_position", "no_position"):
        if isinstance(raw_advice.get(field), str):
            text = raw_advice[field].strip()
            if not is_agent_placeholder_text(text):
                return text
    for value in raw_advice.values():
        if isinstance(value, str):
            text = value.strip()
            if not is_agent_placeholder_text(text):
                return text
    return ""


def is_agent_field_missing(
    value: Any,
    *,
    scalar: bool = False,
    allow_dict: bool = False,
    expect_text: bool = False,
) -> bool:
    """Check if an agent-produced field is missing/invalid."""
    if scalar and isinstance(value, dict):
        if not allow_dict or not value:
            return True
        return not _extract_advice_text_from_dict(value)
    if value is None:
        return True
    if expect_text and scalar:
        if not isinstance(value, str):
            return True
    if isinstance(value, str):
        text = value.strip()
        return is_agent_placeholder_text(text)
    if isinstance(value, dict):
        if scalar:
            return not allow_dict
        return not value
    if scalar and isinstance(value, (list, tuple, set)):
        return True
    return False


def extract_advice_text_from_dict(raw_advice: dict) -> str:
    """Public alias for extracting advice text from a structured dict."""
    return _extract_advice_text_from_dict(raw_advice)


def agent_dashboard_value(
    dash: Dict[str, Any],
    nested_dashboard: Any,
    key: str,
    *,
    scalar: bool = False,
    allow_dict: bool = False,
    expect_text: bool = False,
) -> Any:
    """Read a scalar from top-level agent payload, then nested dashboard fallback."""
    value = dash.get(key) if isinstance(dash, dict) else None
    if isinstance(nested_dashboard, dict) and is_agent_field_missing(
        value, scalar=scalar, allow_dict=allow_dict, expect_text=expect_text,
    ):
        nested_value = nested_dashboard.get(key)
        if not is_agent_field_missing(
            nested_value, scalar=scalar, allow_dict=allow_dict, expect_text=expect_text,
        ):
            value = nested_value
    return value


# ---------------------------------------------------------------------------
# Context snapshot helpers
# ---------------------------------------------------------------------------
def without_runtime_prompt_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy without runtime-only prompt context.

    Market phase and AnalysisContextPack summaries are prompt inputs only.
    P4 stores only the separately rendered public overview at snapshot top level.
    """
    sanitized = dict(context)
    sanitized.pop("market_phase_context", None)
    sanitized.pop("portfolio_context", None)
    sanitized.pop("analysis_context_pack", None)
    sanitized.pop("analysis_context_pack_summary", None)
    sanitized.pop("daily_market_context_summary", None)
    enhanced_context = sanitized.get("enhanced_context")
    if isinstance(enhanced_context, dict):
        enhanced_context = dict(enhanced_context)
        enhanced_context.pop("daily_market_context_summary", None)
        sanitized["enhanced_context"] = enhanced_context
    return sanitized


def safe_to_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Safely convert an object to a dict via to_dict() or __dict__."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return None
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return None
    return None
