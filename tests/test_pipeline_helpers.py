# -*- coding: utf-8 -*-
"""Tests for the pure helper functions extracted from StockAnalysisPipeline."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.core.pipeline_helpers import (
    agent_dashboard_value,
    attach_daily_market_context,
    augment_historical_with_realtime,
    coerce_daily_market_context_date,
    compute_ma_status,
    describe_volume_ratio,
    extract_advice_text_from_dict,
    is_agent_field_missing,
    is_agent_placeholder_text,
    mark_trend_fallback_source,
    safe_to_dict,
    summary_fallback_from_result,
    trend_decision_fallback,
    trend_label_fallback,
    trend_score_fallback,
    trend_signal_fallback,
    without_runtime_prompt_context,
)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
class TestCoerceDailyMarketContextDate:
    def test_datetime(self):
        assert coerce_daily_market_context_date(datetime(2026, 7, 2, 15, 0)) == date(2026, 7, 2)

    def test_date_passthrough(self):
        d = date(2026, 7, 2)
        assert coerce_daily_market_context_date(d) == d

    def test_iso_string(self):
        assert coerce_daily_market_context_date("2026-07-02T15:00:00") == date(2026, 7, 2)

    def test_invalid_string(self):
        assert coerce_daily_market_context_date("not-a-date") is None

    def test_none(self):
        assert coerce_daily_market_context_date(None) is None


# ---------------------------------------------------------------------------
# Volume / MA helpers
# ---------------------------------------------------------------------------
class TestDescribeVolumeRatio:
    @pytest.mark.parametrize(
        "ratio,expected",
        [
            (0.3, "极度萎缩"),
            (0.6, "明显萎缩"),
            (1.0, "正常"),
            (1.5, "温和放量"),
            (2.5, "明显放量"),
            (5.0, "巨量"),
        ],
    )
    def test_buckets(self, ratio, expected):
        assert describe_volume_ratio(ratio) == expected


class TestComputeMaStatus:
    def test_bullish(self):
        assert compute_ma_status(100, 90, 80, 70) == "多头排列 📈"

    def test_bearish(self):
        assert compute_ma_status(70, 80, 90, 100) == "空头排列 📉"

    def test_short_bullish(self):
        assert compute_ma_status(100, 90, 80, 95) == "短期向好 🔼"

    def test_short_bearish(self):
        assert compute_ma_status(80, 90, 100, 95) == "短期走弱 🔽"

    def test_mixed(self):
        assert compute_ma_status(100, 80, 90, 70) == "震荡整理 ↔️"


# ---------------------------------------------------------------------------
# Trend fallback helpers
# ---------------------------------------------------------------------------
class TestTrendFallbacks:
    def _mock_trend(self, signal_score=75, trend_status="uptrend", buy_signal_name="buy"):
        m = MagicMock()
        m.signal_score = signal_score
        m.trend_status = MagicMock()
        m.trend_status.value = trend_status
        m.buy_signal = MagicMock()
        m.buy_signal.value = "买入"
        m.buy_signal.name = buy_signal_name
        return m

    def test_score_fallback(self):
        assert trend_score_fallback(self._mock_trend(80)) == 80

    def test_score_fallback_none(self):
        assert trend_score_fallback(None) is None

    def test_score_fallback_zero(self):
        assert trend_score_fallback(self._mock_trend(0)) is None

    def test_label_zh(self):
        assert trend_label_fallback(self._mock_trend()) == "uptrend"

    def test_decision(self):
        assert trend_decision_fallback(self._mock_trend(buy_signal_name="strong_buy")) == "buy"
        assert trend_decision_fallback(self._mock_trend(buy_signal_name="hold")) == "hold"
        assert trend_decision_fallback(self._mock_trend(buy_signal_name="sell")) == "sell"
        assert trend_decision_fallback(None) is None

    def test_mark_fallback_idempotent(self):
        result = MagicMock()
        result.data_sources = "llm,trend:fallback"
        mark_trend_fallback_source(result)
        assert result.data_sources == "llm,trend:fallback"

    def test_mark_fallback_adds(self):
        result = MagicMock()
        result.data_sources = "llm"
        mark_trend_fallback_source(result)
        assert "trend:fallback" in result.data_sources

    def test_summary(self):
        result = MagicMock()
        result.trend_prediction = "Up"
        result.operation_advice = "Buy"
        assert summary_fallback_from_result(result, "en") == "Trend view: Up; action advice: Buy."

    def test_summary_zh(self):
        result = MagicMock()
        result.trend_prediction = "上涨"
        result.operation_advice = "买入"
        assert summary_fallback_from_result(result, "zh") == "趋势结论：上涨；操作建议：买入。"

    def test_summary_empty(self):
        result = MagicMock()
        result.trend_prediction = ""
        result.operation_advice = ""
        assert summary_fallback_from_result(result, "zh") == ""


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------
class TestAgentPlaceholder:
    @pytest.mark.parametrize("text", ["", "n/a", "N/A", "null", "unknown", "未知", "无"])
    def test_placeholder(self, text):
        assert is_agent_placeholder_text(text) is True

    @pytest.mark.parametrize("text", ["Buy", "买入", "Strong Buy", "85.5"])
    def test_not_placeholder(self, text):
        assert is_agent_placeholder_text(text) is False


class TestAgentFieldMissing:
    def test_none(self):
        assert is_agent_field_missing(None) is True

    def test_placeholder_text(self):
        assert is_agent_field_missing("N/A") is True

    def test_valid_text(self):
        assert is_agent_field_missing("Buy") is False

    def test_scalar_number(self):
        assert is_agent_field_missing(85, scalar=True) is False

    def test_scalar_string_not_number(self):
        # String values are treated as placeholder/missing when scalar=True.
        assert is_agent_field_missing("", scalar=True) is True

    def test_empty_dict(self):
        assert is_agent_field_missing({}, scalar=True) is True


class TestExtractAdvice:
    def test_has_position(self):
        assert extract_advice_text_from_dict({"has_position": "Buy more"}) == "Buy more"

    def test_no_position(self):
        assert extract_advice_text_from_dict({"no_position": "Wait"}) == "Wait"

    def test_fallback_to_any_value(self):
        assert extract_advice_text_from_dict({"other": "Some advice"}) == "Some advice"

    def test_empty(self):
        assert extract_advice_text_from_dict({}) == ""


class TestAgentDashboardValue:
    def test_top_level_hit(self):
        assert agent_dashboard_value({"k": "v"}, {}, "k") == "v"

    def test_fallback_to_nested(self):
        assert agent_dashboard_value({"k": None}, {"k": "nested_v"}, "k") == "nested_v"

    def test_both_missing(self):
        assert agent_dashboard_value({}, {}, "k") is None


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------
class TestWithoutRuntimePromptContext:
    def test_removes_runtime_keys(self):
        ctx = {
            "code": "600519",
            "market_phase_context": {"x": 1},
            "portfolio_context": {"y": 2},
            "analysis_context_pack": {"z": 3},
            "analysis_context_pack_summary": "s",
            "daily_market_context_summary": "d",
        }
        result = without_runtime_prompt_context(ctx)
        assert "market_phase_context" not in result
        assert "portfolio_context" not in result
        assert "analysis_context_pack" not in result
        assert result["code"] == "600519"

    def test_cleans_enhanced_context(self):
        ctx = {
            "enhanced_context": {"daily_market_context_summary": "remove", "keep": "me"},
        }
        result = without_runtime_prompt_context(ctx)
        assert "daily_market_context_summary" not in result["enhanced_context"]
        assert result["enhanced_context"]["keep"] == "me"


class TestSafeToDict:
    def test_none(self):
        assert safe_to_dict(None) is None

    def test_to_dict_method(self):
        obj = MagicMock()
        obj.to_dict.return_value = {"a": 1}
        assert safe_to_dict(obj) == {"a": 1}

    def test_fallback_to_dunder_dict(self):
        class Obj:
            def __init__(self):
                self.x = 5
        result = safe_to_dict(Obj())
        assert result == {"x": 5}


# ---------------------------------------------------------------------------
# attach_daily_market_context (needs mock for DailyMarketContext)
# ---------------------------------------------------------------------------
class TestAttachDailyMarketContext:
    def test_none_noop(self):
        ctx = {}
        attach_daily_market_context(ctx, None, report_language="zh")
        assert ctx == {}

    def test_attaches_context(self):
        from src.services.daily_market_context import format_daily_market_context_prompt_section

        mock_ctx = MagicMock()
        mock_ctx.to_safe_dict.return_value = {"summary": "s"}

        with patch(
            "src.services.daily_market_context.format_daily_market_context_prompt_section",
            return_value="formatted prompt",
        ) as mock_fmt:
            target = {}
            attach_daily_market_context(target, mock_ctx, report_language="zh")

        assert target["daily_market_context"] == {"summary": "s"}
        assert target["daily_market_context_summary"] == "formatted prompt"
        mock_fmt.assert_called_once_with({"summary": "s"}, report_language="zh")
