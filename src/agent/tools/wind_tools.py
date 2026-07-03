# -*- coding: utf-8 -*-
"""
Wind tools — wraps WindFetcher methods as agent-callable tools.

Tools:
- get_wind_fundamentals: financial fundamentals (ROE, margins, cash flow)
- get_wind_holders: shareholder structure
- get_wind_risk_metrics: risk indicators (Beta, volatility, Sharpe)
- get_wind_events: corporate events
- get_wind_announcements: company announcements
- get_wind_financial_news: financial news
- get_wind_economic_data: macro-economic data
"""

import logging
from typing import Any, Dict, Optional

from src.agent.tools.registry import ToolParameter, ToolDefinition

logger = logging.getLogger(__name__)

_wind_fetcher_singleton = None


def _get_wind_fetcher():
    """Return a module-level singleton WindFetcher."""
    global _wind_fetcher_singleton
    if _wind_fetcher_singleton is None:
        from data_provider.wind_fetcher import WindFetcher
        _wind_fetcher_singleton = WindFetcher()
    return _wind_fetcher_singleton


def reset_wind_fetcher() -> None:
    """Clear the cached WindFetcher."""
    global _wind_fetcher_singleton
    _wind_fetcher_singleton = None


# ============================================================
# get_wind_fundamentals
# ============================================================

def _handle_get_wind_fundamentals(stock_code: str) -> dict:
    """Get Wind financial fundamentals."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available. Check WIND_API_KEY."}

    data = fetcher.get_fundamentals(stock_code)
    if not data:
        return {"error": f"No fundamental data available for {stock_code}"}

    return {
        "stock_code": stock_code,
        "source": "wind",
        "data": data,
    }


get_wind_fundamentals_tool = ToolDefinition(
    name="get_wind_fundamentals",
    description="Get Wind financial fundamentals including ROE, margins, cash flow, "
                "leverage ratios, and key financial metrics. Uses Wind institutional data.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519', '000001'",
        ),
    ],
    handler=_handle_get_wind_fundamentals,
    category="data",
)


# ============================================================
# get_wind_holders
# ============================================================

def _handle_get_wind_holders(stock_code: str) -> dict:
    """Get Wind shareholder structure."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available"}

    data = fetcher.get_holders(stock_code)
    if not data:
        return {"error": f"No holder data available for {stock_code}"}

    return {
        "stock_code": stock_code,
        "source": "wind",
        "data": data,
    }


get_wind_holders_tool = ToolDefinition(
    name="get_wind_holders",
    description="Get Wind shareholder structure including top 10 shareholders, "
                "institutional holdings, controller changes, and lock-up expiry info.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519', '000001'",
        ),
    ],
    handler=_handle_get_wind_holders,
    category="data",
)


# ============================================================
# get_wind_risk_metrics
# ============================================================

def _handle_get_wind_risk_metrics(stock_code: str) -> dict:
    """Get Wind risk metrics."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available"}

    data = fetcher.get_risk_metrics(stock_code)
    if not data:
        return {"error": f"No risk metrics available for {stock_code}"}

    return {
        "stock_code": stock_code,
        "source": "wind",
        "data": data,
    }


get_wind_risk_metrics_tool = ToolDefinition(
    name="get_wind_risk_metrics",
    description="Get Wind risk indicators including Beta, annualized volatility, "
                "Sharpe ratio, max drawdown, and VaR metrics.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519', '000001'",
        ),
    ],
    handler=_handle_get_wind_risk_metrics,
    category="analysis",
)


# ============================================================
# get_wind_events
# ============================================================

def _handle_get_wind_events(stock_code: str) -> dict:
    """Get Wind corporate events."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available"}

    data = fetcher.get_events(stock_code)
    if not data:
        return {"error": f"No event data available for {stock_code}"}

    return {
        "stock_code": stock_code,
        "source": "wind",
        "data": data,
    }


get_wind_events_tool = ToolDefinition(
    name="get_wind_events",
    description="Get Wind corporate events including IPO, M&A, ST marking, "
                "compliance events, dividends, and share buybacks.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519', '000001'",
        ),
    ],
    handler=_handle_get_wind_events,
    category="data",
)


# ============================================================
# get_wind_announcements
# ============================================================

def _handle_get_wind_announcements(stock_code: str) -> dict:
    """Get Wind company announcements."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available"}

    data = fetcher.get_announcements(stock_code)
    if not data:
        return {"error": f"No announcement data available for {stock_code}"}

    return {
        "stock_code": stock_code,
        "source": "wind",
        "data": data,
    }


get_wind_announcements_tool = ToolDefinition(
    name="get_wind_announcements",
    description="Search Wind company announcements including annual reports, "
                "prospectuses, major contracts, and regulatory filings.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519', '000001'",
        ),
    ],
    handler=_handle_get_wind_announcements,
    category="search",
)


# ============================================================
# get_wind_financial_news
# ============================================================

def _handle_get_wind_financial_news(stock_code: str) -> dict:
    """Get Wind financial news."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available"}

    data = fetcher.get_financial_news(stock_code)
    if not data:
        return {"error": f"No news data available for {stock_code}"}

    return {
        "stock_code": stock_code,
        "source": "wind",
        "data": data,
    }


get_wind_financial_news_tool = ToolDefinition(
    name="get_wind_financial_news",
    description="Search Wind financial news including company news, industry updates, "
                "and policy changes.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519', '000001'",
        ),
    ],
    handler=_handle_get_wind_financial_news,
    category="search",
)


# ============================================================
# get_wind_economic_data
# ============================================================

def _handle_get_wind_economic_data(query: str) -> dict:
    """Get Wind macro-economic data."""
    fetcher = _get_wind_fetcher()
    if not fetcher.is_available():
        return {"error": "WindFetcher not available"}

    data = fetcher.get_economic_data(query)
    if not data:
        return {"error": f"No economic data available for query: {query}"}

    return {
        "source": "wind",
        "data": data,
    }


get_wind_economic_data_tool = ToolDefinition(
    name="get_wind_economic_data",
    description="Get Wind macro-economic data including GDP, CPI, PPI, PMI, M2, "
                "trade data, and other economic indicators.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language query, e.g., '中国GDP增速', 'CPI数据', 'PMI指数'",
        ),
    ],
    handler=_handle_get_wind_economic_data,
    category="data",
)


# ============================================================
# Export all Wind tools
# ============================================================

ALL_WIND_TOOLS = [
    get_wind_fundamentals_tool,
    get_wind_holders_tool,
    get_wind_risk_metrics_tool,
    get_wind_events_tool,
    get_wind_announcements_tool,
    get_wind_financial_news_tool,
    get_wind_economic_data_tool,
]
