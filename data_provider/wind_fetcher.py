# -*- coding: utf-8 -*-
"""
Wind 万得数据源适配器

通过 WindMCPClient（HTTP JSON-RPC）获取 Wind 金融数据。
需要：
1. WIND_API_KEY 环境变量
2. requests 库（已在依赖中）
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from data_provider.base import BaseFetcher
from data_provider.wind_mcp_client import WindMCPClient, WindMCPError

logger = logging.getLogger(__name__)


def _to_windcode(stock_code: str) -> str:
    """
    转换股票代码为 Wind 格式

    Examples:
        '600519'  -> '600519.SH'
        '000001'  -> '000001.SZ'
        'HK00700' -> '00700.HK'
    """
    code = stock_code.strip().upper()

    # 已经是 Wind 格式
    if '.' in code:
        return code

    # 港股
    if code.startswith('HK'):
        return f"{code[2:]}.HK"

    # A 股
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    elif code.startswith(('0', '2', '3')):
        return f"{code}.SZ"
    elif code.startswith('4') or code.startswith('8'):
        return f"{code}.BJ"

    return code


class WindFetcher(BaseFetcher):
    """Wind 万得数据源 — HTTP/JSON-RPC 模式（无 Node.js 依赖）"""

    name = "WindFetcher"
    priority = 1  # 高优先级

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self._api_key = api_key or os.environ.get("WIND_API_KEY")
        self._client: Optional[WindMCPClient] = None
        self._available: Optional[bool] = None

    def _check_available(self) -> bool:
        """检查 WindFetcher 是否可用"""
        if self._available is not None:
            return self._available

        if not self._api_key:
            logger.warning("[WindFetcher] WIND_API_KEY 未配置")
            self._available = False
            return False

        try:
            self._client = WindMCPClient(api_key=self._api_key)
            self._available = True
            logger.info("[WindFetcher] 可用 (HTTP JSON-RPC 模式)")
            return True
        except Exception as exc:
            logger.warning("[WindFetcher] 初始化失败: %s", exc)
            self._available = False
            return False

    def is_available(self) -> bool:
        return self._check_available()

    def _call_wind(self, server_type: str, tool_name: str, params: Dict[str, Any]) -> Optional[Dict]:
        """调用 Wind MCP 工具（通过 HTTP JSON-RPC）"""
        if not self._check_available() or self._client is None:
            return None

        try:
            return self._client.call(server_type, tool_name, params)
        except WindMCPError as exc:
            logger.warning(
                "[WindFetcher] 调用失败 [%s] %s.%s: %s",
                exc.code, server_type, tool_name, exc.message,
            )
            return None
        except Exception as exc:
            logger.warning(
                "[WindFetcher] 调用异常: %s.%s: %s",
                server_type, tool_name, exc,
            )
            return None

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线 K 线数据"""
        windcode = _to_windcode(stock_code)
        begin_date = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")

        data = self._call_wind("stock_data", "get_stock_kline", {
            "windcode": windcode,
            "begin_date": begin_date,
            "end_date": end_date_fmt,
        })

        if not data:
            return pd.DataFrame()

        # 解析 Wind 返回格式
        columns = [col["name"] for col in data.get("columns", [])]
        rows = data.get("rows", [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)

        # 重命名列
        rename_map = {
            "TIME": "date",
            "OPEN": "open",
            "MATCH": "close",
            "HIGH": "high",
            "LOW": "low",
            "VOLUME": "volume",
            "TURNOVER": "amount",
            "CHANGEHANDRATE": "turnover_rate",
        }
        df = df.rename(columns=rename_map)

        # 保留需要的列
        keep_cols = ["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"]
        df = df[[c for c in keep_cols if c in df.columns]]

        # 转换数据类型
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        df["code"] = stock_code
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化数据列名"""
        if df.empty:
            return df

        # 确保必需列存在
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"[WindFetcher] 缺少必需列: {col}")
                return pd.DataFrame()

        # 添加可选列
        if 'amount' not in df.columns:
            df['amount'] = 0
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = 0
        if 'turnover_rate' not in df.columns:
            df['turnover_rate'] = 0

        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'turnover_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        df['code'] = stock_code
        return df

    def get_fundamentals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取财务基本面数据"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("stock_data", "get_stock_fundamentals", {
            "question": f"{windcode} 财务基本面"
        })

    def get_holders(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取股东结构数据"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("stock_data", "get_stock_equity_holders", {
            "question": f"{windcode} 股本结构和股东"
        })

    def get_events(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取公司事件数据"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("stock_data", "get_stock_events", {
            "question": f"{windcode} 公司事件"
        })

    def get_risk_metrics(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取风险指标"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("stock_data", "get_risk_metrics", {
            "question": f"{windcode} 风险指标 Beta 波动率"
        })

    def get_technicals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取技术指标"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("stock_data", "get_stock_technicals", {
            "question": f"{windcode} 技术指标"
        })

    def get_announcements(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取公司公告"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("financial_docs", "get_company_announcements", {
            "question": f"{windcode} 公司公告"
        })

    def get_financial_news(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取财经新闻"""
        windcode = _to_windcode(stock_code)
        return self._call_wind("financial_docs", "get_financial_news", {
            "question": f"{windcode} 财经新闻"
        })

    def get_economic_data(self, query: str) -> Optional[Dict[str, Any]]:
        """获取宏观经济数据"""
        return self._call_wind("economic_data", "get_economic_data", {
            "question": query
        })
