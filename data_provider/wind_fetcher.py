# -*- coding: utf-8 -*-
"""
Wind 万得数据源适配器

通过 wind-mcp-skill CLI 获取 Wind 金融数据。
需要：
1. Node.js 18+
2. wind-mcp-skill CLI (wind-skills/skills/wind-mcp-skill/scripts/cli.mjs)
3. WIND_API_KEY 环境变量
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data_provider.base import BaseFetcher

logger = logging.getLogger(__name__)


def _resolve_default_skill_dir() -> Path:
    """解析 wind-mcp-skill 目录"""
    env_dir = os.environ.get("WIND_SKILL_DIR")
    if env_dir:
        return Path(env_dir)
    # 从项目根目录查找
    project_root = Path(__file__).parent.parent
    local = project_root / "wind-skills" / "skills" / "wind-mcp-skill"
    if local.exists():
        return local.resolve()
    # 尝试上级目录
    parent = project_root.parent / "wind-skills" / "skills" / "wind-mcp-skill"
    if parent.exists():
        return parent.resolve()
    return local


_DEFAULT_SKILL_DIR = _resolve_default_skill_dir()


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
    """Wind 万得数据源"""

    name = "WindFetcher"
    priority = 1  # 高优先级

    def __init__(self, skill_dir: Optional[Path] = None, api_key: Optional[str] = None):
        super().__init__()
        self._skill_dir = skill_dir or _DEFAULT_SKILL_DIR
        self._api_key = api_key or os.environ.get("WIND_API_KEY")
        self._available = None

    def _check_available(self) -> bool:
        """检查 wind-mcp-skill 是否可用"""
        if self._available is not None:
            return self._available

        # 检查 skill 目录
        cli_path = self._skill_dir / "scripts" / "cli.mjs"
        if not cli_path.exists():
            logger.warning(f"[WindFetcher] CLI 不存在: {cli_path}")
            self._available = False
            return False

        # 检查 Node.js
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                logger.warning("[WindFetcher] Node.js 不可用")
                self._available = False
                return False
        except Exception:
            logger.warning("[WindFetcher] Node.js 不可用")
            self._available = False
            return False

        # 检查 API Key
        if not self._api_key:
            logger.warning("[WindFetcher] WIND_API_KEY 未配置")
            self._available = False
            return False

        self._available = True
        logger.info(f"[WindFetcher] 可用 (skill_dir={self._skill_dir})")
        return True

    def is_available(self) -> bool:
        return self._check_available()

    def _call_wind(self, server_type: str, tool_name: str, params: Dict[str, Any]) -> Optional[Dict]:
        """调用 Wind CLI"""
        if not self._check_available():
            return None

        cli_path = self._skill_dir / "scripts" / "cli.mjs"
        cmd = [
            "node", str(cli_path),
            "call", server_type, tool_name,
            json.dumps(params, ensure_ascii=False)
        ]

        try:
            env = {**os.environ, "WIND_API_KEY": self._api_key, "NODE_NO_WARNINGS": "1"}
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=30, env=env
            )

            if result.returncode != 0:
                logger.warning(f"[WindFetcher] CLI 调用失败: {result.stderr[:200]}")
                return None

            # 解析 JSON 响应（可能跨多行）
            output = result.stdout.strip()
            # 找到 JSON 开始的位置（跳过警告等非 JSON 内容）
            json_start = output.find('{')
            if json_start == -1:
                logger.warning("[WindFetcher] CLI 输出中未找到 JSON")
                return None

            # 尝试从 JSON 开始位置解析
            json_str = output[json_start:]
            try:
                response = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试逐行构建 JSON
                lines = output[json_start:].split('\n')
                json_lines = []
                brace_count = 0
                for line in lines:
                    json_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and json_lines:
                        try:
                            response = json.loads('\n'.join(json_lines))
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    logger.warning("[WindFetcher] 无法解析 CLI 输出 JSON")
                    return None

            if response.get('isError'):
                logger.warning(f"[WindFetcher] API 错误: {response.get('error')}")
                return None

            content = response.get('content', [])
            if content and len(content) > 0:
                text = content[0].get('text', '{}')
                return json.loads(text)

            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"[WindFetcher] CLI 调用超时: {server_type}.{tool_name}")
            return None
        except Exception as e:
            logger.warning(f"[WindFetcher] CLI 调用异常: {e}")
            return None

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线 K 线数据"""
        windcode = _to_windcode(stock_code)
        begin_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        data = self._call_wind("stock_data", "get_stock_kline", {
            "windcode": windcode,
            "begin_date": begin_date,
            "end_date": end_date,
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
