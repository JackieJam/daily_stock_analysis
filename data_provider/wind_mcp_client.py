# -*- coding: utf-8 -*-
"""
Wind MCP HTTP Client — pure Python replacement for the Node.js CLI subprocess.

Speaks JSON-RPC 2.0 over HTTP to Wind MCP servers (mcp.wind.com.cn),
eliminating per-call Node.js cold-start overhead and fragile stdout JSON parsing.

Design:
- requests.Session for connection pooling and keep-alive.
- MCP session reuse via ``Mcp-Session-Id`` response header.
- Thread-safe: uses a reentrant lock around the initialize-once handshake.
- SSE and pure-JSON response parsing (mirrors cli.mjs behaviour).
- HTTP status → error-code mapping for actionable error messages.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server registry — mirrors SERVERS in cli.mjs
# ---------------------------------------------------------------------------
SERVER_ENDPOINTS: Dict[str, str] = {
    "stock_data": "https://mcp.wind.com.cn/vserver_stock_data/mcp/",
    "global_stock_data": "https://mcp.wind.com.cn/vserver_global_stock_data/mcp/",
    "fund_data": "https://mcp.wind.com.cn/vserver_fund_data/mcp/",
    "index_data": "https://mcp.wind.com.cn/vserver_index_data/mcp/",
    "bond_data": "https://mcp.wind.com.cn/vserver_bond_data/mcp/",
    "financial_docs": "https://mcp.wind.com.cn/vserver_financial_docs/mcp/",
    "economic_data": "https://mcp.wind.com.cn/vserver_economic_data/mcp/",
    "analytics_data": "https://mcp.wind.com.cn/vserver_analytics_data/mcp/",
}

# ---------------------------------------------------------------------------
# HTTP status → (error_code, human_message)
# ---------------------------------------------------------------------------
HTTP_ERROR_MAP: Dict[int, tuple[str, str]] = {
    401: ("KEY_INVALID", "API Key 无效或过期 → 开发者中心重新生成"),
    403: ("KEY_FORBIDDEN_SERVER", "API Key 权限不足或该 server 未订阅 → 开发者中心确认"),
    429: ("RATE_LIMIT_QPS", "请求过于频繁 → 等几秒重试"),
    500: ("SERVER_5XX", "服务端异常 → 稍后重试"),
    502: ("SERVER_5XX", "网关异常 → 稍后重试"),
    503: ("SERVER_5XX", "服务暂不可用 → 稍后重试"),
    504: ("SERVER_5XX", "网关超时 → 稍后重试，或减小请求复杂度"),
}

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CALL_TIMEOUT_SECONDS = 60


class WindMCPError(Exception):
    """Raised when the Wind MCP call returns an error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class WindMCPClient:
    """Thread-safe, connection-pooled HTTP client for Wind MCP servers.

    Usage::

        client = WindMCPClient(api_key="...")
        result = client.call("stock_data", "get_stock_kline", {"windcode": "600519.SH", ...})
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("WindMCPClient requires a non-empty api_key")
        self._api_key = api_key.strip()
        self._timeout = timeout
        self._call_timeout = call_timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        # Per-server session id (MCP protocol) — populated after initialize.
        self._server_session_ids: Dict[str, str] = {}
        self._server_session_ids_lock = threading.Lock()
        # Track which servers have completed the initialize handshake.
        self._initialized_servers: set[str] = set()
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def call(
        self,
        server_type: str,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a Wind MCP tool and return the *unwrapped* result content.

        The return value is the ``content[0].text`` JSON payload (already
        parsed into a dict/list), matching what the old subprocess-based
        ``_call_wind`` returned.
        """
        endpoint = self._resolve_endpoint(server_type)
        self._ensure_initialized(server_type, endpoint)

        payload = self._rpc_request(
            endpoint,
            "tools/call",
            {"name": tool_name, "arguments": params},
            timeout=self._call_timeout,
        )
        return self._extract_content(payload, server_type, tool_name)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "WindMCPClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal — JSON-RPC transport
    # ------------------------------------------------------------------
    def _resolve_endpoint(self, server_type: str) -> str:
        try:
            return SERVER_ENDPOINTS[server_type]
        except KeyError:
            valid = ", ".join(SERVER_ENDPOINTS.keys())
            raise WindMCPError(
                "INVALID_SERVER",
                f"Unknown server_type '{server_type}'. Valid: {valid}",
            )

    def _ensure_initialized(self, server_type: str, endpoint: str) -> None:
        """Run the MCP initialize handshake once per server (thread-safe)."""
        if server_type in self._initialized_servers:
            return
        with self._init_lock:
            if server_type in self._initialized_servers:
                return
            self._rpc_request(
                endpoint,
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "daily-stock-analysis", "version": "1.0.0"},
                },
                timeout=self._timeout,
            )
            self._initialized_servers.add(server_type)
            logger.debug("[WindMCP] initialized session for %s", server_type)

    def _rpc_request(
        self,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        extra_headers: Dict[str, str] = {}
        with self._server_session_ids_lock:
            sid = self._server_session_ids.get(endpoint)
        if sid:
            extra_headers["Mcp-Session-Id"] = sid

        try:
            resp = self._session.post(
                endpoint,
                json=body,
                headers=extra_headers or None,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise WindMCPError("NETWORK_ERROR", f"{exc} (server={endpoint})") from exc

        # Capture session id from response for subsequent calls.
        resp_sid = resp.headers.get("Mcp-Session-Id")
        if resp_sid:
            with self._server_session_ids_lock:
                self._server_session_ids[endpoint] = resp_sid

        if not resp.status_code == 200:
            err = HTTP_ERROR_MAP.get(resp.status_code)
            if err:
                code, detail = err
            else:
                code, detail = "UNKNOWN", f"HTTP {resp.status_code} {resp.reason}"
            body_text = ""
            try:
                body_text = resp.text[:200]
            except Exception:
                pass
            raise WindMCPError(code, f"{detail} (server={endpoint}, body={body_text})")

        return self._parse_response(resp.text, endpoint)

    # ------------------------------------------------------------------
    # Internal — response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str, endpoint: str) -> Dict[str, Any]:
        """Parse SSE or pure-JSON response (mirrors parseSSE in cli.mjs)."""
        trimmed = text.strip()
        if trimmed.startswith("{"):
            try:
                return _safe_json_loads(trimmed)
            except ValueError:
                pass

        # SSE format: extract last "data: ..." line
        last_data: Optional[str] = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                last_data = line[5:].strip()
        if last_data:
            try:
                return _safe_json_loads(last_data)
            except ValueError as exc:
                raise WindMCPError(
                    "RESPONSE_PARSE_ERROR",
                    f"SSE data JSON parse failed: {exc} (server={endpoint})",
                ) from exc

        raise WindMCPError(
            "RESPONSE_PARSE_ERROR",
            f"Unrecognised response format (server={endpoint}, head={text[:200]!r})",
        )

    @staticmethod
    def _extract_content(
        payload: Dict[str, Any],
        server_type: str,
        tool_name: str,
    ) -> Dict[str, Any]:
        """Unwrap the JSON-RPC result, surfacing MCP-level errors."""
        if payload.get("error"):
            err_obj = payload["error"]
            msg = err_obj.get("message") or str(err_obj)
            raise WindMCPError("MCP_PROTOCOL_ERROR", f"{msg} (server={server_type})")

        result = payload.get("result")
        if result is None:
            raise WindMCPError(
                "MCP_PROTOCOL_ERROR",
                f"Empty result (server={server_type}, tool={tool_name})",
            )

        if result.get("isError"):
            content_list = result.get("content") or []
            text = content_list[0].get("text", "") if content_list else ""
            raise WindMCPError("MCP_TOOL_ERROR", f"{text} (server={server_type})")

        content_list = result.get("content") or []
        if not content_list:
            return {}
        inner_text = content_list[0].get("text", "{}")
        if isinstance(inner_text, str):
            try:
                inner = _safe_json_loads(inner_text)
            except ValueError:
                return {"raw_text": inner_text}
            # Surface business-level errors embedded in the JSON.
            if isinstance(inner, dict):
                tool_err_code = inner.get("mcp_tool_error_code")
                if isinstance(tool_err_code, int) and not isinstance(tool_err_code, bool) and tool_err_code != 0:
                    raise WindMCPError(
                        "MCP_TOOL_ERROR",
                        f"{inner.get('mcp_tool_error_msg', inner)} (server={server_type})",
                    )
                inner_err = inner.get("error")
                if isinstance(inner_err, dict) and (inner_err.get("code") or inner_err.get("message")):
                    err_code = inner_err.get("code", "")
                    err_msg = inner_err.get("message", "")
                    combined = f"{err_code}: {err_msg}" if err_code else err_msg
                    raise WindMCPError("MCP_TOOL_ERROR", f"{combined} (server={server_type})")
            return inner if inner is not None else {}
        return {"raw_content": inner_text}


def _safe_json_loads(text: str) -> Any:
    """Parse JSON, raising ValueError on failure."""
    import json

    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"JSON parse error: {exc} (input={text[:200]!r})") from exc
