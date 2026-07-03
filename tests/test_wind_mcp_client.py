# -*- coding: utf-8 -*-
"""Tests for the pure-Python Wind MCP HTTP client (wind_mcp_client.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from data_provider.wind_mcp_client import (
    SERVER_ENDPOINTS,
    WindMCPClient,
    WindMCPError,
    _safe_json_loads,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """Return a WindMCPClient with a mocked session (init bypassed)."""
    with patch.object(WindMCPClient, "__init__", lambda self, *a, **kw: None):
        c = WindMCPClient.__new__(WindMCPClient)
    c._api_key = "test-key"
    c._timeout = 10
    c._call_timeout = 30
    c._session = MagicMock()
    c._server_session_ids = {}
    c._server_session_ids_lock = __import__("threading").Lock()
    c._initialized_servers = set()
    c._init_lock = __import__("threading").Lock()
    return c


def _mock_response(status_code: int = 200, text: str = "", headers: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.ok = 200 <= status_code < 300
    resp.headers = headers or {}
    resp.reason = "OK" if resp.ok else "Error"
    return resp


# ---------------------------------------------------------------------------
# SSE / JSON response parsing
# ---------------------------------------------------------------------------
class TestParseResponse:
    def test_pure_json(self, client):
        raw = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"ok": True}})
        result = client._parse_response(raw, "https://test/")
        assert result["result"]["ok"] is True

    def test_sse_format(self, client):
        sse = 'event: message\ndata: {"jsonrpc": "2.0", "id": "1", "result": {"x": 42}}\n'
        result = client._parse_response(sse, "https://test/")
        assert result["result"]["x"] == 42

    def test_sse_multiline_takes_last_data(self, client):
        sse = (
            "data: {\"jsonrpc\": \"2.0\", \"id\": \"0\", \"processing\": true}\n"
            "data: {\"jsonrpc\": \"2.0\", \"id\": \"1\", \"result\": {\"done\": true}}\n"
        )
        result = client._parse_response(sse, "https://test/")
        assert result["result"]["done"] is True

    def test_garbage_raises(self, client):
        with pytest.raises(WindMCPError, match="Unrecognised response format"):
            client._parse_response("not json at all", "https://test/")


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
class TestExtractContent:
    def test_happy_path(self, client):
        payload = {
            "jsonrpc": "2.0",
            "result": {
                "content": [{"type": "text", "text": json.dumps({"key": "value"})}]
            },
        }
        inner = client._extract_content(payload, "stock_data", "get_x")
        assert inner == {"key": "value"}

    def test_isError_raises(self, client):
        payload = {
            "jsonrpc": "2.0",
            "result": {"isError": True, "content": [{"text": "something broke"}]},
        }
        with pytest.raises(WindMCPError, match="MCP_TOOL_ERROR"):
            client._extract_content(payload, "stock_data", "get_x")

    def test_rpc_error_raises(self, client):
        payload = {"jsonrpc": "2.0", "error": {"code": -1, "message": "bad call"}}
        with pytest.raises(WindMCPError, match="MCP_PROTOCOL_ERROR"):
            client._extract_content(payload, "stock_data", "get_x")

    def test_business_error_code(self, client):
        payload = {
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "text": json.dumps({
                        "mcp_tool_error_code": 42,
                        "mcp_tool_error_msg": "quota exceeded",
                    })
                }]
            },
        }
        with pytest.raises(WindMCPError, match="quota exceeded"):
            client._extract_content(payload, "stock_data", "get_x")


# ---------------------------------------------------------------------------
# Full call flow (mocked HTTP)
# ---------------------------------------------------------------------------
class TestCall:
    def test_successful_call(self, client):
        init_resp = _mock_response(
            text=json.dumps({"jsonrpc": "2.0", "id": "i1", "result": {"protocolVersion": "2025-03-26"}}),
            headers={"Mcp-Session-Id": "sess-123"},
        )
        call_resp = _mock_response(
            text=json.dumps({
                "jsonrpc": "2.0",
                "id": "c1",
                "result": {"content": [{"text": json.dumps({"price": 150.0})}]},
            }),
        )
        client._session.post.side_effect = [init_resp, call_resp]

        result = client.call("stock_data", "get_stock_quote", {"windcode": "600519.SH"})
        assert result == {"price": 150.0}

        # Verify Mcp-Session-Id was sent on the second call.
        second_call_kwargs = client._session.post.call_args_list[1]
        sent_headers = second_call_kwargs.kwargs.get("headers") or {}
        assert sent_headers.get("Mcp-Session-Id") == "sess-123"

    def test_initialize_only_once(self, client):
        init_resp = _mock_response(
            text=json.dumps({"jsonrpc": "2.0", "id": "i1", "result": {}}),
        )
        call_resp = _mock_response(
            text=json.dumps({
                "jsonrpc": "2.0", "id": "c1",
                "result": {"content": [{"text": "{}"}]},
            }),
        )
        client._session.post.side_effect = [init_resp, call_resp, call_resp]

        client.call("stock_data", "get_x", {})
        client.call("stock_data", "get_x", {})

        # Only one initialize call (first post), then two tool calls = 3 total.
        assert client._session.post.call_count == 3

    def test_rate_limit_error(self, client):
        init_resp = _mock_response(
            text=json.dumps({"jsonrpc": "2.0", "id": "i1", "result": {}}),
        )
        err_resp = _mock_response(status_code=429, text="")
        client._session.post.side_effect = [init_resp, err_resp]

        with pytest.raises(WindMCPError, match="RATE_LIMIT_QPS"):
            client.call("stock_data", "get_x", {})

    def test_network_error(self, client):
        init_resp = _mock_response(
            text=json.dumps({"jsonrpc": "2.0", "id": "i1", "result": {}}),
        )
        client._session.post.side_effect = [init_resp, __import__("requests").RequestException("connection refused")]

        with pytest.raises(WindMCPError, match="NETWORK_ERROR"):
            client.call("stock_data", "get_x", {})


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
class TestSafeJsonLoads:
    def test_valid(self):
        assert _safe_json_loads('{"a": 1}') == {"a": 1}

    def test_invalid(self):
        with pytest.raises(ValueError, match="JSON parse error"):
            _safe_json_loads("not json")


class TestClientInit:
    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="non-empty api_key"):
            WindMCPClient("")

    def test_valid_key(self):
        c = WindMCPClient("my-key")
        assert c._api_key == "my-key"
        c.close()
