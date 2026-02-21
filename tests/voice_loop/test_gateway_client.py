"""Tests for the OpenClaw Gateway HTTP client."""

import json

import httpx
import pytest
from unittest.mock import AsyncMock, patch


def _sse_lines(chunks: list[str], done: bool = True) -> list[str]:
    """Build SSE data lines from text chunks."""
    lines = []
    for i, text in enumerate(chunks):
        chunk = {
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "created": 1000000,
            "model": "openclaw",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    if done:
        lines.append("data: [DONE]")
        lines.append("")
    return lines


class TestConnect:
    """Verify gateway connection check."""

    async def test_connect_success(self):
        """Successful connect returns True."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="http://test:18789", token="test-token")

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
            result = await client.connect()

        assert result is True
        assert client.connected is True
        await client.close()

    async def test_connect_auth_failure(self):
        """401 response returns False."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="http://test:18789", token="bad-token")

        mock_response = AsyncMock()
        mock_response.status_code = 401

        with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
            result = await client.connect()

        assert result is False
        assert client.connected is False
        await client.close()

    async def test_connect_network_error(self):
        """Network error returns False."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="http://unreachable:18789")

        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("refused")):
            result = await client.connect()

        assert result is False
        assert client.connected is False
        await client.close()

    async def test_ws_url_converted_to_http(self):
        """ws:// URLs are converted to http:// for backwards compat."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="ws://localhost:18789")
        assert client.base_url == "http://localhost:18789"

        client2 = GatewayClient(url="wss://localhost:18789")
        assert client2.base_url == "https://localhost:18789"


class TestSendMessage:
    """Verify send_message yields deltas from SSE stream."""

    async def test_send_message_yields_deltas(self):
        """send_message yields text deltas from SSE streaming response."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="http://test:18789", token="test-token")
        # Manually set up client without real connection
        client._client = httpx.AsyncClient(base_url="http://test:18789")
        client._connected = True

        lines = _sse_lines(["Hello ", "world!"])

        async def async_line_iter():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = async_line_iter
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            collected = []
            async for delta in client.send_message("Test input"):
                collected.append(delta)

        assert collected == ["Hello ", "world!"]
        await client.close()

    async def test_send_message_not_connected_raises(self):
        """send_message raises ConnectionError when not connected."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient()
        with pytest.raises(ConnectionError, match="Not connected"):
            async for _ in client.send_message("test"):
                pass

    async def test_send_message_error_status(self):
        """send_message raises RuntimeError on non-200 response."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="http://test:18789", token="test-token")
        client._client = httpx.AsyncClient(base_url="http://test:18789")
        client._connected = True

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Gateway error 500"):
                async for _ in client.send_message("test"):
                    pass

        await client.close()


class TestAbortRun:
    """Verify abort_run is a no-op in HTTP mode."""

    async def test_abort_run_no_error(self):
        """abort_run does not raise in HTTP mode."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient()
        await client.abort_run("run-123")  # Should not raise


class TestClose:
    """Verify clean shutdown."""

    async def test_close_disconnects(self):
        """close() sets connected to False and closes HTTP client."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient(url="http://test:18789", token="test-token")
        client._client = AsyncMock()
        client._connected = True

        await client.close()

        assert client.connected is False
        assert client._client is None
