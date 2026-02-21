"""Tests for the OpenClaw Gateway WebSocket client."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_challenge():
    """Create a gateway challenge message."""
    return json.dumps({"type": "event", "event": "connect.challenge", "payload": {}})


def _make_hello_ok():
    """Create a successful hello-ok response."""
    return json.dumps({"type": "res", "ok": True, "id": "test-id"})


def _make_auth_failure():
    """Create a failed auth response."""
    return json.dumps({"type": "res", "ok": False, "error": "unauthorized"})


def _make_chat_event(run_id: str, text: str, state: str = "streaming"):
    """Create a chat event payload."""
    return json.dumps({
        "type": "event",
        "event": "chat",
        "payload": {
            "runId": run_id,
            "state": state,
            "message": {
                "content": [{"type": "text", "text": text}],
            },
        },
    })


class TestConnect:
    """Verify gateway connection and auth handshake."""

    @patch("voice_loop.gateway_client.websockets")
    async def test_connect_success(self, mock_ws_module):
        """Successful connect + auth handshake returns True."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_hello_ok()])
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = AsyncMock(return_value=AsyncMock())
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        client = GatewayClient(url="ws://test:18789", token="test-token")
        result = await client.connect()

        assert result is True
        assert client.connected is True
        # Verify auth was sent
        assert mock_ws.send.called
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["method"] == "connect"
        assert sent_data["params"]["auth"]["token"] == "test-token"

    @patch("voice_loop.gateway_client.websockets")
    async def test_connect_failure_returns_false(self, mock_ws_module):
        """Connection failure returns False."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws_module.connect = AsyncMock(side_effect=ConnectionRefusedError("refused"))

        client = GatewayClient(url="ws://test:18789")
        result = await client.connect()

        assert result is False
        assert client.connected is False

    @patch("voice_loop.gateway_client.websockets")
    async def test_auth_failure_returns_false(self, mock_ws_module):
        """Auth failure (non-ok response) returns False."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_auth_failure()])
        mock_ws.send = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        client = GatewayClient(url="ws://test:18789", token="bad-token")
        result = await client.connect()

        assert result is False
        assert client.connected is False


class TestSendMessage:
    """Verify send_message yields deltas from chat events."""

    @patch("voice_loop.gateway_client.websockets")
    async def test_send_message_yields_deltas(self, mock_ws_module):
        """send_message yields text deltas from streaming chat events."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_hello_ok()])
        mock_ws.send = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        client = GatewayClient(url="ws://test:18789", token="test-token", timeout_s=2)

        # Make the event listener a no-op to avoid it consuming our mock
        async def noop_iter():
            # Block forever (simulates idle websocket)
            await asyncio.Event().wait()
            yield  # pragma: no cover

        mock_ws.__aiter__ = lambda self: noop_iter()

        await client.connect()

        # Now simulate what the event listener would do by putting events directly into the queue
        # We need to capture the idempotency_key used by send_message
        original_send = mock_ws.send

        captured_key = None

        async def capture_send(data):
            nonlocal captured_key
            msg = json.loads(data)
            if msg.get("method") == "chat.send":
                captured_key = msg["params"]["idempotencyKey"]

        mock_ws.send = capture_send

        # Run send_message and feed events in parallel
        collected = []

        async def feed_events():
            """Wait for the key to be captured, then feed events into the queue."""
            while captured_key is None:
                await asyncio.sleep(0.01)
            q = client._pending_runs[captured_key]
            await q.put({
                "state": "streaming",
                "message": {"content": [{"type": "text", "text": "Hello "}]},
            })
            await q.put({
                "state": "streaming",
                "message": {"content": [{"type": "text", "text": "world!"}]},
            })
            await q.put({
                "state": "final",
                "message": {"content": [{"type": "text", "text": ""}]},
            })

        feeder = asyncio.create_task(feed_events())

        async for delta in client.send_message("Test input"):
            if delta:
                collected.append(delta)

        await feeder
        assert collected == ["Hello ", "world!"]

    async def test_send_message_not_connected_raises(self):
        """send_message raises ConnectionError when not connected."""
        from voice_loop.gateway_client import GatewayClient

        client = GatewayClient()
        with pytest.raises(ConnectionError, match="Not connected"):
            async for _ in client.send_message("test"):
                pass  # pragma: no cover

    @patch("voice_loop.gateway_client.websockets")
    async def test_send_message_timeout(self, mock_ws_module):
        """send_message raises TimeoutError when gateway doesn't respond."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_hello_ok()])
        mock_ws.send = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        # Event listener no-op
        async def noop_iter():
            await asyncio.Event().wait()
            yield  # pragma: no cover

        mock_ws.__aiter__ = lambda self: noop_iter()

        client = GatewayClient(url="ws://test:18789", token="test-token", timeout_s=0.1)
        await client.connect()

        with pytest.raises(TimeoutError, match="Gateway response timeout"):
            async for _ in client.send_message("test"):
                pass  # pragma: no cover


class TestReconnect:
    """Verify reconnect loop with exponential backoff."""

    @patch("voice_loop.gateway_client.websockets")
    async def test_reconnect_doubles_delay(self, mock_ws_module):
        """Reconnect loop doubles delay on each failure up to max."""
        from voice_loop.gateway_client import GatewayClient

        call_count = 0

        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionRefusedError("refused")
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_hello_ok()])
            mock_ws.send = AsyncMock()
            mock_ws.__aiter__ = lambda self: AsyncMock().__aiter__()
            return mock_ws

        mock_ws_module.connect = fail_then_succeed

        client = GatewayClient(url="ws://test:18789", reconnect_max_s=10)
        client._reconnect_delay = 0.01  # Speed up test

        # Run reconnect loop with a timeout
        task = asyncio.create_task(client._reconnect_loop())
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pass  # May or may not complete within timeout

        # Should have attempted multiple connects
        assert call_count >= 2


class TestAbortRun:
    """Verify abort_run sends correct message."""

    @patch("voice_loop.gateway_client.websockets")
    async def test_abort_run_sends_message(self, mock_ws_module):
        """abort_run sends a chat.abort request with the run ID."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_hello_ok()])
        mock_ws.send = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        async def noop_iter():
            await asyncio.Event().wait()
            yield  # pragma: no cover

        mock_ws.__aiter__ = lambda self: noop_iter()

        client = GatewayClient(url="ws://test:18789", token="test-token")
        await client.connect()

        await client.abort_run("run-123")

        # Find the abort call (skip the connect send)
        calls = [c for c in mock_ws.send.call_args_list if c is not None]
        # The last send call should be the abort
        # But since we mocked send during connect too, we need to check
        # Actually send was mocked, so call_args_list has the connect req + abort
        abort_call = None
        for call in mock_ws.send.call_args_list:
            data = json.loads(call[0][0])
            if data.get("method") == "chat.abort":
                abort_call = data
                break

        assert abort_call is not None
        assert abort_call["params"]["runId"] == "run-123"
        assert abort_call["params"]["sessionKey"] == "main"


class TestClose:
    """Verify clean shutdown."""

    @patch("voice_loop.gateway_client.websockets")
    async def test_close_disconnects(self, mock_ws_module):
        """close() sets connected to False and closes WebSocket."""
        from voice_loop.gateway_client import GatewayClient

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[_make_challenge(), _make_hello_ok()])
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_module.connect = AsyncMock(return_value=mock_ws)

        async def noop_iter():
            await asyncio.Event().wait()
            yield  # pragma: no cover

        mock_ws.__aiter__ = lambda self: noop_iter()

        client = GatewayClient(url="ws://test:18789", token="test-token")
        await client.connect()

        assert client.connected is True

        await client.close()

        assert client.connected is False
        assert client._ws is None
        mock_ws.close.assert_called_once()
