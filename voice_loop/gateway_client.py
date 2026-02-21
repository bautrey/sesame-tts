"""OpenClaw Gateway WebSocket client for chat communication."""

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class GatewayClient:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:18789",
        token: str = "gateway-token",
        timeout_s: int = 10,
        reconnect_max_s: int = 30,
    ):
        self.url = url
        self.token = token
        self.timeout_s = timeout_s
        self.reconnect_max_s = reconnect_max_s
        self._ws = None
        self._connected = False
        self._reconnect_delay = 1.0
        self._pending_runs: dict[str, asyncio.Queue] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to gateway and complete authentication handshake.

        Returns True if connected successfully, False otherwise.
        """
        try:
            self._ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=10)

            # Step 1: Receive connect.challenge
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            challenge = json.loads(raw)
            logger.debug("Received challenge: %s", challenge.get("event"))

            # Step 2: Send connect request with auth
            connect_req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "sesame-voice",
                        "version": "1.0",
                        "platform": "darwin",
                        "mode": "backend",
                    },
                    "role": "operator",
                    "scopes": ["operator.admin", "operator.talk.secrets"],
                    "auth": {"token": self.token},
                },
            }
            await self._ws.send(json.dumps(connect_req))

            # Step 3: Receive hello-ok response
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            response = json.loads(raw)
            if response.get("type") == "res" and response.get("ok"):
                self._connected = True
                self._reconnect_delay = 1.0  # Reset backoff
                logger.info("Gateway connected: %s", self.url)
                # Start background event listener
                asyncio.create_task(self._event_listener())
                return True
            else:
                logger.error("Gateway auth failed: %s", response)
                return False

        except Exception as e:
            logger.error("Gateway connection failed: %s", e)
            self._connected = False
            return False

    async def _event_listener(self) -> None:
        """Background task that routes incoming events to pending run queues."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") == "event" and msg.get("event") == "chat":
                    payload = msg.get("payload", {})
                    run_id = payload.get("runId")
                    if run_id and run_id in self._pending_runs:
                        await self._pending_runs[run_id].put(payload)
                elif msg.get("type") == "res":
                    # Response to a request (e.g., chat.send ack) -- logged but not queued
                    logger.debug("Gateway response: ok=%s", msg.get("ok"))
        except ConnectionClosed:
            logger.warning("Gateway WebSocket closed")
            self._connected = False
            asyncio.create_task(self._reconnect_loop())
        except Exception as e:
            logger.error("Gateway event listener error: %s", e)
            self._connected = False

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff."""
        while not self._connected:
            logger.info("Reconnecting to gateway in %.0fs...", self._reconnect_delay)
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self.reconnect_max_s)
            await self.connect()

    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send a chat message and yield streaming response text deltas.

        Args:
            text: User transcript to send

        Yields:
            Text chunks as they arrive from the assistant

        Raises:
            TimeoutError: If no response within timeout_s
            ConnectionError: If not connected to gateway
        """
        if not self._connected or self._ws is None:
            raise ConnectionError("Not connected to gateway")

        idempotency_key = str(uuid.uuid4())
        run_queue: asyncio.Queue = asyncio.Queue()
        self._pending_runs[idempotency_key] = run_queue

        try:
            # Send chat.send request
            req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "chat.send",
                "params": {
                    "sessionKey": "main",
                    "message": text,
                    "idempotencyKey": idempotency_key,
                },
            }
            await self._ws.send(json.dumps(req))
            logger.info("Sent to gateway: %r", text[:80])

            # Yield streaming deltas until final or error
            while True:
                try:
                    payload = await asyncio.wait_for(run_queue.get(), timeout=self.timeout_s)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Gateway response timeout ({self.timeout_s}s)")

                state = payload.get("state")
                message = payload.get("message", {})
                content = message.get("content", [])

                # Extract text from content blocks
                for block in content:
                    if block.get("type") == "text" and block.get("text"):
                        yield block["text"]

                if state == "final":
                    break
                elif state == "error":
                    error_msg = payload.get("error", "Unknown gateway error")
                    logger.error("Gateway error for run %s: %s", idempotency_key, error_msg)
                    raise RuntimeError(f"Gateway error: {error_msg}")

        finally:
            self._pending_runs.pop(idempotency_key, None)

    async def abort_run(self, run_id: str) -> None:
        """Send abort request for an active run."""
        if self._ws and self._connected:
            req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "chat.abort",
                "params": {"sessionKey": "main", "runId": run_id},
            }
            await self._ws.send(json.dumps(req))
            logger.info("Aborted run: %s", run_id)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None
