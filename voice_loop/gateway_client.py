"""OpenClaw Gateway HTTP client using OpenAI-compatible chat completions endpoint."""

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class GatewayClient:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:18789",
        token: str = "gateway-token",
        timeout_s: int = 60,
        reconnect_max_s: int = 30,
    ):
        # Accept ws:// URLs for backwards compat but convert to http://
        base = url.replace("ws://", "http://").replace("wss://", "https://")
        self.base_url = base.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Verify gateway is reachable by hitting a lightweight endpoint."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
            )
            # Quick connectivity check - POST with empty messages returns 400 but proves auth works
            resp = await self._client.post(
                "/v1/chat/completions",
                json={"model": "openclaw", "messages": [{"role": "user", "content": "ping"}]},
            )
            if resp.status_code == 200:
                self._connected = True
                logger.info("Gateway connected: %s", self.base_url)
                return True
            elif resp.status_code == 401:
                logger.error("Gateway auth failed (401)")
                return False
            else:
                # Any non-401 means the endpoint exists and auth passed
                self._connected = True
                logger.info("Gateway connected: %s (status=%d)", self.base_url, resp.status_code)
                return True
        except Exception as e:
            logger.error("Gateway connection failed: %s", e)
            self._connected = False
            return False

    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send a chat message and yield streaming response text deltas.

        Uses the OpenAI-compatible /v1/chat/completions endpoint with SSE streaming.
        """
        if not self._client:
            raise ConnectionError("Not connected to gateway")

        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "openclaw",
                "stream": True,
                "messages": [{"role": "user", "content": text}],
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Gateway error {resp.status_code}: {body.decode()[:200]}")

            logger.info("Sent to gateway: %r", text[:80])

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    logger.warning("Malformed SSE chunk: %s", data[:100])

    async def abort_run(self, run_id: str) -> None:
        """Abort is not supported via HTTP endpoint."""
        logger.debug("abort_run called but HTTP mode does not support abort")

    async def close(self) -> None:
        """Close the HTTP client."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
