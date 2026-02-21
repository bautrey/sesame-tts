"""Sesame TTS streaming client via ElevenLabs-compatible endpoint."""

import logging
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


class TTSClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8880",
        voice: str = "conversationalB",
        output_format: str = "pcm_24000",
        timeout_s: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.output_format = output_format
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s, connect=5.0))

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks for the given text.

        Uses the ElevenLabs streaming endpoint:
        POST /v1/text-to-speech/{voice_id}/stream

        Args:
            text: Text to synthesize

        Yields:
            Raw PCM audio chunks (int16, 24000 Hz, mono)

        Raises:
            httpx.HTTPError: On connection or HTTP errors
            RuntimeError: If client not started
        """
        if self._client is None:
            raise RuntimeError("TTSClient not started. Call start() first.")

        url = f"{self.base_url}/v1/text-to-speech/{self.voice}/stream"
        payload = {
            "text": text,
            "model_id": "csm-1b",
            "output_format": self.output_format,
        }

        logger.debug(
            "TTS request: voice=%s format=%s text=%r", self.voice, self.output_format, text[:80]
        )

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                yield chunk

    async def health_check(self) -> bool:
        """Check if the TTS server is reachable."""
        if self._client is None:
            return False
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False
