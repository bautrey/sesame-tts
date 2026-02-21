"""Tests for the Sesame TTS streaming client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStreamSpeech:
    """Verify TTS streaming yields audio chunks."""

    @patch("voice_loop.tts_client.httpx")
    async def test_stream_speech_yields_chunks(self, mock_httpx):
        """stream_speech yields byte chunks from the streaming response."""
        from voice_loop.tts_client import TTSClient

        # Create a mock streaming response
        chunks = [b"\x00\x01" * 2048, b"\x02\x03" * 2048, b"\x04\x05" * 1024]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_bytes(chunk_size=4096):
            for c in chunks:
                yield c

        mock_response.aiter_bytes = fake_aiter_bytes

        # Create an async context manager for stream()
        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        client = TTSClient(base_url="http://test:8880", voice="testvoice")
        client._client = mock_client

        collected = []
        async for chunk in client.stream_speech("Hello world"):
            collected.append(chunk)

        assert collected == chunks
        # Verify the correct URL was called
        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "POST"
        assert "/v1/text-to-speech/testvoice/stream" in call_args[0][1]


class TestStreamSpeechHTTPError:
    """Verify HTTP errors are raised."""

    @patch("voice_loop.tts_client.httpx")
    async def test_http_error_raises(self, mock_httpx):
        """HTTP error from TTS server is propagated."""
        import httpx

        from voice_loop.tts_client import TTSClient

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        client = TTSClient()
        client._client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in client.stream_speech("Hello"):
                pass  # pragma: no cover


class TestHealthCheck:
    """Verify health check returns True/False."""

    async def test_health_check_success(self):
        """health_check returns True when server responds 200."""
        from voice_loop.tts_client import TTSClient

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_resp)

        client = TTSClient(base_url="http://test:8880")
        client._client = mock_client

        result = await client.health_check()
        assert result is True

    async def test_health_check_failure(self):
        """health_check returns False when server responds non-200."""
        from voice_loop.tts_client import TTSClient

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_resp)

        client = TTSClient(base_url="http://test:8880")
        client._client = mock_client

        result = await client.health_check()
        assert result is False

    async def test_health_check_connection_error(self):
        """health_check returns False on connection error."""
        from voice_loop.tts_client import TTSClient

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))

        client = TTSClient(base_url="http://test:8880")
        client._client = mock_client

        result = await client.health_check()
        assert result is False


class TestClientNotStarted:
    """Verify RuntimeError when client not started."""

    async def test_stream_speech_not_started_raises(self):
        """stream_speech raises RuntimeError if start() not called."""
        from voice_loop.tts_client import TTSClient

        client = TTSClient()
        with pytest.raises(RuntimeError, match="not started"):
            async for _ in client.stream_speech("Hello"):
                pass  # pragma: no cover

    async def test_health_check_not_started_returns_false(self):
        """health_check returns False if client not started."""
        from voice_loop.tts_client import TTSClient

        client = TTSClient()
        result = await client.health_check()
        assert result is False
