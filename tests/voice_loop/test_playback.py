"""Tests for the audio playback module."""

import asyncio

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestPlayStream:
    """Verify play_stream consumes audio chunks."""

    @patch("voice_loop.playback.sd")
    async def test_play_stream_consumes_chunks(self, mock_sd):
        """play_stream processes all chunks from the async iterator."""
        from voice_loop.playback import AudioPlayer

        # Mock OutputStream
        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        # After we put END_OF_STREAM, active should return False
        mock_stream.active = False
        mock_sd.OutputStream = MagicMock(return_value=mock_stream)

        player = AudioPlayer(sample_rate=24000)

        # Create async iterator of audio chunks
        chunks = [
            np.zeros(4800, dtype=np.int16).tobytes(),  # 200ms of silence
            np.zeros(2400, dtype=np.int16).tobytes(),  # 100ms of silence
        ]

        async def chunk_iter():
            for c in chunks:
                yield c

        await player.play_stream(chunk_iter())

        # Verify stream was created and started
        mock_sd.OutputStream.assert_called_once()
        mock_stream.start.assert_called_once()


class TestStop:
    """Verify stop() sets stop event and stops stream."""

    @patch("voice_loop.playback.sd")
    async def test_stop_sets_event_and_stops_stream(self, mock_sd):
        """stop() sets the stop event and closes the stream."""
        from voice_loop.playback import AudioPlayer

        mock_stream = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()

        player = AudioPlayer()
        player._stream = mock_stream
        player._playing = True

        player.stop()

        assert player._stop_event.is_set()
        assert player._playing is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert player._stream is None

    async def test_stop_with_no_stream_is_safe(self):
        """stop() works even when no stream is active."""
        from voice_loop.playback import AudioPlayer

        player = AudioPlayer()
        player.stop()  # Should not raise

        assert player._stop_event.is_set()
        assert player._playing is False


class TestDeviceResolution:
    """Verify device resolution by substring match."""

    @patch("voice_loop.playback.sd")
    def test_resolve_device_by_substring(self, mock_sd):
        """Device is resolved by substring match on name."""
        from voice_loop.playback import AudioPlayer

        mock_sd.query_devices = MagicMock(
            return_value=[
                {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0},
                {
                    "name": "MacBook Pro Speakers",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
                {
                    "name": "External DAC",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
        )

        player = AudioPlayer(device="MacBook")
        result = player.resolve_device()
        assert result == 1

    @patch("voice_loop.playback.sd")
    def test_resolve_device_not_found_returns_none(self, mock_sd):
        """Unknown device name returns None (system default)."""
        from voice_loop.playback import AudioPlayer

        mock_sd.query_devices = MagicMock(
            return_value=[
                {
                    "name": "MacBook Pro Speakers",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
        )

        player = AudioPlayer(device="Nonexistent")
        result = player.resolve_device()
        assert result is None

    def test_resolve_device_none_returns_none(self):
        """No device configured returns None (system default)."""
        from voice_loop.playback import AudioPlayer

        player = AudioPlayer(device=None)
        result = player.resolve_device()
        assert result is None


class TestIsPlaying:
    """Verify is_playing property."""

    def test_is_playing_initially_false(self):
        """is_playing starts as False."""
        from voice_loop.playback import AudioPlayer

        player = AudioPlayer()
        assert player.is_playing is False
