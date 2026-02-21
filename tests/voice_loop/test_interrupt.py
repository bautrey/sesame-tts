"""Tests for the interrupt monitor."""

import asyncio

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


class TestInterruptDetection:
    """Verify RMS energy-based interrupt detection."""

    @patch("voice_loop.interrupt.sd")
    async def test_energy_above_threshold_sets_event(self, mock_sd):
        """When RMS energy exceeds threshold for consecutive frames, interrupt fires."""
        from voice_loop.interrupt import InterruptMonitor

        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        mock_sd.InputStream = MagicMock(return_value=mock_stream)

        monitor = InterruptMonitor(
            energy_threshold=0.01,
            consecutive_frames=3,
            frame_duration_ms=30,
            sample_rate=16000,
        )

        loop = asyncio.get_event_loop()
        await monitor.start(loop)

        # Get the callback that was passed to InputStream
        callback = mock_sd.InputStream.call_args[1]["callback"]

        # Simulate loud audio frames (high RMS energy)
        # int16 value of ~3277 => float ~0.1 => RMS ~0.1 (well above 0.01 threshold)
        loud_frame = np.full((480, 1), 3277, dtype=np.int16)

        # Feed consecutive frames above threshold
        callback(loud_frame, 480, None, None)
        await asyncio.sleep(0)  # Let event loop process call_soon_threadsafe
        assert not monitor.interrupt_event.is_set()  # 1 frame, need 3

        callback(loud_frame, 480, None, None)
        await asyncio.sleep(0)
        assert not monitor.interrupt_event.is_set()  # 2 frames, need 3

        callback(loud_frame, 480, None, None)
        await asyncio.sleep(0)  # Let event loop process the set()
        assert monitor.interrupt_event.is_set()  # 3 frames, triggered!

        await monitor.stop()

    @patch("voice_loop.interrupt.sd")
    async def test_energy_below_threshold_no_trigger(self, mock_sd):
        """When RMS energy is below threshold, no interrupt is fired."""
        from voice_loop.interrupt import InterruptMonitor

        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        mock_sd.InputStream = MagicMock(return_value=mock_stream)

        monitor = InterruptMonitor(
            energy_threshold=0.1,
            consecutive_frames=3,
            frame_duration_ms=30,
            sample_rate=16000,
        )

        loop = asyncio.get_event_loop()
        await monitor.start(loop)

        callback = mock_sd.InputStream.call_args[1]["callback"]

        # Simulate quiet audio (near silence)
        quiet_frame = np.full((480, 1), 10, dtype=np.int16)

        for _ in range(10):
            callback(quiet_frame, 480, None, None)

        assert not monitor.interrupt_event.is_set()

        await monitor.stop()

    @patch("voice_loop.interrupt.sd")
    async def test_consecutive_frames_requirement(self, mock_sd):
        """Interrupt only fires after consecutive frames above threshold."""
        from voice_loop.interrupt import InterruptMonitor

        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        mock_sd.InputStream = MagicMock(return_value=mock_stream)

        monitor = InterruptMonitor(
            energy_threshold=0.01,
            consecutive_frames=3,
            frame_duration_ms=30,
            sample_rate=16000,
        )

        loop = asyncio.get_event_loop()
        await monitor.start(loop)

        callback = mock_sd.InputStream.call_args[1]["callback"]

        loud_frame = np.full((480, 1), 3277, dtype=np.int16)
        quiet_frame = np.full((480, 1), 1, dtype=np.int16)

        # Two loud frames, then one quiet -- should reset counter
        callback(loud_frame, 480, None, None)
        callback(loud_frame, 480, None, None)
        callback(quiet_frame, 480, None, None)  # Resets count
        await asyncio.sleep(0)

        assert not monitor.interrupt_event.is_set()

        # Two more loud frames -- still not enough (only 2 consecutive)
        callback(loud_frame, 480, None, None)
        callback(loud_frame, 480, None, None)
        await asyncio.sleep(0)

        assert not monitor.interrupt_event.is_set()

        # Third consecutive loud frame -- should trigger
        callback(loud_frame, 480, None, None)
        await asyncio.sleep(0)

        assert monitor.interrupt_event.is_set()

        await monitor.stop()


class TestCapturedAudio:
    """Verify captured audio buffer for seamless transition."""

    @patch("voice_loop.interrupt.sd")
    async def test_captured_audio_returns_frames(self, mock_sd):
        """captured_audio returns concatenated frames from monitoring."""
        from voice_loop.interrupt import InterruptMonitor

        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        mock_sd.InputStream = MagicMock(return_value=mock_stream)

        monitor = InterruptMonitor(
            energy_threshold=0.5,  # High threshold so interrupt doesn't fire
            consecutive_frames=100,
            frame_duration_ms=30,
            sample_rate=16000,
        )

        loop = asyncio.get_event_loop()
        await monitor.start(loop)

        callback = mock_sd.InputStream.call_args[1]["callback"]

        # Feed some frames
        frame1 = np.ones((480, 1), dtype=np.int16) * 100
        frame2 = np.ones((480, 1), dtype=np.int16) * 200

        callback(frame1, 480, None, None)
        callback(frame2, 480, None, None)

        captured = monitor.captured_audio
        assert captured is not None
        assert len(captured) == 960  # 480 + 480

        await monitor.stop()

    async def test_captured_audio_none_when_empty(self):
        """captured_audio returns None when no frames captured."""
        from voice_loop.interrupt import InterruptMonitor

        monitor = InterruptMonitor()
        assert monitor.captured_audio is None
