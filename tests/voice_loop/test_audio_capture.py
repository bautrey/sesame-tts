"""Tests for AudioCapture with mocked sounddevice."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_loop.audio_capture import AudioCapture


@pytest.fixture
def mock_devices():
    """Realistic device list for testing."""
    return [
        {
            "name": "MacBook Pro Microphone",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": "MacBook Pro Speakers",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Blue Yeti Stereo",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
        {
            "name": "External HDMI Output",
            "max_input_channels": 0,
            "max_output_channels": 8,
            "default_samplerate": 48000.0,
        },
    ]


class TestResolveDevice:
    """Test device resolution by substring match."""

    def test_finds_device_by_substring(self, mock_devices):
        with patch("voice_loop.audio_capture.sd.query_devices", return_value=mock_devices):
            cap = AudioCapture(device="yeti")
            idx = cap.resolve_device()
            assert idx == 2

    def test_finds_device_case_insensitive(self, mock_devices):
        with patch("voice_loop.audio_capture.sd.query_devices", return_value=mock_devices):
            cap = AudioCapture(device="MACBOOK PRO MICRO")
            idx = cap.resolve_device()
            assert idx == 0

    def test_returns_none_for_missing_device(self, mock_devices):
        with patch("voice_loop.audio_capture.sd.query_devices", return_value=mock_devices):
            cap = AudioCapture(device="nonexistent")
            idx = cap.resolve_device()
            assert idx is None

    def test_returns_none_when_device_is_none(self):
        cap = AudioCapture(device=None)
        idx = cap.resolve_device()
        assert idx is None

    def test_skips_output_only_devices(self, mock_devices):
        with patch("voice_loop.audio_capture.sd.query_devices", return_value=mock_devices):
            # "External HDMI" exists but has 0 input channels
            cap = AudioCapture(device="HDMI")
            idx = cap.resolve_device()
            assert idx is None


class TestListDevices:
    """Test device listing."""

    def test_list_devices_returns_formatted_list(self, mock_devices):
        with patch("voice_loop.audio_capture.sd.query_devices", return_value=mock_devices):
            result = AudioCapture.list_devices()
            assert len(result) == 4
            assert result[0]["name"] == "MacBook Pro Microphone"
            assert result[0]["index"] == 0
            assert result[0]["max_input_channels"] == 1
            assert result[2]["name"] == "Blue Yeti Stereo"


class TestStartStop:
    """Test start/stop capture cycle with mocked stream."""

    async def test_start_stop_returns_captured_audio(self):
        mock_stream = MagicMock()
        captured_callback = None

        def mock_input_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("callback")
            return mock_stream

        with patch("voice_loop.audio_capture.sd.InputStream", side_effect=mock_input_stream):
            with patch("voice_loop.audio_capture.sd.query_devices", return_value=[]):
                cap = AudioCapture(device=None, sample_rate=16000)
                await cap.start()

                # Simulate audio callback with two frames
                frame1 = np.ones((480, 1), dtype=np.int16) * 100
                frame2 = np.ones((480, 1), dtype=np.int16) * 200
                captured_callback(frame1, 480, None, None)
                captured_callback(frame2, 480, None, None)

                audio = await cap.stop()
                assert len(audio) == 960
                assert audio.dtype == np.int16
                mock_stream.stop.assert_called_once()
                mock_stream.close.assert_called_once()

    async def test_stop_without_start_returns_empty(self):
        cap = AudioCapture()
        audio = await cap.stop()
        assert len(audio) == 0

    async def test_start_clears_previous_buffer(self):
        mock_stream = MagicMock()
        captured_callback = None

        def mock_input_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("callback")
            return mock_stream

        with patch("voice_loop.audio_capture.sd.InputStream", side_effect=mock_input_stream):
            with patch("voice_loop.audio_capture.sd.query_devices", return_value=[]):
                cap = AudioCapture(device=None)
                # Manually add data to buffer
                cap._buffer.append(np.zeros((480, 1), dtype=np.int16))
                assert len(cap._buffer) == 1

                await cap.start()
                # Buffer should be cleared on start
                assert len(cap._buffer) == 0


class TestGetLatestFrame:
    """Test getting the latest frame."""

    def test_returns_none_when_empty(self):
        cap = AudioCapture()
        assert cap.get_latest_frame() is None

    def test_returns_latest_frame(self):
        cap = AudioCapture()
        frame1 = np.ones((480, 1), dtype=np.int16) * 100
        frame2 = np.ones((480, 1), dtype=np.int16) * 200
        cap._buffer.append(frame1)
        cap._buffer.append(frame2)
        latest = cap.get_latest_frame()
        np.testing.assert_array_equal(latest, frame2)
