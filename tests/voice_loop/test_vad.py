"""Tests for VAD with mocked webrtcvad."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock webrtcvad at the module level before importing VAD
_mock_webrtcvad_module = MagicMock()
sys.modules.setdefault("webrtcvad", _mock_webrtcvad_module)

from voice_loop.vad import VAD  # noqa: E402


SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480 samples


def make_frame(value: int = 0) -> np.ndarray:
    """Create a frame of the correct length for 30ms at 16kHz."""
    return np.full(FRAME_SAMPLES, value, dtype=np.int16)


@pytest.fixture
def mock_vad_instance():
    """Create a mock webrtcvad.Vad instance."""
    return MagicMock()


@pytest.fixture
def vad_with_mock(mock_vad_instance):
    """Create a VAD with a mocked webrtcvad.Vad."""
    with patch("webrtcvad.Vad", return_value=mock_vad_instance):
        v = VAD(
            sample_rate=SAMPLE_RATE,
            aggressiveness=3,
            silence_threshold_ms=90,  # 3 frames of silence for easier testing
            frame_duration_ms=FRAME_DURATION_MS,
        )
    # Ensure internal _vad uses our mock
    v._vad = mock_vad_instance
    return v


class TestEndOfSpeech:
    """Speech frames followed by silence frames should trigger end-of-speech."""

    def test_speech_then_silence_triggers(self, vad_with_mock, mock_vad_instance):
        # Set up: 2 speech frames, then 3 silence frames (threshold = 90ms / 30ms = 3)
        speech_silence_sequence = [True, True, False, False, False]
        mock_vad_instance.is_speech.side_effect = speech_silence_sequence

        results = []
        for _ in speech_silence_sequence:
            results.append(vad_with_mock.process_frame(make_frame()))

        # Should NOT trigger on speech frames or first silence frames
        assert results[:4] == [False, False, False, False]
        # Should trigger on the 3rd silence frame (meets threshold)
        assert results[4] is True

    def test_speech_then_more_speech_does_not_trigger(self, vad_with_mock, mock_vad_instance):
        mock_vad_instance.is_speech.return_value = True
        for _ in range(10):
            result = vad_with_mock.process_frame(make_frame())
            assert result is False


class TestSilenceOnly:
    """Silence-only should NOT trigger end-of-speech (must have speech first)."""

    def test_silence_only_does_not_trigger(self, vad_with_mock, mock_vad_instance):
        mock_vad_instance.is_speech.return_value = False
        for _ in range(20):
            result = vad_with_mock.process_frame(make_frame())
            assert result is False

    def test_has_speech_false_when_only_silence(self, vad_with_mock, mock_vad_instance):
        mock_vad_instance.is_speech.return_value = False
        for _ in range(5):
            vad_with_mock.process_frame(make_frame())
        assert vad_with_mock.has_speech is False


class TestReset:
    """Test that reset() clears all speech detection state."""

    def test_reset_clears_speech_detected(self, vad_with_mock, mock_vad_instance):
        # Detect speech
        mock_vad_instance.is_speech.return_value = True
        vad_with_mock.process_frame(make_frame())
        assert vad_with_mock.has_speech is True

        # Reset
        vad_with_mock.reset()
        assert vad_with_mock.has_speech is False

    def test_reset_prevents_trigger_after_prior_speech(self, vad_with_mock, mock_vad_instance):
        # Speech then reset, then silence should NOT trigger
        mock_vad_instance.is_speech.return_value = True
        vad_with_mock.process_frame(make_frame())

        vad_with_mock.reset()

        mock_vad_instance.is_speech.return_value = False
        for _ in range(10):
            result = vad_with_mock.process_frame(make_frame())
            assert result is False


class TestFrameValidation:
    """Test frame length validation."""

    def test_wrong_frame_length_returns_false(self, vad_with_mock):
        # Frame with wrong number of samples
        short_frame = np.zeros(100, dtype=np.int16)
        result = vad_with_mock.process_frame(short_frame)
        assert result is False

    def test_empty_frame_returns_false(self, vad_with_mock):
        empty_frame = np.array([], dtype=np.int16)
        result = vad_with_mock.process_frame(empty_frame)
        assert result is False


class TestHasSpeechProperty:
    """Test the has_speech property."""

    def test_has_speech_true_after_speech(self, vad_with_mock, mock_vad_instance):
        mock_vad_instance.is_speech.return_value = True
        vad_with_mock.process_frame(make_frame())
        assert vad_with_mock.has_speech is True

    def test_has_speech_stays_true_during_silence(self, vad_with_mock, mock_vad_instance):
        # Speech first
        mock_vad_instance.is_speech.return_value = True
        vad_with_mock.process_frame(make_frame())

        # Then silence
        mock_vad_instance.is_speech.return_value = False
        vad_with_mock.process_frame(make_frame())
        assert vad_with_mock.has_speech is True


class TestSilenceCounterReset:
    """Test that speech resets the silence counter."""

    def test_interleaved_speech_resets_silence_counter(self, vad_with_mock, mock_vad_instance):
        # Speech -> 2 silence frames -> speech -> 2 silence frames: should NOT trigger
        sequence = [True, False, False, True, False, False]
        mock_vad_instance.is_speech.side_effect = sequence

        results = [vad_with_mock.process_frame(make_frame()) for _ in sequence]
        # None should trigger because silence never reaches 3 consecutive frames
        assert all(r is False for r in results)
