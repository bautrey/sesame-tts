"""Tests for the MLX Whisper transcriber."""

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_mlx_whisper_module():
    """Insert a mock mlx_whisper into sys.modules for all tests in this file."""
    mock_module = MagicMock()
    mock_module.transcribe = MagicMock(return_value={"text": ""})
    with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
        yield mock_module


class TestTranscribeSuccess:
    """Verify successful transcription returns text."""

    async def test_transcribe_returns_text(self, mock_mlx_whisper_module):
        """Successful transcription returns the text string."""
        from voice_loop.transcriber import Transcriber

        mock_mlx_whisper_module.transcribe = MagicMock(
            return_value={"text": "  Hello, world!  "}
        )

        transcriber = Transcriber(model_name="test-model")
        audio = np.random.randint(-1000, 1000, size=16000, dtype=np.int16)
        result = await transcriber.transcribe(audio)

        assert result == "Hello, world!"


class TestTranscribeEmpty:
    """Verify empty audio returns None."""

    async def test_empty_audio_returns_none(self):
        """Empty audio array returns None without calling mlx_whisper."""
        from voice_loop.transcriber import Transcriber

        transcriber = Transcriber()
        result = await transcriber.transcribe(np.array([], dtype=np.int16))

        assert result is None


class TestTranscribeNoiseFiltering:
    """Verify noise markers are filtered out."""

    @pytest.mark.parametrize(
        "noise_text",
        [
            "[silence]",
            "[SILENCE]",
            "[music]",
            "[noise]",
            "(silence)",
            "...",
            "",
        ],
    )
    async def test_noise_markers_return_none(self, mock_mlx_whisper_module, noise_text):
        """Known noise markers return None."""
        from voice_loop.transcriber import Transcriber

        mock_mlx_whisper_module.transcribe = MagicMock(return_value={"text": noise_text})

        transcriber = Transcriber(model_name="test-model")
        audio = np.zeros(16000, dtype=np.float32)
        result = await transcriber.transcribe(audio)

        assert result is None


class TestTranscribeShortText:
    """Verify short text (<2 chars) is filtered."""

    async def test_single_char_returns_none(self, mock_mlx_whisper_module):
        """Single character transcription is discarded."""
        from voice_loop.transcriber import Transcriber

        mock_mlx_whisper_module.transcribe = MagicMock(return_value={"text": "I"})

        transcriber = Transcriber(model_name="test-model")
        audio = np.zeros(16000, dtype=np.float32)
        result = await transcriber.transcribe(audio)

        assert result is None


class TestInt16ToFloat32Conversion:
    """Verify int16 audio is converted to float32 before transcription."""

    async def test_int16_converted_to_float32(self, mock_mlx_whisper_module):
        """int16 audio is normalized to float32 [-1.0, 1.0] range."""
        from voice_loop.transcriber import Transcriber

        captured_args = {}

        def fake_transcribe(audio, **kwargs):
            captured_args["audio"] = audio
            captured_args["kwargs"] = kwargs
            return {"text": "Hello there"}

        mock_mlx_whisper_module.transcribe = fake_transcribe

        transcriber = Transcriber(model_name="test-model")
        # Create int16 audio with known values
        audio_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        await transcriber.transcribe(audio_int16)

        # Verify the audio passed to mlx_whisper is float32
        passed_audio = captured_args["audio"]
        assert passed_audio.dtype == np.float32
        # Verify normalization: 32767/32768 ~= 1.0, -32768/32768 = -1.0
        np.testing.assert_allclose(passed_audio[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(passed_audio[1], 16384.0 / 32768.0, atol=1e-5)
        np.testing.assert_allclose(passed_audio[3], 32767.0 / 32768.0, atol=1e-5)
        np.testing.assert_allclose(passed_audio[4], -1.0, atol=1e-5)

    async def test_float32_passed_through(self, mock_mlx_whisper_module):
        """float32 audio is passed through without conversion."""
        from voice_loop.transcriber import Transcriber

        captured_args = {}

        def fake_transcribe(audio, **kwargs):
            captured_args["audio"] = audio
            return {"text": "Hello there"}

        mock_mlx_whisper_module.transcribe = fake_transcribe

        transcriber = Transcriber(model_name="test-model")
        audio_float = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        await transcriber.transcribe(audio_float)

        passed_audio = captured_args["audio"]
        assert passed_audio.dtype == np.float32
        np.testing.assert_array_equal(passed_audio, audio_float)


class TestLoadModel:
    """Verify model loading does a dummy transcription to warm up."""

    async def test_load_model_calls_transcribe(self, mock_mlx_whisper_module):
        """load_model() calls mlx_whisper.transcribe with 1s of silence."""
        from voice_loop.transcriber import Transcriber

        captured_calls = []

        def fake_transcribe(audio, **kwargs):
            captured_calls.append({"audio_len": len(audio), "kwargs": kwargs})
            return {"text": ""}

        mock_mlx_whisper_module.transcribe = fake_transcribe

        transcriber = Transcriber(model_name="test-model")
        await transcriber.load_model()

        assert len(captured_calls) == 1
        assert captured_calls[0]["audio_len"] == 16000  # 1s of silence at 16kHz
        assert captured_calls[0]["kwargs"]["path_or_hf_repo"] == "test-model"
        assert captured_calls[0]["kwargs"]["language"] == "en"
