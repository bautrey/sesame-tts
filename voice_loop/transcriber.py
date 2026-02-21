"""MLX Whisper wrapper for speech-to-text transcription."""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self._model = None

    async def load_model(self) -> None:
        """Load the Whisper model. Call once at startup."""
        import mlx_whisper

        loop = asyncio.get_event_loop()
        logger.info("Loading Whisper model: %s", self.model_name)
        start = time.time()
        # Warm up by loading the model -- mlx_whisper.transcribe loads lazily,
        # so we do a dummy transcription to force the model into memory.
        await loop.run_in_executor(
            None,
            lambda: mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32),  # 1s of silence
                path_or_hf_repo=self.model_name,
                language="en",
            ),
        )
        elapsed = time.time() - start
        logger.info("Whisper model loaded in %.1fs", elapsed)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio buffer to text.

        Args:
            audio: int16 or float32 numpy array of audio samples
            sample_rate: sample rate of the audio (should be 16000 for Whisper)

        Returns:
            Transcribed text, or None if empty/noise
        """
        import mlx_whisper

        if len(audio) == 0:
            return None

        # Convert int16 to float32 if needed (mlx_whisper expects float32)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        start = time.time()

        result = await loop.run_in_executor(
            None,
            lambda: mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_name,
                language="en",
            ),
        )

        elapsed = time.time() - start
        text = result.get("text", "").strip()

        # Discard empty or noise-only transcriptions
        noise_markers = {"", "[silence]", "[music]", "[noise]", "(silence)", "..."}
        if text.lower() in noise_markers or len(text) < 2:
            logger.debug("Discarding noise transcript: %r (%.2fs)", text, elapsed)
            return None

        audio_duration = len(audio) / sample_rate
        logger.info(
            "Transcribed %.1fs audio in %.2fs (%.1fx realtime): %r",
            audio_duration,
            elapsed,
            audio_duration / elapsed if elapsed > 0 else 0,
            text,
        )
        return text
