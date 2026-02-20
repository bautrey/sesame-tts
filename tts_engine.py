import logging
import threading
import time
from typing import Generator

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_lm.sample_utils import make_sampler

from config import Settings
from voice_presets import VoicePreset

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.model_loaded = False
        self._load_time: float = 0

    async def load_model(self) -> None:
        """Load CSM-1B model via mlx-audio. Runs in executor to avoid blocking."""
        import asyncio

        loop = asyncio.get_event_loop()
        start = time.time()
        self.model = await loop.run_in_executor(
            None, load_model, self.settings.model_id
        )
        # Point prompt downloads at the mlx-community repo (ungated) instead of
        # the default sesame/csm-1b (gated, requires auth)
        self.model.tokenizer_repo = self.settings.model_id
        self._load_time = time.time() - start
        self.model_loaded = True
        logger.info(
            "Model loaded in %.1fs: %s", self._load_time, self.settings.model_id
        )

    def generate(self, text: str, preset: VoicePreset) -> tuple[np.ndarray, int]:
        """Generate audio from text using a voice preset.

        Returns:
            Tuple of (audio_numpy_float32, sample_rate)
        """
        sampler = make_sampler(temp=preset.temperature, top_k=preset.top_k)

        results = []
        for result in self.model.generate(
            text=text,
            voice=preset.voice_key,
            speaker=preset.speaker_id,
            sampler=sampler,
            max_audio_length_ms=preset.max_audio_length_ms,
            voice_match=True,
        ):
            results.append(result)

        if not results:
            raise RuntimeError("Model produced no audio output")

        if len(results) == 1:
            audio_np = np.array(results[0].audio)
        else:
            audio = mx.concatenate([r.audio for r in results], axis=0)
            audio_np = np.array(audio)

        return audio_np, results[0].sample_rate

    def generate_stream(
        self,
        text: str,
        preset: VoicePreset,
        interrupt_event: threading.Event,
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """Generate audio from text, yielding (audio_np, sample_rate) for each frame.

        This is the streaming variant of generate(). Instead of collecting all
        GenerationResult objects and concatenating, it yields each one individually
        so the streaming pipeline can deliver audio chunks as they are produced.

        In the sentence-splitting pipeline, this method is called once per sentence.
        Each call produces one or more (audio_np, sample_rate) tuples. For a single
        sentence (no newlines), model.generate() typically yields exactly one
        GenerationResult.

        The interrupt_event is checked between each frame yield. If set, generation
        stops immediately and the generator returns. The interrupt_event is shared
        across all sentences in a request -- setting it stops the entire pipeline.

        Args:
            text: The text to generate audio for (typically one sentence)
            preset: Voice preset with generation parameters
            interrupt_event: threading.Event checked between yields; if set, stop

        Yields:
            Tuple of (audio_numpy_float32, sample_rate) for each generated frame
        """
        sampler = make_sampler(temp=preset.temperature, top_k=preset.top_k)

        frame_count = 0
        for result in self.model.generate(
            text=text,
            voice=preset.voice_key,
            speaker=preset.speaker_id,
            sampler=sampler,
            max_audio_length_ms=preset.max_audio_length_ms,
            voice_match=True,
        ):
            # Check interrupt between frames
            if interrupt_event.is_set():
                logger.info(
                    "Generation interrupted after %d frames: voice=%s text=%r",
                    frame_count, preset.voice_key, text[:50],
                )
                break

            audio_np = np.array(result.audio)
            yield (audio_np, result.sample_rate)
            frame_count += 1

        logger.debug("generate_stream completed: %d frames for text=%r", frame_count, text[:50])

    def health(self) -> dict:
        """Return model health status."""
        return {
            "model_loaded": self.model_loaded,
            "model_id": self.settings.model_id,
            "sample_rate": self.model.sample_rate if self.model else None,
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2) if self.model else None,
        }
