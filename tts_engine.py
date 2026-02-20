import logging
import time

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

    def health(self) -> dict:
        """Return model health status."""
        return {
            "model_loaded": self.model_loaded,
            "model_id": self.settings.model_id,
            "sample_rate": self.model.sample_rate if self.model else None,
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2) if self.model else None,
        }
