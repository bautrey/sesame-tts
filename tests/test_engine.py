"""Integration tests for TTSEngine — requires model to be downloaded."""

import numpy as np
import pytest

from config import Settings
from tts_engine import TTSEngine
from voice_presets import VoicePresetManager


@pytest.fixture(scope="module")
def engine():
    import asyncio
    import os

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    settings = Settings()
    eng = TTSEngine(settings)
    asyncio.get_event_loop().run_until_complete(eng.load_model())
    return eng


@pytest.fixture(scope="module")
def preset_manager():
    settings = Settings()
    mgr = VoicePresetManager(settings.presets_dir)
    mgr.load_presets()
    return mgr


def test_model_loads(engine):
    assert engine.model_loaded is True
    assert engine.model is not None


def test_health_reports_loaded(engine):
    health = engine.health()
    assert health["model_loaded"] is True
    assert health["sample_rate"] == 24000


def test_generate_produces_audio(engine, preset_manager):
    preset = preset_manager.get("conversational")
    assert preset is not None

    audio_np, sample_rate = engine.generate("Hello.", preset)
    assert isinstance(audio_np, np.ndarray)
    assert audio_np.dtype == np.float32
    assert len(audio_np) > 0
    assert sample_rate == 24000


def test_generate_with_voice_b(engine, preset_manager):
    preset = preset_manager.get("conversational_b")
    assert preset is not None

    audio_np, sample_rate = engine.generate("Test voice B.", preset)
    assert len(audio_np) > 0
    assert sample_rate == 24000
