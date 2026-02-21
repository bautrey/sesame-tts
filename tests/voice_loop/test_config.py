"""Tests for VoiceLoopSettings configuration loading and resolution."""

import json
from pathlib import Path

import pytest
import yaml

from voice_loop.config import VoiceLoopSettings


class TestDefaultValues:
    """Verify all default field values are correct."""

    def test_defaults(self):
        settings = VoiceLoopSettings()
        assert settings.mic_device is None
        assert settings.speaker_device is None
        assert settings.push_to_talk_hotkey == "cmd+shift+space"
        assert settings.sample_rate == 16000
        assert settings.vad_silence_threshold_ms == 1000
        assert settings.vad_aggressiveness == 3
        assert settings.max_capture_duration_s == 30
        assert settings.whisper_model == "mlx-community/whisper-large-v3-turbo"
        assert settings.gateway_url == "ws://127.0.0.1:18789"
        assert settings.gateway_token == "gateway-token"
        assert settings.gateway_timeout_s == 10
        assert settings.gateway_reconnect_max_s == 30
        assert settings.tts_url == "http://localhost:8880"
        assert settings.tts_voice == "conversationalB"
        assert settings.tts_format == "pcm_24000"
        assert settings.interrupt_energy_threshold == 0.02
        assert settings.interrupt_consecutive_frames == 3
        assert settings.log_level == "info"
        assert settings.log_file == "~/.cache/sesame-voice/voice-loop.log"
        assert settings.run_as_menu_bar is True


class TestYamlOverride:
    """Verify YAML file overrides work correctly."""

    def test_yaml_overrides_defaults(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "sample_rate": 44100,
                    "vad_aggressiveness": 1,
                    "tts_voice": "custom_voice",
                    "log_level": "debug",
                }
            )
        )
        settings = VoiceLoopSettings.from_yaml(path=config_file)
        assert settings.sample_rate == 44100
        assert settings.vad_aggressiveness == 1
        assert settings.tts_voice == "custom_voice"
        assert settings.log_level == "debug"
        # Non-overridden values remain default
        assert settings.gateway_url == "ws://127.0.0.1:18789"

    def test_missing_yaml_returns_defaults(self, tmp_path: Path, monkeypatch):
        # Monkeypatch home to tmp_path so no real openclaw.json is found
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("VOICE_LOOP_GATEWAY_TOKEN", raising=False)
        missing_path = tmp_path / "nonexistent.yaml"
        settings = VoiceLoopSettings.from_yaml(path=missing_path)
        assert settings.sample_rate == 16000
        assert settings.gateway_token == "gateway-token"

    def test_empty_yaml_returns_defaults(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        settings = VoiceLoopSettings.from_yaml(path=config_file)
        assert settings.sample_rate == 16000


class TestEnvVarOverride:
    """Verify environment variable overrides work correctly."""

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("VOICE_LOOP_SAMPLE_RATE", "48000")
        monkeypatch.setenv("VOICE_LOOP_LOG_LEVEL", "debug")
        settings = VoiceLoopSettings()
        assert settings.sample_rate == 48000
        assert settings.log_level == "debug"

    def test_env_var_overrides_yaml(self, tmp_path: Path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"sample_rate": 44100}))
        monkeypatch.setenv("VOICE_LOOP_SAMPLE_RATE", "48000")
        settings = VoiceLoopSettings.from_yaml(path=config_file)
        assert settings.sample_rate == 48000

    def test_gateway_token_env_overrides_all(self, tmp_path: Path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"gateway_token": "yaml-token"}))
        monkeypatch.setenv("VOICE_LOOP_GATEWAY_TOKEN", "env-token")
        settings = VoiceLoopSettings.from_yaml(path=config_file)
        assert settings.gateway_token == "env-token"


class TestOpenclawTokenReading:
    """Verify openclaw.json token auto-reading."""

    def test_reads_nested_config_token(self, tmp_path: Path, monkeypatch):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        openclaw_file = openclaw_dir / "openclaw.json"
        openclaw_file.write_text(
            json.dumps({"config": {"gateway": {"auth": {"token": "oc-token-123"}}}})
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        token = VoiceLoopSettings._read_openclaw_token()
        assert token == "oc-token-123"

    def test_reads_flat_token(self, tmp_path: Path, monkeypatch):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        openclaw_file = openclaw_dir / "openclaw.json"
        openclaw_file.write_text(json.dumps({"token": "flat-token"}))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        token = VoiceLoopSettings._read_openclaw_token()
        assert token == "flat-token"

    def test_missing_file_returns_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        token = VoiceLoopSettings._read_openclaw_token()
        assert token is None

    def test_invalid_json_returns_none(self, tmp_path: Path, monkeypatch):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        openclaw_file = openclaw_dir / "openclaw.json"
        openclaw_file.write_text("not valid json{{{")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        token = VoiceLoopSettings._read_openclaw_token()
        assert token is None

    def test_openclaw_token_used_when_no_yaml_or_env(self, tmp_path: Path, monkeypatch):
        # Set up openclaw.json
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        openclaw_file = openclaw_dir / "openclaw.json"
        openclaw_file.write_text(json.dumps({"token": "auto-token"}))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Ensure no env var is set
        monkeypatch.delenv("VOICE_LOOP_GATEWAY_TOKEN", raising=False)

        # Use a missing YAML path
        missing_yaml = tmp_path / "nonexistent.yaml"
        settings = VoiceLoopSettings.from_yaml(path=missing_yaml)
        assert settings.gateway_token == "auto-token"

    def test_yaml_token_overrides_openclaw(self, tmp_path: Path, monkeypatch):
        # Set up openclaw.json
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        openclaw_file = openclaw_dir / "openclaw.json"
        openclaw_file.write_text(json.dumps({"token": "auto-token"}))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("VOICE_LOOP_GATEWAY_TOKEN", raising=False)

        # YAML explicitly sets gateway_token
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"gateway_token": "yaml-token"}))
        settings = VoiceLoopSettings.from_yaml(path=config_file)
        assert settings.gateway_token == "yaml-token"
