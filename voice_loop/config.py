"""Centralized configuration with YAML file loading, env var overrides, and validation."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
import yaml

CONFIG_PATH = Path.home() / ".config" / "sesame-voice" / "config.yaml"


class VoiceLoopSettings(BaseSettings):
    """Voice loop configuration with layered resolution: env > YAML > openclaw.json > defaults."""

    # Audio devices
    mic_device: Optional[str] = None
    speaker_device: Optional[str] = None

    # Push-to-talk
    push_to_talk_hotkey: str = "cmd+shift+space"

    # Speech capture
    sample_rate: int = 16000
    vad_silence_threshold_ms: int = 1000
    vad_aggressiveness: int = 3
    max_capture_duration_s: int = 30

    # Transcription
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # Gateway
    gateway_url: str = "http://127.0.0.1:18789"
    gateway_token: str = "gateway-token"
    gateway_timeout_s: int = 10
    gateway_reconnect_max_s: int = 30

    # TTS
    tts_url: str = "http://localhost:8880"
    tts_voice: str = "conversationalB"
    tts_format: str = "pcm_24000"

    # Interrupt
    interrupt_energy_threshold: float = 0.02
    interrupt_consecutive_frames: int = 3

    # Logging
    log_level: str = "info"
    log_file: Optional[str] = "~/.cache/sesame-voice/voice-loop.log"

    # Process
    run_as_menu_bar: bool = True

    model_config = {"env_prefix": "VOICE_LOOP_"}

    @classmethod
    def _read_openclaw_token(cls) -> Optional[str]:
        """Read the gateway auth token from ~/.openclaw/openclaw.json.

        Returns the token string if found, or None if the file is missing
        or the token path does not exist in the JSON structure.
        """
        import json

        openclaw_path = Path.home() / ".openclaw" / "openclaw.json"
        if not openclaw_path.exists():
            return None
        try:
            with open(openclaw_path) as f:
                data = json.load(f)
            # Navigate nested path: try common paths in order of likelihood
            for key_path in [
                ("config", "gateway", "auth", "token"),
                ("gateway", "auth", "token"),
                ("auth", "token"),
                ("token",),
            ]:
                node = data
                for key in key_path:
                    if isinstance(node, dict) and key in node:
                        node = node[key]
                    else:
                        node = None
                        break
                if isinstance(node, str) and node:
                    return node
        except (json.JSONDecodeError, OSError):
            pass
        return None

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> "VoiceLoopSettings":
        """Load settings from YAML file, then apply env var overrides.

        Gateway token resolution order (highest priority wins):
          1. VOICE_LOOP_GATEWAY_TOKEN env var
          2. gateway_token in config YAML
          3. Token auto-read from ~/.openclaw/openclaw.json
          4. Hardcoded default ("gateway-token")
        """
        import os

        yaml_values = {}
        if path.exists():
            with open(path) as f:
                yaml_values = yaml.safe_load(f) or {}

        # Auto-read gateway token from openclaw.json if not set in YAML or env
        if "gateway_token" not in yaml_values and not os.environ.get(
            "VOICE_LOOP_GATEWAY_TOKEN"
        ):
            openclaw_token = cls._read_openclaw_token()
            if openclaw_token:
                yaml_values["gateway_token"] = openclaw_token

        # Filter out YAML keys where an env var override exists,
        # so env vars take priority (kwargs would otherwise override env vars)
        env_prefix = "VOICE_LOOP_"
        overrides = {
            k: v
            for k, v in yaml_values.items()
            if not os.environ.get(f"{env_prefix}{k.upper()}")
        }

        return cls(**overrides)
