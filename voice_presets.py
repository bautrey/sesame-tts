import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VoicePreset:
    name: str
    voice_key: str
    speaker_id: int
    temperature: float = 0.9
    top_k: int = 50
    max_audio_length_ms: float = 90_000
    description: str = ""
    speaker_embedding_path: str | None = None


class VoicePresetManager:
    def __init__(self, presets_dir: Path):
        self.presets_dir = presets_dir
        self.presets: dict[str, VoicePreset] = {}

    def load_presets(self) -> None:
        """Load all .json preset files from the presets directory."""
        for path in self.presets_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            preset = VoicePreset(**data)
            self.presets[preset.name] = preset

    def get(self, name: str) -> VoicePreset | None:
        return self.presets.get(name)

    def list_names(self) -> list[str]:
        return list(self.presets.keys())
