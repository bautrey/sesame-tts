from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8880
    log_level: str = "info"

    # Model
    model_id: str = "mlx-community/csm-1b"
    cache_dir: Path = Path.home() / ".cache" / "sesame-tts"

    # Defaults
    default_voice: str = "conversational_b"
    default_format: str = "mp3"
    max_input_length: int = 4096

    # Streaming
    stream_chunk_size: int = 4096
    default_elevenlabs_format: str = "pcm_24000"
    max_queue_depth: int = 3
    warmup_on_start: bool = True

    # Presets directory
    presets_dir: Path = Path(__file__).parent / "presets"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
