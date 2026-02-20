# TRD: Sesame TTS -- OpenAI-Compatible Local TTS Server

**Author:** Burke
**Created:** 2026-02-20
**Status:** Draft
**Version:** 1.1
**PRD Reference:** [docs/PRD.md](./PRD.md) v1.1

---

## 1. Overview

### Project Summary

A local FastAPI server wrapping the Sesame CSM-1B speech model via the `mlx-audio` library (Apple Silicon native inference). The server exposes an OpenAI-compatible `/v1/audio/speech` endpoint on `localhost:8880`, enabling any OpenAI TTS client -- including OpenClaw -- to switch to local generation with a config change only.

### Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Runtime | Python 3.10+ | Matching mlx-audio requirements |
| Package Manager | `uv` | Fast dependency resolution, uses `pyproject.toml` natively |
| Web Framework | FastAPI + uvicorn | Async, OpenAPI docs auto-generated |
| ML Inference | `mlx-audio` 0.3.x (`Blaizzy/mlx-audio`) | CSM-1B ported to MLX natively |
| ML Framework | MLX (Apple Silicon) | No PyTorch in inference path |
| Audio I/O | ffmpeg (subprocess) | All formats (WAV, MP3, Opus, FLAC) encoded via ffmpeg |
| Config | pydantic-settings + python-dotenv | Type-safe settings with `.env` support |
| Deployment | launchd plist | Background service on macOS |
| Hardware | 2025 Mac Studio, Apple M3 Ultra, 256GB unified memory, macOS Tahoe 26.3 |

### Key Architectural Decision: Use mlx-audio Directly

The `mlx-audio` library already provides:
- `Model` class with a `.generate()` method that yields `GenerationResult` objects
- Built-in voice prompts (`conversational_a`, `conversational_b`) with HuggingFace auto-download
- Built-in watermarking (responsible AI compliance)
- Streaming decoder support (future v1.1)

Our server is a thin wrapper that translates OpenAI API requests into mlx-audio calls and manages voice presets, configuration, and error handling. We do NOT reimplement any model loading or inference logic. Audio format encoding is handled entirely by ffmpeg subprocess calls, giving us full control over codec parameters for all output formats.

---

## 2. System Architecture

### Component Diagram

```
+--------------------------------------------------+
|  OpenClaw / Any OpenAI TTS Client                |
+--------------------------------------------------+
          |  POST /v1/audio/speech
          |  GET  /v1/models
          |  GET  /health
          v
+--------------------------------------------------+
|  FastAPI Server  (server.py)                     |
|  - Request validation (Pydantic models)          |
|  - OpenAI-compatible error responses             |
|  - Request queuing (asyncio.Semaphore)           |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  TTS Engine  (tts_engine.py)                     |
|  - Model lifecycle (load, warmup, health)        |
|  - Inference wrapper around mlx-audio Model      |
|  - Voice preset resolution                       |
+--------------------------------------------------+
          |                        |
          v                        v
+-------------------+  +---------------------------+
| mlx-audio Model   |  | Voice Presets             |
| (.generate())     |  | (voice_presets.py)        |
| - CSM-1B weights  |  | - JSON config files       |
| - Mimi codec      |  | - Speaker prompt mapping  |
| - Watermarking    |  +---------------------------+
+-------------------+
          |
          v
+--------------------------------------------------+
|  Audio Converter  (audio_converter.py)           |
|  - ffmpeg subprocess for ALL format conversion   |
|  - WAV, MP3, FLAC, Opus via pipe stdin/stdout    |
+--------------------------------------------------+
          |
          v
      Raw audio bytes (Content-Type: audio/*)
```

### Data Flow

1. Client sends `POST /v1/audio/speech` with JSON body
2. FastAPI validates request via Pydantic model
3. Semaphore ensures serial inference (single-model, one request at a time)
4. `tts_engine` resolves voice preset to speaker prompt + generation parameters
5. `tts_engine` calls `model.generate(text, voice=voice_key, **params)` on the mlx-audio Model
6. Generator yields `GenerationResult` with `audio` (mx.array) and `sample_rate`
7. `audio_converter` converts mx.array to numpy int16 and pipes through ffmpeg for the requested format
8. Server returns raw audio bytes with appropriate Content-Type header

### Module Breakdown

The project uses a flat root layout -- all application modules live at the repository root for simplicity. This is intentional; there is no `src/` or package directory.

```
sesame-tts/
  server.py              # FastAPI application, routes, middleware
  tts_engine.py          # MLX model loading, inference wrapper
  audio_converter.py     # Format conversion (WAV/MP3/FLAC/Opus) via ffmpeg
  voice_presets.py       # Preset loading and management
  config.py              # Settings via pydantic-settings
  errors.py              # OpenAI-compatible error responses
  presets/               # Voice preset JSON files
    conversational.json
    conversational_b.json
  tests/                 # Integration tests (happy path)
    test_engine.py
    test_server.py
  upstream/              # Reference files (not used in production)
    generator.py
    models.py
    run_csm.py
    watermarking.py
    setup.py
    requirements.txt
  docs/
    PRD.md
    TRD.md
  com.burkestudio.sesame-tts.plist   # launchd service definition
  .env.example           # Example configuration
  pyproject.toml         # Project metadata and dependencies (used by uv)
  README.md              # Updated with usage and OpenClaw integration
```

---

## 3. Module Design

### 3.1 `server.py` -- FastAPI Application

**Responsibilities:** HTTP routing, request validation, response formatting, request queuing, application lifecycle.

```python
# Key structures

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

# Application lifespan: load model on startup, cleanup on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    engine = TTSEngine(settings)
    await engine.load_model()
    app.state.engine = engine
    yield
    # Shutdown: cleanup
    del app.state.engine

app = FastAPI(title="Sesame TTS", lifespan=lifespan)

# Request queuing: serial inference via semaphore
inference_semaphore = asyncio.Semaphore(1)
```

**Routes:**

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/v1/audio/speech` | `create_speech()` | TTS generation |
| GET | `/v1/models` | `list_models()` | Model listing (OpenAI format) |
| GET | `/health` | `health_check()` | Server health + model status |

**Request Model:**

```python
class SpeechRequest(BaseModel):
    model: str                                    # Accepted but ignored (single model)
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str                                    # Preset name
    response_format: str = "mp3"                  # mp3, opus, wav, flac
    speed: float = Field(default=1.0, ge=0.5, le=2.0)  # Accepted, logged, not applied in v1
```

**Middleware:**
- Request logging: timestamp, input length, voice, format, generation time
- CORS: disabled by default (localhost only), configurable

**Response Headers:**
- `Content-Type`: `audio/mpeg`, `audio/opus`, `audio/wav`, `audio/flac`
- `Content-Disposition`: `attachment; filename=speech.<format>`
- `X-Generation-Time`: generation duration in seconds (for monitoring)

### 3.2 `tts_engine.py` -- MLX Model Loading and Inference

**Responsibilities:** Model lifecycle management, inference execution, voice preset integration.

```python
from mlx_audio.tts import load as load_tts_model

class TTSEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.model_loaded = False
        self._load_time: float = 0

    async def load_model(self) -> None:
        """Load CSM-1B model via mlx-audio. Runs in executor to avoid blocking."""
        # mlx_audio.tts.load() handles:
        #   - HuggingFace download + caching
        #   - Model config detection (csm -> sesame)
        #   - Weight loading into MLX arrays
        #   - Mimi audio codec initialization
        #   - Watermarker initialization
        self.model = load_tts_model("sesame/csm-1b")
        self.model_loaded = True

    def generate(self, text: str, preset: VoicePreset) -> tuple[mx.array, int]:
        """
        Generate audio from text using a voice preset.

        The mlx-audio Model.generate() method:
        - Accepts: text, voice (str key for built-in prompts), speaker (int),
                   context (List[Segment]), sampler, max_audio_length_ms,
                   ref_audio, ref_text, stream, voice_match
        - Yields: GenerationResult objects with .audio (mx.array) and .sample_rate

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Build generation kwargs from preset
        sampler = make_sampler(temp=preset.temperature, top_k=preset.top_k)

        results = []
        for result in self.model.generate(
            text=text,
            voice=preset.voice_key,      # e.g. "conversational_a"
            speaker=preset.speaker_id,
            sampler=sampler,
            max_audio_length_ms=preset.max_audio_length_ms,
            voice_match=True,
        ):
            results.append(result)

        # Concatenate if multiple segments (newline splitting)
        if len(results) == 1:
            return results[0].audio, results[0].sample_rate
        else:
            audio = mx.concatenate([r.audio for r in results], axis=0)
            return audio, results[0].sample_rate

    def health(self) -> dict:
        """Return model health status."""
        return {
            "model_loaded": self.model_loaded,
            "model_id": self.settings.model_id,
            "sample_rate": self.model.sample_rate if self.model else None,
            "peak_memory_gb": mx.get_peak_memory() / 1e9 if self.model else None,
        }
```

**mlx-audio API Details (from source analysis):**

The `mlx_audio.tts.models.sesame.Model` class:
- Constructor loads: Sesame backbone+decoder (LlamaModel), Mimi audio codec, Llama-3.2-1B tokenizer, watermarker
- `model.sample_rate` property returns 24000 (from Mimi codec)
- `model.generate()` is a **generator** that yields `GenerationResult` objects
- `model.default_speaker_prompt(voice)` loads prompt WAV+text from HuggingFace for the given voice key
- Built-in voice keys: `"conversational_a"`, `"conversational_b"` (speaker prompt WAVs from `sesame/csm-1b` repo)
- Default sampler: `make_sampler(temp=0.9, top_k=50)`
- `model.generate()` splits text on `\n+` by default; pass `split_pattern=None` to disable
- `voice_match=True` (default) prepends the prompt text to generation text for better voice consistency

**GenerationResult fields:**
- `audio`: `mx.array` -- raw audio samples
- `sample_rate`: `int` -- always 24000
- `processing_time_seconds`: `float`
- `peak_memory_usage`: `float` (GB)
- `real_time_factor`: `float`
- `token_count`: `int`

### 3.3 `audio_converter.py` -- Format Conversion

**Responsibilities:** Convert mlx.array audio to the requested output format bytes. All encoding is handled by ffmpeg subprocess, giving us full control over codec parameters for every format.

```python
import io
import subprocess
import numpy as np
import mlx.core as mx

SUPPORTED_FORMATS = {"mp3", "opus", "wav", "flac"}

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "wav": "audio/wav",
    "flac": "audio/flac",
}

# ffmpeg commands for each format (input: raw s16le PCM via stdin)
FFMPEG_ARGS = {
    "wav":  ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0", "-f", "wav", "pipe:1"],
    "mp3":  ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"],
    "opus": ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k", "-f", "opus", "pipe:1"],
    "flac": ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0", "-c:a", "flac", "-f", "flac", "pipe:1"],
}


def convert_audio(audio: mx.array, sample_rate: int, format: str) -> bytes:
    """Convert MLX audio array to the requested format bytes.

    Pipeline: mx.array -> numpy float32 -> int16 PCM -> ffmpeg -> encoded bytes
    All formats use ffmpeg subprocess for consistent encoding control.
    """
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}")

    # Convert mx.array to numpy int16 PCM
    audio_np = np.array(audio.tolist(), dtype=np.float32)
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

    # Pipe through ffmpeg
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + FFMPEG_ARGS[format]
    result = subprocess.run(cmd, input=audio_int16.tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {format} encoding failed: {result.stderr.decode()}")
    return result.stdout
```

**Format Specifications:**

| Format | Encoding | Bitrate | Sample Rate | Channels |
|--------|----------|---------|-------------|----------|
| WAV | PCM 16-bit | N/A (lossless) | 24kHz | Mono |
| MP3 | LAME via ffmpeg | 64kbps | 24kHz | Mono |
| Opus | libopus via ffmpeg | 32kbps | 24kHz (resampled to 48kHz internally by Opus) | Mono |
| FLAC | FLAC via ffmpeg | N/A (lossless) | 24kHz | Mono |

### 3.4 `voice_presets.py` -- Preset Loading and Management

**Responsibilities:** Load, validate, and resolve voice presets from JSON config files.

```python
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class VoicePreset:
    name: str                          # Preset name (e.g., "conversational")
    voice_key: str                     # mlx-audio voice key (e.g., "conversational_a")
    speaker_id: int                    # Speaker ID for generation
    temperature: float = 0.9          # Sampling temperature
    top_k: int = 50                    # Top-k sampling
    max_audio_length_ms: float = 90_000  # Max generation length
    description: str = ""             # Human-readable description
    # Future: speaker_embedding_path for voice cloning
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
```

**v1 Preset Files:**

`presets/conversational.json`:
```json
{
    "name": "conversational",
    "voice_key": "conversational_a",
    "speaker_id": 0,
    "temperature": 0.9,
    "top_k": 50,
    "max_audio_length_ms": 90000,
    "description": "Natural, warm conversational voice (default)"
}
```

`presets/conversational_b.json`:
```json
{
    "name": "conversational_b",
    "voice_key": "conversational_b",
    "speaker_id": 1,
    "temperature": 0.9,
    "top_k": 50,
    "max_audio_length_ms": 90000,
    "description": "Second conversational voice, slightly different character"
}
```

**Extensibility:** Adding a new voice requires only adding a JSON file to `presets/`. The `voice_key` field maps to mlx-audio's built-in speaker prompt system. For custom voices (post-v1), the `speaker_embedding_path` field is reserved.

### 3.5 `config.py` -- Settings via pydantic-settings

**Responsibilities:** Centralized configuration with environment variable + `.env` file support.

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8880
    log_level: str = "info"

    # Model
    model_id: str = "sesame/csm-1b"
    cache_dir: Path = Path.home() / ".cache" / "sesame-tts"

    # Defaults
    default_voice: str = "conversational"
    default_format: str = "mp3"
    max_input_length: int = 4096

    # Presets directory
    presets_dir: Path = Path(__file__).parent / "presets"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

**Environment Variable Mapping:**

| Variable | Settings Field | Default |
|----------|---------------|---------|
| `HOST` | `host` | `0.0.0.0` |
| `PORT` | `port` | `8880` |
| `MODEL_ID` | `model_id` | `sesame/csm-1b` |
| `CACHE_DIR` | `cache_dir` | `~/.cache/sesame-tts` |
| `DEFAULT_VOICE` | `default_voice` | `conversational` |
| `DEFAULT_FORMAT` | `default_format` | `mp3` |
| `MAX_INPUT_LENGTH` | `max_input_length` | `4096` |
| `LOG_LEVEL` | `log_level` | `info` |
| `PRESETS_DIR` | `presets_dir` | `./presets` |

### 3.6 `errors.py` -- OpenAI-Compatible Error Responses

**Responsibilities:** Standardized error formatting matching the OpenAI API error schema.

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class TTSError(Exception):
    def __init__(self, message: str, error_type: str, code: str, status_code: int = 400):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.status_code = status_code

def openai_error_response(message: str, error_type: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )

# Exception handlers registered in server.py:

# TTSError -> OpenAI error JSON
# RequestValidationError -> 400 invalid_request_error
# MemoryError / mx OOM -> 503 server_error / insufficient_memory
# General Exception -> 500 server_error / internal_error
```

**Error Mapping:**

| Condition | HTTP Status | `type` | `code` |
|-----------|-------------|--------|--------|
| Empty input | 400 | `invalid_request_error` | `invalid_input` |
| Input too long | 400 | `invalid_request_error` | `input_too_long` |
| Unknown voice | 400 | `invalid_request_error` | `invalid_voice` |
| Invalid format | 400 | `invalid_request_error` | `invalid_format` |
| Model not loaded | 503 | `server_error` | `model_not_ready` |
| OOM during generation | 503 | `server_error` | `insufficient_memory` |
| ffmpeg failure | 500 | `server_error` | `conversion_failed` |
| General server error | 500 | `server_error` | `internal_error` |

---

## 4. API Specification

### POST /v1/audio/speech

Generate speech audio from text input.

**Request:**
```
POST /v1/audio/speech HTTP/1.1
Content-Type: application/json

{
    "model": "csm-1b",
    "input": "Hello Burke, this is your local TTS server.",
    "voice": "conversational",
    "response_format": "mp3",
    "speed": 1.0
}
```

| Field | Type | Required | Default | Validation |
|-------|------|----------|---------|------------|
| `model` | string | Yes | -- | Accepted, not validated (single model) |
| `input` | string | Yes | -- | 1-4096 characters, non-empty after strip |
| `voice` | string | Yes | -- | Must match a loaded preset name |
| `response_format` | string | No | `"mp3"` | One of: `mp3`, `opus`, `wav`, `flac` |
| `speed` | float | No | `1.0` | Range 0.5-2.0 (accepted, logged, not applied in v1) |

**Response:**
```
HTTP/1.1 200 OK
Content-Type: audio/mpeg
Content-Disposition: attachment; filename=speech.mp3
X-Generation-Time: 1.23

<raw audio bytes>
```

**Error Response:**
```json
{
    "error": {
        "message": "Voice 'unknown_voice' not found. Available voices: conversational, conversational_b",
        "type": "invalid_request_error",
        "code": "invalid_voice"
    }
}
```

### GET /v1/models

List available models in OpenAI format.

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "csm-1b",
            "object": "model",
            "created": 1740000000,
            "owned_by": "sesame"
        }
    ]
}
```

### GET /health

Server health check.

**Response (healthy):**
```json
{
    "status": "ok",
    "model": {
        "id": "sesame/csm-1b",
        "loaded": true,
        "sample_rate": 24000,
        "peak_memory_gb": 2.1
    },
    "ffmpeg": true,
    "voices": ["conversational", "conversational_b"],
    "uptime_seconds": 3600
}
```

**Response (degraded):**
```json
{
    "status": "degraded",
    "model": {
        "id": "sesame/csm-1b",
        "loaded": false,
        "error": "Failed to load model: ..."
    },
    "ffmpeg": true,
    "voices": []
}
```

---

## 5. Master Task List

Each task has a unique ID, estimated hours, dependencies, and completion checkbox.

### Phase 1: Project Setup and MLX Integration

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-001 | Create project structure: `pyproject.toml`, `uv` virtual environment setup (`uv sync`) | 1h | -- | [ ] |
| T-002 | Move upstream files (`generator.py`, `models.py`, `run_csm.py`, `watermarking.py`, `setup.py`, original `requirements.txt`) to `upstream/` directory | 0.5h | -- | [ ] |
| T-003 | Implement `config.py` with pydantic-settings, `.env` support, and `.env.example` | 1h | T-001 | [ ] |
| T-004 | Implement `tts_engine.py`: model loading via `mlx_audio.tts.load()`, health check method | 2h | T-001, T-003 | [ ] |
| T-005 | Validate model loads and generates audio: standalone script that loads CSM-1B and generates a test WAV | 1h | T-004 | [ ] |
| T-006 | Implement `voice_presets.py`: preset loading from JSON, validation, listing | 1.5h | T-001 | [ ] |
| T-007 | Create v1 preset JSON files: `conversational.json`, `conversational_b.json` | 0.5h | T-006 | [ ] |
| T-008 | Integrate voice presets into `tts_engine.py`: preset resolution, sampler configuration | 1h | T-004, T-006, T-007 | [ ] |

### Phase 2: FastAPI Server and OpenAI-Compatible API

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-009 | Implement `server.py` skeleton: FastAPI app with lifespan, Pydantic request models | 1.5h | T-003 | [ ] |
| T-010 | Implement `POST /v1/audio/speech` route: request validation, engine integration, WAV response | 2h | T-008, T-009 | [ ] |
| T-011 | Implement request queuing via `asyncio.Semaphore(1)` for serial inference | 0.5h | T-010 | [ ] |
| T-012 | Implement `GET /v1/models` route: OpenAI-format model listing | 0.5h | T-009 | [ ] |
| T-013 | Implement `GET /health` route: model status, ffmpeg check, voice listing, uptime | 1h | T-009, T-004 | [ ] |
| T-014 | Add request logging middleware: timestamp, input length, voice, format, generation time | 1h | T-010 | [ ] |
| T-015 | End-to-end smoke test: curl generates playable WAV from running server | 0.5h | T-010 | [ ] |

### Phase 3: Audio Format Conversion

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-016 | Implement `audio_converter.py`: ffmpeg-based conversion for all formats (WAV, MP3, Opus, FLAC) | 2h | T-001 | [ ] |
| T-017 | Integrate `audio_converter` into `/v1/audio/speech` route: format parameter handling | 1h | T-010, T-016 | [ ] |
| T-018 | Validate all four formats produce playable output files via curl | 1h | T-017 | [ ] |

### Phase 4: Error Handling, Configuration, and Deployment

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-019 | Implement `errors.py`: TTSError class, OpenAI error response helpers | 1h | T-001 | [ ] |
| T-020 | Register exception handlers in `server.py`: TTSError, ValidationError, MemoryError, general | 1.5h | T-009, T-019 | [ ] |
| T-021 | Implement input validation edge cases: empty input, too-long input, unknown voice, invalid format | 1h | T-020 | [ ] |
| T-022 | Implement structured logging: file logging to `CACHE_DIR/server.log`, configurable level, request metrics | 1.5h | T-003, T-014 | [ ] |
| T-023 | Create launchd plist: auto-start, restart on crash, log file paths, PID file | 1h | T-022 | [ ] |
| T-024 | Write install/deploy script: `uv sync`, launchd load/unload commands | 1h | T-023 | [ ] |
| T-025 | Validate launchd deployment: service start, stop, restart, crash recovery | 1h | T-024 | [ ] |

### Phase 5: Testing, Documentation, and OpenClaw Integration

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-026 | Integration test: model loads and generates audio (`tests/test_engine.py`) | 2h | T-008 | [ ] |
| T-027 | Integration test: API routes happy path -- all endpoints, all formats (`tests/test_server.py`) | 2h | T-017, T-020 | [ ] |
| T-028 | OpenClaw integration test: configure OpenClaw to use local server, generate voice note | 1h | T-017 | [ ] |
| T-029 | Update README.md: installation (uv), usage, OpenClaw configuration, voice presets, API reference | 2h | T-025 | [ ] |

---

## 6. Sprint Planning

### Phase 1: Project Setup and MLX Integration (Days 1-2)

**Goal:** Get CSM-1B loading and generating audio via mlx-audio on the Mac Studio.

**Tasks:** T-001, T-002, T-003, T-004, T-005, T-006, T-007, T-008
**Estimated effort:** 8.5 hours
**Exit criteria:** Standalone script successfully generates a WAV file using mlx-audio with a voice preset.

### Phase 2: FastAPI Server and OpenAI-Compatible API (Days 2-3)

**Goal:** Working HTTP server with OpenAI-compatible speech endpoint returning WAV audio.

**Tasks:** T-009, T-010, T-011, T-012, T-013, T-014, T-015
**Estimated effort:** 7 hours
**Exit criteria:** `curl -X POST localhost:8880/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"csm-1b","input":"Hello Burke.","voice":"conversational"}' --output test.wav` produces playable audio.

### Phase 3: Audio Format Conversion (Day 3-4)

**Goal:** All four audio output formats working via ffmpeg.

**Tasks:** T-016, T-017, T-018
**Estimated effort:** 4 hours
**Exit criteria:** All four formats (MP3/Opus/WAV/FLAC) produce playable files. Voice switching works.

### Phase 4: Error Handling, Configuration, and Deployment (Day 4-5)

**Goal:** Production-ready error handling, logging, and launchd service.

**Tasks:** T-019, T-020, T-021, T-022, T-023, T-024, T-025
**Estimated effort:** 8 hours
**Exit criteria:** Server runs as launchd service, restarts on crash, logs to file. All error cases return OpenAI-format errors.

### Phase 5: Testing, Documentation, and OpenClaw Integration (Day 5-6)

**Goal:** Happy-path integration tests passing, OpenClaw working end-to-end.

**Tasks:** T-026, T-027, T-028, T-029
**Estimated effort:** 7 hours
**Exit criteria:** Integration tests pass (model generates audio, all API routes return correct responses for all formats). OpenClaw generates voice notes via local server.

### Timeline Summary

| Phase | Days | Hours | Tasks |
|-------|------|-------|-------|
| 1. Project Setup + MLX | 1-2 | 8.5h | T-001 to T-008 |
| 2. FastAPI Server | 2-3 | 7h | T-009 to T-015 |
| 3. Audio Formats | 3-4 | 4h | T-016 to T-018 |
| 4. Errors + Deploy | 4-5 | 8h | T-019 to T-025 |
| 5. Testing + Integration | 5-6 | 7h | T-026 to T-029 |
| **Total** | **~6 days** | **~34.5h** | **29 tasks** |

---

## 7. Quality Requirements

### Testing Strategy

| Level | Tool | Scope |
|-------|------|-------|
| Integration | pytest + httpx | Engine (model load/generate), API routes (all endpoints, all formats) |
| End-to-end | curl + manual | Smoke tests, OpenClaw integration |

**Test Approach:**
- Integration tests require the model to be downloaded (run on Mac Studio only)
- Tests focus on happy-path validation: model generates audio, API routes return correct responses for all formats
- No unit tests or mocks -- the codebase is small enough that integration tests provide sufficient coverage
- No performance benchmarks or soak tests in v1 -- monitor via health endpoint and logs

### Performance Benchmarks

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Single sentence latency (10-20 words) | < 2 seconds | `time curl` against running server |
| Format conversion overhead | < 500ms | Timestamp delta between generation and response |
| Startup time (cold) | < 30 seconds | Time from process start to `/health` returning `ok` |
| Steady-state memory | < 4 GB | `mx.get_peak_memory()` via health endpoint |
| Concurrent request handling | Serial (queued) | Semaphore ensures no parallel inference |

### Code Quality

- Type hints on all public functions
- Docstrings on all public classes and functions
- `ruff` for linting and formatting
- No PyTorch imports in production code (only mlx/mlx-audio)

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mlx-audio CSM generation quality is poor | Low | High | Test early in Phase 1 (T-005). mlx-audio is actively maintained. If quality is unacceptable, investigate model quantization or alternative checkpoints. |
| Generation latency exceeds 2s target | Medium | Medium | M3 Ultra should be fast enough. Benchmark early (T-005). If slow: reduce `max_audio_length_ms`, tune `top_k`, or accept longer latency for long text. |
| mlx-audio API changes in future versions | Low | Medium | Pin `mlx-audio` version in `pyproject.toml`. Our wrapper is thin, so updates are manageable. |
| Opus encoding fails or ffmpeg not available | Low | Low | Check ffmpeg presence at startup (health endpoint). Opus is optional -- MP3 is the default format. Document ffmpeg requirement in README. |
| Memory leak during long-running operation | Medium | Medium | mlx-audio calls `mx.clear_cache()` after each generation. Monitor via health endpoint `peak_memory_gb`. launchd auto-restart is the safety net. |
| Voice prompt quality degrades with different text | Medium | Low | `voice_match=True` (default in mlx-audio) prepends prompt text to improve consistency. If needed, experiment with `temperature` and `top_k` in presets. |
| Model weights download fails or is slow on first run | Low | Medium | First run downloads ~4GB from HuggingFace. Document this in README. `CACHE_DIR` setting allows custom cache location. After first download, fully offline. |

---

## References

- [Sesame CSM-1B (GitHub)](https://github.com/SesameAILabs/csm)
- [mlx-audio (GitHub)](https://github.com/Blaizzy/mlx-audio) -- v0.3.1, source-analyzed for this TRD
- [OpenAI TTS API Reference](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [PRD v1.1](./PRD.md)
- [REQUIREMENTS.md](/REQUIREMENTS.md) -- Original requirements specification
- [Sesame Blog](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-20 | Initial TRD based on PRD v1.1 and mlx-audio v0.3.1 source analysis |
| 1.1 | 2026-02-20 | Refined: ffmpeg for all audio formats, uv package manager, simplified to happy-path integration tests only, confirmed flat root layout |
