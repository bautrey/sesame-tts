# Sesame TTS Server

OpenAI-compatible local TTS server wrapping Sesame CSM-1B via [mlx-audio](https://github.com/Blaizzy/mlx-audio) for Apple Silicon Macs.

Exposes `POST /v1/audio/speech` on `localhost:8880`, so any OpenAI TTS client (including OpenClaw) can switch to local generation with a config change.

## Requirements

- Apple Silicon Mac (M1+) with macOS
- Python 3.10-3.13
- `ffmpeg` installed (`brew install ffmpeg`)
- ~4GB disk for model weights (auto-downloaded on first run)

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> sesame-tts
cd sesame-tts
uv sync
```

## Usage

```bash
# Start the server
HF_HUB_DISABLE_XET=1 .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8880
```

First run downloads the model (~4GB). Subsequent starts take ~2 seconds.

### Generate speech

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"csm-1b","input":"Hello, this is a test.","voice":"conversational"}' \
  --output speech.mp3
```

### Supported formats

| Format | Content-Type | Flag |
|--------|-------------|------|
| MP3 | `audio/mpeg` | `"response_format": "mp3"` (default) |
| WAV | `audio/wav` | `"response_format": "wav"` |
| Opus | `audio/opus` | `"response_format": "opus"` |
| FLAC | `audio/flac` | `"response_format": "flac"` |

### Voices

| Voice | Description |
|-------|-------------|
| `conversational` | Natural, warm conversational voice (default) |
| `conversational_b` | Second conversational voice, different character |

Add custom voices by creating JSON files in the `presets/` directory.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | Generate speech (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Server health + model status |

### Request body (`/v1/audio/speech`)

```json
{
  "model": "csm-1b",
  "input": "Text to speak",
  "voice": "conversational",
  "response_format": "mp3",
  "speed": 1.0
}
```

- `model`: Required but ignored (single model)
- `input`: 1-4096 characters
- `voice`: Preset name from `presets/`
- `response_format`: `mp3`, `wav`, `opus`, or `flac`
- `speed`: Accepted for compatibility, not applied in v1

## OpenClaw Integration

Point OpenClaw at the local server:

```
TTS Base URL: http://localhost:8880/v1
TTS Model: csm-1b
TTS Voice: conversational
```

## Running as a Service (launchd)

```bash
# Create log directory
mkdir -p ~/.cache/sesame-tts

# Install the service
cp com.burkestudio.sesame-tts.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.burkestudio.sesame-tts.plist

# Check status
launchctl list | grep sesame

# View logs
tail -f ~/.cache/sesame-tts/server.log

# Stop the service
launchctl unload ~/Library/LaunchAgents/com.burkestudio.sesame-tts.plist
```

## Configuration

Copy `.env.example` to `.env` and modify as needed. All settings can also be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8880` | Server port |
| `MODEL_ID` | `mlx-community/csm-1b` | HuggingFace model ID |
| `DEFAULT_VOICE` | `conversational` | Default voice preset |
| `DEFAULT_FORMAT` | `mp3` | Default output format |
| `LOG_LEVEL` | `info` | Logging level |

## Testing

```bash
uv sync --extra test
HF_HUB_DISABLE_XET=1 .venv/bin/python -m pytest tests/ -v
```

## Architecture

Thin wrapper around mlx-audio's CSM-1B implementation:

```
server.py          → FastAPI routes, request validation, semaphore
tts_engine.py      → Model loading via mlx-audio, generation wrapper
audio_converter.py → ffmpeg subprocess for format conversion
voice_presets.py   → JSON preset loading
config.py          → pydantic-settings configuration
errors.py          → OpenAI-compatible error responses
```

## Credits

- [Sesame CSM-1B](https://github.com/SesameAILabs/csm) — the underlying speech model
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX inference wrapper
- [mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b) — MLX-converted weights
