# Sesame TTS — OpenAI-Compatible Local TTS Server

## Vision

A local text-to-speech server running on Mac Studio (Apple Silicon) that replaces OpenAI's TTS API with Sesame's CSM-1B model. OpenClaw and any other tool that speaks the OpenAI TTS API can point at localhost and get natural, emotional, human-quality speech — for free, forever, with zero API costs.

## Architecture

```
OpenClaw / Any Client
    |
    POST /v1/audio/speech  (OpenAI-compatible)
    |
FastAPI Server (localhost:8880)
    |
    MLX Backend (Apple Silicon native)
    |
    Sesame CSM-1B model (local weights)
    |
    Audio output (mp3/opus/wav)
```

## Requirements

### R1: OpenAI-Compatible API

Implement the OpenAI TTS endpoint spec exactly:

```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "csm-1b",
  "input": "Hello, this is Max.",
  "voice": "conversational",
  "response_format": "mp3",
  "speed": 1.0
}

→ Response: audio binary (Content-Type: audio/mpeg)
```

**Required fields:**
- `model` — accept "csm-1b" (or any value, we only have one model)
- `input` — text to speak (up to 4096 chars)
- `voice` — voice preset name (see R3)
- `response_format` — "mp3" (default), "opus", "wav", "flac"
- `speed` — 0.5 to 2.0 (1.0 = normal)

**Also implement:**
- `GET /v1/models` — list available models
- `GET /health` — service health check

### R2: MLX Backend (Apple Silicon Native)

- Use MLX framework for inference — NOT PyTorch/CUDA
- Reference: `Blaizzy/mlx-audio` has Sesame CSM ported to MLX already
- Model weights downloaded from HuggingFace on first run, cached locally
- Target: real-time or faster generation on M1 Ultra / M2 Ultra
- Memory footprint: keep under 4GB VRAM

### R3: Voice Presets

Define at least 4 built-in voice presets:
- `conversational` — natural, warm, the default "Max" voice
- `professional` — cleaner, more formal delivery
- `fast` — optimized for speed, slightly less natural
- `whisper` — soft, lower energy

Each preset is a configuration of model parameters (temperature, top_k, speaker embedding, etc.). Store presets as JSON/YAML configs so new voices can be added without code changes.

Allow custom speaker embeddings via audio file reference (voice cloning for future use).

### R4: Streaming Support

- Support chunked transfer encoding for long text
- Client receives audio chunks as they're generated
- Header: `Transfer-Encoding: chunked`
- This matters for OpenClaw — Burke doesn't want to wait 10 seconds for a full paragraph before hearing anything

### R5: Audio Format Conversion

- Generate in native format (WAV), convert to requested format on the fly
- Use ffmpeg for format conversion (already installed on the Mac Studio)
- MP3: 64kbps mono (optimized for voice, small file size for Telegram)
- Opus: 32kbps mono (best quality-to-size for voice)
- WAV: 16-bit 24kHz mono (raw)
- FLAC: lossless

### R6: Conversation Context (Future-Ready)

Sesame CSM is a *conversational* speech model — it can take prior audio context to maintain natural prosody across turns. 

For now: single-turn generation is fine.

Design the internal API so context (prior audio segments) can be passed in later. Don't build it yet, just don't make it impossible.

### R7: Configuration

Environment variables or `.env` file:

```
HOST=0.0.0.0
PORT=8880
MODEL_ID=sesame/csm-1b
CACHE_DIR=~/.cache/sesame-tts
DEFAULT_VOICE=conversational
DEFAULT_FORMAT=mp3
MAX_INPUT_LENGTH=4096
LOG_LEVEL=info
```

### R8: Error Handling

- Return OpenAI-compatible error responses:
  ```json
  {"error": {"message": "Input too long", "type": "invalid_request_error", "code": "invalid_input"}}
  ```
- Graceful handling of: model loading failures, OOM, invalid voice names, empty input
- Startup health check: verify model loaded, MLX available, ffmpeg installed

### R9: Deployment

- Runs as a background service on Mac Studio
- launchd plist for auto-start on boot
- Logs to `~/.cache/sesame-tts/server.log`
- PID file for clean shutdown

### R10: OpenClaw Integration

Once running, OpenClaw config change needed:

```json
{
  "tts": {
    "provider": "openai-compatible",
    "baseUrl": "http://localhost:8880/v1",
    "model": "csm-1b",
    "voice": "conversational"
  }
}
```

Document this in a README section so we can wire it up immediately after the server works.

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Runtime | Python 3.10+ | Sesame model is Python |
| Framework | FastAPI + uvicorn | Async, fast, OpenAPI docs |
| ML Backend | MLX | Native Apple Silicon, no CUDA needed |
| Model | Sesame CSM-1B via mlx-audio | Best voice quality, local |
| Audio conversion | ffmpeg (subprocess) | Already installed, proven |
| Config | python-dotenv | Simple, standard |

## Out of Scope (For Now)

- Web UI / playground (just the API)
- Multi-model support (only CSM-1B)
- Voice cloning (structure for it, don't build it)
- GPU/CUDA support (MLX only)
- Docker (runs bare metal on Mac Studio)

## Success Criteria

1. `curl -X POST localhost:8880/v1/audio/speech -H "Content-Type: application/json" -d '{"input":"Hello Burke, this is Max.","voice":"conversational"}' --output test.mp3` produces natural speech
2. Generation speed: < 2 seconds for a typical sentence on Apple Silicon
3. OpenClaw TTS tool produces voice notes via this server instead of OpenAI
4. Zero ongoing API costs

## References

- Sesame CSM-1B: https://github.com/SesameAILabs/csm
- MLX Audio (Sesame port): https://github.com/Blaizzy/mlx-audio
- OpenAI TTS API spec: https://platform.openai.com/docs/api-reference/audio/createSpeech
- Sesame blog: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
