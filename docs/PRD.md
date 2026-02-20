# PRD: Sesame TTS -- OpenAI-Compatible Local TTS Server

**Author:** Burke
**Created:** 2026-02-20
**Status:** Draft
**Version:** 1.1

---

## 1. Product Summary

### Problem

Burke currently pays per-request for OpenAI's TTS API to generate speech in OpenClaw and other tools. This incurs ongoing costs, introduces external latency and availability dependencies, and sends text content to a third-party service. His Mac Studio with Apple Silicon sits underutilized for ML workloads.

### Solution

A local FastAPI server wrapping the Sesame CSM-1B speech model via MLX (Apple Silicon native inference). The server exposes an OpenAI-compatible `/v1/audio/speech` endpoint on `localhost:8880`, allowing any existing OpenAI TTS client -- OpenClaw included -- to switch to local generation with a single config change.

### Value Proposition

- **Zero ongoing API costs** -- model runs locally on hardware Burke already owns
- **No external dependencies** -- works offline, no network latency to cloud APIs
- **High quality speech** -- Sesame CSM-1B produces natural, emotional, human-quality voice output
- **Drop-in replacement** -- OpenAI-compatible API means zero client code changes
- **Privacy** -- text never leaves the local machine

---

## 2. User Analysis

### Primary User: Burke (Developer / Operator)

Burke is a developer who runs OpenClaw and other tools that consume TTS via the OpenAI API. He operates a 2025 Mac Studio with Apple M3 Ultra (256GB unified memory, macOS Tahoe 26.3) and wants to self-host ML inference to eliminate API costs and external dependencies.

**Pain Points:**
- Paying per-request for OpenAI TTS across multiple tools
- Latency from network round-trips to OpenAI servers
- Dependency on OpenAI service availability
- Sending potentially sensitive text to a third-party API

**Needs:**
- A server that starts automatically on boot and runs reliably in the background
- Audio output quality comparable to OpenAI's TTS voices
- Response times fast enough for interactive use (< 2 seconds for a sentence)
- Simple configuration and maintenance

### Secondary Users: OpenAI TTS API Clients

Any application configured to call the OpenAI TTS endpoint can point at this server instead. OpenClaw is the primary client, but the API is generic enough for any compatible tool.

---

## 3. Goals and Non-Goals

### Goals

1. Provide a fully functional OpenAI-compatible TTS API endpoint running locally
2. Achieve natural, high-quality speech output using Sesame CSM-1B via MLX
3. Generate a typical sentence in under 2 seconds on Apple Silicon
4. Support multiple audio output formats (MP3, Opus, WAV, FLAC)
5. Run reliably as a background service with auto-start on boot
6. Eliminate all OpenAI TTS API costs for Burke's use cases

### Non-Goals

- Web UI or audio playground (API only)
- Multi-model support (CSM-1B is the only model)
- Voice cloning (design for it, do not build it)
- GPU/CUDA support (MLX on Apple Silicon only)
- Docker containerization (bare metal on Mac Studio)
- Multi-user authentication or rate limiting
- Production SaaS hardening (this is personal infrastructure)

---

## 4. Functional Requirements

### R1: OpenAI-Compatible TTS Endpoint

Implement `POST /v1/audio/speech` matching the OpenAI TTS API contract.

**Request body:**

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `model` | string | Yes | -- | Accept "csm-1b" or any value (single model) |
| `input` | string | Yes | -- | Text to synthesize, up to 4096 characters |
| `voice` | string | Yes | -- | Voice preset name (see R4) |
| `response_format` | string | No | "mp3" | One of: mp3, opus, wav, flac |
| `speed` | float | No | 1.0 | Range 0.5 to 2.0 |

**Response:** Raw audio bytes with appropriate `Content-Type` header.

**Acceptance Criteria:**
- AC-1.1: `curl -X POST localhost:8880/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"csm-1b","input":"Hello Burke.","voice":"conversational"}' --output test.mp3` produces a playable MP3 file containing the spoken text.
- AC-1.2: Unknown `voice` values return a 400 error with OpenAI-compatible error JSON.
- AC-1.3: Empty `input` returns a 400 error.
- AC-1.4: Input exceeding 4096 characters returns a 400 error.
- AC-1.5: The `model` field is accepted but does not affect behavior (single model).

### R2: Supporting API Endpoints

Implement discovery and health endpoints.

- `GET /v1/models` -- Returns a list of available models in OpenAI format (just "csm-1b").
- `GET /health` -- Returns server health including model load status, MLX availability, and ffmpeg presence.

**Acceptance Criteria:**
- AC-2.1: `/v1/models` returns JSON matching OpenAI's model list schema with "csm-1b" listed.
- AC-2.2: `/health` returns 200 with `{"status": "ok", ...}` when the server is fully operational.
- AC-2.3: `/health` returns a non-200 status or degraded status if the model failed to load.

### R3: MLX Backend (Apple Silicon Native)

Use MLX framework for all model inference. Reference implementation: `Blaizzy/mlx-audio` (Sesame CSM ported to MLX).

**Acceptance Criteria:**
- AC-3.1: Server imports and uses MLX for inference, not PyTorch/CUDA.
- AC-3.2: Model weights are downloaded from HuggingFace on first run and cached to `CACHE_DIR`.
- AC-3.3: Steady-state memory usage stays under 4GB of unified memory. (Conservative target -- actual hardware has 256GB unified memory.)
- AC-3.4: A typical sentence generates in under 2 seconds on M3 Ultra or better.

### R4: Voice Presets

Ship with whatever CSM voices mlx-audio provides out of the box (likely 1-2 conversational voices with different speaker IDs). The preset config system must support adding new voices via config files without code changes.

**v1 voices:** Available CSM speaker IDs from mlx-audio, exposed as named presets.

**Target presets (post-v1):**

| Preset | Character |
|--------|-----------|
| `conversational` | Natural, warm, default voice |
| `professional` | Clean, formal delivery |
| `fast` | Speed-optimized, slightly less natural |
| `whisper` | Soft, low energy |

Each preset defines: temperature, top_k, speaker ID/embedding, and any other relevant generation parameters.

**Acceptance Criteria:**
- AC-4.1: At least 1 voice preset is available and selectable via the `voice` field.
- AC-4.2: Presets are stored as JSON or YAML files in a configurable directory.
- AC-4.3: Adding a new preset requires only adding a config file, no code changes.

### R5: Audio Format Conversion

Generate audio natively as WAV, convert to the requested format via ffmpeg subprocess.

| Format | Spec |
|--------|------|
| MP3 | 64kbps mono |
| Opus | 32kbps mono |
| WAV | 16-bit 24kHz mono |
| FLAC | Lossless |

**Acceptance Criteria:**
- AC-5.1: All four formats produce valid, playable audio files.
- AC-5.2: MP3 output is mono at approximately 64kbps.
- AC-5.3: Opus output is mono at approximately 32kbps.
- AC-5.4: WAV output is 16-bit 24kHz mono.
- AC-5.5: Format conversion adds less than 500ms overhead for a typical sentence.

### R6: Configuration

All settings configurable via environment variables or `.env` file using python-dotenv.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Bind address |
| `PORT` | 8880 | Bind port |
| `MODEL_ID` | sesame/csm-1b | HuggingFace model identifier |
| `CACHE_DIR` | ~/.cache/sesame-tts | Model cache and log directory |
| `DEFAULT_VOICE` | conversational | Default voice preset |
| `DEFAULT_FORMAT` | mp3 | Default audio format |
| `MAX_INPUT_LENGTH` | 4096 | Maximum input text length |
| `LOG_LEVEL` | info | Logging verbosity |

**Acceptance Criteria:**
- AC-6.1: Server reads configuration from environment variables.
- AC-6.2: Server reads `.env` file if present in the working directory.
- AC-6.3: Environment variables override `.env` file values.
- AC-6.4: Server starts with sensible defaults when no configuration is provided.

### R7: Error Handling

Return OpenAI-compatible error responses for all failure cases.

**Error response format:**
```json
{
  "error": {
    "message": "descriptive message",
    "type": "invalid_request_error",
    "code": "error_code"
  }
}
```

**Handled failure cases:**
- Invalid or missing request fields
- Input text too long
- Unknown voice preset
- Model loading failure
- Out of memory during generation
- ffmpeg conversion failure

**Acceptance Criteria:**
- AC-7.1: All error responses match the OpenAI error JSON schema.
- AC-7.2: Invalid requests return 400 with specific error messages.
- AC-7.3: Server errors return 500 without crashing the process.
- AC-7.4: Model OOM errors are caught and return 503 with a retry-appropriate message.

### R8: Deployment as launchd Service

Run as a managed background service on Mac Studio via launchd.

**Acceptance Criteria:**
- AC-8.1: A launchd plist is provided that starts the server on boot.
- AC-8.2: Server logs to `~/.cache/sesame-tts/server.log`.
- AC-8.3: `launchctl` can be used to start, stop, and restart the service.
- AC-8.4: Server writes a PID file for clean shutdown.
- AC-8.5: Service auto-restarts if the process crashes.

### R9: OpenClaw Integration

Document and validate integration with OpenClaw as the primary client.

**OpenClaw configuration:**
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

**Acceptance Criteria:**
- AC-9.1: OpenClaw successfully generates voice notes using this server.
- AC-9.2: README includes integration instructions for OpenClaw.
- AC-9.3: No changes to OpenClaw source code are required -- config-only switch.

---

## 5. Non-Functional Requirements

### Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single sentence latency | < 2 seconds | Time from request to complete audio response for a 10-20 word input |
| Concurrent requests | 1 (serial) | Single-model inference is inherently serial; queue additional requests |
| Memory footprint | < 4GB unified memory | Steady-state after model load (conservative target; hardware has 256GB) |
| Startup time | < 30 seconds | Time from process start to healthy and serving requests |

### Reliability

- Server process auto-restarts on crash via launchd
- Graceful handling of OOM without process termination where possible
- Request queue for concurrent requests (do not reject, queue and process serially)
- Health endpoint for monitoring

### Security

- Binds to localhost by default (configurable to 0.0.0.0 for LAN access)
- No authentication required (local/trusted network only)
- No telemetry or external network calls after model download
- Input sanitization on text field (prevent injection into ffmpeg subprocess)

### Logging

- Structured logging with configurable level (debug, info, warning, error)
- Log to file at `CACHE_DIR/server.log`
- Log rotation or size management to prevent disk fill
- Request logging: timestamp, input length, voice, format, generation time

---

## 6. Acceptance Criteria (End-to-End)

These are the top-level acceptance criteria that validate the product is complete and working.

1. **Smoke test**: `curl` command from REQUIREMENTS.md produces a playable MP3 file with natural-sounding speech of the input text.

2. **Performance**: A 15-word sentence generates in under 2 seconds on the Mac Studio.

3. **OpenClaw integration**: OpenClaw generates voice notes via this server with only a config change -- no code modifications, no degradation in audio quality or response time.

4. **Service reliability**: Server survives 24 hours of continuous uptime without memory leaks or crashes, processing at least 100 requests.

5. **Format coverage**: All four audio formats (MP3, Opus, WAV, FLAC) produce valid, playable output.

6. **Error resilience**: Malformed requests, oversized input, and unknown voices all return proper error responses without crashing the server.

7. **Zero API costs**: No external API calls are made during normal operation (post model download).

---

## 7. Technical Constraints

| Constraint | Detail |
|------------|--------|
| Hardware | 2025 Mac Studio, Apple M3 Ultra, 256GB unified memory, macOS Tahoe 26.3 |
| ML framework | MLX only -- no PyTorch/CUDA in the inference path |
| Python version | 3.10+ |
| Model | Sesame CSM-1B (single model, no multi-model routing) |
| Audio tooling | ffmpeg must be pre-installed on the host |
| Deployment | Bare metal via launchd -- no Docker, no Kubernetes |
| Network | Localhost-first; LAN access optional via config |
| Upstream files | Upstream reference files (generator.py, models.py, watermarking.py, run_csm.py, setup.py, requirements.txt) moved to `upstream/` for reference; not used in production inference path. Server code lives in project root or `src/` |
| Repo ownership | Personal repo under the **bautrey** GitHub account (not FortiumPartners) |

---

## 8. Future Considerations

These items are explicitly out of scope for v1 but should not be designed out of the architecture.

### Streaming Support (v1.1)

v1 delivers complete audio responses synchronously. Streaming via chunked transfer encoding is planned for v1.1. The planned approach is sentence-level chunking: split input text into sentences, generate and stream each chunk independently so audio playback can begin before full generation completes. Target: first-byte latency under 1 second for long-form text (> 200 words).

### Expanded Voice Presets

v1 ships with available CSM speaker IDs from mlx-audio. The target preset table (conversational, professional, fast, whisper) with audibly distinct speech characteristics is a post-v1 goal. The config-driven preset system built in v1 enables adding these without code changes.

### Conversation Context (R6 from REQUIREMENTS.md)

Sesame CSM is a conversational model that accepts prior audio segments as context for more natural prosody across turns. The internal generation interface should accept an optional context parameter (list of prior audio segments with speaker IDs) even if v1 always passes an empty list. This enables multi-turn conversation support without refactoring.

### Voice Cloning

The preset system should accommodate custom speaker embeddings (e.g., from a reference audio file) in the future. The config schema should have a placeholder field for `speaker_embedding_path` even if it is not processed in v1.

### Speed Control

The `speed` parameter is accepted in the API but actual speed modification (via model parameters or post-processing) may be deferred. If deferred, the parameter should be silently accepted and logged.

### Web Playground

A simple HTML page for testing voices and text could be added later. The API is designed to support this without changes.

### Multi-Model Support

If additional TTS models become available for MLX, the `model` field in the API is already present to support routing. No multi-model infrastructure is needed in v1.

---

## References

- [Sesame CSM-1B (GitHub)](https://github.com/SesameAILabs/csm)
- [MLX Audio -- Sesame CSM Port](https://github.com/Blaizzy/mlx-audio)
- [OpenAI TTS API Reference](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [Sesame Blog Post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)
- [REQUIREMENTS.md](/REQUIREMENTS.md) -- Source requirements specification

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-20 | Initial draft |
| 1.1 | 2026-02-20 | Refined: M3 Ultra hardware, pragmatic voice presets, streaming deferred to v1.1, upstream file organization |
