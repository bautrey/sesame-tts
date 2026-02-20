# Sesame TTS Server — The Open Source Voice for AI Agents

## Vision

**Kill the ElevenLabs subscription.** Every AI agent framework — OpenClaw, Open Interpreter, AutoGPT, CrewAI — pays ElevenLabs or OpenAI for voice. We're building the local, free, open source alternative that runs on your hardware and sounds just as good.

One install. Zero API costs. Forever.

This isn't just a TTS server. It's the voice layer the open source AI agent community has been waiting for.

## What We're Building

A production-grade local TTS server that is simultaneously:
1. **OpenAI TTS API compatible** — drop-in replacement for any tool using OpenAI's `/v1/audio/speech`
2. **ElevenLabs API compatible** — drop-in replacement for streaming TTS including OpenClaw Talk Mode
3. **Standalone voice server** — any app, any framework, any language can use it via HTTP

Built on **Sesame CSM-1B** — the model that crossed the uncanny valley of voice. Running on **Apple Silicon via MLX** — native, fast, no CUDA needed.

## Target Users

- **OpenClaw users** who want Talk Mode without ElevenLabs
- **AI agent builders** who need voice output without per-character billing
- **Developers** who want a local TTS API for any project
- **Privacy-conscious users** who don't want their text sent to cloud APIs
- **Hobbyists** who can't justify $22/mo for ElevenLabs Pro

## Core Requirements

### R1: OpenAI-Compatible TTS Endpoint

```
POST /v1/audio/speech
{
  "model": "csm-1b",
  "input": "Hello world",
  "voice": "conversational_b",
  "response_format": "mp3",
  "speed": 1.0
}
→ audio binary
```

- Full OpenAI TTS API spec compliance
- Accept any model name (we only have one, graceful mapping)
- Support: mp3, opus, wav, flac, pcm, aac
- Speed control: 0.5x to 2.0x
- Max input: 4096 characters (chunked generation for longer text)

### R2: ElevenLabs-Compatible Streaming Endpoint

```
POST /v1/text-to-speech/{voice_id}/stream
{
  "text": "Hello world",
  "model_id": "eleven_v3",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": true
  }
}
→ chunked audio stream
```

**Output formats (matching ElevenLabs exactly):**
- `pcm_16000` — 16-bit PCM, 16kHz mono
- `pcm_22050` — 16-bit PCM, 22.05kHz mono
- `pcm_24000` — 16-bit PCM, 24kHz mono (native)
- `pcm_44100` — 16-bit PCM, 44.1kHz mono
- `mp3_22050_32` — MP3, 22.05kHz, 32kbps
- `mp3_44100_32` — MP3, 44.1kHz, 32kbps
- `mp3_44100_64` — MP3, 44.1kHz, 64kbps
- `mp3_44100_96` — MP3, 44.1kHz, 96kbps
- `mp3_44100_128` — MP3, 44.1kHz, 128kbps
- `mp3_44100_192` — MP3, 44.1kHz, 192kbps

**Non-streaming variant too:**
```
POST /v1/text-to-speech/{voice_id}
→ full audio buffer
```

### R3: Streaming Implementation

**Phase 1 (v1): Chunked Buffered Streaming**
- Generate audio frame-by-frame via MLX
- Buffer N frames (configurable, default 6) → decode → stream chunk
- Target: first audio chunk ≤ 500ms on M-series chips
- Stream via `Transfer-Encoding: chunked`
- Support `Connection: keep-alive` for low-latency sequential requests

**Phase 2 (v1.1): True Frame Streaming**
- Stream decoded frames individually as they complete
- Target: first audio chunk ≤ 200ms
- Requires deeper MLX generation loop integration

### R4: Voice Management

**Built-in voices:**
Ship with every voice Sesame/mlx-audio provides. Map them to human-readable names.

**Voice list endpoint:**
```
GET /v1/voices
{
  "voices": [
    {
      "voice_id": "conversational_b",
      "name": "Max (Conversational Male)",
      "category": "generated",
      "labels": {"accent": "american", "gender": "male", "use_case": "conversational"},
      "preview_url": "http://localhost:8880/v1/voices/conversational_b/preview",
      "settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
  ]
}
```

**Voice preview:**
```
GET /v1/voices/{voice_id}/preview
→ 5-second audio sample
```

**Custom voices (future-ready):**
- Accept reference audio file for voice cloning
- Store custom voice configs in `~/.cache/sesame-tts/voices/`
- API: `POST /v1/voices/add` with audio file + name
- Structure the code so this is easy to add but don't build cloning in v1

**ElevenLabs voice ID mapping:**
- Accept any ElevenLabs voice ID → map to default Sesame voice
- Log a friendly message: "Mapped ElevenLabs voice {id} to local voice {name}"
- This means existing ElevenLabs configs work without changes

### R5: Conversation Context (Sesame's Killer Feature)

Sesame CSM is a **conversational** model — it uses prior audio context to maintain natural prosody, rhythm, and emotion across turns. This is what makes it sound human.

**Context endpoint:**
```
POST /v1/text-to-speech/{voice_id}/stream
{
  "text": "That's really interesting, tell me more.",
  "context": [
    {"text": "I just got promoted!", "audio_ref": "turn_001"},
    {"text": "Congratulations! When did you find out?", "audio_ref": "turn_002"}
  ]
}
```

**How it works:**
- Server caches recent audio generations keyed by `audio_ref` or auto-incremented turn ID
- Passes last N turns as context to the model
- Result: Max's voice naturally flows from turn to turn instead of each response sounding cold-started
- Cache eviction: LRU, max 10 turns or 60 seconds of audio

**OpenClaw Talk Mode integration:**
- Talk Mode already sends sequential requests
- If we track turns server-side (by client session), context flows automatically
- No Talk Mode changes needed — just smarter server behavior

### R6: ElevenLabs Compatibility Endpoints

For tools that check account status, model lists, etc.:

```
GET /v1/models
→ [{"model_id": "csm-1b", "name": "Sesame CSM-1B", "can_do_text_to_speech": true, "languages": [{"language_id": "en", "name": "English"}]}]

GET /v1/user/subscription
→ {"tier": "local", "character_count": 0, "character_limit": 999999999, "status": "active"}

GET /v1/user
→ {"subscription": {"tier": "local", "character_limit": 999999999}}
```

Return unlimited everything. You're local. There are no limits.

### R7: Authentication

- Accept `xi-api-key` header (any value, don't validate)
- Accept `Authorization: Bearer` header (any value)
- Accept no auth at all for localhost
- Optional: configurable API key for remote access (future)
- No auth complexity for the default local use case

### R8: Interrupt & Cancellation

Critical for Talk Mode — when the user starts speaking, generation must stop immediately.

- Monitor client disconnect on streaming endpoints
- Cancel MLX inference on disconnect (abort the generation loop)
- Free memory immediately
- Return partial audio up to the interrupt point
- Log interrupts for debugging: "Generation interrupted at frame N of ~M"

### R9: Performance & Resource Management

**Inference queue:**
- Single MLX inference at a time (GPU serialization)
- Async queue for concurrent requests
- Return 429 with `Retry-After` if queue depth > 3
- Priority queue: streaming requests > sync requests

**Memory management:**
- Target: < 8GB peak memory usage
- Unload model after configurable idle timeout (optional, default: never)
- Monitor memory usage, expose in health endpoint

**Warm-up:**
- Pre-generate a short audio clip on startup to warm caches
- Eliminates cold-start latency on first real request

**Benchmarks (include in README):**
- Time to first audio chunk (streaming)
- Full generation time vs text length
- Memory usage (idle, peak)
- Requests per minute (sequential)
- Compare against ElevenLabs API latency

### R10: Observability

**Health endpoint (enhanced):**
```
GET /health
{
  "status": "ok",
  "model": {"loaded": true, "id": "mlx-community/csm-1b", "memory_gb": 6.2},
  "ffmpeg": true,
  "voices": ["conversational_b", "conversational"],
  "queue_depth": 0,
  "uptime_seconds": 3600,
  "requests_served": 42,
  "avg_generation_time_ms": 2100,
  "context_cache_size": 3
}
```

**Structured logging:**
- JSON logs to `~/.cache/sesame-tts/server.log`
- Log every request: voice, format, input length, generation time, streamed/sync
- Log errors with full context
- Configurable log level

**Metrics endpoint (optional):**
```
GET /metrics
→ Prometheus-compatible metrics (future)
```

### R11: Configuration

```env
# Server
HOST=0.0.0.0
PORT=8880
LOG_LEVEL=info
WORKERS=1

# Model
MODEL_ID=mlx-community/csm-1b
CACHE_DIR=~/.cache/sesame-tts
HF_TOKEN=                          # Optional, for faster downloads

# Defaults
DEFAULT_VOICE=conversational_b
DEFAULT_FORMAT=mp3
DEFAULT_OUTPUT_FORMAT=pcm_24000     # For streaming/ElevenLabs endpoint
MAX_INPUT_LENGTH=4096

# Streaming
STREAM_BUFFER_FRAMES=6              # Frames to buffer before first chunk
STREAM_CHUNK_SIZE=4096              # Bytes per stream chunk

# Context (conversation memory)
CONTEXT_MAX_TURNS=10
CONTEXT_MAX_SECONDS=60
CONTEXT_ENABLED=true

# Performance
WARMUP_ON_START=true
MAX_QUEUE_DEPTH=3
IDLE_UNLOAD_MINUTES=0               # 0 = never unload
```

### R12: Installation & Deployment

**One-command install:**
```bash
curl -fsSL https://raw.githubusercontent.com/fortiumpartners/sesame-tts-server/main/install.sh | bash
```

The install script should:
1. Check for Apple Silicon (fail gracefully on Intel/Linux with clear message)
2. Install uv if not present
3. Create venv, install dependencies
4. Download model weights from HuggingFace
5. Prompt for HF token if gated model access fails
6. Generate default `.env`
7. Install launchd plist (optional, prompt user)
8. Run health check
9. Print: "✅ Sesame TTS running at http://localhost:8880"

**Manual install:**
```bash
git clone https://github.com/fortiumpartners/sesame-tts-server
cd sesame-tts-server
uv sync
cp .env.example .env
uv run python server.py
```

**launchd service (macOS):**
- Auto-start on boot
- Restart on crash (max 3 retries, 10s delay)
- Logs to `~/.cache/sesame-tts/server.log`
- `launchctl start/stop com.sesame-tts.server`

**Docker (future, community contribution):**
- Not in v1 (MLX requires macOS)
- Structure code so a PyTorch/CUDA backend could be swapped in

### R13: Error Handling

OpenAI-style errors:
```json
{"error": {"message": "Input exceeds maximum length of 4096 characters", "type": "invalid_request_error", "code": "input_too_long"}}
```

ElevenLabs-style errors:
```json
{"detail": {"status": "invalid_request", "message": "Voice not found"}}
```

Detect which API style the client is using (by endpoint) and return the matching error format.

Handle gracefully:
- Model loading failures → clear message about HF token / disk space
- Out of memory → suggest closing other apps, show memory stats
- Invalid voice → list available voices in error message
- FFmpeg missing → installation instructions in error
- Port already in use → suggest alternative port

### R14: Documentation (Critical for Community Adoption)

**README.md must include:**
- Hero banner / logo
- "What is this" in one sentence
- 30-second GIF/video of Talk Mode working locally
- One-command install
- Configuration for OpenClaw (both TTS and Talk Mode)
- Configuration for other frameworks (Open Interpreter, etc.)
- Voice list with audio samples (link to hosted previews)
- Benchmarks table
- Architecture diagram
- FAQ:
  - "Does this work on Intel Mac?" → No, Apple Silicon required
  - "Does this work on Linux?" → Not yet, MLX is macOS-only. PRs welcome for PyTorch backend.
  - "How does it compare to ElevenLabs?" → Comparable quality, ~2x latency, zero cost
  - "Can I clone my voice?" → Not yet, but the architecture supports it
  - "Can I use this with Home Assistant / Alexa / etc.?" → Yes, any HTTP client works

**CONTRIBUTING.md:**
- How to add new voices
- How to add new output formats
- How to add a PyTorch/CUDA backend
- Architecture overview for contributors

### R15: OpenClaw Integration Guide

**TTS (voice notes in Telegram/chat):**
```bash
# In shell or .env
export OPENAI_TTS_BASE_URL=http://localhost:8880/v1
```

```json
// openclaw.json
{
  "messages": {
    "tts": {
      "provider": "openai",
      "auto": "tagged",
      "openai": {
        "model": "csm-1b",
        "voice": "conversational_b"
      }
    }
  }
}
```

**Talk Mode (real-time voice conversation):**

Check if OpenClaw supports ElevenLabs base URL override. If yes:
```bash
export ELEVENLABS_BASE_URL=http://localhost:8880
```

```json
// openclaw.json
{
  "talk": {
    "voiceId": "conversational_b",
    "modelId": "csm-1b",
    "outputFormat": "pcm_24000",
    "apiKey": "local",
    "interruptOnSpeech": true
  }
}
```

If OpenClaw does NOT support `ELEVENLABS_BASE_URL`:
- Submit a PR to OpenClaw adding base URL override for ElevenLabs TTS provider
- Include PR in our release announcement ("works with OpenClaw v2026.x.x+, PR submitted for Talk Mode support")

### R16: Community Launch Plan

**Pre-launch:**
- [ ] Record a 60-second demo video showing Talk Mode with local Sesame TTS
- [ ] Create audio comparison: ElevenLabs vs Sesame side-by-side
- [ ] Write launch blog post
- [ ] Test on M1, M2, M3 (Pro/Max/Ultra variants)
- [ ] Get 2-3 beta testers from OpenClaw Discord

**Launch targets:**
1. **GitHub** — `fortiumpartners/sesame-tts-server` (public repo)
2. **ClawhHub** — Submit as OpenClaw skill
3. **OpenClaw Discord** — #showcase channel
4. **Reddit** — r/openclaw, r/LocalLLaMA, r/selfhosted
5. **Hacker News** — "Show HN: Local TTS server for AI agents — OpenAI + ElevenLabs compatible"
6. **X/Twitter** — Burke's account + Fortium account
7. **Product Hunt** — If GitHub gets 100+ stars

**Launch messaging:**
- Lead with: "Free Talk Mode for OpenClaw — no ElevenLabs needed"
- Secondary: "Drop-in replacement for OpenAI TTS and ElevenLabs APIs"
- Technical: "Sesame CSM-1B on Apple Silicon via MLX — 20 tokens/sec, < 500ms first chunk"

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Runtime | Python 3.10+ | Sesame ecosystem |
| Framework | FastAPI + uvicorn | Async, fast, auto-docs |
| ML Backend | MLX (mlx-audio) | Native Apple Silicon |
| Model | Sesame CSM-1B | Best open-source voice quality |
| Audio conversion | ffmpeg | Universal, proven |
| Resampling | scipy.signal | For PCM sample rate conversion |
| Package manager | uv | Fast, modern |
| Config | python-dotenv | Simple |
| Service | launchd | macOS native |

## Success Criteria

1. OpenClaw Talk Mode works with zero config changes beyond base URL
2. First audio chunk streams within 500ms
3. Interrupt works — stops mid-sentence when user speaks
4. Audio quality is indistinguishable from ElevenLabs in casual listening
5. One-command install completes in under 5 minutes
6. README is good enough that someone installs it without asking questions
7. 100+ GitHub stars in first month
8. Featured on OpenClaw Discord or blog

## Release Plan

**v1.0 — "It Speaks"**
- OpenAI-compatible endpoint (already done ✅)
- ElevenLabs-compatible streaming endpoint
- Voice management + mapping
- Buffered streaming (target < 500ms first chunk)
- Interrupt support
- launchd service
- Full documentation + install script
- OpenClaw integration guide
- Community launch

**v1.1 — "It Remembers"**
- Conversation context (R5)
- True frame streaming (< 200ms first chunk)
- Voice preview endpoint
- Performance benchmarks suite

**v1.2 — "It's You"**
- Voice cloning from reference audio
- Custom voice management UI (web)
- Additional language support

**v2.0 — "It Runs Everywhere"**
- PyTorch/CUDA backend for Linux/Windows
- Docker container
- Kubernetes helm chart
- WebSocket streaming endpoint
