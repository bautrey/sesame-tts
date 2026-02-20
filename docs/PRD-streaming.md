# PRD: ElevenLabs-Compatible Streaming API for Sesame TTS

**Author:** Burke
**Created:** 2026-02-20
**Status:** Draft
**Version:** 1.1
**PRD Reference:** Extends [docs/PRD.md](./PRD.md) v1.1
**GitHub Issue:** [#1 - ElevenLabs-Compatible Streaming API + OpenClaw Talk Mode Support](https://github.com/bautrey/sesame-tts/issues/1)

---

## 1. Product Summary

### Problem

The sesame-tts server already provides an OpenAI-compatible `/v1/audio/speech` endpoint for synchronous TTS generation. However, OpenClaw Talk Mode -- the real-time voice conversation feature -- uses the **ElevenLabs streaming API**, not the OpenAI TTS API. This means:

1. **Talk Mode still requires an ElevenLabs subscription** ($5-22/month) despite having a working local TTS server
2. **No streaming support** -- the current server generates the entire audio buffer before responding, creating unacceptable latency for conversational use (3-10 seconds for a paragraph)
3. **No conversation context** -- each generation is cold-started, losing the natural prosody and emotional continuity that makes Sesame CSM-1B sound human across conversation turns
4. **No interrupt support** -- if the user starts speaking mid-generation, the server cannot cancel inference, wasting compute and creating awkward overlapping audio

These gaps prevent the server from being a complete ElevenLabs replacement for the AI agent community.

### Solution

Add ElevenLabs API-compatible endpoints to the existing sesame-tts server, implementing chunked audio streaming and inference interruption. The server will accept ElevenLabs API requests verbatim -- same endpoints, same headers, same output formats -- so that any application using the ElevenLabs API (including OpenClaw Talk Mode) can switch to the local server by changing only the base URL. Conversation context caching is designed but deferred to v1.1.

The implementation layers on top of the existing architecture: same FastAPI server, same TTSEngine, same audio converter, same voice preset system. New modules handle streaming chunking, context caching, and the ElevenLabs-specific request/response translation.

### Value Proposition

- **Kill the ElevenLabs subscription** -- Talk Mode works locally, zero API costs, forever
- **Sub-second first audio** -- buffered streaming delivers first chunk in under 500ms (Phase 1), under 200ms (Phase 2)
- **Conversation memory (v1.1)** -- Sesame CSM-1B's context system produces natural turn-to-turn prosody that ElevenLabs cannot match
- **Instant interrupt** -- cancel inference mid-generation when the user starts speaking
- **Universal compatibility** -- ElevenLabs + OpenAI APIs on the same server, same port
- **Community value** -- every OpenClaw user paying ElevenLabs for Talk Mode gets a free local alternative

---

## 2. User Analysis

### 2.1 User Personas

#### Persona 1: Burke (Primary -- Developer/Operator)

**Profile:** Developer running OpenClaw on a 2025 Mac Studio (M3 Ultra, 256GB unified memory). Uses Talk Mode daily for voice conversations with AI agents. Currently paying ElevenLabs for streaming TTS.

**Pain Points:**
- Paying $22/month for ElevenLabs Pro to get Talk Mode streaming
- Already has a working local TTS server but it cannot serve Talk Mode
- Network latency to ElevenLabs adds 200-500ms to every voice response
- Each Talk Mode turn sounds cold-started -- no conversational continuity

**Needs:**
- Drop-in ElevenLabs replacement that requires only a base URL change
- Streaming audio that starts playing within 500ms
- Interrupt support so generation stops when he starts speaking
- Natural conversation flow across turns (context)

**Success Criteria:**
- Talk Mode works identically to ElevenLabs with a config change
- Perceived latency is equal or better than ElevenLabs
- Monthly savings of $22/month (ElevenLabs Pro) or $5/month (Starter)

#### Persona 2: OpenClaw Community Member (Secondary -- End User)

**Profile:** Technical user running OpenClaw for AI agent tasks. Has an Apple Silicon Mac (M1/M2/M3 Pro, Max, or Ultra). Interested in self-hosting but not willing to spend hours configuring.

**Pain Points:**
- ElevenLabs subscription feels expensive for hobby/personal use
- Privacy concerns about sending conversation text to cloud APIs
- Wants Talk Mode but $5-22/month is hard to justify

**Needs:**
- One-command install that works on first try
- Clear README that explains what this is and how to set it up
- Comparable audio quality to ElevenLabs (does not need to be identical)
- Works with their existing OpenClaw configuration

**Success Criteria:**
- Installs and runs in under 5 minutes
- Talk Mode works without reading source code
- Audio quality is acceptable for casual conversation

#### Persona 3: AI Agent Builder (Tertiary -- Developer)

**Profile:** Developer building AI agent applications (Open Interpreter, AutoGPT, CrewAI, custom agents) who needs a TTS API for voice output.

**Pain Points:**
- Per-character billing from ElevenLabs/OpenAI adds up during development
- Testing voice features requires active API subscriptions
- Limited control over voice characteristics with cloud APIs

**Needs:**
- Standard HTTP API compatible with existing ElevenLabs client libraries
- Reliable local server for development and testing
- Streaming support for real-time agent interactions

**Success Criteria:**
- Existing ElevenLabs client code works with a base URL change
- No per-character costs during development
- API is documented well enough to integrate without studying source code

### 2.2 User Journey: Burke Enables Talk Mode Locally

| Step | Action | System Response | Notes |
|------|--------|----------------|-------|
| 1 | Burke's sesame-tts server is already running on localhost:8880 | Server is healthy, OpenAI endpoint working | Pre-condition |
| 2 | Updates server to latest version with ElevenLabs endpoints | Server restarts, new endpoints available | `git pull && uv sync` |
| 3 | Verifies ElevenLabs endpoints work | `curl localhost:8880/v1/voices` returns voice list | Smoke test |
| 4 | Updates `openclaw.json` to point TTS at localhost | Config file change only | Set `messages.tts.elevenlabs.baseUrl` to `http://localhost:8880` |
| 5 | Starts a Talk Mode conversation | Audio streams back within 500ms | First real test |
| 6 | Speaks mid-generation to interrupt | Server cancels inference, audio stops | Interrupt works |
| 7 | Continues multi-turn conversation | Each turn picks up the conversational tone from prior turns | Context caching (v1.1) |
| 8 | Disables ElevenLabs subscription | $22/month saved | The goal |

### 2.3 User Journey: Community Member First Install

| Step | Action | System Response | Notes |
|------|--------|----------------|-------|
| 1 | Finds project on GitHub/Reddit/Discord | Reads README, watches demo video | Discovery |
| 2 | Runs one-command install | Script installs dependencies, downloads model | ~5 minutes |
| 3 | Configures OpenClaw | Edits config per README instructions | Two lines changed |
| 4 | Tests Talk Mode | Audio plays locally | First success |
| 5 | Stars the repo | Community growth | Target: 100+ in first month |

---

## 3. Goals and Non-Goals

### 3.1 Goals

| ID | Goal | Success Metric | Target |
|----|------|---------------|--------|
| G1 | ElevenLabs API compatibility for Talk Mode | OpenClaw Talk Mode works with base URL change only | 100% |
| G2 | Sub-second streaming latency | Time to first audio chunk | <= 500ms (Phase 1), <= 200ms (Phase 2) |
| G3 | Conversation context for natural prosody (v1.1) | Subjective A/B comparison: contextual vs cold-start | Noticeably more natural |
| G4 | Inference interrupt on client disconnect | Time from disconnect to inference cancellation | < 100ms |
| G5 | Talk Mode output formats supported | v1.0 essential formats (pcm_44100, pcm_24000, mp3_44100_128) pass validation | 3/3 formats |
| G6 | Zero configuration changes in client apps | Only base URL/API key changes required | No client code changes |
| G7 | Community adoption | GitHub stars within 30 days of launch | >= 100 |
| G8 | Eliminate ElevenLabs subscription | Monthly API cost after deployment | $0 |

### 3.2 Non-Goals (Explicitly Out of Scope)

| Item | Reason | Future Version |
|------|--------|---------------|
| Voice cloning | Requires significant ML pipeline work; structure for it, do not build | v1.2 |
| WebSocket streaming | ElevenLabs uses HTTP chunked transfer, not WebSocket | v2.0 |
| PyTorch/CUDA backend | MLX is macOS-only; Linux/Windows support is a separate effort | v2.0 |
| Docker container | MLX requires macOS bare metal | v2.0 |
| Web UI / playground | API-only for v1; CLI and curl are sufficient | v1.2 |
| Multi-model routing | Single model (CSM-1B); accept any model_id but route to CSM-1B | v2.0 |
| STS (speech-to-speech) | ElevenLabs offers STS but it requires STT integration | v2.0 |
| Pronunciation dictionaries | ElevenLabs API feature; low priority for local use | v1.2 |
| Real-time authentication | Localhost use case; accept any API key without validation | v1.1 |
| Multi-language support | CSM-1B is English-only; additional languages require model support | v1.2 |

### 3.3 Scope Boundaries

**This PRD covers:**
- ElevenLabs streaming endpoint implementation (server-side)
- Audio chunking and streaming pipeline
- Conversation context caching (deferred to v1.1)
- Voice ID mapping and management endpoints
- Interrupt/cancellation support
- ElevenLabs compatibility headers and error formats
- OpenClaw Talk Mode integration configuration
- Community launch documentation and install script

**This PRD does NOT cover:**
- Changes to OpenClaw source code (may require separate PR to OpenClaw)
- The existing OpenAI-compatible endpoint (covered by PRD.md v1.1)
- Model training, fine-tuning, or weight modification
- Hardware procurement or infrastructure

---

## 4. Functional Requirements

### FR1: ElevenLabs Streaming Endpoint

**Priority:** Must Have (v1.0)

Implement the primary ElevenLabs text-to-speech streaming endpoint.

**Endpoint:** `POST /v1/text-to-speech/{voice_id}/stream`

**Request:**
```json
{
  "text": "Hello, this is a streaming test.",
  "model_id": "eleven_v3",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": true
  },
  "output_format": "pcm_24000"
}
```

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `text` | string | Yes | -- | Text to synthesize, 1-4096 characters |
| `model_id` | string | No | "csm-1b" | Accepted but mapped to CSM-1B regardless of value |
| `voice_settings` | object | No | defaults | Accepted but mapped to preset parameters (see FR1.1) |
| `output_format` | string | No (query param) | "mp3_44100_128" | ElevenLabs format string (see FR3) |
| `optimize_streaming_latency` | integer | No | 0 | Accepted, mapped to stream buffer size (0-4) |

**Response:**
- `Content-Type`: Matches requested output format (e.g., `audio/mpeg` for MP3, `application/octet-stream` for PCM)
- `Transfer-Encoding: chunked`
- Body: Raw audio bytes streamed in chunks as they become available

**FR1.1: Voice Settings Mapping**

ElevenLabs `voice_settings` fields map to Sesame generation parameters:

| ElevenLabs Field | Sesame Mapping | Notes |
|-----------------|----------------|-------|
| `stability` | `1.0 - temperature` | Higher stability = lower temperature |
| `similarity_boost` | Ignored (v1) | Sesame uses voice prompts, not embeddings |
| `style` | Ignored (v1) | No direct equivalent |
| `use_speaker_boost` | `voice_match=True` | Already default behavior |

**FR1.2: Model ID Mapping**

| Client Sends | Server Uses | Log Message |
|-------------|-------------|-------------|
| `eleven_monolingual_v1` | `csm-1b` | "Mapped ElevenLabs model eleven_monolingual_v1 to csm-1b" |
| `eleven_multilingual_v2` | `csm-1b` | "Mapped ElevenLabs model eleven_multilingual_v2 to csm-1b" |
| `eleven_turbo_v2` | `csm-1b` | "Mapped ElevenLabs model eleven_turbo_v2 to csm-1b" |
| `eleven_turbo_v2_5` | `csm-1b` | "Mapped ElevenLabs model eleven_turbo_v2_5 to csm-1b" |
| `eleven_v3` | `csm-1b` | "Mapped ElevenLabs model eleven_v3 to csm-1b" |
| `csm-1b` | `csm-1b` | (no mapping log) |
| Any other | `csm-1b` | "Unknown model {id}, using csm-1b" |

**Acceptance Criteria:**
- AC-FR1.1: `POST /v1/text-to-speech/conversationalB/stream` with valid JSON body returns chunked audio stream
- AC-FR1.2: First audio chunk arrives within 500ms on M3 Ultra
- AC-FR1.3: Complete audio is playable and contains the spoken text
- AC-FR1.4: `voice_settings` are accepted without error regardless of values
- AC-FR1.5: Any `model_id` is accepted and mapped to CSM-1B
- AC-FR1.6: Response includes `Transfer-Encoding: chunked` header

```bash
# AC-FR1.1 + FR1.6: Streaming with chunked transfer
curl -v -N -X POST http://localhost:8880/v1/text-to-speech/conversationalB/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Sesame.","model_id":"csm-1b","output_format":"pcm_24000"}' \
  --output test_fr1.pcm 2>&1 | grep -i "transfer-encoding"
# Expected: "Transfer-Encoding: chunked" in response headers

# AC-FR1.4: voice_settings accepted
curl -N -X POST http://localhost:8880/v1/text-to-speech/conversationalB/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Settings test.","model_id":"csm-1b","voice_settings":{"stability":0.9,"similarity_boost":0.1,"style":1.0},"output_format":"pcm_24000"}' \
  --output test_fr1_settings.pcm
# Expected: 200 OK, playable audio

# AC-FR1.5: Unknown model_id accepted
curl -N -X POST http://localhost:8880/v1/text-to-speech/conversationalB/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Model mapping test.","model_id":"eleven_turbo_v2_5","output_format":"pcm_24000"}' \
  --output test_fr1_model.pcm
# Expected: 200 OK, playable audio, server log shows model mapping
```

### FR2: ElevenLabs Non-Streaming Endpoint

**Priority:** Must Have (v1.0)

Implement the non-streaming variant for clients that prefer full audio buffers.

**Endpoint:** `POST /v1/text-to-speech/{voice_id}`

Same request body as FR1. Response is the complete audio buffer (not chunked).

**Response:**
- `Content-Type`: Matches requested output format
- Body: Complete audio bytes

**Acceptance Criteria:**
- AC-FR2.1: `POST /v1/text-to-speech/conversationalB` returns complete audio buffer
- AC-FR2.2: Response does NOT use chunked transfer encoding
- AC-FR2.3: Audio is identical in quality to the streaming endpoint

```bash
# AC-FR2.1 + FR2.2: Non-streaming (no -N flag)
curl -v -X POST http://localhost:8880/v1/text-to-speech/conversationalB?output_format=pcm_24000 \
  -H "Content-Type: application/json" \
  -d '{"text":"Non-streaming test.","model_id":"csm-1b"}' \
  --output test_fr2.pcm 2>&1 | grep -i "content-length"
# Expected: Content-Length header present (not chunked), playable PCM
ffplay -f s16le -ar 24000 -ac 1 test_fr2.pcm
```

### FR3: ElevenLabs Output Formats

**Priority:** Must Have (v1.0)

Support the 3 output formats required for Talk Mode compatibility. The `output_format` is passed as a query parameter on the streaming endpoint. Additional ElevenLabs format strings are deferred to a later compatibility release.

**v1.0 Formats (covers all Talk Mode defaults):**

| Format String | Sample Rate | Bit Depth / Bitrate | Implementation | Use Case |
|--------------|-------------|---------------------|----------------|----------|
| `pcm_44100` | 44,100 Hz | 16-bit signed LE | Resample from native 24kHz | macOS/iOS Talk Mode default |
| `pcm_24000` | 24,000 Hz | 16-bit signed LE | Native rate, no resampling | Android Talk Mode default |
| `mp3_44100_128` | 44,100 Hz | 128 kbps | ffmpeg encode + resample | Forced MP3 streaming option |

**Deferred Formats (future compatibility release):**
`pcm_16000`, `pcm_22050`, `mp3_22050_32`, `mp3_44100_32`, `mp3_44100_64`, `mp3_44100_96`, `mp3_44100_192`

**Resampling Implementation:**
- Use `scipy.signal.resample` for sample rate conversion
- For PCM formats: resample numpy array, convert to int16, stream raw bytes
- For MP3 formats: pipe resampled PCM through ffmpeg with specified bitrate
- `pcm_24000` is the fast path: no resampling needed (native CSM-1B sample rate is 24000)
- `pcm_44100` requires upsampling from 24kHz to 44.1kHz

**Content-Type Mapping:**

| Format Family | Content-Type |
|--------------|-------------|
| `pcm_*` | `application/octet-stream` |
| `mp3_*` | `audio/mpeg` |

**Acceptance Criteria:**

- AC-FR3.1: All 3 v1.0 output format strings are accepted without error
```bash
# AC-FR3.1a: PCM 44100 (macOS/iOS Talk Mode default)
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_44100" \
  -H "Content-Type: application/json" \
  -d '{"text":"Format test.","model_id":"csm-1b"}' \
  --output test_pcm44100.pcm
# Expected: valid PCM file, playable at 44100 Hz 16-bit mono

# AC-FR3.1b: PCM 24000 (Android Talk Mode default, native rate)
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d '{"text":"Format test.","model_id":"csm-1b"}' \
  --output test_pcm24000.pcm
# Expected: valid PCM file, playable at 24000 Hz 16-bit mono

# AC-FR3.1c: MP3 44100 128kbps
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=mp3_44100_128" \
  -H "Content-Type: application/json" \
  -d '{"text":"Format test.","model_id":"csm-1b"}' \
  --output test_mp3.mp3
# Expected: valid MP3 file, playable in any audio player
```
- AC-FR3.2: `pcm_24000` output has zero resampling artifacts (native pass-through)
- AC-FR3.3: Deferred format strings return 400 error with message indicating the format is not yet supported and listing the 3 available formats
- AC-FR3.4: Unknown format strings return 400 error with list of supported formats

### FR4: Voice Management Endpoints

**Priority:** Must Have (v1.0)

Implement ElevenLabs-compatible voice listing and discovery.

**FR4.1: Voice List**

**Endpoint:** `GET /v1/voices`

**Response:**
```json
{
  "voices": [
    {
      "voice_id": "conversationalB",
      "name": "Conversational B",
      "category": "generated",
      "fine_tuning": {
        "is_allowed_to_fine_tune": false,
        "fine_tuning_requested": false
      },
      "labels": {
        "accent": "american",
        "gender": "neutral",
        "age": "young",
        "use_case": "conversational"
      },
      "description": "Second conversational voice, slightly different character",
      "preview_url": "http://localhost:8880/v1/voices/conversationalB/preview",
      "available_for_tiers": [],
      "settings": {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": true
      },
      "sharing": null,
      "high_quality_base_model_ids": ["csm-1b"]
    }
  ]
}
```

**FR4.2: Voice Preview**

**Endpoint:** `GET /v1/voices/{voice_id}/preview`

**Behavior:**
- Generate a 5-second audio sample using the requested voice
- Cache the preview audio on first request (regenerate only if voice config changes)
- Return as MP3 by default
- Preview text: "Hello, I'm your local text-to-speech voice, powered by Sesame."

**FR4.3: ElevenLabs Voice ID Mapping**

Voice IDs in ElevenLabs are alphanumeric strings (e.g., `21m00Tcm4TlvDq8ikWAM`). The server must handle:

| Scenario | Behavior |
|----------|----------|
| Known Sesame voice ID (e.g., `conversationalB`) | Route to matching preset |
| Unknown ElevenLabs voice ID (any alphanumeric string) | Route to default voice, log mapping |
| Internal preset name with underscores (e.g., `conversational_b`) | Route to matching preset |

**Voice Alias System:**
The existing `VOICE_ALIASES` dict in `server.py` maps ElevenLabs-safe alphanumeric IDs to internal preset names:
- `conversationalB` -> `conversational_b`
- `conversationalA` -> `conversational`

This system extends to the ElevenLabs endpoints. Any voice ID in the URL path is resolved through this alias system first, then falls back to the preset manager, then falls back to the default voice.

**Acceptance Criteria:**
- AC-FR4.1: `GET /v1/voices` returns a JSON array with all available voices in ElevenLabs format
- AC-FR4.2: Each voice entry includes `voice_id`, `name`, `labels`, `preview_url`, and `settings`
- AC-FR4.3: `GET /v1/voices/conversationalB/preview` returns playable audio
- AC-FR4.4: Unknown voice IDs in TTS requests fall back to default voice with a log message
- AC-FR4.5: Voice IDs with underscores (Sesame native) and without (ElevenLabs-safe) both resolve correctly

### FR5: Streaming Implementation

**Priority:** Must Have (v1.0 -- Phase 1)

Implement buffered chunked streaming that delivers audio as it is generated.

**FR5.1: Phase 1 -- Buffered Chunked Streaming (v1.0)**

**Mechanism:**
1. `TTSEngine.generate()` currently collects all `GenerationResult` objects from `model.generate()` before returning
2. Modify to yield `GenerationResult` objects as they are produced (generator pattern)
3. New `StreamingAudioPipeline` module:
   - Receives audio frames from the generator
   - Buffers N frames (configurable, default 6 via `STREAM_BUFFER_FRAMES`)
   - Decodes buffered frames through Mimi codec
   - Converts to requested output format
   - Yields audio chunks for HTTP streaming
4. FastAPI `StreamingResponse` delivers chunks to the client

**Buffer Strategy:**
- First chunk: Buffer `STREAM_BUFFER_FRAMES` frames (default 6), decode, convert, send
- Subsequent chunks: Buffer 1-2 frames, decode, convert, send (lower latency after initial)
- Final chunk: Flush remaining frames, add silence padding if needed

**Latency Budget (Phase 1 target: <= 500ms first chunk):**

| Step | Target Time | Notes |
|------|-------------|-------|
| Request parsing | < 5ms | FastAPI + Pydantic |
| Semaphore acquisition | 0-Nms | Depends on queue depth |
| MLX token generation (6 frames) | ~300ms | M3 Ultra estimate |
| Mimi decode (6 frames) | ~50ms | Audio codec |
| Resample + format convert | ~20ms | For non-native rates |
| HTTP chunk send | < 5ms | Localhost |
| **Total** | **~380ms** | Under 500ms budget |

**FR5.2: Phase 2 -- True Frame Streaming (v1.1)**

**Mechanism:**
1. Stream decoded audio frames individually as they complete
2. Requires hooking into the MLX generation loop at a lower level
3. Target: first audio chunk within 200ms (1-2 frames)
4. May require upstream changes to mlx-audio's generation API

This is a v1.1 goal. The architecture must not prevent this optimization, but it is not required for v1.0.

**Acceptance Criteria:**
- AC-FR5.1: Streaming endpoint delivers first audio chunk within 500ms on M3 Ultra for a short sentence
- AC-FR5.2: Complete streamed audio is identical in quality to non-streaming endpoint
- AC-FR5.3: Streaming works correctly for all 3 v1.0 output formats
- AC-FR5.4: `STREAM_BUFFER_FRAMES` is configurable via environment variable
- AC-FR5.5: Streaming handles empty input gracefully (returns empty response or error)

### FR6: Interrupt Support

**Priority:** Must Have (v1.0)

Cancel MLX inference when the client disconnects mid-stream.

**Mechanism:**
1. Streaming endpoint monitors the client connection via FastAPI's `Request.is_disconnected()`
2. On disconnect detection, set a cancellation flag
3. The generation loop checks the flag between frames and raises `GenerationCancelled`
4. Partial audio generated up to the interrupt point is not sent (client has already disconnected)
5. MLX resources (memory, compute) are freed immediately
6. Semaphore is released so the next request can proceed

**Implementation Detail:**
```
Client connects -> Start generation -> Yield chunks
                                         |
Client disconnects -----> Set cancel flag
                                         |
Generation loop checks flag -> Raise GenerationCancelled
                                         |
Cleanup: free memory, release semaphore, log interrupt
```

**Logging:**
- Log level INFO: "Generation interrupted: voice={voice_id} frames_generated={N} frames_estimated={M} reason=client_disconnect"

**Acceptance Criteria:**
- AC-FR6.1: Client disconnect during streaming stops inference within 100ms
- AC-FR6.2: Memory used by interrupted generation is freed
- AC-FR6.3: Semaphore is released after interruption (next request is not blocked)
- AC-FR6.4: Server does not crash or enter an inconsistent state on interrupt
- AC-FR6.5: Interrupts are logged with frame count and reason

```bash
# AC-FR6.1 + FR6.3: Interrupt and recover
# Terminal 1: start long generation, kill after 1s
timeout 1 curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a very long sentence that should take several seconds to generate fully so we can test interruption behavior.","model_id":"csm-1b"}' \
  --output /dev/null

# Terminal 1: immediately verify server accepts new request
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d '{"text":"Recovery.","model_id":"csm-1b"}' \
  --output test_recovery.pcm
# Expected: second request succeeds, server log shows "Generation interrupted" for first request
```

### FR7: Conversation Context Caching

**Priority:** Deferred to v1.1 (design and implementation)

Cache recent audio turns server-side and pass as context to CSM-1B for natural prosody across conversation turns.

**FR7.1: Context Cache Design**

**Cache Structure:**
- Key: Session identifier (derived from client IP + voice_id, or explicit `session_id` header)
- Value: Ordered list of `ContextTurn` objects (text + audio array + speaker_id + timestamp)
- Eviction: LRU with dual limits:
  - Maximum turns: 10 (configurable via `CONTEXT_MAX_TURNS`)
  - Maximum audio duration: 60 seconds (configurable via `CONTEXT_MAX_SECONDS`)
  - Whichever limit is hit first triggers eviction of oldest turns

**Session Identification:**
- Primary: `X-Session-Id` header (if provided by client)
- Fallback: Hash of `client_ip + voice_id`
- This allows Talk Mode's sequential requests to automatically share context without client changes

**FR7.2: Context Integration with CSM-1B**

The mlx-audio `Model.generate()` method already accepts a `context` parameter (list of `Segment` objects). Each `Segment` contains:
- `text`: The text that was spoken
- `audio`: The audio tensor (mx.array)
- `speaker`: Speaker ID (int)

After each generation, the server:
1. Stores the generated audio + text + speaker as a new `ContextTurn`
2. On the next request in the same session, passes recent turns as `context` to `model.generate()`
3. CSM-1B uses this context to maintain prosody, rhythm, and emotional tone

**FR7.3: Context API (Optional, v1.1)**

An explicit context API for clients that want fine-grained control:

```
POST /v1/text-to-speech/{voice_id}/stream
{
  "text": "That sounds great!",
  "context": [
    {"text": "I just got the job!", "audio_ref": "turn_001"},
    {"text": "Congratulations!", "audio_ref": "turn_002"}
  ]
}
```

**Acceptance Criteria (v1.1 -- Design and Implementation):**
- AC-FR7.5: Sequential requests from the same session automatically use conversation context
- AC-FR7.6: Context cache respects turn and duration limits
- AC-FR7.7: Context-enabled generation produces noticeably more natural multi-turn speech than cold-start
- AC-FR7.8: Context cache memory does not grow unbounded

### FR8: ElevenLabs Compatibility Endpoints

**Priority:** Must Have (v1.0)

Implement supporting endpoints that ElevenLabs clients may query for account status, model info, etc.

**FR8.1: Models Endpoint (ElevenLabs Format)**

**Endpoint:** `GET /v1/models`

This endpoint already exists for OpenAI format. Detect the client context (ElevenLabs vs OpenAI) based on request headers:
- If `xi-api-key` header is present, return ElevenLabs format
- If `Authorization: Bearer` header is present with an OpenAI-style key, return OpenAI format
- Default: Return both formats or use a shared format

**ElevenLabs Response:**
```json
[
  {
    "model_id": "csm-1b",
    "name": "Sesame CSM-1B",
    "can_do_text_to_speech": true,
    "can_do_voice_conversion": false,
    "can_be_finetuned": false,
    "can_use_style": false,
    "can_use_speaker_boost": true,
    "serves_pro_voices": false,
    "token_cost_factor": 0,
    "description": "Sesame CSM-1B -- local conversational speech model via MLX",
    "requires_alpha_access": false,
    "max_characters_request_free_user": 999999999,
    "max_characters_request_subscribed_user": 999999999,
    "languages": [
      {"language_id": "en", "name": "English"}
    ]
  }
]
```

**FR8.2: User Subscription Endpoint**

**Endpoint:** `GET /v1/user/subscription`

**Response:**
```json
{
  "tier": "local",
  "character_count": 0,
  "character_limit": 999999999,
  "can_extend_character_limit": false,
  "allowed_to_extend_character_limit": false,
  "next_character_count_reset_unix": 0,
  "voice_limit": 999,
  "max_voice_add_edits": 999,
  "voice_add_edit_counter": 0,
  "professional_voice_limit": 0,
  "can_extend_voice_limit": false,
  "can_use_instant_voice_cloning": false,
  "can_use_professional_voice_cloning": false,
  "currency": "usd",
  "status": "active",
  "billing_period": "unlimited",
  "character_refresh_period": "unlimited",
  "next_invoice": null
}
```

**FR8.3: User Info Endpoint**

**Endpoint:** `GET /v1/user`

**Response:**
```json
{
  "subscription": {
    "tier": "local",
    "character_limit": 999999999,
    "status": "active"
  },
  "is_new_user": false,
  "xi_api_key": "local"
}
```

**Acceptance Criteria:**
- AC-FR8.1: `GET /v1/models` returns ElevenLabs-format model info when `xi-api-key` header is present
- AC-FR8.2: `GET /v1/user/subscription` returns unlimited subscription info
- AC-FR8.3: `GET /v1/user` returns user info with local subscription
- AC-FR8.4: All endpoints return valid JSON matching ElevenLabs schema structure

```bash
# AC-FR8.1: Models endpoint (ElevenLabs format)
curl -s -H "xi-api-key: local" http://localhost:8880/v1/models | jq '.[0].model_id'
# Expected: "csm-1b"

# AC-FR8.2: Subscription endpoint
curl -s http://localhost:8880/v1/user/subscription | jq '.tier'
# Expected: "local"

# AC-FR8.3: User endpoint
curl -s http://localhost:8880/v1/user | jq '.subscription.status'
# Expected: "active"
```

### FR9: Authentication Compatibility

**Priority:** Must Have (v1.0)

Accept ElevenLabs and OpenAI authentication headers without requiring valid credentials.

**Behavior:**

| Header | Handling |
|--------|---------|
| `xi-api-key: <any value>` | Accept, ignore value, log at DEBUG level |
| `Authorization: Bearer <any value>` | Accept, ignore value, log at DEBUG level |
| No auth headers | Accept (localhost is trusted) |

**Future (v1.1):** Optional configurable API key for remote/LAN access via `API_KEY` environment variable.

**Acceptance Criteria:**
- AC-FR9.1: Requests with `xi-api-key` header succeed regardless of header value
- AC-FR9.2: Requests with `Authorization: Bearer` header succeed regardless of token value
- AC-FR9.3: Requests with no auth headers succeed
- AC-FR9.4: Auth headers are logged at DEBUG level (not INFO, to avoid log noise)

```bash
# AC-FR9.1: xi-api-key
curl -s -H "xi-api-key: literally-anything" http://localhost:8880/v1/voices | jq '.voices | length'
# AC-FR9.2: Bearer token
curl -s -H "Authorization: Bearer sk-fake-key-12345" http://localhost:8880/v1/voices | jq '.voices | length'
# AC-FR9.3: No auth
curl -s http://localhost:8880/v1/voices | jq '.voices | length'
# Expected: all three return the same voice count
```

### FR10: ElevenLabs Error Format

**Priority:** Must Have (v1.0)

Return ElevenLabs-style error responses for ElevenLabs endpoints.

**Error Format:**
```json
{
  "detail": {
    "status": "invalid_request",
    "message": "Voice 'unknown' not found. Available: conversationalB, conversationalA"
  }
}
```

**Endpoint Detection:**
- Requests to `/v1/text-to-speech/*`, `/v1/voices/*`, `/v1/user/*` get ElevenLabs error format
- Requests to `/v1/audio/*`, `/v1/models` (OpenAI-style) get OpenAI error format
- Requests to `/health` get simple JSON errors

**Error Mapping:**

| Condition | HTTP Status | ElevenLabs `status` | Message |
|-----------|-------------|-------------------|---------|
| Empty/missing text | 400 | `invalid_request` | "Text is required and must be non-empty" |
| Text too long | 400 | `invalid_request` | "Text exceeds maximum length of 4096 characters" |
| Invalid output format | 400 | `invalid_request` | "Invalid output format. Supported: pcm_16000, ..." |
| Model not loaded | 503 | `server_error` | "Model not ready. Please wait and retry." |
| OOM during generation | 503 | `server_error` | "Insufficient memory for generation" |
| Generation failed | 500 | `server_error` | "Audio generation failed" |

**Acceptance Criteria:**
- AC-FR10.1: ElevenLabs endpoints return errors in `{"detail": {...}}` format
- AC-FR10.2: OpenAI endpoints continue to return errors in `{"error": {...}}` format
- AC-FR10.3: HTTP status codes match ElevenLabs conventions

### FR11: OpenClaw Talk Mode Integration

**Priority:** Must Have (v1.0)

Ensure the server works as a drop-in replacement for ElevenLabs in OpenClaw Talk Mode.

**Confirmed OpenClaw Configuration (from source):**

OpenClaw reads TTS provider settings from `openclaw.json`. The following config block is already working for TTS voice notes via the existing OpenAI-compatible endpoint. Talk Mode reads `baseUrl` from the same config block. Talk Mode output format is controlled separately via `talk.outputFormat`.

```json
{
  "messages": {
    "tts": {
      "provider": "elevenlabs",
      "elevenlabs": {
        "baseUrl": "http://localhost:8880",
        "apiKey": "local",
        "voiceId": "conversationalB",
        "modelId": "csm1b"
      }
    }
  }
}
```

No environment variable override or OpenClaw source changes are needed. This is a config-file-only integration.

**Required Server Behaviors for Talk Mode:**
1. Accept `POST /v1/text-to-speech/{voice_id}/stream` with chunked response
2. Accept `output_format` query parameter (Talk Mode uses `pcm_24000`)
3. Handle rapid sequential requests (Talk Mode sends many short sentences)
4. Support interrupt (user starts speaking -> cancel current generation)
5. Accept `xi-api-key` header without validation

**Acceptance Criteria:**
- AC-FR11.1: OpenClaw Talk Mode produces voice output using the local server
- AC-FR11.2: Talk Mode interrupt (user speaks during generation) stops audio immediately
- AC-FR11.3: Sequential Talk Mode requests complete without errors or resource leaks
- AC-FR11.4: Audio quality in Talk Mode is comparable to ElevenLabs for conversational speech
- AC-FR11.5: Configuration requires only `openclaw.json` changes (no source modifications to OpenClaw)

```bash
# AC-FR11: Verify server accepts Talk Mode-style request
# This simulates what OpenClaw sends when Talk Mode is active
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_44100" \
  -H "xi-api-key: local" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a Talk Mode test sentence.","model_id":"csm1b"}' \
  --output test_talkmode.pcm
ffplay -f s16le -ar 44100 -ac 1 test_talkmode.pcm
# Expected: playable audio at 44100 Hz (macOS Talk Mode default)
```

### FR12: Enhanced Health Endpoint

**Priority:** Should Have (v1.0)

Extend the existing health endpoint with streaming and queue metrics.

**Response:**
```json
{
  "status": "ok",
  "model": {
    "model_loaded": true,
    "model_id": "mlx-community/csm-1b",
    "sample_rate": 24000,
    "peak_memory_gb": 6.2
  },
  "ffmpeg": true,
  "voices": ["conversationalB", "conversationalA", "conversational_b", "conversational"],
  "queue_depth": 0,
  "uptime_seconds": 3600,
  "requests_served": 42,
  "avg_generation_time_ms": 2100,
  "context_cache_sessions": 0,
  "streaming_enabled": true
}
```

**Acceptance Criteria:**
- AC-FR12.1: Health endpoint includes queue depth, request count, and average generation time
- AC-FR12.2: Health endpoint includes context cache session count (0 until v1.1)
- AC-FR12.3: Health endpoint includes streaming status

### FR13: Enhanced Configuration

**Priority:** Must Have (v1.0)

Extend `config.py` with streaming and ElevenLabs-specific settings.

**New Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_BUFFER_FRAMES` | 6 | Frames to buffer before first streaming chunk |
| `STREAM_CHUNK_SIZE` | 4096 | Bytes per stream chunk for MP3 output |
| `DEFAULT_OUTPUT_FORMAT` | `pcm_24000` | Default ElevenLabs output format |
| `WARMUP_ON_START` | `true` | Pre-generate audio on startup to warm caches |
| `MAX_QUEUE_DEPTH` | 3 | Maximum queued requests before returning 429 |

**Acceptance Criteria:**
- AC-FR13.1: All new settings are configurable via environment variable and `.env` file
- AC-FR13.2: Default values are sensible for the M3 Ultra target hardware
- AC-FR13.3: `.env.example` is updated with all new settings and documentation comments

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| First audio chunk latency (streaming) | <= 500ms (Phase 1) | Timestamp from request to first chunk received by client |
| First audio chunk latency (streaming) | <= 200ms (Phase 2, v1.1) | Same as above |
| Full sentence generation | < 2 seconds | Time from request to complete audio for 10-20 word input |
| Format conversion overhead | < 100ms per chunk | Timestamp delta between raw audio and formatted output |
| PCM resampling overhead | < 50ms per chunk | Benchmark per-chunk resampling time |
| Streaming chunk interval | 50-200ms | Time between consecutive chunks |
| Interrupt response time | < 100ms | Time from client disconnect to inference cancellation |
| Startup time (cold, no warmup) | < 30 seconds | Time from process start to health returning "ok" |
| Startup time (with warmup) | < 45 seconds | Including warmup generation |
| Peak memory usage | < 8GB unified memory | `mx.get_peak_memory()` during generation with context |
| Steady-state memory | < 6GB unified memory | After model load, no active generation |
| Context cache memory | < 500MB | For 10 turns of 6 seconds each |

### 5.2 Reliability

| Requirement | Detail |
|-------------|--------|
| Crash recovery | launchd auto-restart with max 3 retries, 10-second delay |
| Memory leak prevention | Context cache LRU eviction; `mx.clear_cache()` after each generation |
| Queue overflow | Return HTTP 429 with `Retry-After` header when queue depth exceeds `MAX_QUEUE_DEPTH` |
| Interrupt safety | Generation cancellation must not corrupt model state or leak memory |
| Long-running stability | 24+ hours continuous uptime without degradation, processing 500+ requests |
| Graceful degradation | If context cache fails, fall back to context-free generation |

### 5.3 Compatibility

| Requirement | Detail |
|-------------|--------|
| ElevenLabs API version | Compatible with ElevenLabs API v1 (as of 2026-02) |
| OpenAI API version | Maintains existing compatibility (co-exists on same server) |
| OpenClaw version | Compatible with current OpenClaw Talk Mode |
| Python version | 3.10+ |
| macOS version | macOS Tahoe 26.x (Apple Silicon required) |
| mlx-audio version | 0.3.x (pin in pyproject.toml) |
| ffmpeg version | 6.x or 7.x (system-installed) |

### 5.4 Security

| Requirement | Detail |
|-------------|--------|
| Authentication | Accept any API key value; no validation for localhost |
| Input sanitization | Sanitize text input before passing to model and ffmpeg |
| Network exposure | Bind to `0.0.0.0` by default (configurable), but designed for localhost/LAN use |
| No telemetry | Zero external network calls after model download |
| No credential storage | API keys are accepted but never stored or logged at INFO level |
| ffmpeg injection | Text input must not be passed to ffmpeg command line (only piped as audio data) |

### 5.5 Observability

| Requirement | Detail |
|-------------|--------|
| Structured logging | JSON format to `~/.cache/sesame-tts/server.log` |
| Request logging | Every request: voice, format, input_length, generation_time, streamed flag |
| Error logging | Full stack trace for unexpected errors |
| Performance logging | Generation time, first chunk latency, total latency per request |
| Interrupt logging | Log all cancellations with frame count and reason |
| Health endpoint | Enhanced with queue depth, request count, context cache stats |

---

## 6. Acceptance Criteria (End-to-End)

These are the top-level acceptance criteria that validate the product is complete and ready for launch.

### 6.1 Streaming Smoke Test

```bash
# AC1: Streaming smoke test -- chunked transfer, first bytes within 500ms, playable PCM
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "xi-api-key: local" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello Burke, Talk Mode is working locally now.","model_id":"csm-1b"}' \
  --output test_stream.pcm

# Verify: play as raw PCM (macOS)
ffplay -f s16le -ar 24000 -ac 1 test_stream.pcm
```

**Pass criteria:** File is playable as raw PCM (24kHz, 16-bit, mono) and contains the spoken text. First bytes arrive within 500ms.

### 6.2 All Formats Test

Test all 3 v1.0 output formats:

```bash
# AC2a: PCM 44100 (macOS/iOS Talk Mode default)
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_44100" \
  -H "Content-Type: application/json" \
  -d '{"text":"Format test forty-four one hundred.","model_id":"csm-1b"}' \
  --output test_pcm44100.pcm
ffplay -f s16le -ar 44100 -ac 1 test_pcm44100.pcm

# AC2b: PCM 24000 (Android Talk Mode default, native rate)
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d '{"text":"Format test twenty-four thousand.","model_id":"csm-1b"}' \
  --output test_pcm24000.pcm
ffplay -f s16le -ar 24000 -ac 1 test_pcm24000.pcm

# AC2c: MP3 44100 128kbps
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=mp3_44100_128" \
  -H "Content-Type: application/json" \
  -d '{"text":"Format test MP3.","model_id":"csm-1b"}' \
  --output test_mp3.mp3
ffplay test_mp3.mp3

# AC2d: Deferred format returns 400
curl -s -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_16000" \
  -H "Content-Type: application/json" \
  -d '{"text":"Should fail.","model_id":"csm-1b"}' | jq .
# Expected: 400 with message listing supported formats
```

**Pass criteria:** All 3 v1.0 formats produce valid, playable audio. Deferred formats return 400 with a helpful message.

### 6.3 Voice Management Test

```bash
# AC3a: List voices
curl -s http://localhost:8880/v1/voices | jq '.voices[].voice_id'
# Expected: ["conversationalB", "conversationalA", ...]

# AC3b: Voice preview
curl -s http://localhost:8880/v1/voices/conversationalB/preview --output preview.mp3
ffplay preview.mp3
# Expected: playable MP3 with preview speech
```

**Pass criteria:** Returns all configured voices with ElevenLabs-compatible metadata. Preview endpoint returns playable audio.

### 6.4 Interrupt Test

```bash
# AC4: Start long generation, kill after 1 second, verify server recovers
LONG_TEXT="This is a very long paragraph that should take several seconds to generate. It contains many sentences so that the streaming endpoint will be actively generating audio when we disconnect. The server must detect the disconnect, cancel the MLX inference, free memory, release the semaphore, and be ready for the next request immediately."

# Start streaming, kill after 1s
timeout 1 curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$LONG_TEXT\",\"model_id\":\"csm-1b\"}" \
  --output /dev/null 2>/dev/null

# Immediately send a new request -- should not block
curl -N -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d '{"text":"Quick follow-up.","model_id":"csm-1b"}' \
  --output test_interrupt_recovery.pcm
# Expected: second request completes normally, no errors in server logs
```

**Pass criteria:** First request is cancelled. Second request proceeds without delay. No resource leaks.

### 6.5 OpenClaw Talk Mode Test

1. Configure OpenClaw with `ELEVENLABS_BASE_URL=http://localhost:8880`
2. Start a Talk Mode conversation
3. Have a 5-turn back-and-forth conversation
4. Interrupt the AI mid-sentence by speaking

**Pass criteria:** All turns produce natural audio. Interrupt stops generation. No errors in server logs.

### 6.6 Compatibility Headers Test

```bash
# AC6a: ElevenLabs-style auth
curl -s -H "xi-api-key: sk-whatever" http://localhost:8880/v1/voices | jq '.voices | length'

# AC6b: Bearer auth
curl -s -H "Authorization: Bearer sk-whatever" http://localhost:8880/v1/voices | jq '.voices | length'

# AC6c: No auth
curl -s http://localhost:8880/v1/voices | jq '.voices | length'

# Expected: all three return the same count
```

**Pass criteria:** All three requests succeed with identical responses.

### 6.7 Error Resilience Test

```bash
# AC7a: Empty text -> 400
curl -s -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream" \
  -H "Content-Type: application/json" \
  -d '{"text":"","model_id":"csm-1b"}' | jq .
# Expected: {"detail":{"status":"invalid_request","message":"Text is required..."}}

# AC7b: Text too long -> 400
python3 -c "import json; print(json.dumps({'text':'x'*5000,'model_id':'csm-1b'}))" | \
  curl -s -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream" \
  -H "Content-Type: application/json" -d @- | jq .
# Expected: 400 with length error

# AC7c: Unknown voice ID -> falls back to default
curl -s -X POST "http://localhost:8880/v1/text-to-speech/unknownVoice123/stream?output_format=pcm_24000" \
  -H "Content-Type: application/json" \
  -d '{"text":"Fallback test.","model_id":"csm-1b"}' \
  --output test_fallback.pcm
# Expected: 200, plays audio using default voice, server log shows fallback message

# AC7d: Unsupported format -> 400
curl -s -X POST "http://localhost:8880/v1/text-to-speech/conversationalB/stream?output_format=ogg_44100" \
  -H "Content-Type: application/json" \
  -d '{"text":"Bad format.","model_id":"csm-1b"}' | jq .
# Expected: 400 with supported format list
```

**Pass criteria:** All error cases return correct format and status code. Server continues operating.

### 6.8 Performance Benchmark

| Test | Target |
|------|--------|
| Short sentence (10 words), pcm_24000, streaming | First chunk <= 500ms |
| Short sentence (10 words), pcm_24000, non-streaming | Total time < 2s |
| Paragraph (50 words), pcm_24000, streaming | First chunk <= 500ms, complete < 5s |
| 10 sequential short requests | No degradation in latency over time |
| Memory after 100 requests | No growth beyond initial steady state |

### 6.9 Installation Test (Clean Machine)

1. Fresh macOS with Apple Silicon, Homebrew, and Python 3.10+ installed
2. Run one-command install script
3. Verify server starts and responds to health check
4. Configure and test OpenClaw integration

**Pass criteria:** Complete setup in under 5 minutes (excluding model download time).

---

## 7. Release Phases

### Phase v1.0 -- "It Streams" (Target: 1 week from start)

**Scope:**
- ElevenLabs streaming endpoint (`POST /v1/text-to-speech/{voice_id}/stream`)
- ElevenLabs non-streaming endpoint (`POST /v1/text-to-speech/{voice_id}`)
- 3 v1.0 output formats (pcm_44100, pcm_24000, mp3_44100_128)
- Voice management endpoints (`GET /v1/voices`, preview)
- Buffered chunked streaming (first chunk <= 500ms)
- Interrupt support (cancel on disconnect)
- Compatibility endpoints (`/v1/user/subscription`, `/v1/user`, ElevenLabs `/v1/models`)
- Authentication header acceptance
- ElevenLabs-style error responses
- Enhanced health endpoint
- OpenClaw Talk Mode integration and documentation
- Community launch (README, install script, announcements)

**Exit Criteria:**
- All v1.0 acceptance criteria pass
- OpenClaw Talk Mode works end-to-end
- README is complete with install instructions and OpenClaw config
- Install script works on clean Apple Silicon Mac

### Phase v1.1 -- "It Remembers" (Target: 2-3 weeks from v1.0)

**Scope:**
- Conversation context caching design and implementation (LRU, max 10 turns / 60 seconds)
- Automatic session tracking for Talk Mode
- Context cache configuration settings in `config.py`
- True frame streaming (first chunk <= 200ms)
- Voice preview endpoint with cached audio
- Performance benchmark suite
- Optional API key for remote access

**Exit Criteria:**
- Context-enabled conversations are noticeably more natural than cold-start
- First chunk latency <= 200ms
- Benchmark results published in README

### Phase v1.2 -- "It's You" (Target: 4-6 weeks from v1.0)

**Scope:**
- Voice cloning from reference audio (`POST /v1/voices/add`)
- Custom voice management
- Web playground for voice testing
- Additional language support (as CSM-1B models become available)
- Pronunciation dictionary support

**Exit Criteria:**
- Custom voice creation works from a 30-second audio clip
- Web playground allows testing all voices and settings

### Phase v2.0 -- "It Runs Everywhere" (Target: 3-6 months from v1.0)

**Scope:**
- PyTorch/CUDA backend for Linux and Windows
- Docker container
- Kubernetes Helm chart
- WebSocket streaming endpoint
- Multi-model support

**Exit Criteria:**
- Server runs on Ubuntu 22.04 with NVIDIA GPU
- Docker image published to GHCR/Docker Hub

---

## 8. Technical Constraints

| Constraint | Detail |
|------------|--------|
| Hardware | 2025 Mac Studio, Apple M3 Ultra, 256GB unified memory, macOS Tahoe 26.3 |
| ML framework | MLX only (no PyTorch/CUDA in inference path) |
| Model | Sesame CSM-1B via mlx-audio (single model, English only) |
| Native sample rate | 24,000 Hz (from Mimi codec in CSM-1B) |
| Inference concurrency | Serial (single MLX inference at a time via semaphore) |
| Audio codec | ffmpeg required for MP3 encoding; PCM formats use scipy for resampling |
| Deployment | Bare metal via launchd (no Docker -- MLX requires macOS) |
| Existing architecture | Must layer on top of existing server.py, tts_engine.py, audio_converter.py without breaking OpenAI endpoint |
| Port | 8880 (shared between OpenAI and ElevenLabs endpoints) |
| Repo | Personal repo under **bautrey** GitHub account |

---

## 9. Dependencies and Risks

### 9.1 Dependencies

| Dependency | Type | Risk Level | Mitigation |
|-----------|------|-----------|------------|
| mlx-audio 0.3.x | Library | Low | Pin version; thin wrapper makes updates manageable |
| mlx-audio streaming support | Feature | Medium | Phase 1 uses buffered approach that works with current API; Phase 2 may need upstream changes |
| scipy.signal | Library | Low | Well-established library for resampling |
| OpenClaw `baseUrl` config support | External | Low | Confirmed working: `openclaw.json` messages.tts.elevenlabs.baseUrl already supports custom URLs |
| ffmpeg 6.x/7.x | System | Low | Required for MP3; PCM formats work without it |
| CSM-1B model weights | External | Low | Cached after first download; works fully offline |

### 9.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Buffered streaming latency exceeds 500ms | Medium | High | Reduce `STREAM_BUFFER_FRAMES` (min 2); benchmark early; accept higher latency as fallback |
| OpenClaw Talk Mode expects specific ElevenLabs response headers we don't know about | Medium | High | Test early with actual OpenClaw; analyze OpenClaw source for ElevenLabs integration code; iterate quickly |
| MLX generation loop does not support clean cancellation | Low | Medium | Wrap in try/except; worst case, let current frame complete before checking cancel flag |
| Resampling quality (scipy) introduces audible artifacts | Low | Medium | Use `scipy.signal.resample_poly` for quality; benchmark against ffmpeg resampling; allow ffmpeg fallback |
| Context cache causes memory pressure on smaller Macs (M1/M2) | Medium | Medium | Make context configurable and default-off for < 32GB machines; aggressive LRU eviction |
| ElevenLabs API changes format/endpoints after our implementation | Low | Low | Pin to current API version; our compatibility is best-effort for the local use case |
| Community expects Linux/Windows support immediately | High | Low | Clear documentation: "Apple Silicon only in v1; PyTorch backend in v2.0; PRs welcome" |

---

## 10. Community Launch Plan

Ship a polished README with a one-command install script (`curl | bash` or equivalent) and OpenClaw configuration instructions. Create a GitHub release with tagged version and release notes. Announce on OpenClaw Discord (#showcase channel) and Reddit (r/LocalLLaMA, r/selfhosted) with a short post: "Free local Talk Mode for OpenClaw -- no ElevenLabs subscription. Sesame CSM-1B on Apple Silicon with ElevenLabs-compatible streaming API."

Follow-up channels (Hacker News, X/Twitter, Product Hunt) are stretch goals if the initial launch gets traction. The README is the primary launch artifact -- it should include a hero description, install instructions, OpenClaw config snippet, and a short FAQ.

---

## References

- [GitHub Issue #1](https://github.com/bautrey/sesame-tts/issues/1) -- Feature request and full requirements
- [REQUIREMENTS-streaming.md](/REQUIREMENTS-streaming.md) -- Detailed requirements specification
- [PRD.md (v1.1)](./PRD.md) -- Base server PRD (OpenAI-compatible endpoint)
- [TRD.md (v1.1)](./TRD.md) -- Base server technical requirements
- [ElevenLabs API Documentation](https://elevenlabs.io/docs/api-reference/text-to-speech) -- Target API compatibility
- [Sesame CSM-1B (GitHub)](https://github.com/SesameAILabs/csm) -- Model documentation
- [mlx-audio (GitHub)](https://github.com/Blaizzy/mlx-audio) -- MLX inference library
- [OpenClaw](https://github.com/openclaw) -- Primary client application

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-20 | Initial PRD for ElevenLabs-compatible streaming API based on GitHub Issue #1 and REQUIREMENTS-streaming.md |
| 1.1 | 2026-02-20 | Refined with user interview feedback: confirmed OpenClaw config from source (openclaw.json messages.tts block), scoped output formats to 3 v1.0 essentials (pcm_44100, pcm_24000, mp3_44100_128), deferred conversation context entirely to v1.1, trimmed community launch plan, added concrete curl test commands to all acceptance criteria |
