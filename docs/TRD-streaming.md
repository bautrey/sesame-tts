# TRD: ElevenLabs-Compatible Streaming API for Sesame TTS

**Author:** Burke
**Created:** 2026-02-20
**Status:** Draft
**Version:** 1.1
**PRD Reference:** [docs/PRD-streaming.md](./PRD-streaming.md) v1.1
**Base TRD Reference:** [docs/TRD.md](./TRD.md) v1.1
**GitHub Issue:** [#1 - ElevenLabs-Compatible Streaming API + OpenClaw Talk Mode Support](https://github.com/bautrey/sesame-tts/issues/1)

---

## 1. Overview

### What We Are Adding

ElevenLabs API-compatible endpoints layered on top of the existing Sesame TTS server. The server currently exposes an OpenAI-compatible `/v1/audio/speech` endpoint for synchronous TTS generation. This TRD adds:

1. **ElevenLabs streaming endpoint** (`POST /v1/text-to-speech/{voice_id}/stream`) -- chunked audio streaming with sentence-level pipelining
2. **ElevenLabs non-streaming endpoint** (`POST /v1/text-to-speech/{voice_id}`) -- complete audio buffer response
3. **Voice management endpoints** (`GET /v1/voices`, `GET /v1/voices/{voice_id}`) -- ElevenLabs-format voice metadata
4. **Compatibility endpoints** (`GET /v1/models` ElevenLabs format, `GET /v1/user/subscription`, `GET /v1/user`) -- stub responses for client compatibility
5. **New output formats** -- `pcm_44100`, `pcm_24000`, `mp3_44100_128` for Talk Mode compatibility
6. **Sentence-splitting streaming pipeline** -- split input into sentences, generate and stream each independently for low first-chunk latency
7. **Interrupt support** -- detect client disconnect in the streaming loop, stop inference via threading.Event

### Why

OpenClaw Talk Mode uses the ElevenLabs streaming API, not the OpenAI TTS API. Without these endpoints, users must maintain an ElevenLabs subscription ($5-22/month) even though they have a working local TTS server. Adding ElevenLabs API compatibility makes the server a complete drop-in replacement -- change only the `baseUrl` in `openclaw.json` and Talk Mode works locally with zero API costs.

### What We Are NOT Changing

- The existing OpenAI `/v1/audio/speech` endpoint (untouched)
- The existing OpenAI `/v1/models` endpoint (extended, not replaced)
- The model loading pipeline in `tts_engine.py` (extended with streaming method)
- The preset JSON format (extended with optional metadata fields)
- The launchd deployment (unchanged)

---

## 2. System Architecture

### Component Diagram

```
+-----------------------------------------------------------+
|  OpenClaw Talk Mode / ElevenLabs Client                   |
|  (sends xi-api-key header, expects chunked PCM/MP3)       |
+-----------------------------------------------------------+
     |  POST /v1/text-to-speech/{voice_id}/stream
     |  POST /v1/text-to-speech/{voice_id}
     |  GET  /v1/voices
     |  GET  /v1/voices/{voice_id}
     |  GET  /v1/models  (with xi-api-key)
     |  GET  /v1/user/subscription
     |  GET  /v1/user
     v
+-----------------------------------------------------------+
|  FastAPI Server  (server.py)                              |
|  - Mounts ElevenLabs router (elevenlabs_routes.py)        |
|  - Existing OpenAI routes (unchanged)                     |
|  - Exception handlers: ElevenLabs + OpenAI format         |
|  - asyncio.Semaphore for serial inference                  |
+-----------------------------------------------------------+
     |                           |
     v                           v
+------------------------+  +-------------------------------+
|  ElevenLabs Routes     |  | OpenAI Routes (existing)      |
|  (elevenlabs_routes.py)|  | POST /v1/audio/speech         |
|  - TTS streaming       |  | GET  /v1/models               |
|  - TTS non-streaming   |  | GET  /health                  |
|  - Voice management    |  +-------------------------------+
|  - Compatibility stubs |
|  - Error formatting    |
+------------------------+
     |
     v
+-----------------------------------------------------------+
|  Streaming Pipeline  (streaming.py)                       |
|  - Splits input text into sentences                       |
|  - Generates first sentence immediately                   |
|  - Streams audio as each sentence completes               |
|  - Pipelines: generates next sentence while streaming     |
|  - Converts to requested output format                    |
|  - Yields bytes chunks for StreamingResponse              |
|  - Monitors interrupt flag via client disconnect          |
+-----------------------------------------------------------+
     |                           |
     v                           v
+------------------------+  +-------------------------------+
|  TTS Engine            |  | Audio Converter               |
|  (tts_engine.py)       |  | (audio_converter.py)          |
|  MODIFIED:             |  | MODIFIED:                     |
|  + generate_stream()   |  | + pcm_24000 (native)          |
|  + interrupt flag      |  | + pcm_44100 (resample)        |
|  (threading.Event)     |  | + mp3_44100_128 (ffmpeg)      |
+------------------------+  | + convert_audio_chunk()       |
     |                      | + resample_audio()            |
     v                      +-------------------------------+
+------------------------+
| mlx-audio Model        |
| (.generate() yields    |
|  GenerationResult)     |
| - CSM-1B weights       |
| - Mimi codec           |
| - Watermarking         |
+------------------------+

+-----------------------------------------------------------+
|  Voice Presets  (voice_presets.py)                         |
|  MODIFIED:                                                |
|  + elevenlabs_metadata() -- labels, settings, preview_url |
|  + list_elevenlabs_format() -- full voice list response   |
+-----------------------------------------------------------+

+-----------------------------------------------------------+
|  Config  (config.py)                                      |
|  MODIFIED:                                                |
|  + stream_chunk_size: int = 4096                          |
|  + default_elevenlabs_format: str = "pcm_24000"           |
|  + warmup_on_start: bool = True                           |
|  + max_queue_depth: int = 3                               |
+-----------------------------------------------------------+
```

### Data Flow: Streaming Request (Sentence-Splitting Pipeline)

```
1. Client POSTs to /v1/text-to-speech/conversationalB/stream?output_format=pcm_24000
2. elevenlabs_routes.py:
   a. Parses request body (text, model_id, voice_settings, output_format)
   b. Resolves voice_id -> preset via resolve_voice() + preset_manager
   c. Maps voice_settings.stability -> temperature
   d. Acquires inference_semaphore
3. streaming.py.stream_audio():
   a. Splits input text into sentences via regex (split on .!? followed by whitespace)
   b. Sends first sentence to engine.generate_stream() immediately
   c. As first sentence audio completes, converts to output format
   d. Yields first sentence audio as chunked bytes -- client starts playback
   e. Simultaneously begins generating next sentence
   f. Continues until all sentences are generated and streamed
   g. On client disconnect: sets interrupt_event, breaks out of loop
4. tts_engine.py.generate_stream():
   a. Calls self.model.generate() which yields GenerationResult objects
   b. Yields each GenerationResult, checking interrupt_event between yields
5. FastAPI StreamingResponse delivers chunks with Transfer-Encoding: chunked
6. On client disconnect:
   a. request.is_disconnected() returns True (checked in stream_audio loop)
   b. interrupt_event.set()
   c. generate_stream() stops yielding
   d. Semaphore released in finally block
```

---

## 3. Module Design

### 3.1 `elevenlabs_routes.py` (NEW)

**Responsibilities:** All ElevenLabs API routes, request/response translation, ElevenLabs error formatting.

```python
import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from audio_converter import (
    ELEVENLABS_CONTENT_TYPES,
    ELEVENLABS_SUPPORTED_FORMATS,
    ELEVENLABS_DEFERRED_FORMATS,
)
from streaming import stream_audio
from voice_presets import VoicePreset, VoicePresetManager

logger = logging.getLogger(__name__)

router = APIRouter()

# --- ElevenLabs Request Models ---

class VoiceSettings(BaseModel):
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    use_speaker_boost: bool = True

class ElevenLabsTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    model_id: str = "csm-1b"
    voice_settings: Optional[VoiceSettings] = None
    output_format: Optional[str] = None  # Can also come from query param
    optimize_streaming_latency: int = Field(default=0, ge=0, le=4)

# --- ElevenLabs Model ID Mapping ---

ELEVENLABS_MODEL_MAP = {
    "eleven_monolingual_v1",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_turbo_v2_5",
    "eleven_v3",
}

def map_model_id(model_id: str) -> str:
    """Map any ElevenLabs model ID to csm-1b. Log the mapping."""
    if model_id == "csm-1b" or model_id == "csm1b":
        return "csm-1b"
    if model_id in ELEVENLABS_MODEL_MAP:
        logger.info("Mapped ElevenLabs model %s to csm-1b", model_id)
    else:
        logger.info("Unknown model %s, using csm-1b", model_id)
    return "csm-1b"

def map_voice_settings(settings: Optional[VoiceSettings], preset: VoicePreset) -> float:
    """Map ElevenLabs voice_settings to temperature. Returns temperature value."""
    if settings is None:
        return preset.temperature
    # stability is inverse of temperature: high stability = low randomness
    # ElevenLabs stability 0.0-1.0 maps to temperature 1.0-0.0
    return round(1.0 - settings.stability, 2)

# --- ElevenLabs Error Formatting ---

def elevenlabs_error(status_code: int, status: str, message: str) -> JSONResponse:
    """Return an ElevenLabs-format error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "status": status,
                "message": message,
            }
        },
    )

# --- Helper: resolve voice and format ---

def _resolve_format(
    output_format_query: Optional[str],
    output_format_body: Optional[str],
    default_format: str,
) -> tuple[str, Optional[JSONResponse]]:
    """Resolve output format from query param, body, or default. Returns (fmt, error_or_None)."""
    fmt = output_format_query or output_format_body or default_format
    if fmt in ELEVENLABS_DEFERRED_FORMATS:
        return fmt, elevenlabs_error(
            400, "invalid_request",
            f"Format '{fmt}' is not yet supported. Available: {', '.join(sorted(ELEVENLABS_SUPPORTED_FORMATS))}"
        )
    if fmt not in ELEVENLABS_SUPPORTED_FORMATS:
        return fmt, elevenlabs_error(
            400, "invalid_request",
            f"Invalid output format '{fmt}'. Supported: {', '.join(sorted(ELEVENLABS_SUPPORTED_FORMATS))}"
        )
    return fmt, None


def _resolve_preset(
    voice_id: str,
    preset_manager: VoicePresetManager,
    default_voice: str,
) -> tuple[Optional[VoicePreset], Optional[JSONResponse]]:
    """Resolve voice_id to a VoicePreset. Falls back to default. Returns (preset, error_or_None)."""
    from server import resolve_voice
    voice_name = resolve_voice(voice_id)
    preset = preset_manager.get(voice_name)
    if preset is None:
        logger.warning("Voice '%s' not found, falling back to default '%s'", voice_id, default_voice)
        preset = preset_manager.get(default_voice)
        if preset is None:
            return None, elevenlabs_error(503, "server_error", "Default voice not configured")
    return preset, None


# --- TTS Routes ---

@router.post("/v1/text-to-speech/{voice_id}/stream")
async def tts_stream(
    voice_id: str,
    request: Request,
    output_format: str = Query(default=None),
):
    """ElevenLabs streaming TTS endpoint. Returns chunked audio.

    Uses sentence-splitting pipeline: splits input into sentences,
    generates first sentence immediately, streams its audio while
    generating subsequent sentences in parallel.
    """
    engine = request.app.state.engine
    settings = request.app.state.settings
    preset_manager = request.app.state.preset_manager
    inference_semaphore = request.app.state.inference_semaphore

    if not engine.model_loaded:
        return elevenlabs_error(503, "server_error", "Model not ready. Please wait and retry.")

    # Parse body
    try:
        body = await request.json()
        req = ElevenLabsTTSRequest(**body)
    except Exception:
        return elevenlabs_error(400, "invalid_request", "Invalid request body")

    # Resolve output format: query param > body > default
    fmt, fmt_err = _resolve_format(output_format, req.output_format, settings.default_elevenlabs_format)
    if fmt_err is not None:
        return fmt_err

    # Resolve voice
    preset, preset_err = _resolve_preset(voice_id, preset_manager, settings.default_voice)
    if preset_err is not None:
        return preset_err

    # Map model and settings
    map_model_id(req.model_id)
    temperature = map_voice_settings(req.voice_settings, preset)

    # Log auth header at DEBUG level
    xi_key = request.headers.get("xi-api-key")
    if xi_key:
        logger.debug("xi-api-key header present (value not logged)")

    # Check queue depth
    if inference_semaphore.locked():
        queue_depth = getattr(request.app.state, "queue_depth", 0)
        if queue_depth >= settings.max_queue_depth:
            return elevenlabs_error(429, "rate_limit", "Server is busy. Please retry shortly.")

    # Stream response
    import threading
    interrupt_event = threading.Event()

    async def generate_and_stream():
        request.app.state.queue_depth = getattr(request.app.state, "queue_depth", 0) + 1
        try:
            async with inference_semaphore:
                start = time.time()
                loop = asyncio.get_event_loop()

                async for chunk in stream_audio(
                    engine=engine,
                    text=req.text,
                    preset=preset,
                    temperature=temperature,
                    output_format=fmt,
                    chunk_size=settings.stream_chunk_size,
                    interrupt_event=interrupt_event,
                    request=request,
                    loop=loop,
                ):
                    # Check for client disconnect
                    if await request.is_disconnected():
                        interrupt_event.set()
                        logger.info(
                            "Generation interrupted: voice=%s reason=client_disconnect elapsed=%.2fs",
                            voice_id, time.time() - start,
                        )
                        break
                    yield chunk

                elapsed = time.time() - start
                logger.info(
                    "Streamed: voice=%s fmt=%s input_len=%d elapsed=%.2fs",
                    voice_id, fmt, len(req.text), elapsed,
                )
        except MemoryError:
            logger.error("OOM during streaming generation")
        except Exception:
            logger.exception("Streaming generation failed")
        finally:
            request.app.state.queue_depth = max(0, getattr(request.app.state, "queue_depth", 1) - 1)
            interrupt_event.set()  # Ensure generation stops

    content_type = ELEVENLABS_CONTENT_TYPES.get(fmt, "application/octet-stream")
    return StreamingResponse(
        generate_and_stream(),
        media_type=content_type,
        headers={"Transfer-Encoding": "chunked"},
    )


@router.post("/v1/text-to-speech/{voice_id}")
async def tts_non_stream(
    voice_id: str,
    request: Request,
    output_format: str = Query(default=None),
):
    """ElevenLabs non-streaming TTS endpoint. Returns complete audio buffer."""
    engine = request.app.state.engine
    settings = request.app.state.settings
    preset_manager = request.app.state.preset_manager
    inference_semaphore = request.app.state.inference_semaphore

    if not engine.model_loaded:
        return elevenlabs_error(503, "server_error", "Model not ready. Please wait and retry.")

    # Parse body
    try:
        body = await request.json()
        req = ElevenLabsTTSRequest(**body)
    except Exception:
        return elevenlabs_error(400, "invalid_request", "Invalid request body")

    # Resolve output format
    fmt, fmt_err = _resolve_format(output_format, req.output_format, settings.default_elevenlabs_format)
    if fmt_err is not None:
        return fmt_err

    # Resolve voice
    preset, preset_err = _resolve_preset(voice_id, preset_manager, settings.default_voice)
    if preset_err is not None:
        return preset_err

    # Map settings
    map_model_id(req.model_id)
    temperature = map_voice_settings(req.voice_settings, preset)

    # Generate complete audio
    async with inference_semaphore:
        start = time.time()
        loop = asyncio.get_event_loop()
        try:
            # Create a temporary preset copy with mapped temperature
            gen_preset = VoicePreset(
                name=preset.name,
                voice_key=preset.voice_key,
                speaker_id=preset.speaker_id,
                temperature=temperature,
                top_k=preset.top_k,
                max_audio_length_ms=preset.max_audio_length_ms,
                description=preset.description,
            )
            audio_np, sample_rate = await loop.run_in_executor(
                None, engine.generate, req.text, gen_preset
            )
        except MemoryError:
            return elevenlabs_error(503, "server_error", "Insufficient memory for generation")
        except RuntimeError as e:
            return elevenlabs_error(500, "server_error", f"Audio generation failed: {e}")
        gen_time = time.time() - start

    # Convert to requested format
    from audio_converter import convert_audio_elevenlabs
    try:
        audio_bytes = convert_audio_elevenlabs(audio_np, sample_rate, fmt)
    except RuntimeError as e:
        return elevenlabs_error(500, "server_error", f"Audio conversion failed: {e}")

    logger.info(
        "Generated (non-stream): voice=%s fmt=%s input_len=%d gen=%.2fs",
        voice_id, fmt, len(req.text), gen_time,
    )

    content_type = ELEVENLABS_CONTENT_TYPES.get(fmt, "application/octet-stream")
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )


# --- Voice Management Routes ---

@router.get("/v1/voices")
async def list_voices(request: Request):
    """List all available voices in ElevenLabs format."""
    preset_manager = request.app.state.preset_manager
    settings = request.app.state.settings
    voices = preset_manager.list_elevenlabs_format(
        base_url=f"http://{settings.host}:{settings.port}"
    )
    return {"voices": voices}


@router.get("/v1/voices/{voice_id}")
async def get_voice(voice_id: str, request: Request):
    """Get a single voice's metadata in ElevenLabs format."""
    preset_manager = request.app.state.preset_manager
    settings = request.app.state.settings

    from server import resolve_voice
    voice_name = resolve_voice(voice_id)
    preset = preset_manager.get(voice_name)
    if preset is None:
        return elevenlabs_error(
            404, "voice_not_found",
            f"Voice '{voice_id}' not found. Available: {', '.join(preset_manager.list_names())}"
        )

    meta = preset.elevenlabs_metadata(
        base_url=f"http://{settings.host}:{settings.port}"
    )
    return meta


# --- Compatibility Routes ---

@router.get("/v1/user/subscription")
async def user_subscription():
    """Return unlimited local subscription info."""
    return {
        "tier": "local",
        "character_count": 0,
        "character_limit": 999999999,
        "can_extend_character_limit": False,
        "allowed_to_extend_character_limit": False,
        "next_character_count_reset_unix": 0,
        "voice_limit": 999,
        "max_voice_add_edits": 999,
        "voice_add_edit_counter": 0,
        "professional_voice_limit": 0,
        "can_extend_voice_limit": False,
        "can_use_instant_voice_cloning": False,
        "can_use_professional_voice_cloning": False,
        "currency": "usd",
        "status": "active",
        "billing_period": "unlimited",
        "character_refresh_period": "unlimited",
        "next_invoice": None,
    }


@router.get("/v1/user")
async def user_info():
    """Return local user info."""
    return {
        "subscription": {
            "tier": "local",
            "character_limit": 999999999,
            "status": "active",
        },
        "is_new_user": False,
        "xi_api_key": "local",
    }
```

**Key Design Decisions:**

1. **Separate router, not inline in server.py.** The ElevenLabs routes are mounted via `app.include_router(router)` in `server.py`. This keeps the OpenAI routes clean and allows independent testing.

2. **Voice fallback behavior.** Unknown voice IDs fall back to the default voice with a warning log, rather than returning 404. This matches ElevenLabs' behavior where Talk Mode may send arbitrary voice IDs.

3. **Output format resolution order.** Query parameter takes priority over body field, which takes priority over the server default. This matches how OpenClaw sends `output_format` as a query parameter.

4. **ElevenLabs error format.** All error responses from ElevenLabs routes use `{"detail": {"status": ..., "message": ...}}` instead of the OpenAI `{"error": {...}}` format.

5. **No explicit cancel endpoint.** Interrupt is handled entirely via client disconnect detection. When the client closes the connection, `request.is_disconnected()` returns True and the streaming loop sets the interrupt event. This is simpler and matches how Talk Mode actually works -- it just drops the connection when the user interrupts.

6. **Helper functions for format/voice resolution.** `_resolve_format()` and `_resolve_preset()` are shared between streaming and non-streaming endpoints to eliminate duplication.

### 3.2 `streaming.py` (NEW)

**Responsibilities:** Sentence-splitting streaming pipeline. Splits input text into sentences, generates each sentence independently via the engine, converts to output format, yields byte chunks. Pipelines generation of subsequent sentences while streaming earlier ones.

```python
import asyncio
import logging
import re
import threading
from typing import AsyncGenerator

import numpy as np
from fastapi import Request

from audio_converter import convert_audio_chunk
from tts_engine import TTSEngine
from voice_presets import VoicePreset

logger = logging.getLogger(__name__)

# Native sample rate from CSM-1B / Mimi codec
NATIVE_SAMPLE_RATE = 24000

# Sentence splitting regex: split on sentence-ending punctuation followed by whitespace.
# Keeps the punctuation attached to the sentence. Handles .!? followed by one or more
# whitespace characters (space, newline, etc.).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming pipeline.

    Uses simple regex: split on .!? followed by whitespace.
    Each sentence retains its terminal punctuation.
    Single sentences (no split points) return as a one-element list.

    Examples:
        "Hello there. How are you?" -> ["Hello there.", "How are you?"]
        "Hello there" -> ["Hello there"]
        "Wait! Really? Yes." -> ["Wait!", "Really?", "Yes."]
        "Dr. Smith went home. He was tired." -> ["Dr. Smith went home.", "He was tired."]

    Note: The regex does not handle abbreviations like "Dr." perfectly.
    For Talk Mode's typical short inputs (single sentences), this is fine.
    Multi-sentence paragraphs may occasionally mis-split on abbreviations,
    but the audio result is still correct (just split differently).
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    # Filter out empty strings from splitting
    return [s.strip() for s in sentences if s.strip()]


async def stream_audio(
    engine: TTSEngine,
    text: str,
    preset: VoicePreset,
    temperature: float,
    output_format: str,
    chunk_size: int,
    interrupt_event: threading.Event,
    request: Request,
    loop: asyncio.AbstractEventLoop,
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio chunks from the TTS engine using sentence-splitting pipeline.

    Strategy (sentence-level pipelining):
    1. Split input text into sentences (regex: split on .!? + whitespace)
    2. Start generating first sentence immediately
    3. As first sentence audio completes, stream it as chunked PCM/MP3
    4. Simultaneously begin generating next sentence
    5. Continue until all sentences are generated and streamed

    This gives first-chunk latency of one short sentence (~1s or less)
    instead of the whole paragraph. For single-sentence input (typical
    Talk Mode), behavior is identical to generating the full text.

    The mlx-audio model.generate() yields GenerationResult objects.
    Each GenerationResult contains:
      - audio: mx.array of raw audio samples at 24000 Hz
      - sample_rate: int (always 24000)

    For pcm_24000: extract int16 bytes directly (no resampling)
    For pcm_44100: resample 24000 -> 44100, then extract int16 bytes
    For mp3_44100_128: resample 24000 -> 44100, encode via ffmpeg subprocess
    """
    # Create a modified preset with the mapped temperature
    gen_preset = VoicePreset(
        name=preset.name,
        voice_key=preset.voice_key,
        speaker_id=preset.speaker_id,
        temperature=temperature,
        top_k=preset.top_k,
        max_audio_length_ms=preset.max_audio_length_ms,
        description=preset.description,
    )

    # Determine if we need resampling
    target_sample_rate = _parse_sample_rate(output_format)
    needs_resample = target_sample_rate != NATIVE_SAMPLE_RATE

    # Split text into sentences for pipelined generation
    sentences = split_sentences(text)
    logger.debug("Split text into %d sentence(s) for streaming", len(sentences))

    # Queue for receiving audio from the generation thread.
    # Each item is either (audio_np, sample_rate) or SENTINEL to indicate
    # that the current sentence is complete.
    audio_queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()  # Marks end of one sentence's generation
    DONE = object()  # Marks all sentences complete

    total_frames = 0

    def _run_generation():
        """Run sentence-by-sentence generation in a background thread.

        For each sentence, calls engine.generate_stream() which yields
        (audio_np, sample_rate) tuples. After all frames for a sentence
        are pushed, pushes SENTINEL. After all sentences, pushes DONE.
        """
        nonlocal total_frames
        try:
            for i, sentence in enumerate(sentences):
                if interrupt_event.is_set():
                    break

                logger.debug("Generating sentence %d/%d: %r", i + 1, len(sentences), sentence[:50])

                for audio_np, sr in engine.generate_stream(sentence, gen_preset, interrupt_event):
                    if interrupt_event.is_set():
                        break
                    loop.call_soon_threadsafe(audio_queue.put_nowait, (audio_np, sr))
                    total_frames += 1

                if interrupt_event.is_set():
                    break

                # Signal that this sentence is complete
                loop.call_soon_threadsafe(audio_queue.put_nowait, SENTINEL)

        except Exception as e:
            logger.exception("Generation thread error: %s", e)
        finally:
            # Always signal completion
            loop.call_soon_threadsafe(audio_queue.put_nowait, DONE)

    # Start generation in background thread
    gen_future = loop.run_in_executor(None, _run_generation)

    try:
        sentence_buffer: list[np.ndarray] = []

        while True:
            # Check interrupt
            if interrupt_event.is_set():
                break

            # Get next item from queue (audio frame, SENTINEL, or DONE)
            try:
                item = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # Check disconnect while waiting
                if await request.is_disconnected():
                    interrupt_event.set()
                    logger.info("Client disconnected during streaming, interrupting generation")
                    break
                continue

            if item is DONE:
                # All sentences complete -- flush any remaining buffer
                if sentence_buffer:
                    chunk_audio = np.concatenate(sentence_buffer, axis=0)
                    sentence_buffer.clear()
                    formatted_bytes = _convert_chunk(
                        chunk_audio, NATIVE_SAMPLE_RATE, output_format,
                        target_sample_rate, needs_resample,
                    )
                    for piece in _split_bytes(formatted_bytes, chunk_size):
                        yield piece
                break

            if item is SENTINEL:
                # One sentence complete -- flush its buffer and yield audio
                if sentence_buffer:
                    chunk_audio = np.concatenate(sentence_buffer, axis=0)
                    sentence_buffer.clear()
                    formatted_bytes = _convert_chunk(
                        chunk_audio, NATIVE_SAMPLE_RATE, output_format,
                        target_sample_rate, needs_resample,
                    )
                    for piece in _split_bytes(formatted_bytes, chunk_size):
                        yield piece
                continue

            # Regular audio frame -- accumulate into sentence buffer
            audio_np, sr = item
            sentence_buffer.append(audio_np)

    finally:
        interrupt_event.set()  # Ensure generation thread stops
        await gen_future  # Wait for generation thread to finish
        logger.debug("Streaming complete: %d total frames generated", total_frames)


def _parse_sample_rate(output_format: str) -> int:
    """Extract target sample rate from ElevenLabs format string."""
    if output_format == "pcm_24000":
        return 24000
    elif output_format == "pcm_44100":
        return 44100
    elif output_format == "mp3_44100_128":
        return 44100
    else:
        return 24000  # Default to native


def _convert_chunk(
    audio_np: np.ndarray,
    source_rate: int,
    output_format: str,
    target_rate: int,
    needs_resample: bool,
) -> bytes:
    """Convert a numpy float32 audio chunk to the requested format bytes.

    For PCM formats: resample if needed, convert to int16, return raw bytes.
    For MP3 formats: resample if needed, pipe through ffmpeg.
    """
    return convert_audio_chunk(audio_np, source_rate, output_format, target_rate, needs_resample)


def _split_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    """Split a byte buffer into fixed-size chunks for smooth HTTP delivery.

    For PCM formats, chunk_size should align to sample boundaries (multiples of 2
    for int16). The caller is responsible for choosing appropriate chunk_size.

    Returns a list of byte chunks, each at most chunk_size bytes.
    The last chunk may be smaller.
    """
    if len(data) <= chunk_size:
        return [data]
    chunks = []
    offset = 0
    while offset < len(data):
        end = min(offset + chunk_size, len(data))
        chunks.append(data[offset:end])
        offset = end
    return chunks
```

**Sentence-Splitting Streaming Architecture:**

```
Input text: "Hello there. How are you? I'm doing great."
                    |
                    v
           split_sentences()
                    |
     +--------------+--------------+
     |              |              |
     v              v              v
"Hello there."  "How are you?"  "I'm doing great."
     |
     v (immediate)
engine.generate_stream("Hello there.")
     |
     | yields (audio_np, sr) frames
     | then SENTINEL
     v
convert + yield audio bytes  -----> Client starts playback
     |
     | (simultaneous with client playback)
     v
engine.generate_stream("How are you?")
     |
     | yields frames, then SENTINEL
     v
convert + yield audio bytes  -----> Client continues playback
     |
     v
engine.generate_stream("I'm doing great.")
     |
     | yields frames, then SENTINEL
     v
convert + yield audio bytes  -----> Client finishes playback
     |
     v
DONE sentinel -> flush remaining -> stream complete
```

**First-Chunk Latency Comparison:**

```
WITHOUT sentence splitting (v1.0 original):
  Input: "Hello there. How are you? I'm doing great."
  Generate ALL text -> ~4-6s for 3 sentences
  First audio chunk arrives at: ~4-6s

WITH sentence splitting (v1.1):
  Input: "Hello there. How are you? I'm doing great."
  Generate "Hello there." only -> ~0.8-1.2s for 1 short sentence
  First audio chunk arrives at: ~0.8-1.2s
  "How are you?" generates while "Hello there." plays
  Seamless audio delivery to client

  Single sentence input (Talk Mode typical):
  Input: "Hello there."
  Identical to without splitting -- one sentence, one generation
  First audio chunk arrives at: ~0.8-1.2s (same either way)
```

**Key Design Decision: `threading.Event` over `asyncio.Event`.**

The mlx-audio `model.generate()` call runs in a synchronous thread (via `run_in_executor`). An `asyncio.Event` cannot be checked from a synchronous thread without an event loop reference. A `threading.Event` is thread-safe and can be checked with zero overhead in the generation loop. The tradeoff is that we bridge async/sync via an `asyncio.Queue` -- the background thread pushes frames, the async generator consumes them.

**Key Design Decision: sentence-level buffering, not frame-level.**

Each sentence's audio frames are accumulated in `sentence_buffer` and flushed together when the SENTINEL arrives. This ensures each yielded audio chunk is a complete sentence -- no mid-word cuts. The audio converter receives one coherent audio segment per sentence, which produces better resampling quality (no edge artifacts from splitting mid-audio) and cleaner MP3 encoding (complete audio segments encode better than tiny fragments).

### 3.3 `audio_converter.py` (MODIFIED)

**New responsibilities:** ElevenLabs output formats (PCM raw, MP3 streaming), per-chunk conversion, resampling.

```python
# --- Existing code (unchanged) ---

SUPPORTED_FORMATS = {"mp3", "opus", "wav", "flac"}
CONTENT_TYPES = { ... }  # existing
FFMPEG_ARGS = { ... }    # existing
def convert_audio(audio_np, sample_rate, fmt): ...  # existing

# --- New: ElevenLabs format support ---

import subprocess
from math import gcd

import numpy as np

ELEVENLABS_SUPPORTED_FORMATS = {"pcm_24000", "pcm_44100", "mp3_44100_128"}

ELEVENLABS_DEFERRED_FORMATS = {
    "pcm_16000", "pcm_22050",
    "mp3_22050_32", "mp3_44100_32", "mp3_44100_64", "mp3_44100_96", "mp3_44100_192",
}

ELEVENLABS_CONTENT_TYPES = {
    "pcm_24000": "application/octet-stream",
    "pcm_44100": "application/octet-stream",
    "mp3_44100_128": "audio/mpeg",
}


def resample_audio(audio_np: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio from source_rate to target_rate using scipy.signal.resample_poly.

    Uses polyphase FIR resampling for high-quality conversion without the
    edge artifacts that FFT-based scipy.signal.resample can introduce on
    short audio chunks.

    scipy is already a transitive dependency of the project (via mlx-audio),
    so no new dependency is needed.

    Args:
        audio_np: float32 numpy array of audio samples (mono)
        source_rate: source sample rate (e.g., 24000)
        target_rate: target sample rate (e.g., 44100)

    Returns:
        Resampled float32 numpy array

    Resampling math for 24000 -> 44100:
        GCD(44100, 24000) = 300
        up   = 44100 / 300 = 147
        down = 24000 / 300 = 80
        resample_poly(audio, up=147, down=80)
        Produces exact 44100 Hz output with no floating-point sample rate drift.
    """
    if source_rate == target_rate:
        return audio_np

    from scipy.signal import resample_poly

    g = gcd(target_rate, source_rate)
    up = target_rate // g
    down = source_rate // g

    return resample_poly(audio_np, up, down).astype(np.float32)


def convert_audio_chunk(
    audio_np: np.ndarray,
    source_rate: int,
    output_format: str,
    target_rate: int,
    needs_resample: bool,
) -> bytes:
    """Convert a numpy float32 audio chunk to ElevenLabs output format bytes.

    For PCM formats: resample if needed, convert to int16, return raw bytes.
    For MP3 formats: resample if needed, pipe through ffmpeg subprocess.

    Args:
        audio_np: float32 numpy array, values in [-1.0, 1.0]
        source_rate: source sample rate (e.g., 24000)
        output_format: one of ELEVENLABS_SUPPORTED_FORMATS
        target_rate: target sample rate after resampling
        needs_resample: whether resampling is needed

    Returns:
        Raw bytes in the requested format
    """
    # Resample if needed
    if needs_resample:
        audio_np = resample_audio(audio_np, source_rate, target_rate)

    # Convert float32 to int16 PCM
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

    if output_format.startswith("pcm_"):
        # Raw PCM: just return the int16 bytes
        return audio_int16.tobytes()

    elif output_format == "mp3_44100_128":
        # MP3 via ffmpeg subprocess
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(target_rate), "-ac", "1", "-i", "pipe:0",
            "-c:a", "libmp3lame", "-b:a", "128k",
            "-f", "mp3", "pipe:1",
        ]
        result = subprocess.run(cmd, input=audio_int16.tobytes(), capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg MP3 encoding failed: {result.stderr.decode()}")
        return result.stdout

    else:
        raise ValueError(f"Unsupported ElevenLabs format: {output_format}")


def convert_audio_elevenlabs(
    audio_np: np.ndarray,
    sample_rate: int,
    output_format: str,
) -> bytes:
    """Convert complete audio buffer to ElevenLabs output format.

    Used by the non-streaming endpoint. Same logic as convert_audio_chunk
    but for a complete audio buffer. Determines resampling needs automatically.

    Args:
        audio_np: float32 numpy array of complete audio
        sample_rate: source sample rate (typically 24000)
        output_format: one of ELEVENLABS_SUPPORTED_FORMATS

    Returns:
        Complete audio bytes in the requested format
    """
    target_rate = 24000
    needs_resample = False

    if output_format == "pcm_44100":
        target_rate = 44100
        needs_resample = True
    elif output_format == "mp3_44100_128":
        target_rate = 44100
        needs_resample = True
    # pcm_24000: no resample needed

    return convert_audio_chunk(audio_np, sample_rate, output_format, target_rate, needs_resample)
```

**Key Design Decision: `scipy.signal.resample_poly` for resampling (confirmed).**

scipy is already a transitive dependency of the project via mlx-audio, so no new dependency is added. We use `scipy.signal.resample_poly` for sample rate conversion rather than ffmpeg for two reasons:

1. **Per-chunk resampling.** Each streaming chunk must be resampled independently. Spawning an ffmpeg subprocess per chunk adds ~10-20ms of process overhead. scipy operates in-process with zero subprocess overhead.

2. **Quality.** `resample_poly` uses a polyphase FIR filter which handles short chunks cleanly. FFT-based `resample` can introduce edge artifacts on short audio segments. ffmpeg's internal resampler (libsoxr) is excellent but the subprocess overhead negates its benefit for small chunks.

3. **MP3 encoding still uses ffmpeg.** Only the resampling step uses scipy. The MP3 encoding step pipes the already-resampled PCM through ffmpeg with `-b:a 128k`. For PCM output, ffmpeg is not involved at all.

### 3.4 `tts_engine.py` (MODIFIED)

**New responsibilities:** Streaming generation method, interrupt flag support.

```python
# --- Existing code (unchanged) ---

import logging
import threading
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, settings): ...
    async def load_model(self): ...
    def generate(self, text, preset): ...  # existing, unchanged
    def health(self): ...

    # --- New: Streaming generation ---

    def generate_stream(
        self,
        text: str,
        preset: VoicePreset,
        interrupt_event: threading.Event,
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """Generate audio from text, yielding (audio_np, sample_rate) for each frame.

        This is the streaming variant of generate(). Instead of collecting all
        GenerationResult objects and concatenating, it yields each one individually
        so the streaming pipeline can deliver audio chunks as they are produced.

        In the sentence-splitting pipeline, this method is called once per sentence.
        Each call produces one or more (audio_np, sample_rate) tuples. For a single
        sentence (no newlines), model.generate() typically yields exactly one
        GenerationResult.

        The interrupt_event is checked between each frame yield. If set, generation
        stops immediately and the generator returns. The interrupt_event is shared
        across all sentences in a request -- setting it stops the entire pipeline.

        Args:
            text: The text to generate audio for (typically one sentence)
            preset: Voice preset with generation parameters
            interrupt_event: threading.Event checked between yields; if set, stop

        Yields:
            Tuple of (audio_numpy_float32, sample_rate) for each generated frame
        """
        sampler = make_sampler(temp=preset.temperature, top_k=preset.top_k)

        frame_count = 0
        for result in self.model.generate(
            text=text,
            voice=preset.voice_key,
            speaker=preset.speaker_id,
            sampler=sampler,
            max_audio_length_ms=preset.max_audio_length_ms,
            voice_match=True,
        ):
            # Check interrupt between frames
            if interrupt_event.is_set():
                logger.info(
                    "Generation interrupted after %d frames: voice=%s text=%r",
                    frame_count, preset.voice_key, text[:50],
                )
                break

            audio_np = np.array(result.audio)
            yield (audio_np, result.sample_rate)
            frame_count += 1

        logger.debug("generate_stream completed: %d frames for text=%r", frame_count, text[:50])
```

**Important:** The existing `generate()` method is NOT modified. It continues to work for the OpenAI endpoint and the ElevenLabs non-streaming endpoint. The new `generate_stream()` method provides the same model interaction but yields frames individually.

**Interrupt integration with sentence-splitting:** The same `interrupt_event` is passed to every `generate_stream()` call across all sentences in a request. When a client disconnects:

1. `stream_audio()` detects disconnect via `request.is_disconnected()`
2. Sets `interrupt_event`
3. The currently-running `generate_stream()` checks the event between yields and breaks
4. `_run_generation()` in the background thread sees the event and stops iterating sentences
5. DONE sentinel is pushed to the queue
6. `stream_audio()` exits, releasing the semaphore

### 3.5 `server.py` (MODIFIED)

**Changes:** Mount ElevenLabs router, expose shared state, update `/v1/models` for ElevenLabs format, enhance health endpoint.

```python
# --- New imports ---
from elevenlabs_routes import router as elevenlabs_router

# --- Modified lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    preset_manager.load_presets()
    logger.info("Loaded voice presets: %s", preset_manager.list_names())

    engine = TTSEngine(settings)
    await engine.load_model()

    # Expose shared state for ElevenLabs router
    app.state.engine = engine
    app.state.settings = settings
    app.state.preset_manager = preset_manager
    app.state.inference_semaphore = inference_semaphore
    app.state.queue_depth = 0
    app.state.requests_served = 0

    yield

    del app.state.engine

app = FastAPI(title="Sesame TTS", lifespan=lifespan)

# Mount ElevenLabs router
app.include_router(elevenlabs_router)

# --- Modified /v1/models route ---
@app.get("/v1/models")
async def list_models(request: Request):
    """List models. Returns ElevenLabs format if xi-api-key header is present."""
    xi_key = request.headers.get("xi-api-key")
    if xi_key is not None:
        # ElevenLabs format
        return [
            {
                "model_id": "csm-1b",
                "name": "Sesame CSM-1B",
                "can_do_text_to_speech": True,
                "can_do_voice_conversion": False,
                "can_be_finetuned": False,
                "can_use_style": False,
                "can_use_speaker_boost": True,
                "serves_pro_voices": False,
                "token_cost_factor": 0,
                "description": "Sesame CSM-1B -- local conversational speech model via MLX",
                "requires_alpha_access": False,
                "max_characters_request_free_user": 999999999,
                "max_characters_request_subscribed_user": 999999999,
                "languages": [
                    {"language_id": "en", "name": "English"}
                ],
            }
        ]
    # OpenAI format (existing)
    return {
        "object": "list",
        "data": [
            {
                "id": "csm-1b",
                "object": "model",
                "created": 1740000000,
                "owned_by": "sesame",
            }
        ],
    }

# --- Enhanced /health route ---
@app.get("/health")
async def health_check():
    engine: TTSEngine = app.state.engine
    model_health = engine.health()
    ffmpeg_available = shutil.which("ffmpeg") is not None
    voices = preset_manager.list_names()
    all_voice_ids = voices + [alias for alias in VOICE_ALIASES if VOICE_ALIASES[alias] in voices]

    status = "ok" if model_health["model_loaded"] and ffmpeg_available else "degraded"

    return {
        "status": status,
        "model": model_health,
        "ffmpeg": ffmpeg_available,
        "voices": all_voice_ids,
        "uptime_seconds": round(time.time() - _start_time),
        "queue_depth": getattr(app.state, "queue_depth", 0),
        "requests_served": getattr(app.state, "requests_served", 0),
        "streaming_enabled": True,
        "context_cache_sessions": 0,  # Placeholder for v1.1
    }
```

**Key Design Decision: Shared state via `app.state`.**

The ElevenLabs router needs access to the engine, settings, preset manager, and inference semaphore. Rather than using global variables (which would create circular imports), we attach these to `app.state` during lifespan setup. The router accesses them via `request.app.state`. This is the standard FastAPI pattern for dependency injection without DI frameworks.

### 3.6 `voice_presets.py` (MODIFIED)

**New responsibilities:** ElevenLabs-format voice metadata, voice list in ElevenLabs format.

```python
# --- Existing VoicePreset dataclass, add method ---

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
    # New optional fields for ElevenLabs metadata
    labels_accent: str = "american"
    labels_gender: str = "neutral"
    labels_age: str = "young"
    labels_use_case: str = "conversational"

    def elevenlabs_voice_id(self) -> str:
        """Return the ElevenLabs-safe voice ID (no underscores)."""
        # Map internal names to ElevenLabs-safe IDs
        # e.g., "conversational_b" -> "conversationalB"
        #        "conversational" -> "conversationalA"
        from server import VOICE_ALIASES
        # Reverse lookup: find alias that maps to this preset name
        for alias, preset_name in VOICE_ALIASES.items():
            if preset_name == self.name:
                return alias
        # Fallback: use the preset name as-is
        return self.name

    def elevenlabs_metadata(self, base_url: str = "http://localhost:8880") -> dict:
        """Return this voice's metadata in ElevenLabs API format."""
        voice_id = self.elevenlabs_voice_id()
        return {
            "voice_id": voice_id,
            "name": self.description or self.name,
            "category": "generated",
            "fine_tuning": {
                "is_allowed_to_fine_tune": False,
                "fine_tuning_requested": False,
            },
            "labels": {
                "accent": self.labels_accent,
                "gender": self.labels_gender,
                "age": self.labels_age,
                "use_case": self.labels_use_case,
            },
            "description": self.description,
            "preview_url": f"{base_url}/v1/voices/{voice_id}/preview",
            "available_for_tiers": [],
            "settings": {
                "stability": round(1.0 - self.temperature, 2),
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
            "sharing": None,
            "high_quality_base_model_ids": ["csm-1b"],
        }


# --- Existing VoicePresetManager, add method ---

class VoicePresetManager:
    # ... existing methods ...

    def list_elevenlabs_format(self, base_url: str = "http://localhost:8880") -> list[dict]:
        """Return all voices in ElevenLabs API format."""
        return [
            preset.elevenlabs_metadata(base_url)
            for preset in self.presets.values()
        ]
```

**Note on VoicePreset fields:** The new `labels_*` fields are optional with defaults. Existing preset JSON files will continue to work without modification. To customize labels, add the fields to the JSON file.

### 3.7 `config.py` (MODIFIED)

**New fields for streaming and ElevenLabs support.**

```python
class Settings(BaseSettings):
    # --- Existing fields (unchanged) ---
    host: str = "0.0.0.0"
    port: int = 8880
    log_level: str = "info"
    model_id: str = "mlx-community/csm-1b"
    cache_dir: Path = Path.home() / ".cache" / "sesame-tts"
    default_voice: str = "conversational_b"
    default_format: str = "mp3"
    max_input_length: int = 4096
    presets_dir: Path = Path(__file__).parent / "presets"

    # --- New: Streaming settings ---
    stream_chunk_size: int = 4096
    """Target chunk size in bytes for HTTP streaming output.
    PCM chunks are split into pieces of this size for smooth delivery.
    MP3 chunks are sent as-is since ffmpeg produces complete frames.
    Should be a multiple of 2 for PCM (int16 sample alignment)."""

    # --- New: ElevenLabs settings ---
    default_elevenlabs_format: str = "pcm_24000"
    """Default output format for ElevenLabs endpoints when not specified by client."""

    warmup_on_start: bool = True
    """Pre-generate a short audio clip on startup to warm MLX caches and JIT compilation.
    Adds ~5s to startup time but makes the first real request faster."""

    max_queue_depth: int = 3
    """Maximum number of queued requests before returning HTTP 429.
    Since inference is serial, deep queues mean long waits."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

**Removed:** `stream_buffer_frames` -- no longer needed. The sentence-splitting pipeline buffers at the sentence level, not the frame level. Each sentence's audio is accumulated and flushed as a complete unit, so there is no configurable frame buffer count.

**Environment variable names** (auto-derived by pydantic-settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_CHUNK_SIZE` | `4096` | Target bytes per HTTP streaming chunk |
| `DEFAULT_ELEVENLABS_FORMAT` | `pcm_24000` | Default ElevenLabs output format |
| `WARMUP_ON_START` | `true` | Pre-generate audio on startup |
| `MAX_QUEUE_DEPTH` | `3` | Max queued requests before 429 |

### 3.8 `errors.py` (MODIFIED)

**New: ElevenLabs error response helper.**

```python
# --- Existing code (unchanged) ---

class TTSError(Exception): ...
def openai_error_response(...): ...

# --- New: ElevenLabs error format ---

def elevenlabs_error_response(message: str, status: str, status_code: int) -> JSONResponse:
    """Return an ElevenLabs-format error response.

    ElevenLabs uses {"detail": {"status": ..., "message": ...}} format,
    distinct from OpenAI's {"error": {"message": ..., "type": ..., "code": ...}}.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "status": status,
                "message": message,
            }
        },
    )
```

**Note:** The `elevenlabs_error()` helper is defined inline in `elevenlabs_routes.py` for now (see section 3.1). During implementation, we may consolidate it into `errors.py` alongside `openai_error_response()`. The TRD shows both locations -- the implementer should choose one and keep it consistent.

---

## 4. ElevenLabs API Specification

### 4.1 POST /v1/text-to-speech/{voice_id}/stream

Generate speech audio and stream it back in chunks. Uses sentence-splitting pipeline for low first-chunk latency.

**Request:**
```
POST /v1/text-to-speech/conversationalB/stream?output_format=pcm_24000 HTTP/1.1
Host: localhost:8880
Content-Type: application/json
xi-api-key: local

{
    "text": "Hello Burke, Talk Mode is working locally now.",
    "model_id": "csm-1b",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": true
    },
    "output_format": "pcm_24000"
}
```

| Field | Type | Required | Default | Validation |
|-------|------|----------|---------|------------|
| `text` | string | Yes | -- | 1-4096 characters |
| `model_id` | string | No | `"csm-1b"` | Any value accepted, mapped to csm-1b |
| `voice_settings` | object | No | preset defaults | All fields optional, see mapping table |
| `output_format` | string | No (query param preferred) | `"pcm_24000"` | One of: `pcm_24000`, `pcm_44100`, `mp3_44100_128` |
| `optimize_streaming_latency` | int | No | `0` | 0-4, accepted but not used in v1.0 |

**Path parameter:** `voice_id` -- resolved via `VOICE_ALIASES` dict, then preset manager, then default voice fallback.

**Query parameter:** `output_format` -- takes priority over the body field.

**Response:**
```
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Transfer-Encoding: chunked

<chunked raw audio bytes>
```

**Content-Type by format:**

| Format | Content-Type |
|--------|-------------|
| `pcm_24000` | `application/octet-stream` |
| `pcm_44100` | `application/octet-stream` |
| `mp3_44100_128` | `audio/mpeg` |

**Interrupt behavior:** No explicit cancel endpoint. When the client closes the connection (e.g., Talk Mode interrupts), the server detects the disconnect via `request.is_disconnected()`, sets the interrupt event to stop inference, and cleans up resources. The next request can proceed immediately.

### 4.2 POST /v1/text-to-speech/{voice_id}

Generate speech audio and return complete buffer (non-streaming).

**Request:** Same as 4.1.

**Response:**
```
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Length: 48000

<complete audio bytes>
```

**Difference from streaming:** Response includes `Content-Length` header instead of `Transfer-Encoding: chunked`.

### 4.3 GET /v1/voices

List all available voices in ElevenLabs format.

**Request:**
```
GET /v1/voices HTTP/1.1
Host: localhost:8880
xi-api-key: local
```

**Response:**
```json
{
    "voices": [
        {
            "voice_id": "conversationalA",
            "name": "Natural, warm conversational voice (default)",
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
            "description": "Natural, warm conversational voice (default)",
            "preview_url": "http://localhost:8880/v1/voices/conversationalA/preview",
            "available_for_tiers": [],
            "settings": {
                "stability": 0.1,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": true
            },
            "sharing": null,
            "high_quality_base_model_ids": ["csm-1b"]
        },
        {
            "voice_id": "conversationalB",
            "name": "Second conversational voice, slightly different character",
            "category": "generated",
            ...
        }
    ]
}
```

### 4.4 GET /v1/voices/{voice_id}

Get metadata for a single voice.

**Request:**
```
GET /v1/voices/conversationalB HTTP/1.1
Host: localhost:8880
```

**Response:** Single voice object (same structure as one element of the `voices` array in 4.3).

**Error (voice not found):**
```json
{
    "detail": {
        "status": "voice_not_found",
        "message": "Voice 'unknownVoice' not found. Available: conversational, conversational_b"
    }
}
```

### 4.5 GET /v1/models (ElevenLabs format)

**Request:**
```
GET /v1/models HTTP/1.1
Host: localhost:8880
xi-api-key: local
```

**Response:**
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

**Note:** The same `/v1/models` endpoint returns OpenAI format when `xi-api-key` header is absent (existing behavior preserved).

### 4.6 GET /v1/user/subscription

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

### 4.7 GET /v1/user

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

---

## 5. Streaming Architecture

### 5.1 Sentence-Splitting Pipeline (v1.1 Core Strategy)

The key v1.0 streaming strategy is **sentence-level pipelining**: split input text into sentences before generating, send the first sentence to the model immediately, stream its audio as soon as it's ready, then generate subsequent sentences while the first is playing.

**Why this works:** mlx-audio's `model.generate()` yields `GenerationResult` objects at sentence granularity (it splits on `\n+` internally). For a single sentence with no newlines, it yields exactly ONE `GenerationResult` with the complete audio. By splitting the input text into sentences ourselves (before passing to the model), we control the granularity and can pipeline generation with streaming delivery.

**Pipeline stages:**

```
Stage 1: SPLIT
  Input text -> split_sentences() -> ["Sentence 1.", "Sentence 2.", "Sentence 3."]
  Time: <1ms (regex split)

Stage 2: GENERATE FIRST SENTENCE
  "Sentence 1." -> engine.generate_stream() -> (audio_np, sr)
  Time: ~0.8-1.2s for a short sentence (10 words)

Stage 3: STREAM FIRST SENTENCE AUDIO
  (audio_np, sr) -> convert_audio_chunk() -> yield bytes to client
  Client begins playback immediately
  Time: ~10ms (conversion) + network latency

Stage 4: GENERATE NEXT SENTENCE (overlaps with Stage 3 playback)
  "Sentence 2." -> engine.generate_stream() -> (audio_np, sr)
  This happens while the client is playing Sentence 1's audio
  Time: ~0.8-1.2s (but client is hearing audio during this time)

Stage 5: STREAM NEXT SENTENCE AUDIO
  Repeat Stage 3-4 for each remaining sentence

Stage 6: DONE
  All sentences generated and streamed
```

**First-chunk latency improvement:**

```
Without sentence splitting:
  "Hello there. How are you? I'm doing great."
  -> Generate entire text as one unit: ~3-5s
  -> First audio arrives at: ~3-5s

With sentence splitting:
  "Hello there. How are you? I'm doing great."
  -> Split into 3 sentences
  -> Generate "Hello there." only: ~0.8-1.2s
  -> First audio arrives at: ~0.8-1.2s (3-4x improvement)
  -> Remaining sentences generate while client plays first sentence
```

**Single-sentence input (typical Talk Mode):**

For single-sentence input (the most common Talk Mode case), the pipeline has no splitting to do -- it generates the one sentence and streams it. Behavior is identical to generating the full text. No overhead from the splitting step.

### 5.2 Sentence Splitting Implementation

```python
import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    """Split on .!? followed by whitespace. Keeps punctuation with sentence."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]
```

**Splitting behavior examples:**

| Input | Output |
|-------|--------|
| `"Hello there."` | `["Hello there."]` |
| `"Hello there. How are you?"` | `["Hello there.", "How are you?"]` |
| `"Wait! Really? Yes."` | `["Wait!", "Really?", "Yes."]` |
| `"Dr. Smith went home."` | `["Dr.", "Smith went home."]` |
| `"Hello there"` | `["Hello there"]` |

**Note on abbreviations:** The regex will incorrectly split on abbreviations like "Dr." or "U.S." followed by whitespace. For Talk Mode's typical short inputs (single sentences), this is irrelevant. For multi-sentence paragraphs, occasional mis-splits on abbreviations produce correct audio -- just split into more pieces than intended, which actually improves streaming latency. This is an acceptable tradeoff for the simplicity of the regex approach.

### 5.3 Disconnect Detection (Interrupt Mechanism)

There is no explicit cancel endpoint. Interrupt is handled entirely via client disconnect detection in the streaming response loop.

```
                    Main Thread (asyncio)           Background Thread (sync)
                    ========================        ========================

request arrives --> create threading.Event() -----> _run_generation() starts
                    |                               |
                    | stream_audio() async gen      | for sentence in sentences:
                    | yields chunks to client       |   engine.generate_stream(sentence)
                    |                               |   yields frames to queue
                    |                               |   pushes SENTINEL after each sentence
                    |                               |
client disconnects  |                               |
  detected via      |                               |
  is_disconnected() |                               |
  in stream_audio() |                               |
                    |                               |
                    interrupt_event.set() ---------> checked between yields
                    |                               checked between sentences
                    |                               |
                    | break from yield loop         | breaks from generate loop
                    |                               | pushes DONE, returns
                    |                               |
                    await gen_future <-------------- thread completes
                    |
                    semaphore released (finally)
                    |
                    next request can proceed
```

**How disconnect detection works in the streaming response:**

The disconnect check happens in two places:

1. **In `stream_audio()` (primary):** During the `asyncio.wait_for(audio_queue.get(), timeout=0.1)` timeout loop. Every 100ms while waiting for the next audio frame, `request.is_disconnected()` is checked. If True, `interrupt_event.set()` is called and the loop breaks.

2. **In `generate_and_stream()` (secondary):** After each `chunk` is yielded from `stream_audio()`, the route handler checks `request.is_disconnected()`. If True, it sets the interrupt event and breaks. This catches disconnects that happen between chunk yields rather than during queue waits.

**Why `threading.Event` instead of `asyncio.Event`:**

1. `model.generate()` runs in a thread pool executor (synchronous MLX code)
2. `asyncio.Event` can only be waited on from async code; checking `.is_set()` from a sync thread is technically safe but semantically wrong
3. `threading.Event` is designed for cross-thread signaling, has zero overhead for `.is_set()` checks, and is the natural choice when one side is sync

**Interrupt latency:** With sentence-splitting, the interrupt check occurs between sentences (not just between `GenerationResult` yields within a sentence). For a sentence currently being generated, the interrupt takes effect after that sentence's generation completes. For typical Talk Mode sentences (10-20 words, ~1s generation), this means worst-case ~1s interrupt latency. This is acceptable because the client has already stopped listening (it disconnected).

### 5.4 PCM Chunk Size for Smooth Playback

For smooth audio playback, chunks should align to sample boundaries:

```
pcm_24000:
  2 bytes per sample (int16)
  Chunk size should be a multiple of 2
  Recommended: 4800 bytes = 100ms of audio at 24000 Hz
  Or: 2400 bytes = 50ms of audio

pcm_44100:
  2 bytes per sample (int16)
  Recommended: 8820 bytes = 100ms of audio at 44100 Hz
  Or: 4410 bytes = 50ms of audio

mp3_44100_128:
  MP3 frames are variable size (~418 bytes at 128kbps/44100Hz)
  Chunk size of 4096 bytes fits ~9-10 MP3 frames
  ffmpeg output is already framed, so any chunk size works
```

The `STREAM_CHUNK_SIZE` setting (default 4096) controls the maximum bytes per HTTP chunk. For PCM, we round down to the nearest sample boundary. For MP3, we use the setting as-is.

---

## 6. Output Format Specifications

### 6.1 pcm_44100

- **Description:** Raw signed 16-bit little-endian PCM at 44,100 Hz, mono
- **Content-Type:** `application/octet-stream`
- **Use case:** macOS/iOS Talk Mode default
- **Implementation:** Resample from native 24,000 Hz using `scipy.signal.resample_poly(audio, up=147, down=80)`, then convert float32 to int16
- **Byte layout:** Interleaved int16 samples, little-endian. Each sample is 2 bytes.
- **Bytes per second:** 88,200

**Playback verification:**
```bash
ffplay -f s16le -ar 44100 -ac 1 output.pcm
```

### 6.2 pcm_24000

- **Description:** Raw signed 16-bit little-endian PCM at 24,000 Hz, mono
- **Content-Type:** `application/octet-stream`
- **Use case:** Android Talk Mode default, native CSM-1B sample rate
- **Implementation:** No resampling needed. Convert float32 directly to int16.
- **Byte layout:** Interleaved int16 samples, little-endian. Each sample is 2 bytes.
- **Bytes per second:** 48,000
- **Performance note:** This is the fastest format -- zero resampling overhead.

**Playback verification:**
```bash
ffplay -f s16le -ar 24000 -ac 1 output.pcm
```

### 6.3 mp3_44100_128

- **Description:** MPEG Layer 3 at 44,100 Hz, 128 kbps, mono
- **Content-Type:** `audio/mpeg`
- **Use case:** MP3 streaming option for clients that prefer compressed audio
- **Implementation:**
  1. Resample from 24,000 Hz to 44,100 Hz using `scipy.signal.resample_poly`
  2. Convert float32 to int16 PCM
  3. Pipe int16 bytes through ffmpeg: `-c:a libmp3lame -b:a 128k -f mp3`
- **Bytes per second:** ~16,000 (128 kbps / 8)
- **Performance note:** Each chunk requires an ffmpeg subprocess call. For streaming, this adds ~10-20ms per chunk.

**Playback verification:**
```bash
ffplay output.mp3
# or
ffprobe output.mp3  # should show: 128 kb/s, 44100 Hz, mono
```

### 6.4 Deferred Formats

The following ElevenLabs format strings are recognized but return HTTP 400:

`pcm_16000`, `pcm_22050`, `mp3_22050_32`, `mp3_44100_32`, `mp3_44100_64`, `mp3_44100_96`, `mp3_44100_192`

**Error response:**
```json
{
    "detail": {
        "status": "invalid_request",
        "message": "Format 'pcm_16000' is not yet supported. Available: mp3_44100_128, pcm_24000, pcm_44100"
    }
}
```

---

## 7. Master Task List

Each task has a unique ID (T-101+), estimated hours, dependencies, and completion checkbox. Tasks are organized by implementation phase.

### Phase 1: Audio Format Support (PCM + MP3 Streaming Formats)

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-101 | Add `resample_audio()` function to `audio_converter.py` using `scipy.signal.resample_poly` for 24000->44100 conversion. scipy is already a transitive dependency -- no new dependency needed. Verify import works. | 2h | -- | [ ] |
| T-102 | Add `convert_audio_chunk()` function to `audio_converter.py` for per-chunk format conversion (PCM raw bytes, MP3 via ffmpeg). | 2h | T-101 | [ ] |
| T-103 | Add `convert_audio_elevenlabs()` function to `audio_converter.py` for complete-buffer ElevenLabs format conversion. | 1h | T-102 | [ ] |
| T-104 | Add `ELEVENLABS_SUPPORTED_FORMATS`, `ELEVENLABS_DEFERRED_FORMATS`, and `ELEVENLABS_CONTENT_TYPES` constants to `audio_converter.py`. | 0.5h | -- | [ ] |
| T-105 | Test: Validate `resample_audio()` produces correct 44100 Hz output from 24000 Hz input. Verify sample count ratio matches 147/80. Verify no audible artifacts on a 5-second speech clip. | 1h | T-101 | [ ] |
| T-106 | Test: Validate `convert_audio_chunk()` for all 3 formats. Verify PCM byte alignment (even byte count), MP3 playability via ffprobe. | 1h | T-102 | [ ] |

**Phase 1 total: 7.5h**

### Phase 2: ElevenLabs Routes (Non-Streaming First)

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-201 | Create `elevenlabs_routes.py` with `APIRouter`, request models (`ElevenLabsTTSRequest`, `VoiceSettings`), model ID mapping, error helpers, and format/voice resolution helpers (`_resolve_format`, `_resolve_preset`). | 2h | -- | [ ] |
| T-202 | Implement `POST /v1/text-to-speech/{voice_id}` (non-streaming). Wire up voice resolution, settings mapping, engine.generate(), and `convert_audio_elevenlabs()`. | 3h | T-103, T-201 | [ ] |
| T-203 | Implement `GET /v1/voices` and `GET /v1/voices/{voice_id}`. | 1.5h | T-201, T-302 | [ ] |
| T-204 | Implement `GET /v1/user/subscription` and `GET /v1/user` stub endpoints. | 0.5h | T-201 | [ ] |
| T-205 | Modify `GET /v1/models` in `server.py` to return ElevenLabs format when `xi-api-key` header is present. | 1h | -- | [ ] |
| T-206 | Mount ElevenLabs router in `server.py` via `app.include_router()`. Expose engine, settings, preset_manager, semaphore on `app.state`. | 1h | T-201 | [ ] |
| T-207 | Test: Non-streaming TTS endpoint produces valid audio in all 3 formats. Test voice resolution and fallback. Test all compatibility endpoints return valid JSON. | 2h | T-202, T-203, T-204, T-205 | [ ] |

**Phase 2 total: 11h**

### Phase 3: Streaming Implementation (includes sentence-splitting pipeline)

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-301 | Add `generate_stream()` method to `TTSEngine` class. Yields `(audio_np, sample_rate)` tuples. Accepts `threading.Event` for interrupt. Designed to be called once per sentence in the pipeline. | 2h | -- | [ ] |
| T-302 | Add ElevenLabs metadata methods to `VoicePreset` (`elevenlabs_voice_id()`, `elevenlabs_metadata()`) and `VoicePresetManager` (`list_elevenlabs_format()`). | 1.5h | -- | [ ] |
| T-303 | Create `streaming.py` with `split_sentences()` function (regex: split on `.!?` followed by whitespace), `stream_audio()` async generator with sentence-splitting pipeline, `_parse_sample_rate()`, `_convert_chunk()`, and `_split_bytes()` helpers. Implement: sentence splitting, per-sentence generation via background thread, async queue bridge, sentence-level buffering and flushing, chunk-size byte splitting, disconnect detection in queue timeout loop. | 4h | T-301, T-102 | [ ] |
| T-304 | Implement `POST /v1/text-to-speech/{voice_id}/stream` in `elevenlabs_routes.py`. Wire up streaming pipeline with `StreamingResponse`. Include disconnect detection in `generate_and_stream()` wrapper. | 2h | T-303, T-201 | [ ] |
| T-305 | Add streaming config fields to `config.py`: `stream_chunk_size`, `default_elevenlabs_format`, `warmup_on_start`, `max_queue_depth`. Remove `stream_buffer_frames` (no longer needed with sentence-level buffering). | 1h | -- | [ ] |
| T-306 | Update `.env.example` with all new environment variables and documentation comments. | 0.5h | T-305 | [ ] |
| T-307 | Test: Streaming endpoint returns chunked audio for all 3 formats. Verify `Transfer-Encoding: chunked` header. Verify audio is playable. | 2h | T-304 | [ ] |
| T-308 | Test: Multi-sentence input delivers first sentence audio significantly faster than full-text generation time. Verify via timing that first chunk arrives in ~1s for a 3-sentence paragraph. | 1.5h | T-304 | [ ] |
| T-309 | Test: `split_sentences()` unit tests covering: single sentence, multiple sentences, sentences with `!` and `?`, abbreviations like "Dr.", empty input, input with no terminal punctuation. | 1h | T-303 | [ ] |

**Phase 3 total: 15.5h**

### Phase 4: Interrupt Support (Disconnect Detection Only)

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-401 | Implement client disconnect detection in `stream_audio()` via `request.is_disconnected()` polling during `asyncio.wait_for` timeout (every 100ms). Set `interrupt_event` on disconnect. | 1.5h | T-303 | [ ] |
| T-402 | Verify `generate_stream()` exits cleanly when `interrupt_event` is set mid-sentence. Verify `_run_generation()` stops iterating remaining sentences when event is set between sentences. | 1.5h | T-401, T-301 | [ ] |
| T-403 | Add `finally` block to `stream_audio()` ensuring interrupt event is set and generation thread is awaited on any exit path. Add `finally` block to `generate_and_stream()` ensuring semaphore release and queue depth decrement. | 1h | T-402 | [ ] |
| T-404 | Add interrupt logging: log voice_id, frames generated, elapsed time, which sentence was interrupted, reason (client_disconnect). | 0.5h | T-402 | [ ] |
| T-405 | Test: Start long generation (3+ sentences), disconnect after first sentence, verify server recovers and accepts new request immediately. Verify no memory leak (check health endpoint). | 2h | T-403 | [ ] |
| T-406 | Test: Rapid sequential requests (simulate Talk Mode pattern). Send 5 short sentences back-to-back with 500ms delays. Verify all complete without errors. | 1.5h | T-403 | [ ] |

**Phase 4 total: 8h**

### Phase 5: Integration Testing + OpenClaw Talk Mode Verification

| ID | Task | Est. | Deps | Status |
|----|------|------|------|--------|
| T-501 | Enhance health endpoint with `queue_depth`, `requests_served`, `streaming_enabled`, `context_cache_sessions`. | 1h | T-206 | [ ] |
| T-502 | Add ElevenLabs exception handlers in `server.py` for ElevenLabs routes: detect path prefix, return ElevenLabs error format. Verify OpenAI routes still return OpenAI format. | 2h | T-206 | [ ] |
| T-503 | Integration test: Full ElevenLabs API smoke test. All endpoints, all formats, all auth header combinations. Validate response schemas match ElevenLabs format. | 3h | T-307, T-405 | [ ] |
| T-504 | Integration test: Simulate Talk Mode session. 5 sequential streaming requests with pcm_44100 format, xi-api-key header, short sentences. Verify all complete successfully. | 2h | T-503 | [ ] |
| T-505 | OpenClaw Talk Mode integration: Configure `openclaw.json`, start Talk Mode conversation, verify audio plays, test interrupt via disconnect. Document results and any issues. | 2h | T-504 | [ ] |
| T-506 | Performance benchmark: Measure time-to-first-chunk for single sentence (10 words) and multi-sentence (3 sentences) across all 3 formats. Verify sentence-splitting provides measurable improvement for multi-sentence input. Record results in README. | 1.5h | T-503 | [ ] |
| T-507 | Update README.md: Add ElevenLabs API documentation, OpenClaw Talk Mode setup instructions, streaming configuration, output format reference, sentence-splitting explanation. | 2h | T-505 | [ ] |
| T-508 | Update `.env.example`: Add all new streaming/ElevenLabs environment variables with comments. | 0.5h | T-306 | [ ] |

**Phase 5 total: 14h**

### Task Summary

| Phase | Tasks | Estimated Hours |
|-------|-------|----------------|
| Phase 1: Audio Format Support | T-101 to T-106 | 7.5h |
| Phase 2: ElevenLabs Routes | T-201 to T-207 | 11h |
| Phase 3: Streaming + Sentence Splitting | T-301 to T-309 | 15.5h |
| Phase 4: Interrupt (Disconnect Detection) | T-401 to T-406 | 8h |
| Phase 5: Integration + OpenClaw | T-501 to T-508 | 14h |
| **Total** | **30 tasks** | **56h** |

---

## 8. Sprint Planning

### Sprint 1: Foundation (Days 1-2)

**Goal:** Audio format conversion and ElevenLabs route skeleton working end-to-end with non-streaming TTS.

**Tasks:** T-101 through T-106 (Phase 1), T-201, T-204, T-205, T-206, T-305, T-306

**Estimated effort:** ~14h

**Exit criteria:**
- `scipy.signal.resample_poly` correctly resamples 24000->44100 with no audible artifacts
- `convert_audio_chunk()` produces valid PCM and MP3 output
- ElevenLabs router is mounted and compatibility stubs return valid JSON
- Config extended with streaming fields
- All Phase 1 tests pass

### Sprint 2: Non-Streaming TTS + Voice Management (Days 3-4)

**Goal:** Complete ElevenLabs non-streaming TTS and voice management endpoints.

**Tasks:** T-202, T-203, T-301, T-302, T-207

**Estimated effort:** ~10h

**Exit criteria:**
- `POST /v1/text-to-speech/{voice_id}` generates audio in all 3 formats
- `GET /v1/voices` returns valid ElevenLabs-format voice list
- Voice ID resolution works (aliases, preset names, fallback to default)
- `generate_stream()` method exists on TTSEngine (needed for Sprint 3)
- All non-streaming tests pass

### Sprint 3: Streaming + Sentence Splitting Pipeline (Days 5-6)

**Goal:** Sentence-splitting chunked streaming working end-to-end for all formats.

**Tasks:** T-303, T-304, T-307, T-308, T-309

**Estimated effort:** ~10.5h

**Exit criteria:**
- `POST /v1/text-to-speech/{voice_id}/stream` returns chunked audio
- All 3 output formats work in streaming mode
- `split_sentences()` correctly splits multi-sentence input
- Multi-sentence input delivers first sentence audio in ~1s (not ~3-5s)
- `Transfer-Encoding: chunked` header present in response
- Sentence splitting unit tests pass

### Sprint 4: Interrupt + Integration (Days 7-8)

**Goal:** Disconnect-based interrupt working, all integration tests passing, OpenClaw Talk Mode verified.

**Tasks:** T-401 through T-406 (Phase 4), T-501 through T-508 (Phase 5)

**Estimated effort:** ~22h

**Exit criteria:**
- Client disconnect stops generation within one sentence
- Server recovers immediately after interrupt
- Rapid sequential requests complete without errors
- OpenClaw Talk Mode works with base URL change only
- README updated with ElevenLabs API docs and Talk Mode instructions
- Performance benchmarks recorded (single-sentence and multi-sentence)

### Timeline Summary

| Sprint | Days | Hours | Tasks | Exit Criteria |
|--------|------|-------|-------|---------------|
| 1. Foundation | 1-2 | ~14h | T-101-106, T-201, T-204-206, T-305-306 | Audio formats + route skeleton |
| 2. Non-Streaming | 3-4 | ~10h | T-202-203, T-207, T-301-302 | Full non-streaming TTS |
| 3. Streaming | 5-6 | ~10.5h | T-303-304, T-307-309 | Sentence-splitting streaming working |
| 4. Integration | 7-8 | ~22h | T-401-406, T-501-508 | Talk Mode verified |
| **Total** | **~8 days** | **~56h** | **30 tasks** | |

---

## 9. Quality Requirements

### 9.1 Testing Strategy

| Level | Tool | Scope |
|-------|------|-------|
| Unit (targeted) | pytest | `resample_audio()`, `convert_audio_chunk()`, `split_sentences()`, voice settings mapping, model ID mapping |
| Integration | pytest + httpx | All ElevenLabs endpoints, streaming, interrupt, format conversion |
| End-to-end | curl + manual | Talk Mode session, interrupt behavior, sequential requests |

**Test files:**

| File | Coverage |
|------|----------|
| `tests/test_audio_converter.py` | `resample_audio()`, `convert_audio_chunk()`, `convert_audio_elevenlabs()`, format validation |
| `tests/test_elevenlabs_routes.py` | All ElevenLabs endpoints: TTS streaming/non-streaming, voices, models, user, error cases |
| `tests/test_streaming.py` | Streaming pipeline: sentence splitting, sentence-level buffering, chunking, interrupt, disconnect handling |

### 9.2 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first chunk (streaming, single 10-word sentence) | < 2s | `time curl` against streaming endpoint |
| Time to first chunk (streaming, 3-sentence paragraph) | < 1.5s (first sentence only) | Timestamp of first chunk delivery vs. full-text generation time |
| Full sentence generation (non-streaming, 10 words) | < 2s | `time curl` against non-streaming endpoint |
| Resampling overhead per sentence | < 20ms | Benchmark `resample_audio()` on typical sentence audio |
| MP3 encoding overhead per sentence | < 30ms | Benchmark ffmpeg subprocess on typical sentence audio |
| Interrupt response (generation stop) | < 1 sentence generation time | Time from disconnect to generator exit |
| Memory after 100 requests | No growth > 100MB above steady state | Health endpoint `peak_memory_gb` |

**Note on first-chunk latency:** The 2s target for single-sentence input reflects the generation time for one short sentence. For multi-sentence input, the sentence-splitting pipeline achieves ~1-1.5s first-chunk latency because only the first sentence needs to generate before streaming begins. This is the key improvement over v1.0's approach of generating all text before streaming.

### 9.3 Code Quality

- Type hints on all public functions
- Docstrings on all public classes and functions
- `ruff` for linting and formatting
- No circular imports between modules
- ElevenLabs routes in separate file from OpenAI routes
- All new dependencies pinned in `pyproject.toml`

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Sentence splitting on abbreviations ("Dr. Smith") produces extra splits** | Medium | Low | Extra splits only affect multi-sentence paragraphs. Each split still generates correct audio, just in smaller pieces. Actually improves streaming latency. Acceptable for v1.0. |
| **Resampling quality (scipy) introduces audible artifacts** | Low | Medium | Use `resample_poly` (polyphase FIR) instead of `resample` (FFT). Benchmark and A/B test against ffmpeg resampling. If artifacts detected, fall back to ffmpeg subprocess for resampling. scipy is already a transitive dependency -- confirmed. |
| **ffmpeg subprocess per MP3 sentence adds too much latency** | Medium | Low | Benchmark early (T-106). If >30ms per sentence: consider persistent ffmpeg subprocess with stdin/stdout pipe, or batch multiple sentences before encoding. PCM formats are unaffected (no ffmpeg). |
| **OpenClaw Talk Mode expects specific ElevenLabs response headers we don't implement** | Medium | High | Test with actual OpenClaw early (T-505, Sprint 4). Analyze OpenClaw source for ElevenLabs integration code. Most likely required: `Transfer-Encoding: chunked`, `Content-Type`. |
| **`asyncio.Queue` bridge between sync generator and async streaming introduces latency** | Low | Low | Queue operations are O(1). The bottleneck is MLX inference (~300ms per frame), not the queue. If needed, switch to a thread-safe deque with condition variable. |
| **threading.Event interrupt doesn't stop MLX inference mid-sentence** | Low | Medium | The event is checked between `GenerationResult` yields within a sentence and between sentences. For single-sentence calls (which is what the pipeline sends), the check happens after the sentence completes. Acceptable for v1.0 since each sentence is short (<2s generation). |
| **Memory leak from interrupted generations** | Low | Medium | Ensure `finally` block calls `interrupt_event.set()`, awaits generator thread, and releases semaphore. MLX's garbage collection handles tensor cleanup. Monitor via health endpoint. |
| **Circular import between `elevenlabs_routes.py` and `server.py`** | Medium | Low | `elevenlabs_routes.py` imports `resolve_voice` and `VOICE_ALIASES` from `server.py`. Move these to a shared module (e.g., `voice_resolution.py`) if circular import occurs during implementation. |
| **Sentence boundary audio discontinuity (clicks/pops between sentences)** | Low | Medium | Each sentence generates independently, so there may be minor discontinuities at sentence boundaries. For Talk Mode (short single sentences), this is irrelevant. For multi-sentence input, monitor during testing and consider cross-fade if needed. |

---

## References

- [PRD-streaming.md](./PRD-streaming.md) v1.1 -- Product requirements for ElevenLabs streaming
- [TRD.md](./TRD.md) v1.1 -- Base server technical requirements (style reference)
- [GitHub Issue #1](https://github.com/bautrey/sesame-tts/issues/1) -- Feature request
- [ElevenLabs API Reference](https://elevenlabs.io/docs/api-reference/text-to-speech) -- Target API compatibility
- [mlx-audio (GitHub)](https://github.com/Blaizzy/mlx-audio) -- MLX inference library
- [scipy.signal.resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html) -- Polyphase resampling
- [OpenClaw](https://github.com/openclaw) -- Primary client application

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-20 | Initial TRD. Architecture design, module specifications with code sketches, API specification, streaming pipeline design, output format specs, 28-task breakdown across 5 phases. Key finding: mlx-audio yields sentence-level GenerationResults (not codec frames), so v1.0 streaming provides chunked HTTP delivery but not sub-sentence latency reduction. |
| 1.1 | 2026-02-20 | Refined with user interview feedback. Added sentence-splitting streaming optimization: split input into sentences before generating, stream first sentence audio while generating subsequent sentences, reducing first-chunk latency from full-text generation time to single-sentence generation time (~1s). Confirmed scipy.signal.resample_poly for 24000->44100 resampling (already a transitive dependency). Simplified interrupt mechanism to disconnect-detection-only (no explicit cancel endpoint). Removed `stream_buffer_frames` config (replaced by sentence-level buffering). Added `split_sentences()` with regex splitting. Updated all code sketches for implementation completeness. Added T-309 (sentence splitting tests). Updated task count to 30, estimate to 56h. Added sentence boundary audio discontinuity risk. |
