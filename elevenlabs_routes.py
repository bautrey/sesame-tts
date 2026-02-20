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
    convert_audio_elevenlabs,
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
