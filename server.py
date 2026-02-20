import asyncio
import logging
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from audio_converter import CONTENT_TYPES, SUPPORTED_FORMATS, convert_audio
from config import Settings
from errors import TTSError, openai_error_response
from tts_engine import TTSEngine
from voice_presets import VoicePresetManager

logger = logging.getLogger(__name__)

settings = Settings()
preset_manager = VoicePresetManager(settings.presets_dir)

# Alphanumeric aliases for ElevenLabs compatibility (no underscores allowed)
VOICE_ALIASES = {
    "conversationalB": "conversational_b",
    "conversationalA": "conversational",
}


def resolve_voice(voice_id: str) -> str:
    """Resolve a voice alias to its real preset name."""
    return VOICE_ALIASES.get(voice_id, voice_id)

# Serial inference via semaphore
inference_semaphore = asyncio.Semaphore(1)

_start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load voice presets
    preset_manager.load_presets()
    logger.info("Loaded voice presets: %s", preset_manager.list_names())

    # Load model
    engine = TTSEngine(settings)
    await engine.load_model()
    app.state.engine = engine

    yield

    del app.state.engine


app = FastAPI(title="Sesame TTS", lifespan=lifespan)


# --- Request/Response Models ---


class SpeechRequest(BaseModel):
    model: str
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


# --- Exception Handlers ---


@app.exception_handler(TTSError)
async def tts_error_handler(request: Request, exc: TTSError):
    return openai_error_response(exc.message, exc.error_type, exc.code, exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        msg = "; ".join(e.get("msg", str(e)) for e in errors)
    else:
        msg = str(exc)
    return openai_error_response(msg, "invalid_request_error", "invalid_input", 400)


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return openai_error_response(
        "Internal server error", "server_error", "internal_error", 500
    )


# --- Routes ---


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    engine: TTSEngine = app.state.engine

    if not engine.model_loaded:
        raise TTSError("Model not loaded", "server_error", "model_not_ready", 503)

    # Validate format
    fmt = req.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise TTSError(
            f"Invalid format '{req.response_format}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
            "invalid_request_error",
            "invalid_format",
        )

    # Validate voice
    voice_name = resolve_voice(req.voice)
    preset = preset_manager.get(voice_name)
    if preset is None:
        raise TTSError(
            f"Voice '{voice_name}' not found. Available voices: {', '.join(preset_manager.list_names())}",
            "invalid_request_error",
            "invalid_voice",
        )

    # Log speed if non-default (accepted but not applied in v1)
    if req.speed != 1.0:
        logger.info("Speed %.1f requested but not applied in v1", req.speed)

    # Generate audio with semaphore for serial inference
    async with inference_semaphore:
        start = time.time()
        loop = asyncio.get_event_loop()
        try:
            audio_np, sample_rate = await loop.run_in_executor(
                None, engine.generate, req.input, preset
            )
        except MemoryError:
            raise TTSError(
                "Insufficient memory for generation",
                "server_error",
                "insufficient_memory",
                503,
            )
        except RuntimeError as e:
            raise TTSError(str(e), "server_error", "generation_failed", 500)
        gen_time = time.time() - start

    # Convert to requested format
    try:
        audio_bytes = convert_audio(audio_np, sample_rate, fmt)
    except RuntimeError as e:
        raise TTSError(str(e), "server_error", "conversion_failed", 500)

    total_time = time.time() - start
    logger.info(
        "Generated: voice=%s fmt=%s input_len=%d gen=%.2fs total=%.2fs",
        req.voice, fmt, len(req.input), gen_time, total_time,
    )

    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES[fmt],
        headers={
            "Content-Disposition": f"attachment; filename=speech.{fmt}",
            "X-Generation-Time": f"{gen_time:.2f}",
        },
    )


@app.get("/v1/models")
async def list_models():
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
    }
