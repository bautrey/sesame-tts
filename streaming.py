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
