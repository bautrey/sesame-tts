import subprocess
from math import gcd

import numpy as np

SUPPORTED_FORMATS = {"mp3", "opus", "wav", "flac"}

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "wav": "audio/wav",
    "flac": "audio/flac",
}

# ffmpeg commands for each format (input: raw s16le PCM via stdin)
FFMPEG_ARGS = {
    "wav": [
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
        "-f", "wav", "pipe:1",
    ],
    "mp3": [
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
        "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1",
    ],
    "opus": [
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", "32k", "-f", "opus", "pipe:1",
    ],
    "flac": [
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
        "-c:a", "flac", "-f", "flac", "pipe:1",
    ],
}


def convert_audio(audio_np: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Convert numpy float32 audio array to the requested format bytes via ffmpeg.

    Pipeline: numpy float32 -> int16 PCM -> ffmpeg -> encoded bytes
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")

    # Convert float32 to int16 PCM
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

    # Update sample rate in ffmpeg args
    args = []
    for arg in FFMPEG_ARGS[fmt]:
        if arg == "24000":
            args.append(str(sample_rate))
        else:
            args.append(arg)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, input=audio_int16.tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {fmt} encoding failed: {result.stderr.decode()}")
    return result.stdout


# --- ElevenLabs format support ---

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
