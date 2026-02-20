import subprocess

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
