"""Record audio from a specific microphone device into an in-memory buffer using sounddevice."""

import asyncio
import logging
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        max_duration_s: int = 30,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.max_duration_s = max_duration_s
        self._buffer: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._stop_event = asyncio.Event()

    def resolve_device(self) -> Optional[int]:
        """Find device index by substring match on device name. Returns None for default."""
        if self.device is None:
            return None
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if (
                self.device.lower() in dev["name"].lower()
                and dev["max_input_channels"] > 0
            ):
                logger.info(
                    "Resolved mic device: '%s' -> index %d ('%s')",
                    self.device,
                    i,
                    dev["name"],
                )
                return i
        logger.warning("Mic device '%s' not found, using system default", self.device)
        return None

    @staticmethod
    def list_devices() -> list[dict]:
        """List all audio devices with their capabilities."""
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            result.append(
                {
                    "index": i,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                    "max_output_channels": dev["max_output_channels"],
                    "default_samplerate": dev["default_samplerate"],
                }
            )
        return result

    async def start(self) -> None:
        """Open the mic stream and begin buffering audio."""
        self._buffer.clear()
        self._stop_event.clear()
        device_idx = self.resolve_device()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio capture status: %s", status)
            self._buffer.append(indata.copy())

        self._stream = sd.InputStream(
            device=device_idx,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=callback,
            blocksize=int(self.sample_rate * 0.03),  # 30ms frames for VAD compat
        )
        self._stream.start()

    async def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a numpy array."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._buffer:
            return np.array([], dtype=self.dtype)
        return np.concatenate(self._buffer, axis=0)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently captured frame (for VAD processing)."""
        if self._buffer:
            return self._buffer[-1]
        return None
