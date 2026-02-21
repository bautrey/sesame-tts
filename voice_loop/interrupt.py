"""Interrupt detection during playback via microphone RMS energy monitoring."""

import asyncio
import logging
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class InterruptMonitor:
    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        energy_threshold: float = 0.02,
        consecutive_frames: int = 3,
        frame_duration_ms: int = 30,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_duration_ms = frame_duration_ms
        self._stream: Optional[sd.InputStream] = None
        self._interrupt_event = asyncio.Event()
        self._above_threshold_count = 0
        self._running = False
        self._captured_frames: list[np.ndarray] = []

    @property
    def interrupt_event(self) -> asyncio.Event:
        return self._interrupt_event

    @property
    def captured_audio(self) -> Optional[np.ndarray]:
        """Return any audio captured during interrupt monitoring (for seamless transition)."""
        if self._captured_frames:
            return np.concatenate(self._captured_frames, axis=0)
        return None

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Begin monitoring the mic for speech."""
        self._interrupt_event.clear()
        self._above_threshold_count = 0
        self._captured_frames.clear()
        self._running = True

        # Resolve device (reuse same logic as AudioCapture)
        device_idx = None
        if self.device:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if self.device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                    device_idx = i
                    break

        def callback(indata, frames, time_info, status):
            if not self._running:
                return
            # Calculate RMS energy
            audio_float = indata.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float**2))

            # Store frames for seamless transition to LISTENING
            self._captured_frames.append(indata.copy())
            # Keep only last 1 second of audio
            max_frames = int(1000 / self.frame_duration_ms)
            if len(self._captured_frames) > max_frames:
                self._captured_frames.pop(0)

            if rms > self.energy_threshold:
                self._above_threshold_count += 1
                if self._above_threshold_count >= self.consecutive_frames:
                    logger.info(
                        "Interrupt detected: RMS=%.4f (threshold=%.4f)",
                        rms,
                        self.energy_threshold,
                    )
                    loop.call_soon_threadsafe(self._interrupt_event.set)
            else:
                self._above_threshold_count = 0

        blocksize = int(self.sample_rate * self.frame_duration_ms / 1000)
        self._stream = sd.InputStream(
            device=device_idx,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
            blocksize=blocksize,
        )
        self._stream.start()

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
