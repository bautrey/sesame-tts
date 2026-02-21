"""Audio playback via sounddevice OutputStream with queue-based streaming."""

import asyncio
import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Sentinel value to signal end of stream
_END_OF_STREAM = None


class AudioPlayer:
    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 24000,
        channels: int = 1,
        dtype: str = "int16",
        buffer_size_ms: int = 100,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.buffer_size_ms = buffer_size_ms
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.OutputStream] = None
        self._playing = False
        self._stop_event = asyncio.Event()

    def resolve_device(self) -> Optional[int]:
        """Find output device index by substring match."""
        if self.device is None:
            return None
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if self.device.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
                logger.info(
                    "Resolved speaker device: '%s' -> index %d ('%s')",
                    self.device,
                    i,
                    dev["name"],
                )
                return i
        logger.warning("Speaker device '%s' not found, using system default", self.device)
        return None

    async def play_stream(self, audio_chunks) -> None:
        """Play an async iterator of PCM audio chunks.

        Args:
            audio_chunks: AsyncIterator yielding bytes (int16, mono, sample_rate Hz)
        """
        self._stop_event.clear()
        self._playing = True
        device_idx = self.resolve_device()

        # Thread-safe queue for feeding the sounddevice callback
        self._audio_queue = queue.Queue()

        def callback(outdata, frames, time_info, status):
            if status:
                logger.warning("Playback status: %s", status)
            try:
                data = self._audio_queue.get_nowait()
                if data is _END_OF_STREAM:
                    outdata[:] = 0
                    raise sd.CallbackStop
                # Pad if chunk is smaller than requested frames
                if len(data) < len(outdata):
                    outdata[: len(data)] = data
                    outdata[len(data) :] = 0
                else:
                    outdata[:] = data[: len(outdata)]
            except queue.Empty:
                outdata[:] = 0  # Underrun -- output silence

        blocksize = int(self.sample_rate * self.buffer_size_ms / 1000)

        self._stream = sd.OutputStream(
            device=device_idx,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=callback,
            blocksize=blocksize,
        )
        self._stream.start()

        try:
            async for chunk in audio_chunks:
                if self._stop_event.is_set():
                    break
                # Convert raw bytes to numpy array for sounddevice
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                # Feed in blocksize-aligned pieces
                offset = 0
                while offset < len(audio_data):
                    end = min(offset + blocksize, len(audio_data))
                    frame = audio_data[offset:end]
                    if len(frame) < blocksize:
                        frame = np.pad(frame, (0, blocksize - len(frame)))
                    self._audio_queue.put(frame.reshape(-1, 1))
                    offset = end

            # Signal end of stream
            self._audio_queue.put(_END_OF_STREAM)

            # Wait for playback to finish (drain the queue)
            if not self._stop_event.is_set():
                while self._stream and self._stream.active:
                    await asyncio.sleep(0.05)

        finally:
            self.stop()

    def stop(self) -> None:
        """Immediately stop playback."""
        self._stop_event.set()
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # Drain the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_playing(self) -> bool:
        return self._playing
