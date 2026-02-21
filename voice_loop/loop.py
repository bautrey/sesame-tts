"""Main async orchestrator for the voice conversation loop."""

import asyncio
import logging
import re
import time
from typing import Optional

from .audio_capture import AudioCapture
from .config import VoiceLoopSettings
from .gateway_client import GatewayClient
from .interrupt import InterruptMonitor
from .playback import AudioPlayer
from .sounds import play_error_chime, play_listening_chime
from .state import StateMachine, VoiceLoopState
from .transcriber import Transcriber
from .tts_client import TTSClient
from .vad import VAD

logger = logging.getLogger(__name__)

# Sentence boundary pattern: split on . ! ? followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceBuffer:
    """Accumulates text chunks from gateway deltas and yields complete sentences.

    Detects sentence boundaries at `. `, `! `, `? ` (punctuation followed by
    whitespace). Holds incomplete text until a boundary is found or flush()
    is called to yield any remaining text.
    """

    def __init__(self):
        self._buffer: str = ""

    def add(self, text: str) -> list[str]:
        """Add a text chunk. Returns a list of complete sentences (may be empty)."""
        self._buffer += text
        sentences = []
        # Split on sentence boundaries
        parts = _SENTENCE_END.split(self._buffer)
        if len(parts) > 1:
            # All parts except the last are complete sentences
            sentences = parts[:-1]
            self._buffer = parts[-1]
        return sentences

    def flush(self) -> Optional[str]:
        """Return any remaining text in the buffer (for the final chunk)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None


class VoiceLoop:
    """Coordinates all voice loop modules through the state machine."""

    def __init__(self, settings: VoiceLoopSettings):
        self.settings = settings
        self.state = StateMachine()
        self.capture = AudioCapture(
            device=settings.mic_device,
            sample_rate=settings.sample_rate,
            max_duration_s=settings.max_capture_duration_s,
        )
        self.vad = VAD(
            sample_rate=settings.sample_rate,
            aggressiveness=settings.vad_aggressiveness,
            silence_threshold_ms=settings.vad_silence_threshold_ms,
        )
        self.transcriber = Transcriber(model_name=settings.whisper_model)
        self.gateway = GatewayClient(
            url=settings.gateway_url,
            token=settings.gateway_token,
            timeout_s=settings.gateway_timeout_s,
        )
        self.tts = TTSClient(
            base_url=settings.tts_url,
            voice=settings.tts_voice,
            output_format=settings.tts_format,
        )
        self.player = AudioPlayer(
            device=settings.speaker_device,
            sample_rate=24000,  # PCM 24000 from TTS
        )
        self.interrupt = InterruptMonitor(
            device=settings.mic_device,
            energy_threshold=settings.interrupt_energy_threshold,
            consecutive_frames=settings.interrupt_consecutive_frames,
        )
        self._ptt_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._audio_buffer = None
        self._transcript = None
        self._tts_audio_queue: asyncio.Queue = asyncio.Queue()

    def trigger_ptt(self) -> None:
        """Called from menu bar or hotkey handler to trigger push-to-talk."""
        self._ptt_event.set()

    async def start(self) -> None:
        """Initialize all components and start the main loop."""
        logger.info("Starting Voice Loop...")
        logger.info(
            "Config: mic=%s, gateway=%s, tts=%s",
            self.settings.mic_device or "system default",
            self.settings.gateway_url,
            self.settings.tts_url,
        )

        # Log available devices
        for dev in AudioCapture.list_devices():
            if dev["max_input_channels"] > 0:
                logger.info("  Input device: [%d] %s", dev["index"], dev["name"])
            if dev["max_output_channels"] > 0:
                logger.info("  Output device: [%d] %s", dev["index"], dev["name"])

        # Load models
        await self.transcriber.load_model()

        # Connect to services
        await self.tts.start()
        await self.gateway.connect()

        # Main loop
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main state machine loop."""
        while not self._shutdown_event.is_set():
            try:
                current = self.state.state

                if current == VoiceLoopState.IDLE:
                    await self._handle_idle()
                elif current == VoiceLoopState.LISTENING:
                    await self._handle_listening()
                elif current == VoiceLoopState.TRANSCRIBING:
                    await self._handle_transcribing()
                elif current == VoiceLoopState.PROCESSING:
                    await self._handle_processing()
                elif current == VoiceLoopState.SPEAKING:
                    await self._handle_speaking()
                elif current == VoiceLoopState.ERROR:
                    await self._handle_error()

            except Exception as e:
                logger.exception("Unhandled error in voice loop: %s", e)
                if self.state.state != VoiceLoopState.IDLE:
                    try:
                        await self.state.transition(VoiceLoopState.ERROR)
                    except ValueError:
                        # If we can't transition to ERROR (e.g., already in ERROR),
                        # force back to IDLE via ERROR path
                        pass

    async def _handle_idle(self) -> None:
        """Wait for PTT activation."""
        self._ptt_event.clear()
        await self._ptt_event.wait()
        await self.state.transition(VoiceLoopState.LISTENING)

    async def _handle_listening(self) -> None:
        """Capture audio until VAD detects silence."""
        t_start = time.monotonic()
        play_listening_chime()
        self.vad.reset()
        await self.capture.start()

        try:
            while True:
                await asyncio.sleep(0.03)  # Match 30ms frame duration
                frame = self.capture.get_latest_frame()
                if frame is not None and self.vad.process_frame(frame.flatten()):
                    break
                # Check max capture duration
                if time.monotonic() - t_start > self.settings.max_capture_duration_s:
                    logger.warning("Max capture duration reached")
                    break
        finally:
            self._audio_buffer = await self.capture.stop()

        elapsed = time.monotonic() - t_start
        logger.info("Captured %.1fs of audio (%d samples)", elapsed, len(self._audio_buffer))

        if len(self._audio_buffer) == 0 or not self.vad.has_speech:
            await self.state.transition(VoiceLoopState.IDLE)
        else:
            await self.state.transition(VoiceLoopState.TRANSCRIBING)

    async def _handle_transcribing(self) -> None:
        """Transcribe captured audio."""
        t_start = time.monotonic()
        self._transcript = await self.transcriber.transcribe(
            self._audio_buffer, self.settings.sample_rate
        )
        elapsed = time.monotonic() - t_start
        logger.info("Transcription took %.2fs: %r", elapsed, self._transcript)

        if self._transcript:
            await self.state.transition(VoiceLoopState.PROCESSING)
        else:
            await self.state.transition(VoiceLoopState.IDLE)

    async def _handle_processing(self) -> None:
        """Stream gateway deltas, detect sentence boundaries, and send each
        complete sentence to TTS immediately for lowest perceived latency.

        Uses SentenceBuffer to accumulate gateway text deltas and yield
        complete sentences at `. ! ?` boundaries. Each sentence is sent to
        TTS via tts_client.stream_speech() and its audio chunks are queued
        into self._tts_audio_queue for the SPEAKING state handler.
        """
        t_start = time.monotonic()
        self._tts_audio_queue = asyncio.Queue()
        sentence_buf = SentenceBuffer()

        try:
            sentence_count = 0
            async for delta in self.gateway.send_message(self._transcript):
                sentences = sentence_buf.add(delta)
                for sentence in sentences:
                    sentence_count += 1
                    logger.debug(
                        "Sentence %d ready for TTS: %r", sentence_count, sentence[:80]
                    )
                    # Stream TTS audio for this sentence into the queue
                    async for audio_chunk in self.tts.stream_speech(sentence):
                        await self._tts_audio_queue.put(audio_chunk)

            # Flush any remaining text (final partial sentence)
            remaining = sentence_buf.flush()
            if remaining:
                sentence_count += 1
                logger.debug(
                    "Final sentence %d for TTS: %r", sentence_count, remaining[:80]
                )
                async for audio_chunk in self.tts.stream_speech(remaining):
                    await self._tts_audio_queue.put(audio_chunk)

            # Signal end of TTS audio stream
            await self._tts_audio_queue.put(None)

            elapsed = time.monotonic() - t_start
            logger.info(
                "Gateway + TTS streaming completed in %.2fs (%d sentences)",
                elapsed,
                sentence_count,
            )

            if sentence_count > 0:
                await self.state.transition(VoiceLoopState.SPEAKING)
            else:
                logger.warning("Empty gateway response")
                await self.state.transition(VoiceLoopState.ERROR)

        except (TimeoutError, ConnectionError, RuntimeError) as e:
            logger.error("Gateway/TTS error: %s", e)
            await self._tts_audio_queue.put(None)  # Unblock any waiting consumer
            await self.state.transition(VoiceLoopState.ERROR)

    async def _tts_audio_generator(self):
        """Async generator that yields audio chunks from the TTS audio queue.

        Used by AudioPlayer.play_stream() to consume audio chunks produced
        by _handle_processing as sentences are streamed to TTS.
        """
        while True:
            chunk = await self._tts_audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def _handle_speaking(self) -> None:
        """Play queued TTS audio chunks with interrupt monitoring.

        The TTS audio queue is populated by _handle_processing, which streams
        sentences to TTS as they arrive from the gateway. This handler drains
        the queue through the AudioPlayer while monitoring for user interrupts.
        """
        loop = asyncio.get_event_loop()
        t_start = time.monotonic()

        try:
            # Start interrupt monitor
            await self.interrupt.start(loop)

            # Play audio from the TTS queue
            audio_stream = self._tts_audio_generator()

            # Race: playback vs interrupt
            playback_task = asyncio.create_task(self.player.play_stream(audio_stream))
            interrupt_task = asyncio.create_task(self.interrupt.interrupt_event.wait())

            done, pending = await asyncio.wait(
                {playback_task, interrupt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if interrupt_task in done:
                # User interrupted -- stop playback and re-enter listening
                self.player.stop()
                logger.info(
                    "Playback interrupted after %.2fs", time.monotonic() - t_start
                )
                await self.state.transition(VoiceLoopState.LISTENING)
            else:
                # Playback completed normally
                logger.info("Playback completed in %.2fs", time.monotonic() - t_start)
                await self.state.transition(VoiceLoopState.IDLE)

        except Exception as e:
            logger.error("Speaking error: %s", e)
            self.player.stop()
            await self.state.transition(VoiceLoopState.ERROR)
        finally:
            await self.interrupt.stop()

    async def _handle_error(self) -> None:
        """Play error chime and return to idle."""
        play_error_chime()
        await asyncio.sleep(1.0)  # Let chime play
        await self.state.transition(VoiceLoopState.IDLE)

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self._shutdown_event.set()
        self._ptt_event.set()  # Unblock _handle_idle if waiting
        self.player.stop()
        await self.interrupt.stop()
        await self.tts.close()
        await self.gateway.close()
        logger.info("Voice Loop shut down.")
