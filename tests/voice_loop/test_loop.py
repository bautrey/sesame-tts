"""Integration tests for VoiceLoop orchestrator with all I/O mocked."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_loop.config import VoiceLoopSettings
from voice_loop.loop import SentenceBuffer, VoiceLoop
from voice_loop.state import VoiceLoopState


# ---------------------------------------------------------------------------
# SentenceBuffer unit tests
# ---------------------------------------------------------------------------


class TestSentenceBuffer:
    """Verify SentenceBuffer splits text on sentence boundaries."""

    def test_no_boundary(self):
        buf = SentenceBuffer()
        assert buf.add("Hello world") == []

    def test_single_sentence(self):
        buf = SentenceBuffer()
        result = buf.add("Hello world. ")
        assert result == ["Hello world."]

    def test_multiple_sentences(self):
        buf = SentenceBuffer()
        result = buf.add("First sentence. Second sentence! Third? ")
        assert result == ["First sentence.", "Second sentence!", "Third?"]

    def test_accumulation_across_chunks(self):
        buf = SentenceBuffer()
        assert buf.add("Hello ") == []
        assert buf.add("world") == []
        result = buf.add(". Next sentence. ")
        assert result == ["Hello world.", "Next sentence."]

    def test_flush_returns_remaining(self):
        buf = SentenceBuffer()
        buf.add("Hello world")
        assert buf.flush() == "Hello world"

    def test_flush_empty_returns_none(self):
        buf = SentenceBuffer()
        assert buf.flush() is None

    def test_flush_whitespace_only_returns_none(self):
        buf = SentenceBuffer()
        buf.add("   ")
        assert buf.flush() is None

    def test_question_mark_boundary(self):
        buf = SentenceBuffer()
        result = buf.add("How are you? I am fine. ")
        assert result == ["How are you?", "I am fine."]

    def test_exclamation_boundary(self):
        buf = SentenceBuffer()
        result = buf.add("Wow! That is great. ")
        assert result == ["Wow!", "That is great."]


# ---------------------------------------------------------------------------
# Helpers for building a mocked VoiceLoop
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> VoiceLoopSettings:
    """Create settings with sensible test defaults."""
    defaults = {
        "mic_device": None,
        "speaker_device": None,
        "sample_rate": 16000,
        "vad_silence_threshold_ms": 1000,
        "vad_aggressiveness": 3,
        "max_capture_duration_s": 30,
        "whisper_model": "test-model",
        "gateway_url": "ws://localhost:18789",
        "gateway_token": "test-token",
        "gateway_timeout_s": 10,
        "gateway_reconnect_max_s": 30,
        "tts_url": "http://localhost:8880",
        "tts_voice": "test-voice",
        "tts_format": "pcm_24000",
        "interrupt_energy_threshold": 0.02,
        "interrupt_consecutive_frames": 3,
        "log_level": "warning",
        "log_file": None,
        "run_as_menu_bar": False,
    }
    defaults.update(overrides)
    return VoiceLoopSettings(**defaults)


def _build_loop(settings=None) -> VoiceLoop:
    """Build a VoiceLoop with all heavy dependencies mocked out."""
    if settings is None:
        settings = _make_settings()

    with (
        patch("voice_loop.loop.AudioCapture"),
        patch("voice_loop.loop.VAD"),
        patch("voice_loop.loop.Transcriber"),
        patch("voice_loop.loop.GatewayClient"),
        patch("voice_loop.loop.TTSClient"),
        patch("voice_loop.loop.AudioPlayer"),
        patch("voice_loop.loop.InterruptMonitor"),
    ):
        loop = VoiceLoop(settings)

    return loop


async def _drive_loop_through(vl: VoiceLoop, expected_final_state: VoiceLoopState, timeout: float = 5.0):
    """Run the voice loop until it reaches the expected final state, then stop it."""
    transitions = []

    def track(old, new):
        transitions.append((old, new))
        if new == expected_final_state:
            vl._shutdown_event.set()

    vl.state.on_transition(track)

    # Run the loop with a timeout
    try:
        await asyncio.wait_for(vl._run_loop(), timeout=timeout)
    except asyncio.TimeoutError:
        vl._shutdown_event.set()
        raise AssertionError(
            f"Loop did not reach {expected_final_state} within {timeout}s. "
            f"Transitions: {transitions}, final state: {vl.state.state}"
        )

    return transitions


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Full IDLE -> LISTENING -> TRANSCRIBING -> PROCESSING -> SPEAKING -> IDLE cycle."""

    async def test_happy_path(self):
        vl = _build_loop()

        # -- Mock AudioCapture --
        fake_audio = np.zeros(16000, dtype=np.int16)  # 1s of audio
        vl.capture.start = AsyncMock()
        vl.capture.stop = AsyncMock(return_value=fake_audio)
        vl.capture.get_latest_frame = MagicMock(
            return_value=np.zeros(480, dtype=np.int16)
        )

        # -- Mock VAD: first call returns False (speech), second returns True (end of speech) --
        call_count = 0

        def vad_process(frame):
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # End of speech on second frame

        vl.vad.reset = MagicMock()
        vl.vad.process_frame = MagicMock(side_effect=vad_process)
        vl.vad.has_speech = True

        # -- Mock Transcriber --
        vl.transcriber.transcribe = AsyncMock(return_value="Hello there")

        # -- Mock Gateway: yield one complete sentence --
        async def fake_send_message(text):
            yield "Hello back to you. "

        vl.gateway.send_message = MagicMock(side_effect=fake_send_message)

        # -- Mock TTS: yield one audio chunk --
        async def fake_stream_speech(text):
            yield b"\x00" * 4096

        vl.tts.stream_speech = MagicMock(side_effect=fake_stream_speech)

        # -- Mock AudioPlayer: play completes immediately --
        vl.player.play_stream = AsyncMock()
        vl.player.stop = MagicMock()

        # -- Mock InterruptMonitor: no interrupt --
        vl.interrupt.start = AsyncMock()
        vl.interrupt.stop = AsyncMock()
        vl.interrupt.interrupt_event = asyncio.Event()  # Never set

        # -- Mock sounds --
        with patch("voice_loop.loop.play_listening_chime"), patch(
            "voice_loop.loop.play_error_chime"
        ):
            # Trigger PTT after a short delay
            async def trigger():
                await asyncio.sleep(0.05)
                vl.trigger_ptt()

            asyncio.create_task(trigger())

            transitions = await _drive_loop_through(vl, VoiceLoopState.IDLE)

        # Verify we went through all states
        states_visited = [VoiceLoopState.IDLE]  # Starting state
        for _, new in transitions:
            states_visited.append(new)

        assert VoiceLoopState.LISTENING in states_visited
        assert VoiceLoopState.TRANSCRIBING in states_visited
        assert VoiceLoopState.PROCESSING in states_visited
        assert VoiceLoopState.SPEAKING in states_visited
        # Final state should be IDLE (after SPEAKING)
        assert states_visited[-1] == VoiceLoopState.IDLE


class TestInterruptDuringPlayback:
    """SPEAKING -> LISTENING on interrupt."""

    async def test_interrupt_during_playback(self):
        vl = _build_loop()

        # Pre-populate TTS audio queue and advance state to SPEAKING
        vl._tts_audio_queue = asyncio.Queue()
        await vl._tts_audio_queue.put(b"\x00" * 4096)
        await vl._tts_audio_queue.put(None)

        # Advance state machine to SPEAKING
        await vl.state.transition(VoiceLoopState.LISTENING)
        await vl.state.transition(VoiceLoopState.TRANSCRIBING)
        await vl.state.transition(VoiceLoopState.PROCESSING)
        await vl.state.transition(VoiceLoopState.SPEAKING)

        # Mock player: play_stream blocks until cancelled
        async def slow_playback(audio_chunks):
            async for _ in audio_chunks:
                pass
            await asyncio.sleep(10)  # Block long enough for interrupt to fire

        vl.player.play_stream = AsyncMock(side_effect=slow_playback)
        vl.player.stop = MagicMock()

        # Mock interrupt: fires quickly
        interrupt_event = asyncio.Event()
        vl.interrupt.start = AsyncMock()
        vl.interrupt.stop = AsyncMock()
        vl.interrupt.interrupt_event = interrupt_event

        async def fire_interrupt():
            await asyncio.sleep(0.05)
            interrupt_event.set()

        asyncio.create_task(fire_interrupt())

        # Run the speaking handler directly
        await vl._handle_speaking()

        assert vl.state.state == VoiceLoopState.LISTENING
        vl.player.stop.assert_called_once()


class TestEmptyTranscript:
    """TRANSCRIBING -> IDLE when transcript is empty."""

    async def test_empty_transcript(self):
        vl = _build_loop()

        # Set up state at TRANSCRIBING
        await vl.state.transition(VoiceLoopState.LISTENING)
        await vl.state.transition(VoiceLoopState.TRANSCRIBING)

        vl._audio_buffer = np.zeros(16000, dtype=np.int16)
        vl.transcriber.transcribe = AsyncMock(return_value=None)

        await vl._handle_transcribing()

        assert vl.state.state == VoiceLoopState.IDLE


class TestGatewayTimeout:
    """PROCESSING -> ERROR on gateway timeout."""

    async def test_gateway_timeout(self):
        vl = _build_loop()

        # Set up state at PROCESSING
        await vl.state.transition(VoiceLoopState.LISTENING)
        await vl.state.transition(VoiceLoopState.TRANSCRIBING)
        await vl.state.transition(VoiceLoopState.PROCESSING)

        vl._transcript = "Hello"

        # Mock gateway to raise TimeoutError
        async def timeout_send(text):
            raise TimeoutError("Gateway response timeout (10s)")
            # Make it an async generator that raises
            yield  # pragma: no cover -- unreachable, needed for generator syntax

        vl.gateway.send_message = MagicMock(side_effect=timeout_send)

        with patch("voice_loop.loop.play_error_chime"):
            await vl._handle_processing()

        assert vl.state.state == VoiceLoopState.ERROR


class TestTTSUnavailable:
    """PROCESSING -> ERROR when TTS server is unreachable."""

    async def test_tts_unavailable(self):
        vl = _build_loop()

        # Set up state at PROCESSING
        await vl.state.transition(VoiceLoopState.LISTENING)
        await vl.state.transition(VoiceLoopState.TRANSCRIBING)
        await vl.state.transition(VoiceLoopState.PROCESSING)

        vl._transcript = "Hello"

        # Mock gateway: yields text that triggers TTS
        async def fake_send_message(text):
            yield "Response text. "

        vl.gateway.send_message = MagicMock(side_effect=fake_send_message)

        # Mock TTS to raise RuntimeError
        async def failing_tts(text):
            raise RuntimeError("TTS connection refused")
            yield  # pragma: no cover -- unreachable, needed for generator syntax

        vl.tts.stream_speech = MagicMock(side_effect=failing_tts)

        with patch("voice_loop.loop.play_error_chime"):
            await vl._handle_processing()

        assert vl.state.state == VoiceLoopState.ERROR


class TestErrorRecovery:
    """ERROR -> IDLE after error chime."""

    async def test_error_returns_to_idle(self):
        vl = _build_loop()

        # Set up state at ERROR
        await vl.state.transition(VoiceLoopState.LISTENING)
        await vl.state.transition(VoiceLoopState.TRANSCRIBING)
        await vl.state.transition(VoiceLoopState.PROCESSING)
        await vl.state.transition(VoiceLoopState.ERROR)

        with patch("voice_loop.loop.play_error_chime") as mock_chime:
            await vl._handle_error()

        assert vl.state.state == VoiceLoopState.IDLE
        mock_chime.assert_called_once()
