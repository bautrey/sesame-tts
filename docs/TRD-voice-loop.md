# TRD: Voice Loop -- Standalone Voice Conversation Agent

**Author:** Burke (with Claude Code)
**Created:** 2026-02-21
**Status:** Refined
**Version:** 1.1
**PRD Reference:** [docs/PRD-voice-loop.md](./PRD-voice-loop.md) v1.1

---

## 1. Overview

### What We Are Building

A standalone Python voice conversation loop that runs as a macOS menu bar application, orchestrating the full pipeline from push-to-talk activation through speech capture, local transcription, OpenClaw gateway communication, Sesame TTS generation, and audio playback -- all running locally on Apple Silicon.

### Why

OpenClaw's built-in Talk Mode is broken (hardcoded ElevenLabs URL, system voice fallback, flaky PTT, double audio, device conflicts). Rather than waiting for upstream fixes, this voice loop owns the entire pipeline end-to-end, bypassing every Talk Mode bug.

### What We Are NOT Changing

- The existing Sesame TTS server (`server.py`, `tts_engine.py`, `streaming.py`) -- consumed as-is via its ElevenLabs streaming endpoint
- The OpenClaw gateway -- consumed as-is via its WebSocket protocol on port 18789
- The existing `pyproject.toml` project configuration for the TTS server itself

---

## 2. System Architecture

### 2.1 Component Diagram

```
+====================================================================+
|                     Voice Loop Process (voice_loop/)                |
|                                                                     |
|  +-----------------+      +-------------------+                     |
|  | Menu Bar (rumps)|      | Global Hotkey     |                     |
|  | PTT click       |      | (Quartz Events)   |                     |
|  +--------+--------+      +--------+----------+                     |
|           |                         |                               |
|           +-------+     +-----------+                               |
|                   v     v                                           |
|           +-------+-----+--------+                                  |
|           | State Manager        |  voice_loop/state.py             |
|           | (IDLE -> LISTENING   |                                  |
|           |  -> TRANSCRIBING     |                                  |
|           |  -> PROCESSING       |                                  |
|           |  -> SPEAKING -> IDLE)|                                  |
|           +-------+--------------+                                  |
|                   |                                                 |
|  +----------------+------------------------------------------+      |
|  |                                                           |      |
|  v                                                           |      |
|  +-------------------+     +--------------------+            |      |
|  | Audio Capture     |     | VAD                |            |      |
|  | (sounddevice)     +---->| (webrtcvad)        |            |      |
|  | voice_loop/       |     | voice_loop/vad.py  |            |      |
|  | audio_capture.py  |     +--------+-----------+            |      |
|  +-------------------+              |                        |      |
|                            silence detected                  |      |
|                                     v                        |      |
|                          +----------+-----------+            |      |
|                          | Transcriber          |            |      |
|                          | (MLX Whisper)        |            |      |
|                          | voice_loop/          |            |      |
|                          | transcriber.py       |            |      |
|                          +----------+-----------+            |      |
|                                     |                        |      |
|                               transcript                     |      |
|                                     v                        |      |
|  +--------------------+  +----------+-----------+            |      |
|  | OpenClaw Gateway   |  | Gateway Client       |            |      |
|  | ws://localhost:     |<>| (websockets)         |            |      |
|  | 18789              |  | voice_loop/           |            |      |
|  +--------------------+  | gateway_client.py     |            |      |
|                          +----------+-----------+            |      |
|                                     |                        |      |
|                            response text                     |      |
|                                     v                        |      |
|  +--------------------+  +----------+-----------+            |      |
|  | Sesame TTS Server  |  | TTS Client           |            |      |
|  | http://localhost:   |<>| (httpx streaming)    |            |      |
|  | 8880               |  | voice_loop/           |            |      |
|  +--------------------+  | tts_client.py         |            |      |
|                          +----------+-----------+            |      |
|                                     |                        |      |
|                          audio chunks (PCM)                  |      |
|                                     v                        |      |
|                          +----------+-----------+            |      |
|                          | Audio Playback       |<--+        |      |
|                          | (sounddevice         |   |        |      |
|                          |  OutputStream)       |   |        |      |
|                          | voice_loop/          |   |        |      |
|                          | playback.py          |   |        |      |
|                          +----------+-----------+   |        |      |
|                                     |               |        |      |
|                                     v               |        |      |
|                          +----------+-----------+   |        |      |
|                          | Interrupt Monitor    +---+        |      |
|                          | (mic during playback)|            |      |
|                          | voice_loop/          |            |      |
|                          | interrupt.py         +------------+      |
|                          +----------------------+ (re-enter         |
|                                                    LISTENING)       |
|  +-----------------------+    +-------------------+                 |
|  | Sounds               |    | Config             |                 |
|  | voice_loop/sounds.py |    | voice_loop/         |                 |
|  +-----------------------+    | config.py          |                 |
|                               +-------------------+                 |
+====================================================================+
```

### 2.2 State Machine

```
States:
  IDLE          -- Waiting for PTT activation. Minimal CPU. Menu bar shows idle icon.
  LISTENING     -- Capturing mic audio. VAD monitoring for silence. Menu bar shows listening icon.
  TRANSCRIBING  -- Running MLX Whisper on captured audio buffer. Menu bar shows processing icon.
  PROCESSING    -- Sent transcript to gateway, waiting for response. Menu bar shows processing icon.
  SPEAKING      -- Receiving TTS audio and playing back. Interrupt monitor active. Menu bar shows speaking icon.
  ERROR         -- Transient error state. Plays error chime, then returns to IDLE.

Transitions:
  IDLE         -> LISTENING      : PTT activated (menu bar click or hotkey)
  LISTENING    -> TRANSCRIBING   : VAD detects silence (end of speech)
  LISTENING    -> IDLE           : Capture timeout (30s) or empty capture
  TRANSCRIBING -> PROCESSING     : Transcript ready (non-empty)
  TRANSCRIBING -> IDLE           : Empty/noise transcript discarded
  PROCESSING   -> SPEAKING       : First TTS audio chunk received
  PROCESSING   -> ERROR          : Gateway timeout (10s) or TTS unavailable
  SPEAKING     -> IDLE           : Playback complete
  SPEAKING     -> LISTENING      : Interrupt detected (user speaks during playback)
  ERROR        -> IDLE           : After error chime plays (automatic, ~1s)
```

### 2.3 Data Flow with Latency Budget

Target: sub-4-second from PTT activation to first audio byte, assuming a 2-second user utterance.

```
Stage                          Duration    Cumulative   Module
------------------------------ ----------- ------------ -------------------------
PTT activation + capture start   50ms        0.05s      menu_bar.py / loop.py
User speech                     2000ms       2.05s      audio_capture.py
VAD silence detection           1000ms       3.05s      vad.py (tuned to 1.0s)
MLX Whisper transcription        500ms       3.55s      transcriber.py
Gateway send + first delta      1500ms       5.05s      gateway_client.py
Sentence buffer fills            200ms       5.25s      loop.py (SentenceBuffer)
TTS first chunk received         800ms       6.05s      tts_client.py
Audio playback begins              5ms       6.05s      playback.py
                                             ------
From PTT activation:                         ~6.05s
From end of speech (useful):                 ~3.0s
```

Notes:
- The 4-second target is measured from **end of user speech** to first audio byte, which is ~3.0s above.
- VAD silence threshold is set to 1.0s (not 1.5s) for Phase 1 to maximize responsiveness.
- Gateway latency is LLM-dependent; 1.5s assumes streaming first delta arrives quickly.
- Sentence-streaming TTS: the first complete sentence is sent to TTS as soon as a sentence boundary (`. ! ?`) is detected in gateway deltas. This means the TTS request fires **during** gateway streaming, not after it completes. Perceived latency is significantly lower because TTS generation overlaps with gateway generation of subsequent sentences.
- Subsequent sentences are queued to TTS as they complete, with audio chunks fed into the player sequentially.

### 2.4 Threading Model

```
Main Thread (macOS requirement for rumps):
  - rumps.App.run() -- menu bar event loop
  - Spawns asyncio event loop in a daemon thread

Asyncio Thread (daemon):
  - voice_loop/loop.py -- main orchestrator coroutine
  - gateway_client.py -- WebSocket send/receive
  - tts_client.py -- httpx streaming
  - interrupt.py -- mic monitoring during playback

Sounddevice Callback Threads (managed by PortAudio):
  - audio_capture.py -- mic input callback fills buffer
  - playback.py -- output callback drains buffer
  - interrupt.py -- mic monitoring callback during SPEAKING state

MLX Whisper Thread (run_in_executor):
  - transcriber.py -- CPU/GPU-bound transcription runs in executor to avoid blocking asyncio
```

---

## 3. Module Design

All modules live in `voice_loop/` package within the existing `sesame-tts` project root.

```
sesame-tts/
  voice_loop/
    __init__.py
    __main__.py        # Entry point: python -m voice_loop
    config.py          # VoiceLoopSettings (pydantic-settings + YAML)
    state.py           # VoiceLoopState enum + state machine
    audio_capture.py   # Mic capture via sounddevice
    vad.py             # Voice Activity Detection
    transcriber.py     # MLX Whisper wrapper
    gateway_client.py  # OpenClaw gateway WebSocket client
    tts_client.py      # Sesame TTS HTTP streaming client
    playback.py        # Audio output via sounddevice OutputStream
    interrupt.py       # Interrupt detection during playback
    menu_bar.py        # rumps menu bar app
    loop.py            # Main async orchestrator
    sounds.py          # Chime/feedback sounds
  tests/
    voice_loop/
      __init__.py
      test_config.py
      test_state.py
      test_vad.py
      test_transcriber.py
      test_gateway_client.py
      test_tts_client.py
      test_playback.py
      test_interrupt.py
      test_loop.py
      conftest.py       # Shared fixtures
```

### 3.1 voice_loop/config.py -- VoiceLoopSettings

**Purpose:** Centralized configuration with YAML file loading, environment variable overrides, and validation.

**Design:**

```python
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import yaml

CONFIG_PATH = Path.home() / ".config" / "sesame-voice" / "config.yaml"

class VoiceLoopSettings(BaseSettings):
    # Audio devices
    mic_device: Optional[str] = None          # None = system default
    speaker_device: Optional[str] = None      # None = system default

    # Push-to-talk
    push_to_talk_hotkey: str = "cmd+shift+space"

    # Speech capture
    sample_rate: int = 16000                  # Whisper-native sample rate
    vad_silence_threshold_ms: int = 1000      # Silence to end capture (tuned down from 1500)
    vad_aggressiveness: int = 3               # webrtcvad 0-3 (3 = most aggressive)
    max_capture_duration_s: int = 30

    # Transcription
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # Gateway
    gateway_url: str = "ws://127.0.0.1:18789"
    gateway_token: str = "gateway-token"       # Auth token for connect handshake
    gateway_timeout_s: int = 10
    gateway_reconnect_max_s: int = 30

    # TTS
    tts_url: str = "http://localhost:8880"
    tts_voice: str = "conversationalB"
    tts_format: str = "pcm_24000"

    # Interrupt
    interrupt_energy_threshold: float = 0.02   # RMS energy threshold for speech detection
    interrupt_consecutive_frames: int = 3      # Consecutive frames above threshold to trigger

    # Logging
    log_level: str = "info"
    log_file: Optional[str] = "~/.cache/sesame-voice/voice-loop.log"

    # Process
    run_as_menu_bar: bool = True

    model_config = {"env_prefix": "VOICE_LOOP_"}

    @classmethod
    def _read_openclaw_token(cls) -> Optional[str]:
        """Read the gateway auth token from ~/.openclaw/openclaw.json.

        Returns the token string if found, or None if the file is missing
        or the token path does not exist in the JSON structure.
        """
        import json
        openclaw_path = Path.home() / ".openclaw" / "openclaw.json"
        if not openclaw_path.exists():
            return None
        try:
            with open(openclaw_path) as f:
                data = json.load(f)
            # Navigate nested path: config.gateway.auth.token (or similar)
            # Try common paths in order of likelihood
            for key_path in [
                ("config", "gateway", "auth", "token"),
                ("gateway", "auth", "token"),
                ("auth", "token"),
                ("token",),
            ]:
                node = data
                for key in key_path:
                    if isinstance(node, dict) and key in node:
                        node = node[key]
                    else:
                        node = None
                        break
                if isinstance(node, str) and node:
                    return node
        except (json.JSONDecodeError, OSError):
            pass
        return None

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> "VoiceLoopSettings":
        """Load settings from YAML file, then apply env var overrides.

        Gateway token resolution order (highest priority wins):
          1. VOICE_LOOP_GATEWAY_TOKEN env var
          2. gateway_token in config YAML
          3. Token auto-read from ~/.openclaw/openclaw.json
          4. Hardcoded default ("gateway-token")
        """
        overrides = {}
        if path.exists():
            with open(path) as f:
                overrides = yaml.safe_load(f) or {}

        # Auto-read gateway token from openclaw.json if not set in YAML or env
        import os
        if "gateway_token" not in overrides and not os.environ.get("VOICE_LOOP_GATEWAY_TOKEN"):
            openclaw_token = cls._read_openclaw_token()
            if openclaw_token:
                overrides["gateway_token"] = openclaw_token

        return cls(**overrides)
```

**Key decisions:**
- Uses `pydantic-settings` (already a project dependency) for validation and env var support.
- YAML loaded as constructor kwargs, so env vars (`VOICE_LOOP_*`) still override.
- `sample_rate` defaults to 16000 (Whisper native) to avoid resampling overhead.
- VAD silence threshold tuned to 1000ms (not 1500ms) per PRD optimization guidance.
- Gateway token auto-reads from `~/.openclaw/openclaw.json` when not explicitly configured. Resolution order: env var > config YAML > openclaw.json > hardcoded default. This avoids requiring users to manually copy the token into the voice loop config.

**PRD Coverage:** CFG-1, CFG-2, CFG-3, CFG-4, CFG-5

---

### 3.2 voice_loop/state.py -- State Machine

**Purpose:** Enum-based state machine with validated transitions and observer callbacks.

**Design:**

```python
import asyncio
import enum
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class VoiceLoopState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

# Valid state transitions
TRANSITIONS: dict[VoiceLoopState, set[VoiceLoopState]] = {
    VoiceLoopState.IDLE:         {VoiceLoopState.LISTENING},
    VoiceLoopState.LISTENING:    {VoiceLoopState.TRANSCRIBING, VoiceLoopState.IDLE},
    VoiceLoopState.TRANSCRIBING: {VoiceLoopState.PROCESSING, VoiceLoopState.IDLE},
    VoiceLoopState.PROCESSING:   {VoiceLoopState.SPEAKING, VoiceLoopState.ERROR},
    VoiceLoopState.SPEAKING:     {VoiceLoopState.IDLE, VoiceLoopState.LISTENING},
    VoiceLoopState.ERROR:        {VoiceLoopState.IDLE},
}

StateCallback = Callable[[VoiceLoopState, VoiceLoopState], None]

class StateMachine:
    def __init__(self):
        self._state = VoiceLoopState.IDLE
        self._callbacks: list[StateCallback] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> VoiceLoopState:
        return self._state

    def on_transition(self, callback: StateCallback) -> None:
        """Register a callback invoked on every state transition."""
        self._callbacks.append(callback)

    async def transition(self, new_state: VoiceLoopState) -> None:
        """Transition to a new state. Raises ValueError for invalid transitions."""
        async with self._lock:
            old = self._state
            if new_state not in TRANSITIONS.get(old, set()):
                raise ValueError(f"Invalid transition: {old.value} -> {new_state.value}")
            self._state = new_state
            logger.info("State: %s -> %s", old.value, new_state.value)
            for cb in self._callbacks:
                cb(old, new_state)
```

**Key decisions:**
- Async lock prevents concurrent transitions (e.g., interrupt racing with playback complete).
- Callbacks are synchronous (for menu bar icon updates on main thread via `rumps.Timer` or queue).
- ERROR state always transitions back to IDLE (automatic recovery).

---

### 3.3 voice_loop/audio_capture.py -- Mic Capture

**Purpose:** Record audio from a specific microphone device into an in-memory buffer using sounddevice.

**Design:**

```python
import asyncio
import logging
import numpy as np
import sounddevice as sd
from typing import Optional

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
        self.device = device          # Device name (substring match) or None for default
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
            if self.device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                logger.info("Resolved mic device: '%s' -> index %d ('%s')", self.device, i, dev["name"])
                return i
        logger.warning("Mic device '%s' not found, using system default", self.device)
        return None

    @staticmethod
    def list_devices() -> list[dict]:
        """List all audio devices with their capabilities."""
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            result.append({
                "index": i,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
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
```

**Key decisions:**
- 30ms blocksize for webrtcvad compatibility (webrtcvad requires 10/20/30ms frames).
- int16 dtype for direct webrtcvad consumption (avoids float-to-int conversion).
- Substring match for device name (handles "MacBook Pro Microphone" vs exact name differences).
- In-memory buffer only (no temp files), per PRD SC-6.

**PRD Coverage:** SC-1, SC-2, SC-3, SC-5, SC-6, SC-7, SC-8

---

### 3.4 voice_loop/vad.py -- Voice Activity Detection

**Purpose:** Detect speech start and silence end-of-speech in the audio stream.

**Design:**

```python
import logging
import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)

class VAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 3,
        silence_threshold_ms: int = 1000,
        frame_duration_ms: int = 30,
    ):
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.silence_threshold_ms = silence_threshold_ms
        self.frame_duration_ms = frame_duration_ms
        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_frames = 0
        self._speech_detected = False
        self._frames_for_silence = silence_threshold_ms // frame_duration_ms

    def reset(self) -> None:
        """Reset VAD state for a new capture session."""
        self._silence_frames = 0
        self._speech_detected = False

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process one audio frame. Returns True if end-of-speech detected.

        Args:
            frame: int16 numpy array, exactly frame_duration_ms of audio

        Returns:
            True if silence has been detected for silence_threshold_ms after speech
        """
        # webrtcvad expects bytes
        frame_bytes = frame.tobytes()

        # Ensure correct frame length
        expected_samples = self.sample_rate * self.frame_duration_ms // 1000
        if len(frame) != expected_samples:
            return False

        is_speech = self._vad.is_speech(frame_bytes, self.sample_rate)

        if is_speech:
            self._speech_detected = True
            self._silence_frames = 0
        else:
            if self._speech_detected:
                self._silence_frames += 1

        # End-of-speech: speech was detected, then silence for threshold duration
        return self._speech_detected and self._silence_frames >= self._frames_for_silence

    @property
    def has_speech(self) -> bool:
        """Whether any speech has been detected in this session."""
        return self._speech_detected
```

**Key decisions:**
- webrtcvad aggressiveness=3 (most aggressive filtering) to minimize false speech detection from ambient noise.
- Frame-based counting for silence detection (not wall-clock time) for deterministic behavior.
- Requires speech to be detected first before silence triggers end-of-speech (prevents immediate trigger on startup silence).
- int16 input matches audio_capture.py output directly.

**PRD Coverage:** SC-4, INT-5

---

### 3.5 voice_loop/transcriber.py -- MLX Whisper Wrapper

**Purpose:** Load MLX Whisper model once and transcribe audio buffers to text.

**Design:**

```python
import asyncio
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self._model = None

    async def load_model(self) -> None:
        """Load the Whisper model. Call once at startup."""
        import mlx_whisper
        loop = asyncio.get_event_loop()
        logger.info("Loading Whisper model: %s", self.model_name)
        start = time.time()
        # Warm up by loading the model -- mlx_whisper.transcribe loads lazily,
        # so we do a dummy transcription to force the model into memory.
        await loop.run_in_executor(
            None,
            lambda: mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32),  # 1s of silence
                path_or_hf_repo=self.model_name,
                language="en",
            ),
        )
        elapsed = time.time() - start
        logger.info("Whisper model loaded in %.1fs", elapsed)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio buffer to text.

        Args:
            audio: int16 or float32 numpy array of audio samples
            sample_rate: sample rate of the audio (should be 16000 for Whisper)

        Returns:
            Transcribed text, or None if empty/noise
        """
        import mlx_whisper

        if len(audio) == 0:
            return None

        # Convert int16 to float32 if needed (mlx_whisper expects float32)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        start = time.time()

        result = await loop.run_in_executor(
            None,
            lambda: mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_name,
                language="en",
            ),
        )

        elapsed = time.time() - start
        text = result.get("text", "").strip()

        # Discard empty or noise-only transcriptions
        noise_markers = {"", "[silence]", "[music]", "[noise]", "(silence)", "..."}
        if text.lower() in noise_markers or len(text) < 2:
            logger.debug("Discarding noise transcript: %r (%.2fs)", text, elapsed)
            return None

        audio_duration = len(audio) / sample_rate
        logger.info(
            "Transcribed %.1fs audio in %.2fs (%.1fx realtime): %r",
            audio_duration, elapsed, audio_duration / elapsed if elapsed > 0 else 0, text,
        )
        return text
```

**Key decisions:**
- Model loaded once via dummy transcription (mlx_whisper loads lazily on first call).
- `run_in_executor` to avoid blocking the asyncio loop during GPU-bound transcription.
- Noise filtering for common Whisper hallucination tokens (`[silence]`, `[music]`, etc.).
- Accepts int16 from audio_capture and converts to float32 for Whisper.
- Logs realtime factor for performance monitoring.

**PRD Coverage:** STT-1, STT-2, STT-3, STT-4, STT-5, STT-6

---

### 3.6 voice_loop/gateway_client.py -- OpenClaw Gateway WebSocket Client

**Purpose:** Connect to the OpenClaw gateway, authenticate, send chat messages, and receive streaming response events.

**Protocol:** JSON-RPC over WebSocket (reverse-engineered from OpenClaw source).

**Design:**

```python
import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

class GatewayClient:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:18789",
        token: str = "gateway-token",
        timeout_s: int = 10,
        reconnect_max_s: int = 30,
    ):
        self.url = url
        self.token = token
        self.timeout_s = timeout_s
        self.reconnect_max_s = reconnect_max_s
        self._ws = None
        self._connected = False
        self._reconnect_delay = 1.0
        self._pending_runs: dict[str, asyncio.Queue] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to gateway and complete authentication handshake.

        Returns True if connected successfully, False otherwise.
        """
        try:
            self._ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=10)

            # Step 1: Receive connect.challenge
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            challenge = json.loads(raw)
            logger.debug("Received challenge: %s", challenge.get("event"))

            # Step 2: Send connect request with auth
            connect_req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "sesame-voice",
                        "version": "1.0",
                        "platform": "darwin",
                        "mode": "backend",
                    },
                    "role": "operator",
                    "scopes": ["operator.admin", "operator.talk.secrets"],
                    "auth": {"token": self.token},
                },
            }
            await self._ws.send(json.dumps(connect_req))

            # Step 3: Receive hello-ok response
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            response = json.loads(raw)
            if response.get("type") == "res" and response.get("ok"):
                self._connected = True
                self._reconnect_delay = 1.0  # Reset backoff
                logger.info("Gateway connected: %s", self.url)
                # Start background event listener
                asyncio.create_task(self._event_listener())
                return True
            else:
                logger.error("Gateway auth failed: %s", response)
                return False

        except Exception as e:
            logger.error("Gateway connection failed: %s", e)
            self._connected = False
            return False

    async def _event_listener(self) -> None:
        """Background task that routes incoming events to pending run queues."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") == "event" and msg.get("event") == "chat":
                    payload = msg.get("payload", {})
                    run_id = payload.get("runId")
                    if run_id and run_id in self._pending_runs:
                        await self._pending_runs[run_id].put(payload)
                elif msg.get("type") == "res":
                    # Response to a request (e.g., chat.send ack) -- logged but not queued
                    logger.debug("Gateway response: ok=%s", msg.get("ok"))
        except ConnectionClosed:
            logger.warning("Gateway WebSocket closed")
            self._connected = False
            asyncio.create_task(self._reconnect_loop())
        except Exception as e:
            logger.error("Gateway event listener error: %s", e)
            self._connected = False

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff."""
        while not self._connected:
            logger.info("Reconnecting to gateway in %.0fs...", self._reconnect_delay)
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self.reconnect_max_s)
            await self.connect()

    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send a chat message and yield streaming response text deltas.

        Args:
            text: User transcript to send

        Yields:
            Text chunks as they arrive from the assistant

        Raises:
            TimeoutError: If no response within timeout_s
            ConnectionError: If not connected to gateway
        """
        if not self._connected or self._ws is None:
            raise ConnectionError("Not connected to gateway")

        idempotency_key = str(uuid.uuid4())
        run_queue: asyncio.Queue = asyncio.Queue()
        self._pending_runs[idempotency_key] = run_queue

        try:
            # Send chat.send request
            req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "chat.send",
                "params": {
                    "sessionKey": "main",
                    "message": text,
                    "idempotencyKey": idempotency_key,
                },
            }
            await self._ws.send(json.dumps(req))
            logger.info("Sent to gateway: %r", text[:80])

            # Yield streaming deltas until final or error
            while True:
                try:
                    payload = await asyncio.wait_for(run_queue.get(), timeout=self.timeout_s)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Gateway response timeout ({self.timeout_s}s)")

                state = payload.get("state")
                message = payload.get("message", {})
                content = message.get("content", [])

                # Extract text from content blocks
                for block in content:
                    if block.get("type") == "text" and block.get("text"):
                        yield block["text"]

                if state == "final":
                    break
                elif state == "error":
                    error_msg = payload.get("error", "Unknown gateway error")
                    logger.error("Gateway error for run %s: %s", idempotency_key, error_msg)
                    raise RuntimeError(f"Gateway error: {error_msg}")

        finally:
            self._pending_runs.pop(idempotency_key, None)

    async def abort_run(self, run_id: str) -> None:
        """Send abort request for an active run."""
        if self._ws and self._connected:
            req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "chat.abort",
                "params": {"sessionKey": "main", "runId": run_id},
            }
            await self._ws.send(json.dumps(req))
            logger.info("Aborted run: %s", run_id)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None
```

**Key decisions:**
- `send_message` is an async generator yielding text deltas -- enables streaming to TTS.
- Pending runs tracked by idempotencyKey, routed by background event listener.
- Exponential backoff reconnection (1s, 2s, 4s... max 30s) per PRD GW-4.
- Abort support for interrupting in-progress runs.
- Protocol version 3 hardcoded per current OpenClaw gateway.

**PRD Coverage:** GW-1, GW-2, GW-3, GW-4, GW-5, GW-6, GW-7

---

### 3.7 voice_loop/tts_client.py -- Sesame TTS Streaming Client

**Purpose:** Stream audio from the Sesame TTS server via ElevenLabs-compatible streaming endpoint.

**Design:**

```python
import asyncio
import logging
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8880",
        voice: str = "conversationalB",
        output_format: str = "pcm_24000",
        timeout_s: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.output_format = output_format
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s, connect=5.0))

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks for the given text.

        Uses the ElevenLabs streaming endpoint:
        POST /v1/text-to-speech/{voice_id}/stream

        Args:
            text: Text to synthesize

        Yields:
            Raw PCM audio chunks (int16, 24000 Hz, mono)

        Raises:
            httpx.HTTPError: On connection or HTTP errors
        """
        if self._client is None:
            raise RuntimeError("TTSClient not started. Call start() first.")

        url = f"{self.base_url}/v1/text-to-speech/{self.voice}/stream"
        payload = {
            "text": text,
            "model_id": "csm-1b",
            "output_format": self.output_format,
        }

        logger.debug("TTS request: voice=%s format=%s text=%r", self.voice, self.output_format, text[:80])

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                yield chunk

    async def health_check(self) -> bool:
        """Check if the TTS server is reachable."""
        if self._client is None:
            return False
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False
```

**Key decisions:**
- Uses httpx async streaming (already a test dependency, promoted to main dependency).
- ElevenLabs streaming endpoint (`/v1/text-to-speech/{voice_id}/stream`) for lowest latency.
- PCM 24000 format avoids transcoding overhead (native CSM-1B sample rate).
- 4096-byte chunks match the existing Sesame TTS `stream_chunk_size` default.
- Client disconnect (closing the stream) acts as an implicit interrupt signal to the TTS server.

**PRD Coverage:** TTS-1, TTS-2, TTS-3, TTS-4, TTS-5, TTS-6

---

### 3.8 voice_loop/playback.py -- Audio Playback

**Purpose:** Stream PCM audio to speakers using sounddevice OutputStream. Supports streaming playback (play while receiving) and instant stop for interrupt.

**Design:**

```python
import asyncio
import logging
import queue
import threading
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
                logger.info("Resolved speaker device: '%s' -> index %d ('%s')", self.device, i, dev["name"])
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
                    outdata[:len(data)] = data
                    outdata[len(data):] = 0
                else:
                    outdata[:] = data[:len(outdata)]
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
                while self._stream.active:
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
```

**Key decisions:**
- Callback-based OutputStream for low-latency streaming playback.
- Queue-based feeding: async producer pushes chunks, callback consumes them.
- `stop()` is synchronous and immediate -- called by interrupt handler.
- Blocksize of 100ms provides good balance between latency and underrun prevention.
- PCM int16 at 24000 Hz matches TTS client output directly (no conversion).

**PRD Coverage:** PB-1, PB-2, PB-3

---

### 3.9 voice_loop/interrupt.py -- Interrupt Detection

**Purpose:** Monitor the microphone during playback to detect user speech and signal an interrupt.

**Design:**

```python
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
            rms = np.sqrt(np.mean(audio_float ** 2))

            # Store frames for seamless transition to LISTENING
            self._captured_frames.append(indata.copy())
            # Keep only last 1 second of audio
            max_frames = int(1000 / self.frame_duration_ms)
            if len(self._captured_frames) > max_frames:
                self._captured_frames.pop(0)

            if rms > self.energy_threshold:
                self._above_threshold_count += 1
                if self._above_threshold_count >= self.consecutive_frames:
                    logger.info("Interrupt detected: RMS=%.4f (threshold=%.4f)", rms, self.energy_threshold)
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
```

**Key decisions:**
- RMS energy-based detection (simpler than webrtcvad for interrupt, which only needs to detect "is someone speaking loudly").
- Consecutive frames requirement prevents single-frame noise spikes from triggering.
- Captured audio is retained for seamless transition: when interrupt fires, the interrupting speech is already buffered and can seed the next LISTENING phase.
- `loop.call_soon_threadsafe` bridges the sounddevice callback thread to the asyncio event loop.

**PRD Coverage:** INT-1, INT-2, INT-3, INT-5, INT-6

---

### 3.10 voice_loop/menu_bar.py -- Menu Bar Application

**Purpose:** macOS menu bar app using rumps. Shows state, provides PTT click trigger, and dropdown with status/controls.

**Design:**

```python
import logging
import threading
from typing import Callable, Optional

import rumps

logger = logging.getLogger(__name__)

# State icons (macOS system symbols or emoji fallback)
STATE_ICONS = {
    "idle": "🎙",           # Microphone
    "listening": "🔴",      # Red circle (recording)
    "transcribing": "⚡",    # Processing
    "processing": "💭",      # Thinking
    "speaking": "🔊",       # Speaker
    "error": "⚠️",          # Warning
    "disconnected": "❌",    # Disconnected
}

class VoiceLoopMenuBar(rumps.App):
    def __init__(
        self,
        on_ptt_click: Optional[Callable] = None,
        on_quit: Optional[Callable] = None,
    ):
        super().__init__(
            name="Sesame Voice",
            icon=None,
            title=STATE_ICONS["idle"],
            quit_button=None,  # Custom quit handler
        )
        self._on_ptt_click = on_ptt_click
        self._on_quit = on_quit
        self._state = "idle"
        self._mic_device = "System Default"
        self._gateway_status = "Disconnected"

        # Build menu items
        self._status_item = rumps.MenuItem("Status: Idle", callback=None)
        self._status_item.set_callback(None)
        self._mic_item = rumps.MenuItem(f"Mic: {self._mic_device}", callback=None)
        self._gateway_item = rumps.MenuItem(f"Gateway: {self._gateway_status}", callback=None)
        self._ptt_item = rumps.MenuItem("Push to Talk", callback=self._handle_ptt)
        self._quit_item = rumps.MenuItem("Quit", callback=self._handle_quit)

        self.menu = [
            self._status_item,
            self._mic_item,
            self._gateway_item,
            None,  # Separator
            self._ptt_item,
            None,  # Separator
            self._quit_item,
        ]

    def _handle_ptt(self, sender):
        """Handle push-to-talk menu item click."""
        if self._on_ptt_click:
            self._on_ptt_click()

    def _handle_quit(self, sender):
        """Handle quit menu item click."""
        if self._on_quit:
            self._on_quit()
        rumps.quit_application()

    def update_state(self, state: str) -> None:
        """Update the menu bar to reflect the current voice loop state."""
        self._state = state
        self.title = STATE_ICONS.get(state, STATE_ICONS["idle"])
        self._status_item.title = f"Status: {state.capitalize()}"

    def update_mic_device(self, device_name: str) -> None:
        """Update the displayed mic device name."""
        self._mic_device = device_name
        self._mic_item.title = f"Mic: {device_name}"

    def update_gateway_status(self, status: str) -> None:
        """Update the gateway connection status display."""
        self._gateway_status = status
        self._gateway_item.title = f"Gateway: {status}"
```

**Key decisions:**
- rumps runs on the main thread (macOS requirement for menu bar apps).
- State updates are called from the asyncio thread via rumps-safe methods (rumps handles thread safety for title/menu updates).
- PTT click fires a callback that signals the asyncio loop to enter LISTENING state.
- Emoji icons as fallback; could be replaced with proper .icns files in Phase 3.
- Custom quit handler ensures clean shutdown of asyncio loop and all resources.

**PRD Coverage:** MB-1, MB-2, MB-3

---

### 3.11 voice_loop/sounds.py -- Chime Sounds

**Purpose:** Play short feedback sounds (listening chime, error chime) using sounddevice.

**Design:**

```python
import logging
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

def _generate_tone(
    frequency: float,
    duration_s: float,
    sample_rate: int = 24000,
    amplitude: float = 0.3,
    fade_ms: float = 20,
) -> np.ndarray:
    """Generate a sine wave tone with fade-in/fade-out."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    # Apply fade
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and fade_samples < len(tone) // 2:
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

    return tone

def play_listening_chime(device: int | str | None = None) -> None:
    """Play a short ascending two-tone chime (C5 + E5) to indicate listening started."""
    sr = 24000
    tone1 = _generate_tone(523.25, 0.08, sr)  # C5
    tone2 = _generate_tone(659.25, 0.12, sr)  # E5
    silence = np.zeros(int(sr * 0.02), dtype=np.float32)
    chime = np.concatenate([tone1, silence, tone2])
    try:
        sd.play(chime, samplerate=sr, device=device, blocking=False)
    except Exception as e:
        logger.warning("Failed to play listening chime: %s", e)

def play_error_chime(device: int | str | None = None) -> None:
    """Play a short descending two-tone chime (E4 + C4) to indicate an error."""
    sr = 24000
    tone1 = _generate_tone(329.63, 0.12, sr)  # E4
    tone2 = _generate_tone(261.63, 0.15, sr)  # C4
    silence = np.zeros(int(sr * 0.03), dtype=np.float32)
    chime = np.concatenate([tone1, silence, tone2])
    try:
        sd.play(chime, samplerate=sr, device=device, blocking=False)
    except Exception as e:
        logger.warning("Failed to play error chime: %s", e)

def play_complete_chime(device: int | str | None = None) -> None:
    """Play a single short tone to indicate response complete."""
    sr = 24000
    tone = _generate_tone(880.0, 0.06, sr, amplitude=0.15)  # A5, quiet
    try:
        sd.play(tone, samplerate=sr, device=device, blocking=False)
    except Exception as e:
        logger.warning("Failed to play complete chime: %s", e)
```

**Key decisions:**
- Procedurally generated tones (no external audio files to manage).
- Non-blocking playback so chimes do not delay the main loop.
- Low amplitude (0.15-0.3) to avoid startling the user.
- 24000 Hz sample rate matches the TTS output.

---

### 3.12 voice_loop/loop.py -- Main Orchestrator

**Purpose:** Async event loop that runs the state machine, coordinating all modules through each state transition.

**Design (pseudocode with key async patterns):**

```python
import asyncio
import logging
import time
from typing import Optional

import re
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

# Sentence boundary pattern: split on . ! ? followed by whitespace or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

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
            reconnect_max_s=settings.gateway_reconnect_max_s,
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

    def trigger_ptt(self) -> None:
        """Called from menu bar or hotkey handler to trigger push-to-talk."""
        self._ptt_event.set()

    async def start(self) -> None:
        """Initialize all components and start the main loop."""
        logger.info("Starting Voice Loop...")
        logger.info("Config: mic=%s, gateway=%s, tts=%s",
                     self.settings.mic_device or "system default",
                     self.settings.gateway_url,
                     self.settings.tts_url)

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
                    await self.state.transition(VoiceLoopState.ERROR)

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
        self._tts_audio_queue: asyncio.Queue = asyncio.Queue()
        sentence_buf = SentenceBuffer()

        try:
            sentence_count = 0
            async for delta in self.gateway.send_message(self._transcript):
                sentences = sentence_buf.add(delta)
                for sentence in sentences:
                    sentence_count += 1
                    logger.debug("Sentence %d ready for TTS: %r", sentence_count, sentence[:80])
                    # Stream TTS audio for this sentence into the queue
                    async for audio_chunk in self.tts.stream_speech(sentence):
                        await self._tts_audio_queue.put(audio_chunk)

            # Flush any remaining text (final partial sentence)
            remaining = sentence_buf.flush()
            if remaining:
                sentence_count += 1
                logger.debug("Final sentence %d for TTS: %r", sentence_count, remaining[:80])
                async for audio_chunk in self.tts.stream_speech(remaining):
                    await self._tts_audio_queue.put(audio_chunk)

            # Signal end of TTS audio stream
            await self._tts_audio_queue.put(None)

            elapsed = time.monotonic() - t_start
            logger.info("Gateway + TTS streaming completed in %.2fs (%d sentences)", elapsed, sentence_count)

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

            if interrupt_task in done:
                # User interrupted -- stop playback and re-enter listening
                self.player.stop()
                logger.info("Playback interrupted after %.2fs", time.monotonic() - t_start)
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
        self.player.stop()
        await self.interrupt.stop()
        await self.tts.close()
        await self.gateway.close()
        logger.info("Voice Loop shut down.")
```

**Key decisions:**
- State handler pattern: each state has a `_handle_*` method that runs until state transition.
- `asyncio.wait` with `FIRST_COMPLETED` for clean playback vs interrupt racing.
- Sentence-streaming architecture: `_handle_processing` accumulates gateway deltas in a `SentenceBuffer`, detects sentence boundaries (`. ! ?`), and sends each complete sentence to TTS immediately. Audio chunks are queued into `_tts_audio_queue` for the SPEAKING state to consume. This significantly reduces perceived latency because TTS generation starts during gateway streaming, not after it completes.
- `SentenceBuffer` is a simple accumulator that splits on punctuation followed by whitespace. It holds partial text until a boundary is found or `flush()` is called for the final fragment.
- Error handler plays chime and auto-returns to IDLE after 1 second.
- All timing logged for performance analysis.

---

### 3.13 voice_loop/__main__.py -- Entry Point

**Purpose:** CLI entry point for `python -m voice_loop`.

**Design:**

```python
import argparse
import asyncio
import logging
import sys
import threading

from .audio_capture import AudioCapture
from .config import VoiceLoopSettings
from .loop import VoiceLoop
from .menu_bar import VoiceLoopMenuBar


def main():
    parser = argparse.ArgumentParser(description="Sesame Voice Loop")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--headless", action="store_true", help="Run without menu bar (daemon mode)")
    args = parser.parse_args()

    if args.list_devices:
        devices = AudioCapture.list_devices()
        print("\nAudio Devices:")
        print("-" * 60)
        for dev in devices:
            direction = []
            if dev["max_input_channels"] > 0:
                direction.append(f"IN({dev['max_input_channels']}ch)")
            if dev["max_output_channels"] > 0:
                direction.append(f"OUT({dev['max_output_channels']}ch)")
            print(f"  [{dev['index']:2d}] {dev['name']:<40s} {', '.join(direction)}  {dev['default_samplerate']:.0f}Hz")
        print()
        sys.exit(0)

    # Load config
    from pathlib import Path
    config_path = Path(args.config) if args.config else None
    settings = VoiceLoopSettings.from_yaml(config_path) if config_path else VoiceLoopSettings.from_yaml()

    # Configure logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if settings.log_file:
        from pathlib import Path as P
        log_path = P(settings.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    # Create voice loop
    voice_loop = VoiceLoop(settings)

    if args.headless or not settings.run_as_menu_bar:
        # Headless mode: just run the async loop
        asyncio.run(voice_loop.start())
    else:
        # Menu bar mode: run rumps on main thread, asyncio in daemon thread
        def run_async_loop():
            asyncio.run(voice_loop.start())

        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()

        menu_app = VoiceLoopMenuBar(
            on_ptt_click=voice_loop.trigger_ptt,
            on_quit=lambda: asyncio.run_coroutine_threadsafe(
                voice_loop.shutdown(), asyncio.get_event_loop()
            ),
        )

        # Register state change callback to update menu bar
        voice_loop.state.on_transition(
            lambda old, new: menu_app.update_state(new.value)
        )

        menu_app.run()


if __name__ == "__main__":
    main()
```

**Key decisions:**
- `--list-devices` flag exits immediately after printing (per PRD SC-8).
- rumps runs on main thread (macOS requirement), asyncio in daemon thread.
- PTT and quit callbacks bridge menu bar to asyncio loop.
- State transition callback updates menu bar icon automatically.

**PRD Coverage:** SC-8, MB-4, CFG-1

---

## 4. Master Task List

### Phase 1: MVP -- Push-to-Talk + Menu Bar

| ID | Task | Module | Est. Hours | Dependencies | Priority | Status |
|----|------|--------|-----------|-------------|----------|--------|
| T-001 | Create `voice_loop/` package structure with `__init__.py` | Package | 0.5 | None | Must | ☐ |
| T-002 | Implement `VoiceLoopSettings` with YAML loading and env var overrides | config.py | 2.0 | T-001 | Must | ☐ |
| T-003 | Write unit tests for config loading (YAML, env vars, defaults) | test_config.py | 1.5 | T-002 | Must | ☐ |
| T-004 | Create default config YAML in `voice_loop/default_config.yaml` | config | 0.5 | T-002 | Must | ☐ |
| T-005 | Implement `VoiceLoopState` enum and `StateMachine` with transitions | state.py | 2.0 | T-001 | Must | ☐ |
| T-006 | Write unit tests for state machine (valid/invalid transitions, callbacks) | test_state.py | 1.5 | T-005 | Must | ☐ |
| T-007 | Implement `AudioCapture` with device selection and list_devices | audio_capture.py | 3.0 | T-001 | Must | ☐ |
| T-008 | Write unit tests for AudioCapture (mock sounddevice) | test_audio_capture.py | 1.5 | T-007 | Must | ☐ |
| T-009 | Implement `VAD` with webrtcvad integration | vad.py | 2.0 | T-001 | Must | ☐ |
| T-010 | Write unit tests for VAD (speech detection, silence detection, reset) | test_vad.py | 1.5 | T-009 | Must | ☐ |
| T-011 | Implement `Transcriber` with MLX Whisper model loading and transcription | transcriber.py | 3.0 | T-001 | Must | ☐ |
| T-012 | Write unit tests for Transcriber (mock mlx_whisper, noise filtering) | test_transcriber.py | 1.5 | T-011 | Must | ☐ |
| T-013 | Implement `GatewayClient` with connect, auth, send, receive, reconnect | gateway_client.py | 5.0 | T-001 | Must | ☐ |
| T-014 | Write unit tests for GatewayClient (mock websocket, auth flow, streaming) | test_gateway_client.py | 3.0 | T-013 | Must | ☐ |
| T-015 | Implement `TTSClient` with httpx streaming | tts_client.py | 2.0 | T-001 | Must | ☐ |
| T-016 | Write unit tests for TTSClient (mock httpx, streaming chunks) | test_tts_client.py | 1.5 | T-015 | Must | ☐ |
| T-017 | Implement `AudioPlayer` with streaming playback and stop | playback.py | 3.0 | T-001 | Must | ☐ |
| T-018 | Write unit tests for AudioPlayer (mock sounddevice) | test_playback.py | 1.5 | T-017 | Must | ☐ |
| T-019 | Implement `InterruptMonitor` with energy-based detection | interrupt.py | 2.5 | T-001 | Must | ☐ |
| T-020 | Write unit tests for InterruptMonitor | test_interrupt.py | 1.5 | T-019 | Must | ☐ |
| T-021 | Implement `sounds.py` with chime generation | sounds.py | 1.0 | T-001 | Should | ☐ |
| T-022 | Implement `VoiceLoopMenuBar` with rumps | menu_bar.py | 2.5 | T-001 | Must | ☐ |
| T-023 | Implement `VoiceLoop` main orchestrator (all state handlers) including `SentenceBuffer` and sentence-streaming TTS pipeline | loop.py | 7.0 | T-005, T-007, T-009, T-011, T-013, T-015, T-017, T-019 | Must | ☐ |
| T-024 | Write integration tests for VoiceLoop (mock all I/O) | test_loop.py | 4.0 | T-023 | Must | ☐ |
| T-025 | Implement `__main__.py` with CLI args and entry point | __main__.py | 2.0 | T-023, T-022 | Must | ☐ |
| T-026 | Add voice_loop dependencies to `pyproject.toml` (optional-dependencies group) | pyproject.toml | 0.5 | T-001 | Must | ☐ |
| T-027 | End-to-end manual test: full happy path with real hardware | Manual | 3.0 | T-025 | Must | ☐ |
| T-028 | Performance profiling: log and verify latency budget per stage | Manual | 2.0 | T-027 | Should | ☐ |

### Phase 2: Wake Word + Polish

| ID | Task | Module | Est. Hours | Dependencies | Priority | Status |
|----|------|--------|-----------|-------------|----------|--------|
| T-029 | Implement wake word detection via Apple Speech framework | wake_word.py | 6.0 | T-025 | Phase 2 | ☐ |
| T-030 | Implement Vosk fallback wake word detection | wake_word.py | 4.0 | T-029 | Phase 2 | ☐ |
| T-031 | Add wake word muting during playback (echo cancellation) | interrupt.py | 2.0 | T-029, T-019 | Phase 2 | ☐ |
| T-032 | Upgrade interrupt to speech-based detection (not just hotkey) | interrupt.py | 3.0 | T-019 | Phase 2 | ☐ |
| T-034 | Add TTS server health checking and graceful degradation | tts_client.py | 2.0 | T-015 | Phase 2 | ☐ |
| T-035 | Refine VAD tuning with real-environment testing | vad.py | 2.0 | T-009 | Phase 2 | ☐ |

### Phase 3: launchd Service + Polish

| ID | Task | Module | Est. Hours | Dependencies | Priority | Status |
|----|------|--------|-----------|-------------|----------|--------|
| T-036 | Create launchd plist for background service | launchd | 2.0 | T-027 | Phase 3 | ☐ |
| T-037 | Configure log rotation | config | 1.0 | T-036 | Phase 3 | ☐ |
| T-038 | Write README for voice_loop module | docs | 2.0 | T-027 | Phase 3 | ☐ |
| T-039 | Improve config validation with helpful error messages | config.py | 1.5 | T-002 | Phase 3 | ☐ |

**Phase 1 Total:** ~58 hours (approximately 7.5 working days)
**Phase 2 Total:** ~19 hours (approximately 2.5 working days)
**Phase 3 Total:** ~6.5 hours (approximately 1 working day)

---

## 5. Sprint Planning (Phase 1)

### Sprint 1: Foundation (Days 1-2)

**Goal:** Config, state machine, audio capture, and VAD working independently.

| Task | Description | Est. |
|------|-------------|------|
| T-001 | Package structure | 0.5h |
| T-026 | Dependencies in pyproject.toml | 0.5h |
| T-002 | VoiceLoopSettings | 2.0h |
| T-003 | Config tests | 1.5h |
| T-004 | Default config YAML | 0.5h |
| T-005 | StateMachine | 2.0h |
| T-006 | State machine tests | 1.5h |
| T-007 | AudioCapture | 3.0h |
| T-008 | AudioCapture tests | 1.5h |
| T-009 | VAD | 2.0h |
| T-010 | VAD tests | 1.5h |
| **Total** | | **16.5h** |

**Exit Criteria:**
- `python -m voice_loop --list-devices` prints available audio devices
- Config loads from YAML and env vars
- State machine validates transitions correctly
- AudioCapture records from selected mic
- VAD detects speech start and silence end-of-speech

### Sprint 2: Pipeline (Days 3-5)

**Goal:** Transcriber, gateway client, TTS client, and playback working independently.

| Task | Description | Est. |
|------|-------------|------|
| T-011 | Transcriber (MLX Whisper) | 3.0h |
| T-012 | Transcriber tests | 1.5h |
| T-013 | GatewayClient | 5.0h |
| T-014 | GatewayClient tests | 3.0h |
| T-015 | TTSClient | 2.0h |
| T-016 | TTSClient tests | 1.5h |
| T-017 | AudioPlayer | 3.0h |
| T-018 | AudioPlayer tests | 1.5h |
| T-019 | InterruptMonitor | 2.5h |
| T-020 | InterruptMonitor tests | 1.5h |
| T-021 | Chime sounds | 1.0h |
| **Total** | | **26.0h** |

**Exit Criteria:**
- Transcriber transcribes a WAV file correctly
- GatewayClient connects, authenticates, sends/receives messages
- TTSClient streams audio from Sesame TTS server
- AudioPlayer plays streaming PCM audio
- InterruptMonitor detects speech during playback

### Sprint 3: Integration (Days 6-7)

**Goal:** Everything wired together into a working menu bar app.

| Task | Description | Est. |
|------|-------------|------|
| T-022 | VoiceLoopMenuBar (rumps) | 2.5h |
| T-023 | VoiceLoop orchestrator (incl. SentenceBuffer + sentence-streaming TTS) | 7.0h |
| T-024 | Integration tests | 4.0h |
| T-025 | __main__.py entry point | 2.0h |
| T-027 | End-to-end manual test | 3.0h |
| T-028 | Performance profiling | 2.0h |
| **Total** | | **20.5h** |

**Exit Criteria (AC-1 Happy Path):**
- Click menu bar icon -> icon changes to listening
- Speak -> VAD detects silence -> Whisper transcribes
- Transcript sent to gateway -> response received
- TTS generates audio -> playback begins streaming
- After playback -> icon returns to idle
- Total time from end of speech to first audio < 4 seconds

---

## 6. Testing Strategy

### 6.1 Unit Tests

Each module gets its own test file with mocked external dependencies.

| Test File | Mocked Dependencies | Key Test Cases |
|-----------|---------------------|----------------|
| `test_config.py` | Filesystem (YAML file) | Default values, YAML override, env var override, missing file, invalid values |
| `test_state.py` | None (pure logic) | All valid transitions, invalid transition raises ValueError, callback invocation, concurrent transition safety |
| `test_vad.py` | None (pure logic, mock audio data) | Speech detection, silence after speech, no false end-of-speech on initial silence, reset clears state |
| `test_transcriber.py` | `mlx_whisper.transcribe` | Successful transcription, empty audio, noise filtering ("[silence]", whitespace), int16-to-float32 conversion |
| `test_gateway_client.py` | `websockets` | Connect + auth handshake, send message + receive deltas, timeout handling, reconnect backoff, abort run |
| `test_tts_client.py` | `httpx.AsyncClient` | Streaming response, HTTP error handling, health check |
| `test_playback.py` | `sounddevice` | Streaming playback, stop mid-playback, empty stream |
| `test_interrupt.py` | `sounddevice` | Energy threshold detection, consecutive frames requirement, captured audio buffer |

### 6.2 Integration Tests

| Test | Description | Mocked |
|------|-------------|--------|
| `test_loop.py::test_happy_path` | Full IDLE->LISTENING->TRANSCRIBING->PROCESSING->SPEAKING->IDLE cycle | sounddevice, mlx_whisper, websockets, httpx |
| `test_loop.py::test_interrupt_during_playback` | SPEAKING->LISTENING transition on interrupt | sounddevice, mlx_whisper, websockets, httpx |
| `test_loop.py::test_empty_transcript` | TRANSCRIBING->IDLE on empty transcript | sounddevice, mlx_whisper |
| `test_loop.py::test_gateway_timeout` | PROCESSING->ERROR on gateway timeout | sounddevice, mlx_whisper, websockets |
| `test_loop.py::test_tts_unavailable` | PROCESSING->ERROR on TTS connection failure | sounddevice, mlx_whisper, websockets, httpx |

### 6.3 Manual Test Plan

These tests require real hardware and running services.

| Test | Preconditions | Steps | Expected Result |
|------|---------------|-------|-----------------|
| MT-1: Happy path | TTS server running, gateway running | Click menu bar icon, say "Hello", wait | Hear TTS response, icon returns to idle |
| MT-2: Device selection | Specific mic configured | Set `mic_device` in config, restart | Correct device shown in logs and menu bar |
| MT-3: Interrupt | TTS server running, gateway running | Click icon, ask long question, speak during response | Playback stops, new speech captured |
| MT-4: Gateway down | Gateway not running | Click icon, speak | Error chime plays, icon shows error then idle |
| MT-5: TTS down | TTS server not running, gateway running | Click icon, speak | Error chime plays after gateway response |
| MT-6: Reconnection | Gateway running | Start voice loop, stop gateway, restart gateway | Auto-reconnects, next interaction works |
| MT-7: Coexistence | Krisp and Wispr Flow running | Use voice loop normally | No audio conflicts |
| MT-8: Uptime | All services running | Run for 24 hours with periodic interactions | No crashes, stable memory |

---

## 7. Dependency List

### New Dependencies (Phase 1)

| Package | Version | Purpose | License | Already in pyproject.toml? |
|---------|---------|---------|---------|---------------------------|
| `mlx-whisper` | >=0.4.0 | Local speech-to-text via MLX | MIT | No |
| `sounddevice` | >=0.5.0 | Mic capture + audio playback | MIT | No |
| `webrtcvad` | >=2.0.10 | Voice Activity Detection | MIT | No |
| `websockets` | >=13.0 | Gateway WebSocket client | BSD | No |
| `httpx` | >=0.27.0 | TTS HTTP streaming client | BSD | In `[test]` only |
| `rumps` | >=0.4.0 | macOS menu bar application | MIT | No |
| `PyYAML` | >=6.0 | YAML config file parsing | MIT | No |

### Already Present (reused from sesame-tts)

| Package | Purpose |
|---------|---------|
| `pydantic-settings` | Settings validation |
| `numpy` | Audio array manipulation |

### Phase 2 Additional Dependencies

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `pyobjc-framework-Speech` | >=10.0 | Apple Speech framework for wake word | MIT |
| `vosk` | >=0.3.45 | Fallback wake word detection | Apache 2.0 |

### Proposed pyproject.toml Addition

```toml
[project.optional-dependencies]
voice-loop = [
    "mlx-whisper>=0.4.0",
    "sounddevice>=0.5.0",
    "webrtcvad>=2.0.10",
    "websockets>=13.0",
    "httpx>=0.27.0",
    "rumps>=0.4.0",
    "PyYAML>=6.0",
]
```

Install with: `uv pip install -e ".[voice-loop]"`

---

## 8. Gateway WebSocket Protocol Reference

This section documents the reverse-engineered OpenClaw gateway protocol for implementation reference.

### Connection Sequence

```
Client                                    Gateway (ws://127.0.0.1:18789)
  |                                          |
  |  --- WebSocket connect ----------------> |
  |                                          |
  |  <-- event: connect.challenge ---------- |  (contains nonce)
  |                                          |
  |  --- req: connect (auth + client info) > |
  |                                          |
  |  <-- res: ok=true, payload: hello-ok --- |  (features, snapshot, auth)
  |                                          |
  |  === Connection established ===========  |
```

### Message Format

All messages are JSON with a `type` field:

- `req` -- Client request (has `id`, `method`, `params`)
- `res` -- Server response (has `id`, `ok`, `payload` or `error`)
- `event` -- Server push event (has `event`, `payload`)

### Chat Flow

```
Client                                    Gateway
  |                                          |
  |  --- req: chat.send (message, key) ----> |
  |                                          |
  |  <-- res: ok, runId, status:accepted --- |  (immediate ack)
  |                                          |
  |  <-- event: chat (state:delta, text) --- |  (streaming chunk)
  |  <-- event: chat (state:delta, text) --- |  (streaming chunk)
  |  <-- event: chat (state:delta, text) --- |  (streaming chunk)
  |  <-- event: chat (state:final, text) --- |  (last chunk)
  |                                          |
```

### Abort Flow

```
Client                                    Gateway
  |                                          |
  |  --- req: chat.abort (runId) ----------> |
  |                                          |
  |  <-- res: ok -------------------------  |
  |                                          |
```

---

## 9. Open Questions and Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Global hotkey implementation | Deferred to post-MVP | Quartz event tap requires Accessibility permissions and is complex. Menu bar click is sufficient for MVP. Hotkey can be added as a follow-up. |
| 2 | VAD silence threshold | 1000ms (not 1500ms) | Per PRD optimization guidance, tuned down for responsiveness. Can be adjusted via config. |
| 3 | Gateway response: stream to TTS or collect full response? | Phase 1: stream sentences to TTS as gateway deltas arrive. | Per user interview feedback, sentence-streaming promoted to Phase 1. SentenceBuffer detects boundaries at `. ! ?` and sends each complete sentence to TTS immediately, significantly reducing perceived latency. |
| 4 | httpx: promote from test dep to main dep? | Yes, via `[voice-loop]` optional group | httpx is already pinned in test deps. Voice loop needs it for async streaming. Using optional deps keeps TTS server install lean. |
| 5 | Separate process or same process as TTS server? | Separate process | Voice loop has different lifecycle (menu bar app vs server), different resource profile, and should not crash the TTS server. |

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-21 | Burke + Claude Code | Initial TRD from PRD v1.1 |
| 1.1 | 2026-02-21 | Burke + Claude Code | Refined: gateway token auto-read from openclaw.json, sentence-streaming to TTS promoted to Phase 1, global hotkey confirmed deferred |
