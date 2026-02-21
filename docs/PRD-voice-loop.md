# PRD: Voice Loop -- Standalone Voice Conversation Agent for Sesame TTS

**Author:** Burke
**Created:** 2026-02-20
**Status:** Refined
**Version:** 1.1

---

## 1. Product Summary

### Problem

OpenClaw includes a built-in "Talk Mode" for voice conversations, but it is broken in multiple ways that make it unusable as a daily voice assistant:

1. **Hardcoded ElevenLabs URL** -- The Talk Mode baseUrl is hardcoded to `api.elevenlabs.io`. An upstream PR to make this configurable has been submitted but is not merged. This means Talk Mode cannot use the local Sesame TTS server.
2. **System robot voice fallback** -- When ElevenLabs fails (no API key configured or network issues), OpenClaw falls back to the macOS system `say` command, producing a robotic voice.
3. **Unreliable wake word detection** -- OpenClaw's built-in wake word detection is flaky and frequently misses or false-triggers.
4. **Flaky push-to-talk** -- The push-to-talk hotkey in OpenClaw is unreliable, sometimes failing to register or releasing prematurely.
5. **Double audio responses** -- When Sesame TTS is partially working via OpenClaw, both the system voice and Sesame `afplay` output can fire simultaneously, producing garbled overlapping audio.
6. **Audio device conflicts** -- OpenClaw does not handle virtual audio devices (Krisp, Wispr Flow) gracefully, often selecting the wrong mic input or causing audio routing conflicts.

These are upstream bugs in OpenClaw that Burke cannot fix on his own timeline. Waiting for upstream fixes blocks daily use of voice conversations with the local AI assistant.

### Solution

Build a standalone voice conversation loop as a Python application that runs independently of OpenClaw's Talk Mode. This loop orchestrates the full voice pipeline:

1. **Push-to-talk activation** -- Burke clicks the menu bar icon or presses a global hotkey to begin speech capture. Wake word detection is deferred to Phase 2.
2. **Speech capture and transcription** -- Records user speech and transcribes it locally using MLX Whisper. No cloud services, no API costs.
3. **Gateway communication** -- Sends the transcript to the OpenClaw gateway (WebSocket on port 18789) and receives the assistant's text response.
4. **TTS generation** -- Sends the response text to the Sesame TTS server (localhost:8880) for high-quality local speech synthesis.
5. **Audio playback with interrupt** -- Plays the generated audio through speakers. If the user speaks during playback, audio stops immediately and the loop re-enters listening mode.
6. **Loop restart** -- After playback completes (or is interrupted), returns to idle state awaiting the next push-to-talk activation.

This bypasses every OpenClaw Talk Mode bug by owning the entire pipeline end-to-end.

### Value Proposition

- **Unblocked by upstream** -- No dependency on OpenClaw fixing Talk Mode bugs
- **Fully local** -- Zero cloud dependencies; STT, LLM gateway, and TTS all run on the Mac Studio
- **Fast round-trip** -- Target under 4 seconds from push-to-talk activation to audio playing, leveraging local inference on M3 Ultra
- **Reliable** -- Explicit mic device selection eliminates Krisp/virtual device conflicts
- **Interrupt support** -- Natural conversation flow; speak to stop the assistant mid-sentence
- **Background operation** -- Runs as a launchd service or menu bar app, always ready

---

## 2. User Analysis

### Primary User: Burke (Developer / Operator)

Burke is a solo developer running a Mac Studio (M3 Ultra, 256GB unified memory, macOS Tahoe). He uses OpenClaw as an AI assistant and has already deployed the Sesame TTS server for local speech synthesis. He wants a voice-first interaction mode that works reliably without cloud dependencies.

**Current Journey (Broken):**

1. Burke activates OpenClaw Talk Mode
2. Wake word detection fails intermittently -- sometimes picks up background noise, sometimes ignores direct speech
3. When speech is detected, OpenClaw attempts to use ElevenLabs (hardcoded URL) and fails
4. macOS system `say` command fires as fallback, producing robotic voice
5. Meanwhile, if Sesame TTS is also triggered via a workaround, both audio outputs overlap
6. Krisp virtual mic is selected instead of the physical mic, causing silence or distortion
7. Burke gives up and types instead

**Target Journey:**

1. Burke clicks the menu bar icon or presses a global hotkey (push-to-talk)
2. The menu bar icon changes to "listening" state, confirming the system is capturing
3. Burke speaks naturally; speech is captured and transcribed locally in under 1 second
4. Transcript is sent to OpenClaw gateway; response arrives in 1-2 seconds
5. Sesame TTS generates and streams high-quality audio; playback begins within 1 second of response
6. If Burke speaks during playback, audio stops and the loop captures new input
7. After playback, the system returns to idle state (menu bar shows idle icon)
8. Total time from push-to-talk to hearing the response: under 4 seconds for a typical exchange

**Pain Points:**

- Cannot use voice conversations with OpenClaw due to upstream bugs
- Does not want to depend on upstream PRs being merged on any timeline
- Needs explicit mic device control (Mac Studio has multiple audio sources including Krisp)
- Wants the assistant to feel responsive and interruptible, not robotic and slow

### Persona Summary

| Attribute | Value |
|-----------|-------|
| Name | Burke |
| Role | Solo developer and daily AI assistant user |
| Hardware | Mac Studio, M3 Ultra, 256GB RAM |
| OS | macOS Tahoe 26.3 |
| Audio environment | Built-in speakers, studio mic, Krisp virtual device, Wispr Flow |
| Primary use case | Voice conversations with OpenClaw AI assistant |
| Technical skill | Expert -- comfortable with Python, launchd, system configuration |

---

## 3. Goals and Non-Goals

### Goals

1. **Reliable voice loop** -- Deliver a complete push-to-talk-to-response-to-idle cycle that works every time without manual intervention.
2. **Sub-4-second round-trip** -- From push-to-talk activation to the first audio of the response playing through speakers, targeting under 4 seconds for a typical single-sentence exchange.
3. **Fully local execution** -- Every component (STT, LLM communication, TTS) runs on the local machine with zero cloud API calls.
4. **Interrupt support** -- User can speak at any time during playback to stop the current response and start a new input.
5. **Audio device control** -- Explicit mic and speaker device selection to avoid conflicts with virtual audio devices (Krisp, Wispr Flow).
6. **Background operation** -- Runs as a daemon (launchd) or menu bar app, surviving reboots and running without a terminal window.
7. **Coexistence** -- Must not conflict with Krisp, Wispr Flow, or other audio applications running simultaneously.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Round-trip latency | < 4 seconds | Time from push-to-talk activation to first audio byte playing |
| STT accuracy | > 95% word accuracy | Compare transcripts to intended speech over 50 test utterances |
| Interrupt latency | < 500ms | Time from user speech onset to playback stop |
| Uptime | > 99% during active hours | Background process does not crash or hang over a 7-day period |
| Audio device conflict rate | 0% | No interference with Krisp, Wispr Flow, or other audio apps |

### Non-Goals (Explicit Scope Boundaries)

1. **Multi-user support** -- This is a single-user system for Burke's Mac Studio. No authentication, multi-tenancy, or network access beyond localhost.
2. **Cross-platform** -- macOS only. No Linux or Windows support.
3. **Custom wake word training** -- Uses fixed wake words ("Max", "Hey Max"). No user-configurable wake word training.
4. **Conversation history in the voice loop** -- The voice loop is stateless per utterance. Conversation history and context management is handled by the OpenClaw gateway, not the voice loop.
5. **GUI beyond menu bar** -- No full desktop application. The menu bar icon provides status and basic controls only.
6. **Replacing OpenClaw** -- This does not replace OpenClaw as the AI assistant. It replaces only the broken Talk Mode input/output pipeline.
7. **Multi-language support** -- English only for wake word and STT in v1.
8. **Voice cloning or custom TTS voices** -- Uses existing Sesame TTS voice presets. No voice cloning capability.

---

## 4. Functional Requirements

### 4.1 Wake Word Detection (Phase 2)

> **Note:** Wake word detection is deferred to Phase 2. Phase 1 uses push-to-talk only.

| ID | Requirement | Priority |
|----|-------------|----------|
| WW-1 | Detect wake words "Max" and "Hey Max" from microphone audio stream | Phase 2 |
| WW-2 | Use local speech recognition (Apple Speech framework via PyObjC or Vosk) -- no cloud APIs | Phase 2 |
| WW-3 | Emit an audible chime or visual indicator (menu bar flash) when wake word is detected | Phase 2 |
| WW-4 | Achieve > 95% true positive rate for wake words spoken at normal volume within 2 meters | Phase 2 |
| WW-5 | Achieve < 2% false positive rate during normal ambient noise | Phase 2 |
| WW-6 | Ignore wake words embedded in Sesame TTS playback (echo cancellation / muting during playback) | Phase 2 |
| WW-7 | Configurable sensitivity threshold via config file | Phase 2 |

### 4.2 Push-to-Talk (Global Hotkey)

| ID | Requirement | Priority |
|----|-------------|----------|
| PTT-1 | Support a configurable global hotkey (default: Fn key or Cmd+Shift+Space) to begin speech capture | Must |
| PTT-2 | Hotkey press begins capture immediately, bypassing wake word detection | Must |
| PTT-3 | Release of hotkey (or a configurable timeout) ends capture and triggers transcription | Should |
| PTT-4 | Hotkey must work even when the voice loop app is not in the foreground | Must |
| PTT-5 | Visual feedback (menu bar icon change) while push-to-talk is active | Should |

### 4.3 Speech Capture

| ID | Requirement | Priority |
|----|-------------|----------|
| SC-1 | Capture audio from a configurable microphone device (by name in config file) | Must |
| SC-2 | Default to the system default mic if no device is configured in the config file | Must |
| SC-3 | Support sample rates of 16kHz (Whisper native) or 44.1/48kHz with automatic resampling | Must |
| SC-4 | Detect end-of-speech using voice activity detection (VAD) -- stop capture after 1.5 seconds of silence | Must |
| SC-5 | Maximum capture duration of 30 seconds to prevent runaway recording | Must |
| SC-6 | Buffer audio in memory (no temp files for typical utterances) | Should |
| SC-7 | Log the selected microphone device name and all available audio devices on startup | Must |
| SC-8 | Provide a `--list-devices` CLI flag that prints all available audio input/output devices and exits | Must |

### 4.4 Speech-to-Text (Local Transcription)

| ID | Requirement | Priority |
|----|-------------|----------|
| STT-1 | Transcribe captured audio using MLX Whisper (mlx-community/whisper-large-v3-turbo or equivalent) | Must |
| STT-2 | Run transcription fully locally on Apple Silicon -- no cloud APIs | Must |
| STT-3 | Transcription latency under 1 second for a typical 5-second utterance on M3 Ultra | Must |
| STT-4 | Support English language transcription | Must |
| STT-5 | Discard empty or noise-only transcriptions (e.g., "[silence]", "", whitespace-only) | Must |
| STT-6 | Log transcription text and duration for debugging | Should |

### 4.5 OpenClaw Gateway Integration

> **TRD Task:** The gateway WebSocket protocol (message format, authentication, streaming behavior) must be reverse-engineered from the OpenClaw source code at `/tmp/openclaw`. This should be documented as part of the TRD before implementation begins.

| ID | Requirement | Priority |
|----|-------------|----------|
| GW-1 | Connect to OpenClaw gateway via WebSocket at configurable URL (default: ws://localhost:18789) | Must |
| GW-2 | Send user transcript as a message and receive assistant response text | Must |
| GW-3 | Handle WebSocket connection lifecycle: connect, reconnect on failure, graceful close | Must |
| GW-4 | Reconnect automatically with exponential backoff (1s, 2s, 4s, max 30s) if gateway is unavailable | Must |
| GW-5 | Timeout if no response received within 10 seconds; play an error chime and return to listening | Must |
| GW-6 | Support streaming responses from gateway (process text as it arrives for faster TTS start) | Should |
| GW-7 | Log gateway connection status, message send/receive, and errors | Must |

### 4.6 TTS Generation (Sesame TTS Integration)

| ID | Requirement | Priority |
|----|-------------|----------|
| TTS-1 | Send assistant response text to Sesame TTS server at configurable URL (default: http://localhost:8880) | Must |
| TTS-2 | Use the ElevenLabs streaming endpoint (`/v1/text-to-speech/{voice_id}/stream`) for lowest latency | Must |
| TTS-3 | Use configurable voice preset (default: `conversationalB`) | Must |
| TTS-4 | Request PCM format (pcm_24000) to avoid transcoding overhead | Should |
| TTS-5 | Begin playback as soon as the first audio chunk arrives (streaming playback) | Must |
| TTS-6 | Handle Sesame TTS server unavailability gracefully: log error, play error chime, return to listening | Must |
| TTS-7 | If gateway response is multi-sentence, leverage Sesame's sentence-splitting pipeline for faster first-chunk delivery | Should |

### 4.7 Audio Playback

| ID | Requirement | Priority |
|----|-------------|----------|
| PB-1 | Play generated audio through a configurable output device (or system default) | Must |
| PB-2 | Support streaming playback: begin playing while audio chunks are still being received | Must |
| PB-3 | Use sounddevice or pyaudio for programmatic playback (not afplay subprocess) to enable interrupt | Must |
| PB-4 | Mute or disable wake word listening during playback to prevent self-triggering | Phase 2 |
| PB-5 | Unmute wake word listening after playback completes | Phase 2 |

### 4.8 Interrupt Support

| ID | Requirement | Priority |
|----|-------------|----------|
| INT-1 | Monitor microphone for speech during assistant playback | Must |
| INT-2 | If user speech is detected during playback, stop playback within 500ms | Must |
| INT-3 | After interrupt, immediately enter speech capture mode (the interrupting speech becomes new input) | Must |
| INT-4 | Send interrupt/cancel signal to Sesame TTS server (HTTP disconnect) to stop generation | Should |
| INT-5 | Distinguish between intentional speech and ambient noise during interrupt detection | Must |
| INT-6 | Configurable interrupt sensitivity threshold | Should |

### 4.9 Menu Bar Application

| ID | Requirement | Priority |
|----|-------------|----------|
| MB-1 | Display a menu bar icon indicating current state: idle, listening, processing, speaking | Must |
| MB-2 | Menu bar dropdown shows: current status, mic device name, gateway connection status, quit option | Must |
| MB-3 | Click menu bar icon to activate push-to-talk (primary PTT trigger alongside global hotkey) | Must |
| MB-4 | Support running as a headless daemon without menu bar (for launchd service mode) | Should |

### 4.10 Configuration

| ID | Requirement | Priority |
|----|-------------|----------|
| CFG-1 | All configurable values loaded from a YAML or TOML config file (default: `~/.config/sesame-voice/config.yaml`) | Must |
| CFG-2 | Support environment variable overrides for all config values | Should |
| CFG-3 | Configuration options include: mic device name/ID, speaker device name/ID, gateway WebSocket URL, TTS server URL, TTS voice ID, wake words, push-to-talk hotkey, VAD silence threshold, interrupt sensitivity, log level | Must |
| CFG-4 | Provide a documented default config file in the repository | Must |
| CFG-5 | Validate configuration on startup; log warnings for invalid or missing optional values, exit with clear error for required values | Must |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Target |
|-------------|--------|
| Full round-trip latency (PTT activation to first audio byte playing) | < 4 seconds |
| Speech capture + VAD end detection | < 1.5 seconds after user stops speaking |
| MLX Whisper transcription (5-second utterance) | < 1 second |
| Gateway round-trip (send transcript, receive response) | < 2 seconds (dependent on LLM) |
| TTS first-chunk latency (text to first audio chunk from Sesame) | < 1.5 seconds |
| Playback start from first chunk received | < 50ms |
| Interrupt detection to playback stop | < 500ms |

### 5.2 Reliability

| Requirement | Target |
|-------------|--------|
| Process uptime without crash or hang | > 99% over 7 days |
| Auto-recovery from Sesame TTS server restart | Reconnect within 5 seconds |
| Auto-recovery from gateway disconnect | Reconnect with exponential backoff |
| Memory stability (no unbounded growth) | Stable within 200MB after 1000 conversations |
| Graceful degradation if a component is down | Return to listening with error indicator |

### 5.3 Resource Usage

| Requirement | Target |
|-------------|--------|
| Idle CPU usage (waiting for PTT) | < 1% of one core |
| Peak CPU during transcription | Acceptable (MLX Whisper uses GPU/Neural Engine) |
| Idle memory footprint | < 500MB (including loaded Whisper model) |
| No interference with Sesame TTS server GPU memory | Whisper and CSM-1B must coexist in unified memory |
| Disk usage for logs | < 100MB with log rotation |

### 5.4 Compatibility

| Requirement | Details |
|-------------|---------|
| macOS version | macOS 14+ (Sonoma and later) |
| Python version | 3.10+ |
| Apple Silicon | M1+ required (MLX dependency) |
| Audio coexistence | Must not interfere with Krisp, Wispr Flow, or other audio apps |
| Sesame TTS server version | Compatible with current server (v1.0.0) at localhost:8880 |
| OpenClaw gateway | Compatible with WebSocket protocol on port 18789 |

---

## 6. Acceptance Criteria

### AC-1: Happy Path -- Complete Voice Loop

**Given** the voice loop is running, Sesame TTS server is healthy, and OpenClaw gateway is connected
**When** Burke clicks the menu bar icon (or presses the global hotkey) and says "What is the weather like today?"
**Then:**
- The menu bar icon changes to "listening" state
- Speech is captured until 1.5 seconds of silence
- MLX Whisper transcribes the speech locally
- Transcript is sent to the OpenClaw gateway via WebSocket
- Gateway response is received
- Response text is sent to Sesame TTS server
- Audio playback begins streaming through speakers
- After playback, the system returns to idle state (menu bar shows idle icon)
- Total time from push-to-talk activation to first audio byte playing is under 4 seconds

### AC-2: Push-to-Talk Alternative

**Given** the voice loop is running
**When** Burke presses the configured global hotkey and speaks
**Then:**
- Speech capture begins immediately (no wake word required)
- On hotkey release (or silence timeout), capture ends and transcription begins
- The rest of the pipeline proceeds identically to AC-1

### AC-3: Interrupt During Playback

**Given** the assistant is currently speaking (audio playing)
**When** Burke begins speaking
**Then:**
- Playback stops within 500ms of speech onset
- The voice loop enters capture mode for the new utterance
- The interrupted response is abandoned
- The new utterance is processed through the full pipeline

### AC-4: Mic Device Selection

**Given** the system has multiple audio input devices (built-in mic, studio mic, Krisp virtual device)
**When** the config file specifies `mic_device: "Studio Mic"` (or equivalent device name)
**Then:**
- The voice loop uses only the specified device for all audio capture
- Krisp virtual device is not selected or interfered with
- The selected device name is logged on startup

### AC-5: Sesame TTS Server Unavailable

**Given** the Sesame TTS server at localhost:8880 is not running
**When** the voice loop attempts to generate audio for a gateway response
**Then:**
- An error is logged with the connection failure details
- An error chime plays to indicate the failure
- The voice loop returns to listening state without crashing
- When the TTS server becomes available again, the next request succeeds

### AC-6: Gateway Disconnection and Reconnection

**Given** the OpenClaw gateway WebSocket connection drops
**When** the voice loop detects the disconnection
**Then:**
- An error is logged
- Reconnection attempts begin with exponential backoff (1s, 2s, 4s, ..., max 30s)
- The menu bar icon (if running) shows a disconnected indicator
- Once reconnected, the voice loop resumes normal operation
- Push-to-talk continues to function during disconnection (captures are queued or discarded with user feedback)

### AC-7: No Self-Triggering (Phase 2)

> **Note:** This acceptance criterion applies to Phase 2 when wake word detection is implemented.

**Given** the assistant is speaking through the speakers
**When** the wake word "Max" appears in the assistant's spoken output
**Then:**
- The wake word detector does not trigger
- No false activation occurs
- This is achieved by disabling wake word detection during playback

### AC-8: Background Process Stability

**Given** the voice loop is installed as a launchd service
**When** the system has been running for 7 days with regular voice interactions
**Then:**
- The process has not crashed or hung
- Memory usage remains stable (within 200MB of startup baseline)
- No audio device conflicts have occurred
- Logs show clean operation with no unhandled exceptions

### AC-9: Audio Coexistence

**Given** Krisp and Wispr Flow are both running
**When** the voice loop is actively processing voice conversations
**Then:**
- Krisp noise cancellation continues to function for its own clients
- Wispr Flow dictation continues to work independently
- The voice loop uses its configured mic device without routing conflicts
- No audio glitches, dropouts, or device contention errors

---

## 7. Technical Architecture

### High-Level Component Diagram

```
+------------------------------------------------------------------+
|                        Voice Loop Process                         |
|                                                                   |
|  +------------------+     +-------------------+                   |
|  | Menu Bar PTT     |     | Push-to-Talk      |                   |
|  | (rumps icon      |     | (Global Hotkey)   |                   |
|  |  click)          |     | (Quartz Events)   |                   |
|  |                  |     |                   |                   |
|  +--------+---------+     +--------+----------+                   |
|           |                        |                              |
|           +--------+   +-----------+                              |
|                    v   v                                          |
|           +--------+---+----------+                               |
|           | Speech Capture        |                               |
|           | (sounddevice/pyaudio) |                               |
|           | + VAD (silence detect) |                              |
|           +--------+--------------+                               |
|                    |                                              |
|                    v                                              |
|           +--------+--------------+                               |
|           | MLX Whisper           |                               |
|           | (Local Transcription) |                               |
|           +--------+--------------+                               |
|                    |                                              |
|                    v                                              |
|           +--------+--------------+       +--------------------+  |
|           | Gateway Client        | <---> | OpenClaw Gateway   |  |
|           | (WebSocket)           |       | ws://localhost:18789|  |
|           +--------+--------------+       +--------------------+  |
|                    |                                              |
|                    v                                              |
|           +--------+--------------+       +--------------------+  |
|           | TTS Client            | <---> | Sesame TTS Server  |  |
|           | (httpx streaming)     |       | localhost:8880     |  |
|           +--------+--------------+       +--------------------+  |
|                    |                                              |
|                    v                                              |
|           +--------+--------------+                               |
|           | Audio Playback        |                               |
|           | (sounddevice stream)  |                               |
|           | + Interrupt Monitor   |                               |
|           +--------+--------------+                               |
|                    |                                              |
|                    v                                              |
|           +--------+--------------+                               |
|           | State Manager         |                               |
|           | (idle -> listening ->  |                               |
|           |  processing ->        |                               |
|           |  speaking -> idle)    |                               |
|           +-----------------------+                               |
|                                                                   |
|  +------------------+                                             |
|  | Menu Bar UI      |  (rumps -- PTT trigger + state display)     |
|  | Status + Controls|                                             |
|  +------------------+                                             |
+------------------------------------------------------------------+
```

### State Machine

```
                    +-------+
                    | IDLE  | <---------------------------+
                    | (wait |                             |
                    | for   |                             |
                    | PTT)  |                             |
                    +---+---+                             |
                        |                                 |
            menu bar click                                |
            or hotkey pressed                             |
                        |                                 |
                        v                                 |
                  +-----+------+                          |
                  | LISTENING  |                          |
                  | (capturing |                          |
                  |  speech)   |                          |
                  +-----+------+                          |
                        |                                 |
                silence detected                          |
                (VAD timeout)                              |
                        |                                 |
                        v                                 |
                 +------+-------+                         |
                 | TRANSCRIBING |                         |
                 | (MLX Whisper)|                         |
                 +------+-------+                         |
                        |                                 |
                  transcript ready                        |
                        |                                 |
                        v                                 |
                 +------+-------+                         |
                 | PROCESSING   |                         |
                 | (gateway     |                         |
                 |  round-trip) |                         |
                 +------+-------+                         |
                        |                                 |
                  response received                       |
                        |                                 |
                        v                                 |
                 +------+-------+                         |
                 | SPEAKING     |---- interrupt ----+     |
                 | (TTS + play) |    detected       |     |
                 +------+-------+                   |     |
                        |                           v     |
                  playback complete           +-----+---+ |
                        |                    | LISTENING | |
                        +--------------------+--> (new)  +-+
                                             +-----------+
```

### Key Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Wake word (Phase 2) | Apple Speech framework (PyObjC) | Deferred to Phase 2. Native macOS integration, low latency, no model download. Fallback: Vosk with small model. |
| Speech capture | sounddevice | Pure Python, supports device selection by name, low-latency callbacks, good macOS support |
| VAD | WebRTC VAD (py-webrtcvad) or energy-based | Lightweight, well-tested, no GPU needed |
| STT | MLX Whisper (mlx-community/whisper-large-v3-turbo) | Local, fast on Apple Silicon, high accuracy, already available on the machine |
| Gateway comm | websockets library | Async, lightweight, handles reconnection cleanly |
| TTS client | httpx (async, streaming) | Already a project dependency, supports streaming responses |
| Audio playback | sounddevice (OutputStream) | Supports streaming playback, device selection, can be stopped instantly for interrupt |
| Menu bar | rumps | Lightweight macOS menu bar framework, pure Python |
| Configuration | YAML (PyYAML) or TOML (tomllib, stdlib in 3.11+) | Human-readable, simple, widely understood |
| Process management | launchd (plist) | Native macOS service management, auto-restart, survives reboots |

### Data Flow (Latency Budget)

```
PTT activated                t=0.0s
Speech capture begins        t=0.05s
User speaks for ~3s          t=0.05s - 3.05s
Silence detected (VAD)       t=4.55s  (3.05s speech + 1.5s silence)
Whisper transcription        t=5.25s  (0.7s for ~3s audio on M3 Ultra)
Gateway send + LLM response  t=7.25s  (2.0s for gateway round-trip)
TTS first chunk received     t=8.25s  (1.0s for first sentence TTS)
Audio playback begins        t=8.25s  (immediate on chunk receipt)
```

Note: The 4-second target is measured from push-to-talk activation to first audio playing, assuming a short (1-2 second) user utterance. The budget for a 2-second utterance:

```
PTT activation + capture      0.05s
User speech                   2.0s
VAD silence detection         1.5s  (can be tuned down to 1.0s)
Whisper transcription         0.5s
Gateway round-trip            1.5s  (depends on LLM)
TTS first chunk               1.0s
                              -----
Total from PTT activation:   ~6.55s (from PTT)
Total from end of speech:     ~4.5s (from silence detection)
```

The 4-second target is achievable by:
- Removing wake word detection overhead (PTT is instant)
- Reducing VAD silence threshold to 1.0s for the initial release
- Using Whisper turbo model for faster transcription
- Leveraging Sesame TTS streaming for first-chunk delivery
- Optimizing gateway protocol for minimal overhead

---

## 8. Risks and Mitigations

### R1: MLX Whisper + CSM-1B Memory Contention

**Risk:** Both MLX Whisper and CSM-1B run on the same Apple Silicon GPU/Neural Engine. Loading both models simultaneously could exceed unified memory or cause contention during concurrent inference.

**Likelihood:** Medium
**Impact:** High (crashes or severe latency)

**Mitigation:**
- M3 Ultra has 256GB unified memory -- both models fit comfortably (Whisper ~3GB + CSM-1B ~4GB)
- The voice loop pipeline is sequential: Whisper runs, then TTS runs. They do not need to infer simultaneously.
- Monitor peak memory usage via `mx.get_peak_memory()` and log it
- If contention is observed, implement explicit model unloading between steps (unlikely to be needed)

### R2: Wake Word False Positives from TV / Music / Conversation (Phase 2)

> **Note:** This risk applies to Phase 2 when wake word detection is implemented. Not applicable to Phase 1 (push-to-talk only).

**Risk:** The word "Max" is common in everyday English, appearing in names, brand names ("Max" streaming service), and conversation. This could cause frequent false activations.

**Likelihood:** High
**Impact:** Medium (annoying but not breaking)

**Mitigation:**
- Prefer "Hey Max" (two-word phrase) as the primary wake word for higher specificity
- Implement confidence threshold tuning in config
- Use push-to-talk as the primary mode if false positives are too frequent
- Consider adding a confirmation sound + short timeout: if no speech follows within 2 seconds, return to idle
- Disable wake word during playback (already required for echo prevention)

### R3: Apple Speech Framework API Instability (Phase 2)

> **Note:** This risk applies to Phase 2 when wake word detection is implemented. Not applicable to Phase 1 (push-to-talk only).

**Risk:** The Apple Speech framework accessed via PyObjC may have undocumented behavior, version-specific bugs, or require entitlements that complicate deployment.

**Likelihood:** Medium
**Impact:** Medium (blocks wake word feature)

**Mitigation:**
- Vosk with a small English model is the fallback. Vosk is well-tested, MIT-licensed, and works offline.
- Abstract wake word detection behind an interface so implementations can be swapped
- Test Apple Speech approach in a prototype before committing to it

### R4: OpenClaw Gateway WebSocket Protocol Changes

**Risk:** The OpenClaw gateway WebSocket protocol on port 18789 may change in future OpenClaw updates, breaking the voice loop integration.

**Likelihood:** Low
**Impact:** High (voice loop stops working)

**Mitigation:**
- Document the current WebSocket message format in the codebase
- Version the gateway client implementation
- Monitor OpenClaw releases for protocol changes
- Gateway client is a small, isolated module -- easy to update

### R5: Audio Device Enumeration Fragility

**Risk:** macOS audio device names can change across OS updates, or when devices are plugged/unplugged. Hardcoded device names in config may stop matching.

**Likelihood:** Medium
**Impact:** Medium (falls back to wrong device)

**Mitigation:**
- Support matching by partial name (substring match) in addition to exact match
- Log all available devices on startup for easy debugging
- Fall back to system default with a warning if configured device is not found
- Provide a CLI command to list available audio devices: `python -m voice_loop --list-devices`

### R6: Latency Budget Exceeded

**Risk:** The 4-second round-trip target may not be achievable if the LLM response time from the OpenClaw gateway is slow, or if component latencies stack up worse than estimated.

**Likelihood:** Medium
**Impact:** Medium (functional but feels sluggish)

**Mitigation:**
- Log timing for every pipeline stage to identify bottlenecks
- Reduce VAD silence threshold (1.0s instead of 1.5s) to shave time
- Use Whisper turbo model for faster transcription
- Leverage TTS streaming to overlap generation with playback
- Consider streaming gateway responses to TTS (start TTS before full response is received) as a Phase 2 optimization

---

## 9. Phasing

### Phase 1: MVP -- Push-to-Talk + Menu Bar (Target: 1 week)

**Goal:** End-to-end voice loop working from push-to-talk (menu bar click or global hotkey) through to audio playback.

**Scope:**
- Menu bar application (rumps) showing state: idle, listening, processing, speaking
- Menu bar icon click acts as push-to-talk trigger
- Menu bar dropdown: current status, mic device, gateway connection status, quit
- Push-to-talk via global hotkey (alternative to menu bar click)
- Speech capture with configurable mic device (by name in config, system default if unset)
- `--list-devices` CLI flag to enumerate available audio devices
- Log all available audio devices on startup
- VAD-based end-of-speech detection
- MLX Whisper local transcription
- OpenClaw gateway WebSocket integration (basic: connect, send, receive)
- Sesame TTS streaming integration (via ElevenLabs streaming endpoint)
- Audio playback via sounddevice
- Basic interrupt support (stop playback on hotkey press)
- YAML configuration file
- CLI entry point (`python -m voice_loop`)
- Logging with timing for all pipeline stages

**Deliverable:** A menu bar app with push-to-talk that Burke can click, speak, and hear the response through Sesame TTS.

### Phase 2: Wake Word + Polish (Target: 1 week)

**Goal:** Add wake word detection for hands-free activation and improve the conversational experience.

**Scope:**
- Wake word detection ("Max", "Hey Max") via Apple Speech or Vosk (WW-1 through WW-7)
- Automatic muting of wake word during playback (echo prevention / no self-triggering)
- Speech-based interrupt detection (stop playback when user speaks, not just hotkey)
- Gateway reconnection with exponential backoff
- Sesame TTS server health checking and graceful degradation
- Audio feedback (chimes for wake word detected, error, etc.)
- Refined VAD tuning for optimal silence threshold

**Deliverable:** Hands-free voice loop that responds to "Max" or "Hey Max" with full interrupt support, in addition to push-to-talk.

### Phase 3: launchd Service + Polish (Target: 3-5 days)

**Goal:** Production-quality background operation that survives reboots.

**Scope:**
- launchd plist for running as a service
- Log rotation configuration
- Configuration validation with helpful error messages
- Documentation (README for the voice loop module)

**Deliverable:** A polished background service that Burke installs once and forgets about.

### Phase 4: Optimization (Ongoing)

**Goal:** Tune latency and reliability based on real-world usage.

**Scope:**
- Gateway streaming response integration (start TTS before full response arrives)
- Whisper model size tuning (turbo vs. large tradeoffs)
- VAD parameter optimization based on Burke's environment acoustics
- Conversation context hints (send previous exchange context to improve gateway responses)
- Metrics dashboard (optional: log pipeline timings to a local SQLite for analysis)

**Deliverable:** Sub-3-second round-trip for typical exchanges.

---

## Appendix A: Dependency Summary

| Package | Purpose | Phase | License |
|---------|---------|-------|---------|
| mlx-whisper | Local speech-to-text | 1 | MIT |
| sounddevice | Mic capture + playback | 1 | MIT |
| py-webrtcvad | Voice activity detection | 1 | MIT |
| websockets | Gateway WebSocket client | 1 | BSD |
| httpx | TTS server HTTP client (async streaming) | 1 | BSD |
| rumps | macOS menu bar app (PTT trigger + state display) | 1 | MIT |
| PyYAML | Configuration file parsing | 1 | MIT |
| pyobjc-framework-Speech | Apple Speech framework (wake word) | 2 | MIT |
| vosk | Fallback wake word detection | 2 | Apache 2.0 |

All Phase 1 dependencies are MIT or BSD licensed. All inference runs locally. No cloud API calls.

## Appendix B: Configuration File Example

```yaml
# ~/.config/sesame-voice/config.yaml

# Audio devices
mic_device: "MacBook Pro Microphone"  # or "Studio Display Microphone", etc.
# speaker_device: null  # null = system default

# Wake word (Phase 2 -- not used in v1)
# wake_words:
#   - "Max"
#   - "Hey Max"
# wake_word_engine: "apple_speech"  # or "vosk"
# wake_word_confidence: 0.7

# Push-to-talk (primary activation method in v1)
push_to_talk_hotkey: "cmd+shift+space"

# Speech capture
vad_silence_threshold_ms: 1500
max_capture_duration_s: 30

# Transcription
whisper_model: "mlx-community/whisper-large-v3-turbo"

# Gateway
gateway_url: "ws://localhost:18789"
gateway_timeout_s: 10

# TTS
tts_url: "http://localhost:8880"
tts_voice: "conversationalB"
tts_format: "pcm_24000"

# Interrupt
interrupt_sensitivity: 0.5  # 0.0 (never interrupt) to 1.0 (very sensitive)

# Logging
log_level: "info"
log_file: "~/.cache/sesame-voice/voice-loop.log"

# Process
run_as_menu_bar: true  # false = headless daemon mode
```

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-20 | Burke | Initial draft |
| 1.1 | 2026-02-21 | Burke | Refined based on user interview: deferred wake word to Phase 2, added menu bar with PTT to Phase 1, gateway protocol to be reverse-engineered from OpenClaw source, audio device discovery at runtime with `--list-devices` flag |
