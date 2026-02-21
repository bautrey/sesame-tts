"""Enum-based state machine with validated transitions and observer callbacks."""

import asyncio
import enum
import logging
from typing import Callable

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
    VoiceLoopState.IDLE: {VoiceLoopState.LISTENING},
    VoiceLoopState.LISTENING: {VoiceLoopState.TRANSCRIBING, VoiceLoopState.IDLE},
    VoiceLoopState.TRANSCRIBING: {VoiceLoopState.PROCESSING, VoiceLoopState.IDLE},
    VoiceLoopState.PROCESSING: {VoiceLoopState.SPEAKING, VoiceLoopState.ERROR},
    VoiceLoopState.SPEAKING: {VoiceLoopState.IDLE, VoiceLoopState.LISTENING},
    VoiceLoopState.ERROR: {VoiceLoopState.IDLE},
}

StateCallback = Callable[[VoiceLoopState, VoiceLoopState], None]


class StateMachine:
    """Async state machine with transition validation and observer callbacks."""

    def __init__(self) -> None:
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
                raise ValueError(
                    f"Invalid transition: {old.value} -> {new_state.value}"
                )
            self._state = new_state
            logger.info("State: %s -> %s", old.value, new_state.value)
            for cb in self._callbacks:
                cb(old, new_state)
