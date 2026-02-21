"""Tests for the voice loop state machine."""

import pytest

from voice_loop.state import StateMachine, VoiceLoopState, TRANSITIONS


class TestInitialState:
    """Verify initial state is IDLE."""

    async def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state == VoiceLoopState.IDLE


class TestValidTransitions:
    """Verify all valid transitions succeed."""

    @pytest.mark.parametrize(
        "from_state,to_state",
        [
            (VoiceLoopState.IDLE, VoiceLoopState.LISTENING),
            (VoiceLoopState.LISTENING, VoiceLoopState.TRANSCRIBING),
            (VoiceLoopState.LISTENING, VoiceLoopState.IDLE),
            (VoiceLoopState.TRANSCRIBING, VoiceLoopState.PROCESSING),
            (VoiceLoopState.TRANSCRIBING, VoiceLoopState.IDLE),
            (VoiceLoopState.PROCESSING, VoiceLoopState.SPEAKING),
            (VoiceLoopState.PROCESSING, VoiceLoopState.ERROR),
            (VoiceLoopState.SPEAKING, VoiceLoopState.IDLE),
            (VoiceLoopState.SPEAKING, VoiceLoopState.LISTENING),
            (VoiceLoopState.ERROR, VoiceLoopState.IDLE),
        ],
    )
    async def test_valid_transition(self, from_state, to_state):
        sm = StateMachine()
        # Navigate to from_state first by walking a valid path
        path = _path_to_state(from_state)
        for state in path:
            await sm.transition(state)
        assert sm.state == from_state
        await sm.transition(to_state)
        assert sm.state == to_state


class TestInvalidTransitions:
    """Verify invalid transitions raise ValueError."""

    @pytest.mark.parametrize(
        "from_state,to_state",
        [
            (VoiceLoopState.IDLE, VoiceLoopState.SPEAKING),
            (VoiceLoopState.IDLE, VoiceLoopState.PROCESSING),
            (VoiceLoopState.IDLE, VoiceLoopState.ERROR),
            (VoiceLoopState.LISTENING, VoiceLoopState.SPEAKING),
            (VoiceLoopState.TRANSCRIBING, VoiceLoopState.SPEAKING),
            (VoiceLoopState.PROCESSING, VoiceLoopState.IDLE),
            (VoiceLoopState.ERROR, VoiceLoopState.SPEAKING),
        ],
    )
    async def test_invalid_transition_raises(self, from_state, to_state):
        sm = StateMachine()
        path = _path_to_state(from_state)
        for state in path:
            await sm.transition(state)
        with pytest.raises(ValueError, match="Invalid transition"):
            await sm.transition(to_state)


class TestCallbacks:
    """Verify callbacks fire on transitions."""

    async def test_callback_fires_on_transition(self):
        sm = StateMachine()
        transitions_log = []
        sm.on_transition(lambda old, new: transitions_log.append((old, new)))

        await sm.transition(VoiceLoopState.LISTENING)
        assert transitions_log == [
            (VoiceLoopState.IDLE, VoiceLoopState.LISTENING),
        ]

    async def test_multiple_callbacks_all_fire(self):
        sm = StateMachine()
        log1 = []
        log2 = []
        sm.on_transition(lambda old, new: log1.append((old, new)))
        sm.on_transition(lambda old, new: log2.append((old, new)))

        await sm.transition(VoiceLoopState.LISTENING)
        assert len(log1) == 1
        assert len(log2) == 1

    async def test_callback_receives_correct_states(self):
        sm = StateMachine()
        transitions_log = []
        sm.on_transition(lambda old, new: transitions_log.append((old, new)))

        await sm.transition(VoiceLoopState.LISTENING)
        await sm.transition(VoiceLoopState.TRANSCRIBING)
        await sm.transition(VoiceLoopState.PROCESSING)

        assert transitions_log == [
            (VoiceLoopState.IDLE, VoiceLoopState.LISTENING),
            (VoiceLoopState.LISTENING, VoiceLoopState.TRANSCRIBING),
            (VoiceLoopState.TRANSCRIBING, VoiceLoopState.PROCESSING),
        ]


class TestTransitionsDict:
    """Verify the TRANSITIONS dict is complete."""

    def test_all_states_have_transitions(self):
        for state in VoiceLoopState:
            assert state in TRANSITIONS, f"Missing transitions for {state}"


def _path_to_state(target: VoiceLoopState) -> list[VoiceLoopState]:
    """Return a valid transition path from IDLE to the target state."""
    paths = {
        VoiceLoopState.IDLE: [],
        VoiceLoopState.LISTENING: [VoiceLoopState.LISTENING],
        VoiceLoopState.TRANSCRIBING: [
            VoiceLoopState.LISTENING,
            VoiceLoopState.TRANSCRIBING,
        ],
        VoiceLoopState.PROCESSING: [
            VoiceLoopState.LISTENING,
            VoiceLoopState.TRANSCRIBING,
            VoiceLoopState.PROCESSING,
        ],
        VoiceLoopState.SPEAKING: [
            VoiceLoopState.LISTENING,
            VoiceLoopState.TRANSCRIBING,
            VoiceLoopState.PROCESSING,
            VoiceLoopState.SPEAKING,
        ],
        VoiceLoopState.ERROR: [
            VoiceLoopState.LISTENING,
            VoiceLoopState.TRANSCRIBING,
            VoiceLoopState.PROCESSING,
            VoiceLoopState.ERROR,
        ],
    }
    return paths[target]
